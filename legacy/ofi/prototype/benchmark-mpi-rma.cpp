/*
 * benchmark-mpi-rma.cpp - GPU-to-GPU MPI RMA Latency Benchmark with GPU-Direct
 *
 * This program measures MPI RMA (Remote Memory Access) latency between two
 * processes using AMD GPU memory with GPU-Direct support. It uses MPI_Put
 * for one-sided data transfer.
 *
 * Key Features:
 * - Uses AMD GPU memory (HIP/ROCm) with MPI GPU-Direct
 * - Uses MPI RMA (MPI_Put) for one-sided communication
 * - Tests multiple message sizes: 16KB to 1MB
 * - Each size tested 10 times to compute average latency
 * - Requires MPICH_GPU_SUPPORT_ENABLED=1 environment variable
 * - Each process uses GPU 0
 *
 * MPI RMA Concepts:
 * - MPI_Win_create: Create window to expose GPU memory
 * - MPI_Put: One-sided write to remote GPU memory
 * - MPI_Win_fence: Synchronization for active target mode
 *
 * Environment Requirements:
 * - export MPICH_GPU_SUPPORT_ENABLED=1
 *
 * Compile:
 * - CC -std=c++11 -o benchmark-mpi-rma benchmark-mpi-rma.cpp -L/opt/rocm/lib -lamdhip64
 *
 * Run:
 * - srun -N 2 -n 2 ./benchmark-mpi-rma
 */

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK_HIP(x, msg) do { \
    hipError_t err = (x); \
    if (err != hipSuccess) { \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        fprintf(stderr, "Rank %d: %s failed: %s (%d)\n", rank, msg, hipGetErrorString(err), err); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CHECK_MPI(x, msg) do { \
    int err = (x); \
    if (err != MPI_SUCCESS) { \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        char error_string[MPI_MAX_ERROR_STRING]; \
        int length; \
        MPI_Error_string(err, error_string, &length); \
        fprintf(stderr, "Rank %d: %s failed: %s (%d)\n", rank, msg, error_string, err); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Test parameters
#define NUM_ITERATIONS 10

const size_t test_sizes[] = {
    16*1024,            // 16KB
    32*1024,            // 32KB
    64*1024,            // 64KB
    128*1024,           // 128KB
    256*1024,           // 256KB
    512*1024,           // 512KB
    1024*1024           // 1MB
};
const int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

// =============================================================================
// Utility Functions
// =============================================================================

// Get current time in microseconds
static double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// Format size for display
static const char* format_size(size_t size, char* buf) {
    if (size < 1024) {
        snprintf(buf, 32, "%zuB", size);
    } else if (size < 1024*1024) {
        snprintf(buf, 32, "%zuKB", size/1024);
    } else {
        snprintf(buf, 32, "%zuMB", size/(1024*1024));
    }
    return buf;
}

// =============================================================================
// Main Program
// =============================================================================

int main(int argc, char** argv) {
    int rank, size;
    int provided;

    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This benchmark requires exactly 2 MPI processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("=== MPI RMA (MPI_Put) GPU-to-GPU Latency Benchmark ===\n");
        printf("MPI GPU-Direct support: %s\n",
               getenv("MPICH_GPU_SUPPORT_ENABLED") ? "Enabled" : "Disabled (Set MPICH_GPU_SUPPORT_ENABLED=1)");
    }

    // =============================================================================
    // GPU Initialization
    // =============================================================================
    CHECK_HIP(hipSetDevice(0), "hipSetDevice");

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
    printf("Rank %d: Using GPU: %s\n", rank, prop.name);

    // =============================================================================
    // Allocate GPU Memory
    // =============================================================================
    size_t max_buf_size = test_sizes[num_test_sizes - 1];

    char *d_local_buf = NULL;   // Local GPU buffer for sending
    char *d_remote_buf = NULL;  // Remote GPU buffer (exposed via MPI_Win)

    CHECK_HIP(hipMalloc(&d_local_buf, max_buf_size), "hipMalloc(local)");
    CHECK_HIP(hipMalloc(&d_remote_buf, max_buf_size), "hipMalloc(remote)");

    // Initialize GPU buffers
    char *h_temp = (char *)malloc(max_buf_size);
    memset(h_temp, 0xAA, max_buf_size);
    CHECK_HIP(hipMemcpy(d_local_buf, h_temp, max_buf_size, hipMemcpyHostToDevice), "hipMemcpy H2D local");

    memset(h_temp, 0, max_buf_size);
    CHECK_HIP(hipMemcpy(d_remote_buf, h_temp, max_buf_size, hipMemcpyHostToDevice), "hipMemcpy H2D remote");

    CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");
    free(h_temp);

    if (rank == 0) {
        printf("\nGPU buffers allocated (%zu bytes)\n", max_buf_size);
        printf("Local buffer:  %p\n", d_local_buf);
        printf("Remote buffer: %p\n\n", d_remote_buf);
    }

    // =============================================================================
    // Create MPI Window for RMA
    // =============================================================================
    MPI_Win win;
    MPI_Info info;
    MPI_Info_create(&info);

    // Hint that this is GPU memory
    MPI_Info_set(info, "which_accumulate_ops", "same_op");

    // Create window exposing remote_buf for RMA access
    CHECK_MPI(MPI_Win_create(d_remote_buf, max_buf_size, 1, info, MPI_COMM_WORLD, &win),
              "MPI_Win_create");

    MPI_Info_free(&info);

    if (rank == 0) {
        printf("MPI Window created for RMA operations\n\n");
    }

    // Synchronization before benchmark
    MPI_Barrier(MPI_COMM_WORLD);

    // =============================================================================
    // Latency Benchmark
    // =============================================================================
    if (rank == 0) {
        printf("%-10s %10s %10s\n", "Size", "Iterations", "Avg Latency");
        printf("%-10s %10s %10s\n", "", "", "(us)");
        printf("---------------------------------------------\n");
    }

    int peer_rank = (rank == 0) ? 1 : 0;

    for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
        size_t current_size = test_sizes[size_idx];
        double total_time = 0.0;

        // Synchronization before each size test
        MPI_Barrier(MPI_COMM_WORLD);

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            if (rank == 0) {
                // ===== Rank 0: Sender using MPI_Put =====
                double start_time = get_time_us();

                // Start RMA access epoch
                CHECK_MPI(MPI_Win_fence(0, win), "MPI_Win_fence(start)");

                // Perform one-sided Put operation
                // MPI_Put(origin_addr, origin_count, origin_datatype, target_rank,
                //         target_disp, target_count, target_datatype, win)
                CHECK_MPI(MPI_Put(d_local_buf, current_size, MPI_BYTE,
                                 peer_rank, 0, current_size, MPI_BYTE, win),
                         "MPI_Put");

                // Complete RMA access epoch (ensures completion)
                CHECK_MPI(MPI_Win_fence(0, win), "MPI_Win_fence(complete)");

                double end_time = get_time_us();
                total_time += (end_time - start_time);

            } else {
                // ===== Rank 1: Receiver (passive) =====
                // Just participate in fence synchronization
                CHECK_MPI(MPI_Win_fence(0, win), "MPI_Win_fence(start)");
                CHECK_MPI(MPI_Win_fence(0, win), "MPI_Win_fence(complete)");
            }
        }

        // Print results
        if (rank == 0) {
            double avg_latency = total_time / NUM_ITERATIONS;
            char size_str[32];
            format_size(current_size, size_str);
            printf("%-10s %10d %10.2f\n", size_str, NUM_ITERATIONS, avg_latency);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("---------------------------------------------\n");
    }

    // =============================================================================
    // Cleanup
    // =============================================================================
    MPI_Win_free(&win);

    CHECK_HIP(hipFree(d_local_buf), "hipFree(local)");
    CHECK_HIP(hipFree(d_remote_buf), "hipFree(remote)");

    MPI_Finalize();

    if (rank == 0) {
        printf("\nBenchmark completed successfully.\n");
    }

    return 0;
}
