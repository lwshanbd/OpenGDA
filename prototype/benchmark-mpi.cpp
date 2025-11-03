/*
 * benchmark-mpi.cpp - GPU-to-GPU MPI Latency Benchmark with GPU-Direct
 *
 * This program measures MPI point-to-point latency between two processes using
 * AMD GPU memory with GPU-Direct support. It tests various message sizes and
 * reports average latency for each.
 *
 * Key Features:
 * - Uses AMD GPU memory (HIP/ROCm) with MPI GPU-Direct
 * - Tests multiple message sizes: 1B to 128MB (14 different sizes)
 * - Each size tested 10 times to compute average latency
 * - Requires MPICH_GPU_SUPPORT_ENABLED=1 environment variable
 * - Each process uses GPU 0
 *
 * Environment Requirements:
 * - export MPICH_GPU_SUPPORT_ENABLED=1
 *
 * Compile:
 * - CC -std=c++11 -o benchmark-mpi benchmark-mpi.cpp -L/opt/rocm/lib -lamdhip64
 *
 * Run:
 * - srun -N 2 -n 2 ./benchmark-mpi
 */

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_HIP(x, msg) do { \
    hipError_t err = (x); \
    if (err != hipSuccess) { \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        fprintf(stderr, "Rank %d: %s failed: %s (%d)\n", rank, msg, hipGetErrorString(err), err); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Test parameters
#define NUM_ITERATIONS 10
#define MPI_TAG 100

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

    // =============================================================================
    // GPU Initialization
    // =============================================================================
    CHECK_HIP(hipSetDevice(0), "hipSetDevice");

    int device;
    CHECK_HIP(hipGetDevice(&device), "hipGetDevice");

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");

    if (rank == 0) {
        printf("Rank %d: Using GPU %d: %s\n", rank, device, prop.name);

        // Check GPU-Direct environment variable
        char *gpu_support = getenv("MPICH_GPU_SUPPORT_ENABLED");
        if (gpu_support && atoi(gpu_support) == 1) {
            printf("GPU-Direct: ENABLED (MPICH_GPU_SUPPORT_ENABLED=1)\n");
        } else {
            printf("WARNING: MPICH_GPU_SUPPORT_ENABLED not set to 1\n");
            printf("         GPU-Direct may not be active!\n");
        }
        printf("\n");
    }

    // =============================================================================
    // Allocate GPU Memory
    // =============================================================================
    size_t max_buf_size = test_sizes[num_test_sizes - 1];

    char *d_send_buf = NULL;
    char *d_recv_buf = NULL;

    CHECK_HIP(hipMalloc(&d_send_buf, max_buf_size), "hipMalloc(send)");
    CHECK_HIP(hipMalloc(&d_recv_buf, max_buf_size), "hipMalloc(recv)");

    // Initialize GPU buffers
    char *h_temp = (char *)malloc(max_buf_size);
    memset(h_temp, 0xAB, max_buf_size);
    CHECK_HIP(hipMemcpy(d_send_buf, h_temp, max_buf_size, hipMemcpyHostToDevice), "hipMemcpy H2D send");

    memset(h_temp, 0, max_buf_size);
    CHECK_HIP(hipMemcpy(d_recv_buf, h_temp, max_buf_size, hipMemcpyHostToDevice), "hipMemcpy H2D recv");

    CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");

    if (rank == 0) {
        printf("GPU buffers allocated (%zu bytes)\n", max_buf_size);
        printf("Send buffer: %p\n", d_send_buf);
        printf("Recv buffer: %p\n\n", d_recv_buf);
    }

    // =============================================================================
    // Latency Benchmark
    // =============================================================================
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("========================================\n");
        printf("GPU-to-GPU MPI Latency Benchmark\n");
        printf("========================================\n");
        printf("%-10s %10s %10s\n", "Size", "Iterations", "Avg Latency");
        printf("%-10s %10s %10s\n", "", "", "(us)");
        printf("----------------------------------------\n");
    }

    // Determine peer rank
    int peer = (rank == 0) ? 1 : 0;

    // Loop through different message sizes
    for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
        size_t current_size = test_sizes[size_idx];
        double total_time = 0.0;
        int successful_iterations = 0;

        // Warmup iteration (not counted)
        if (rank == 0) {
            MPI_Send(d_send_buf, current_size, MPI_BYTE, peer, MPI_TAG, MPI_COMM_WORLD);
        } else {
            MPI_Recv(d_recv_buf, current_size, MPI_BYTE, peer, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Timed iterations
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            MPI_Barrier(MPI_COMM_WORLD);

            double start_time = 0.0;
            if (rank == 0) {
                start_time = MPI_Wtime();
            }

            // Perform send/recv
            if (rank == 0) {
                MPI_Send(d_send_buf, current_size, MPI_BYTE, peer, MPI_TAG, MPI_COMM_WORLD);
            } else {
                MPI_Recv(d_recv_buf, current_size, MPI_BYTE, peer, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            if (rank == 0) {
                double end_time = MPI_Wtime();
                double elapsed = (end_time - start_time) * 1e6; // Convert to microseconds
                total_time += elapsed;
                successful_iterations++;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Print results for this size
        if (rank == 0 && successful_iterations > 0) {
            double avg_latency = total_time / successful_iterations;
            char size_str[32];
            format_size(current_size, size_str);
            printf("%-10s %10d %10.2f\n", size_str, successful_iterations, avg_latency);
            fflush(stdout);
        }
    }

    if (rank == 0) {
        printf("========================================\n\n");
    }

    // =============================================================================
    // Cleanup
    // =============================================================================
    CHECK_HIP(hipFree(d_send_buf), "hipFree(send)");
    CHECK_HIP(hipFree(d_recv_buf), "hipFree(recv)");
    free(h_temp);

    MPI_Finalize();
    return 0;
}
