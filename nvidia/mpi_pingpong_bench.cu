/**
 * mpi_pingpong_bench.cu - MPI Ping-Pong Benchmark for Comparison
 *
 * This benchmark uses standard MPI_Send/MPI_Recv to measure baseline
 * communication latency and bandwidth for comparison with GPU-triggered RDMA.
 *
 * Tests:
 *   1. MPI with host memory (standard)
 *   2. MPI with GPU memory (CUDA-aware MPI)
 *   3. GPU-triggered RDMA (DevX) - from gpu_pingpong_bench
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <cuda_runtime.h>
#include <mpi.h>

using namespace std::chrono;

// Test configurations - same as GPU benchmark
constexpr int WARMUP_ITERS = 50;
constexpr int TEST_ITERS = 1000;
constexpr size_t MSG_SIZES[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    128*1024, 256*1024, 512*1024, 1024*1024,
    2*1024*1024, 4*1024*1024, 8*1024*1024, 16*1024*1024
};
constexpr int NUM_SIZES = sizeof(MSG_SIZES) / sizeof(MSG_SIZES[0]);

//==============================================================================
// Helper functions
//==============================================================================

void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

const char* format_size(size_t size, char* buf) {
    if (size < 1024) {
        snprintf(buf, 32, "%zuB", size);
    } else if (size < 1024 * 1024) {
        snprintf(buf, 32, "%zuKB", size / 1024);
    } else {
        snprintf(buf, 32, "%zuMB", size / (1024 * 1024));
    }
    return buf;
}

//==============================================================================
// MPI Ping-Pong with Host Memory
//==============================================================================

void run_mpi_host_pingpong(int rank, int peer, void* send_buf, void* recv_buf,
                           size_t msg_size, int iterations,
                           double* avg_rtt, double* min_rtt, double* max_rtt) {
    MPI_Status status;
    double total_time = 0.0;
    double local_min = 1e9;
    double local_max = 0.0;

    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();

        if (rank == 0) {
            MPI_Send(send_buf, msg_size, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
            MPI_Recv(recv_buf, msg_size, MPI_BYTE, peer, 0, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(recv_buf, msg_size, MPI_BYTE, peer, 0, MPI_COMM_WORLD, &status);
            MPI_Send(send_buf, msg_size, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        }

        auto end = high_resolution_clock::now();
        double iter_time = duration_cast<nanoseconds>(end - start).count() / 1000.0;

        total_time += iter_time;
        if (iter_time < local_min) local_min = iter_time;
        if (iter_time > local_max) local_max = iter_time;
    }

    *avg_rtt = total_time / iterations;
    *min_rtt = local_min;
    *max_rtt = local_max;
}

//==============================================================================
// MPI Ping-Pong with GPU Memory (CUDA-aware MPI)
//==============================================================================

void run_mpi_gpu_pingpong(int rank, int peer, void* d_send_buf, void* d_recv_buf,
                          size_t msg_size, int iterations,
                          double* avg_rtt, double* min_rtt, double* max_rtt) {
    MPI_Status status;
    double total_time = 0.0;
    double local_min = 1e9;
    double local_max = 0.0;

    // Ensure GPU operations are complete before timing
    cudaDeviceSynchronize();

    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();

        if (rank == 0) {
            MPI_Send(d_send_buf, msg_size, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
            MPI_Recv(d_recv_buf, msg_size, MPI_BYTE, peer, 0, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(d_recv_buf, msg_size, MPI_BYTE, peer, 0, MPI_COMM_WORLD, &status);
            MPI_Send(d_send_buf, msg_size, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        }

        auto end = high_resolution_clock::now();
        double iter_time = duration_cast<nanoseconds>(end - start).count() / 1000.0;

        total_time += iter_time;
        if (iter_time < local_min) local_min = iter_time;
        if (iter_time > local_max) local_max = iter_time;
    }

    *avg_rtt = total_time / iterations;
    *min_rtt = local_min;
    *max_rtt = local_max;
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires exactly 2 ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    int peer = (rank == 0) ? 1 : 0;

    // Get local rank for GPU selection
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    // Initialize CUDA
    int num_gpus;
    cuda_check(cudaGetDeviceCount(&num_gpus), "get device count");
    int gpu_id = local_rank % num_gpus;
    cuda_check(cudaSetDevice(gpu_id), "set device");

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, gpu_id);

    // Allocate buffers
    size_t max_size = 16 * 1024 * 1024;  // 16MB

    // Host buffers
    void* h_send_buf = malloc(max_size);
    void* h_recv_buf = malloc(max_size);
    memset(h_send_buf, rank + 1, max_size);
    memset(h_recv_buf, 0, max_size);

    // GPU buffers
    void* d_send_buf = nullptr;
    void* d_recv_buf = nullptr;
    cuda_check(cudaMalloc(&d_send_buf, max_size), "alloc gpu send");
    cuda_check(cudaMalloc(&d_recv_buf, max_size), "alloc gpu recv");
    cuda_check(cudaMemset(d_send_buf, rank + 1, max_size), "memset gpu send");
    cuda_check(cudaMemset(d_recv_buf, 0, max_size), "memset gpu recv");

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Print header
    //==========================================================================

    if (rank == 0) {
        printf("=======================================================\n");
        printf("   MPI Ping-Pong Benchmark (Comparison)\n");
        printf("=======================================================\n");
        printf("GPU: %s\n", props.name);
        printf("Warmup: %d iterations\n", WARMUP_ITERS);
        printf("Test: %d iterations per message size\n", TEST_ITERS);
        printf("-------------------------------------------------------\n\n");
        fflush(stdout);
    }

    //==========================================================================
    // Test 1: MPI with Host Memory
    //==========================================================================

    if (rank == 0) {
        printf("=== MPI Ping-Pong (Host Memory) ===\n");
        printf("%-8s %12s %12s %12s %12s %12s\n",
               "Size", "RTT (us)", "Half-RTT", "Min RTT", "Max RTT", "BW (Gbps)");
        printf("-----------------------------------------------------------------------\n");
        fflush(stdout);
    }

    // Warmup
    double dummy_avg, dummy_min, dummy_max;
    run_mpi_host_pingpong(rank, peer, h_send_buf, h_recv_buf, 64,
                          WARMUP_ITERS, &dummy_avg, &dummy_min, &dummy_max);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int sz_idx = 0; sz_idx < NUM_SIZES; sz_idx++) {
        size_t msg_size = MSG_SIZES[sz_idx];

        double avg_rtt, min_rtt, max_rtt;
        run_mpi_host_pingpong(rank, peer, h_send_buf, h_recv_buf, msg_size,
                              TEST_ITERS, &avg_rtt, &min_rtt, &max_rtt);

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            double half_rtt = avg_rtt / 2.0;
            double bw_gbps = (msg_size * 2.0 * 8.0) / (avg_rtt * 1000.0);

            char size_buf[32];
            printf("%-8s %12.3f %12.3f %12.3f %12.3f %12.2f\n",
                   format_size(msg_size, size_buf),
                   avg_rtt, half_rtt, min_rtt, max_rtt, bw_gbps);
            fflush(stdout);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Test 2: MPI with GPU Memory (CUDA-aware MPI)
    //==========================================================================

    if (rank == 0) {
        printf("=== MPI Ping-Pong (GPU Memory, CUDA-aware MPI) ===\n");
        printf("%-8s %12s %12s %12s %12s %12s\n",
               "Size", "RTT (us)", "Half-RTT", "Min RTT", "Max RTT", "BW (Gbps)");
        printf("-----------------------------------------------------------------------\n");
        fflush(stdout);
    }

    // Warmup
    run_mpi_gpu_pingpong(rank, peer, d_send_buf, d_recv_buf, 64,
                         WARMUP_ITERS, &dummy_avg, &dummy_min, &dummy_max);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int sz_idx = 0; sz_idx < NUM_SIZES; sz_idx++) {
        size_t msg_size = MSG_SIZES[sz_idx];

        double avg_rtt, min_rtt, max_rtt;
        run_mpi_gpu_pingpong(rank, peer, d_send_buf, d_recv_buf, msg_size,
                             TEST_ITERS, &avg_rtt, &min_rtt, &max_rtt);

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            double half_rtt = avg_rtt / 2.0;
            double bw_gbps = (msg_size * 2.0 * 8.0) / (avg_rtt * 1000.0);

            char size_buf[32];
            printf("%-8s %12.3f %12.3f %12.3f %12.3f %12.2f\n",
                   format_size(msg_size, size_buf),
                   avg_rtt, half_rtt, min_rtt, max_rtt, bw_gbps);
            fflush(stdout);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    //==========================================================================
    // Summary
    //==========================================================================

    if (rank == 0) {
        printf("\n=======================================================\n");
        printf("Benchmark complete!\n");
        printf("\nTo compare with GPU-triggered RDMA, run:\n");
        printf("  srun -N 2 --gres=gpu:1 ./gpu_pingpong_bench\n");
        printf("=======================================================\n");
        fflush(stdout);
    }

    // Cleanup
    free(h_send_buf);
    free(h_recv_buf);
    cudaFree(d_send_buf);
    cudaFree(d_recv_buf);

    MPI_Finalize();
    return 0;
}
