/**
 * nvshmem_simple_bench.cu - Simple NVSHMEM latency benchmark
 * Uses regular kernel launch instead of collective_launch
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_CHECK(stmt) do {                                           \
    cudaError_t err = (stmt);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                    \
                cudaGetErrorString(err), __FILE__, __LINE__);           \
        exit(1);                                                        \
    }                                                                   \
} while (0)

constexpr int WARMUP_ITERS = 50;
constexpr int BENCH_ITERS = 500;

// Simple ping-pong using nvshmem_int_p (8-byte put)
__global__ void ping_pong_p(volatile long* flag_d, int pe, int iterations) {
    int peer = !pe;

    for (int i = 0; i < iterations; i++) {
        if (pe == 0) {
            // Send to peer
            nvshmem_long_p((long*)flag_d, i + 1, peer);
            // Wait for reply
            nvshmem_long_wait_until((long*)flag_d, NVSHMEM_CMP_EQ, (long)(i + 1));
        } else {
            // Wait for data
            nvshmem_long_wait_until((long*)flag_d, NVSHMEM_CMP_EQ, (long)(i + 1));
            // Send reply
            nvshmem_long_p((long*)flag_d, i + 1, peer);
        }
    }
}

// Ping-pong with variable size using nvshmem_putmem
__global__ void ping_pong_put(void* data_d, volatile uint64_t* flag_d,
                               size_t size, int pe, int iterations) {
    int peer = !pe;

    for (int i = 0; i < iterations; i++) {
        if (pe == 0) {
            nvshmem_putmem(data_d, data_d, size, peer);
            nvshmem_fence();
            nvshmem_uint64_p((uint64_t*)flag_d, i + 1, peer);
            nvshmem_uint64_wait_until((uint64_t*)flag_d, NVSHMEM_CMP_EQ, (uint64_t)(i + 1));
        } else {
            nvshmem_uint64_wait_until((uint64_t*)flag_d, NVSHMEM_CMP_EQ, (uint64_t)(i + 1));
            nvshmem_putmem(data_d, data_d, size, peer);
            nvshmem_fence();
            nvshmem_uint64_p((uint64_t*)flag_d, i + 1, peer);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set device before nvshmem init
    int local_rank = 0;
    char* lr = getenv("SLURM_LOCALID");
    if (!lr) lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (lr) local_rank = atoi(lr);

    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(local_rank % dev_count));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, local_rank % dev_count));

    // Init nvshmem
    nvshmemx_init_attr_t attr;
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    if (npes != 2) {
        if (mype == 0) fprintf(stderr, "Need exactly 2 PEs\n");
        nvshmem_finalize();
        MPI_Finalize();
        return 1;
    }

    if (mype == 0) {
        printf("GPU: %s\n", prop.name);
    }

    // Allocate symmetric memory
    size_t max_size = 16 * 1024 * 1024;
    void* data_d = nvshmem_malloc(max_size);
    uint64_t* flag_d = (uint64_t*)nvshmem_malloc(sizeof(uint64_t));

    if (!data_d || !flag_d) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", mype);
        nvshmem_finalize();
        MPI_Finalize();
        return 1;
    }

    CUDA_CHECK(cudaMemset(data_d, 0, max_size));
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    nvshmem_barrier_all();

    if (mype == 0) {
        printf("\n=== NVSHMEM Simple Ping-Pong Benchmark ===\n\n");
    }

    // Test 1: Simple 8-byte ping-pong using nvshmem_long_p
    {
        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        // Warmup
        ping_pong_p<<<1, 1, 0, stream>>>((volatile long*)flag_d, mype, WARMUP_ITERS);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        nvshmem_barrier_all();

        // Benchmark
        CUDA_CHECK(cudaEventRecord(start, stream));
        ping_pong_p<<<1, 1, 0, stream>>>((volatile long*)flag_d, mype, BENCH_ITERS);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double rtt_us = (ms * 1000.0) / BENCH_ITERS;

        if (mype == 0) {
            printf("nvshmem_long_p (8B): RTT = %.2f us, one-way = %.2f us\n",
                   rtt_us, rtt_us / 2);
        }
        nvshmem_barrier_all();
    }

    // Test 2: Variable size ping-pong
    if (mype == 0) {
        printf("\n%-10s %12s %12s %12s\n", "Size", "RTT (us)", "One-way", "BW (Gbps)");
        printf("%-10s %12s %12s %12s\n", "----", "--------", "-------", "---------");
    }

    for (size_t sz = 8; sz <= max_size; sz *= 2) {
        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        // Warmup
        ping_pong_put<<<1, 1, 0, stream>>>(data_d, (volatile uint64_t*)flag_d,
                                            sz, mype, WARMUP_ITERS);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        nvshmem_barrier_all();

        // Benchmark
        CUDA_CHECK(cudaEventRecord(start, stream));
        ping_pong_put<<<1, 1, 0, stream>>>(data_d, (volatile uint64_t*)flag_d,
                                            sz, mype, BENCH_ITERS);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double rtt_us = (ms * 1000.0) / BENCH_ITERS;
        double bw_gbps = (sz * 8.0 * 2.0) / (rtt_us * 1000.0);

        if (mype == 0) {
            if (sz < 1024)
                printf("%-10zu %12.2f %12.2f %12.2f\n", sz, rtt_us, rtt_us/2, bw_gbps);
            else if (sz < 1024*1024)
                printf("%-9zuK %12.2f %12.2f %12.2f\n", sz/1024, rtt_us, rtt_us/2, bw_gbps);
            else
                printf("%-9zuM %12.2f %12.2f %12.2f\n", sz/1024/1024, rtt_us, rtt_us/2, bw_gbps);
        }
        nvshmem_barrier_all();
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    nvshmem_free(flag_d);
    nvshmem_free(data_d);

    nvshmem_finalize();
    MPI_Finalize();

    return 0;
}
