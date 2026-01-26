/**
 * nvshmem_pingpong_bench.cu - NVSHMEM ping-pong latency benchmark
 *
 * Compares NVSHMEM performance with our GPU-triggered RDMA implementation.
 *
 * Build:
 *   module load nvhpc/24.5.cuda_12.4
 *   nvcc -dc -ccbin mpic++ -std=c++17 -gencode=arch=compute_90,code=sm_90 \
 *        -I$NVSHMEM_HOME/include nvshmem_pingpong_bench.cu -c -o nvshmem_pingpong_bench.o
 *   nvcc -gencode=arch=compute_90,code=sm_90 nvshmem_pingpong_bench.o \
 *        -cuda -cudalib=nvshmem -lnvshmem_host -lnvshmem_device -o nvshmem_pingpong_bench
 *
 * Run:
 *   srun -N 2 --gres=gpu:1 ./nvshmem_pingpong_bench
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

// Maximum message size
constexpr size_t MAX_MSG_SIZE = 16 * 1024 * 1024;  // 16 MB

// Iterations
constexpr int WARMUP_ITERS = 50;
constexpr int BENCH_ITERS = 500;

/**
 * Ping-pong kernel using nvshmem_putmem_nbi + signal
 *
 * PE 0 sends to PE 1, then waits for signal
 * PE 1 waits for signal, then sends back to PE 0
 */
__global__ void ping_pong_kernel(void* data_d, uint64_t* flag_d,
                                  size_t size, int pe, int iterations) {
    int peer = !pe;

    for (int i = 0; i < iterations; i++) {
        if (pe == 0) {
            // PE 0: Send data, then wait for reply
            nvshmem_putmem_nbi(data_d, data_d, size, peer);
            nvshmem_fence();
            nvshmemx_signal_op(flag_d, i + 1, NVSHMEM_SIGNAL_SET, peer);

            // Wait for reply
            nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (uint64_t)(i + 1));
        } else {
            // PE 1: Wait for data, then send back
            nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (uint64_t)(i + 1));

            nvshmem_putmem_nbi(data_d, data_d, size, peer);
            nvshmem_fence();
            nvshmemx_signal_op(flag_d, i + 1, NVSHMEM_SIGNAL_SET, peer);
        }
    }
    nvshmem_quiet();
}

/**
 * Simple put kernel (no signal, measures nvshmem_putmem latency only)
 */
__global__ void put_only_kernel(void* data_d, size_t size, int pe, int iterations) {
    int peer = !pe;

    for (int i = 0; i < iterations; i++) {
        nvshmem_putmem_nbi(data_d, data_d, size, peer);
    }
    nvshmem_quiet();
}

void run_benchmark() {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    if (npes != 2) {
        if (mype == 0) {
            fprintf(stderr, "Error: This benchmark requires exactly 2 PEs\n");
        }
        return;
    }

    // Allocate symmetric memory
    void* data_d = nvshmem_malloc(MAX_MSG_SIZE);
    uint64_t* flag_d = (uint64_t*)nvshmem_malloc(sizeof(uint64_t));

    if (!data_d || !flag_d) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", mype);
        return;
    }

    CUDA_CHECK(cudaMemset(data_d, 0, MAX_MSG_SIZE));
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    nvshmem_barrier_all();

    if (mype == 0) {
        printf("\n");
        printf("=============================================================\n");
        printf("           NVSHMEM Ping-Pong Latency Benchmark\n");
        printf("=============================================================\n");
        printf("\n");
        printf("Configuration:\n");
        printf("  PEs: %d\n", npes);
        printf("  Warmup iterations: %d\n", WARMUP_ITERS);
        printf("  Benchmark iterations: %d\n", BENCH_ITERS);
        printf("\n");
        printf("Note: RTT = Round-Trip Time (full ping-pong)\n");
        printf("\n");
        printf("%-10s %12s %12s %12s\n", "Size", "RTT (us)", "One-way (us)", "BW (Gbps)");
        printf("%-10s %12s %12s %12s\n", "----", "--------", "-----------", "---------");
    }

    // Test different sizes
    for (size_t size = 1; size <= MAX_MSG_SIZE; size *= 2) {
        // Reset flag
        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        // Warmup
        void* warmup_args[] = {&data_d, &flag_d, &size, &mype, (void*)&WARMUP_ITERS};
        int status = nvshmemx_collective_launch((const void*)ping_pong_kernel,
                                                 1, 1, warmup_args, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "PE %d: collective_launch failed (warmup)\n", mype);
            break;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Reset flag
        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        nvshmem_barrier_all();

        // Benchmark
        CUDA_CHECK(cudaEventRecord(start, stream));

        void* bench_args[] = {&data_d, &flag_d, &size, &mype, (void*)&BENCH_ITERS};
        status = nvshmemx_collective_launch((const void*)ping_pong_kernel,
                                             1, 1, bench_args, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "PE %d: collective_launch failed (benchmark)\n", mype);
            break;
        }

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        double rtt_us = (ms * 1000.0) / BENCH_ITERS;
        double oneway_us = rtt_us / 2.0;
        double bw_gbps = (size * 8.0 * 2.0) / (rtt_us * 1000.0);  // 2x for both directions

        if (mype == 0) {
            if (size < 1024) {
                printf("%-10zu %12.2f %12.2f %12.2f\n", size, rtt_us, oneway_us, bw_gbps);
            } else if (size < 1024 * 1024) {
                printf("%-9zuK %12.2f %12.2f %12.2f\n", size / 1024, rtt_us, oneway_us, bw_gbps);
            } else {
                printf("%-9zuM %12.2f %12.2f %12.2f\n", size / (1024 * 1024), rtt_us, oneway_us, bw_gbps);
            }
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        printf("\n");
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    nvshmem_free(flag_d);
    nvshmem_free(data_d);
}

int main(int argc, char* argv[]) {
    // Initialize MPI first
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set CUDA device based on local rank BEFORE nvshmem init
    int local_rank = 0;
    char* local_rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!local_rank_str) local_rank_str = getenv("SLURM_LOCALID");
    if (!local_rank_str) local_rank_str = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (local_rank_str) local_rank = atoi(local_rank_str);

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (rank == 0) {
        printf("Rank %d: Using GPU %d: %s\n", rank, device, prop.name);
    }

    // Initialize NVSHMEM with MPI
    nvshmemx_init_attr_t attr;
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = &comm;

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Run benchmark
    run_benchmark();

    // Finalize
    nvshmem_finalize();
    MPI_Finalize();

    return 0;
}
