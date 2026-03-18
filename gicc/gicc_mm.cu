/**
 * gicc_mm.cu - Distributed Matrix Multiplication using GICC API
 *
 * Rewrite of nvidia/mm_gda_nv.cu. All IB/QP/MR boilerplate replaced by
 * gicc::Runtime. GPU kernels use gicc::put() for RDMA.
 *
 * Run: mpirun -np <npes> ./gicc_mm [matrix_size]
 */

#include "gicc.hpp"
#include "gicc_device.cuh"

#include <iostream>
#include <ctime>
#include <cstring>
#include <unistd.h>
#include <cuda_runtime.h>
#include <mpi.h>

#define CUDA_CHECK(cmd) do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

void print_matrix(const float* mat, const int Is, const int Js)
{
    for (int i = 0; i < Is; i++) {
        for (int j = 0; j < Js; j++)
            std::cout << mat[i * Js + j] << ' ';
        std::cout << '\n';
    }
}

float timediff_us(const timespec& t_start, const timespec& t_end)
{
    return (t_end.tv_sec - t_start.tv_sec) * 1.0e6 +
           (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3;
}

// =============================================================================
// CUDA kernel for matrix stripe multiplication (unchanged from original)
// =============================================================================

__global__ void matmul_stripe_kernel(
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    float* __restrict__ Cs,
    int N, int Ns, int col_offset)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (k < N && j < Ns) {
        float b_kj = Bs[k * Ns + j];
        for (int i = 0; i < Ns; i++) {
            atomicAdd(&Cs[i * N + col_offset + j], As[i * N + k] * b_kj);
        }
    }
}

// =============================================================================
// GPU RDMA kernels using GICC API
// =============================================================================

__global__ void gpu_rdma_write_kernel(
    gicc::DeviceCtx* ctx,
    uint64_t local_addr, uint32_t local_lkey,
    uint64_t remote_addr, uint32_t remote_rkey,
    uint32_t size, bool signaled)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::put(ctx, local_addr, local_lkey,
                  remote_addr, remote_rkey, size, signaled);
    }
}

__global__ void gpu_rdma_write_and_wait_kernel(
    gicc::DeviceCtx* ctx,
    uint64_t local_addr, uint32_t local_lkey,
    uint64_t remote_addr, uint32_t remote_rkey,
    uint32_t size)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::put(ctx, local_addr, local_lkey,
                  remote_addr, remote_rkey, size, true);
        gicc::quiet(ctx);
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    // --- All IB/QP setup in one object ---
    gicc::Runtime rt;

    int mype = rt.rank();
    int npes = rt.size();

    if (npes < 2) {
        if (mype == 0) std::cerr << "Requires at least 2 processes\n";
        MPI_Finalize();
        return 1;
    }

    // Ring neighbors
    // Original code: QP connected to right neighbor, RDMA writes go right
    int right_neighbor = (mype + 1) % npes;

    // Matrix size
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    int Ns_arg = (argc > 2) ? atoi(argv[2]) : 0;
    N = (N / npes) * npes;
    const int Ns = (Ns_arg > 0) ? Ns_arg : N / npes;
    const size_t stripe_size = N * Ns * sizeof(float);

    if (mype == 0) {
        std::cout << "Matrix stripe: " << N << 'x' << Ns
                  << ", " << stripe_size << " bytes\n";
        std::cout << "Ring pattern: " << npes << " processes\n";
    }

    // Allocate host arrays
    auto h_As = new float[N * Ns];
    auto h_Bs = new float[N * Ns];
    auto h_Cs = new float[N * Ns];

    for (int i = 0; i < N * Ns; i++) {
        h_As[i] = (i + mype) % 11 + 7;
        h_Bs[i] = (i + mype) % 13 + 5;
        h_Cs[i] = 0;
    }

    // Allocate device arrays - double buffering for B
    float *d_As, *d_Cs, *d_B[2];
    CUDA_CHECK(cudaMalloc(&d_As, stripe_size));
    CUDA_CHECK(cudaMalloc(&d_Cs, stripe_size));
    CUDA_CHECK(cudaMalloc(&d_B[0], stripe_size));
    CUDA_CHECK(cudaMalloc(&d_B[1], stripe_size));

    CUDA_CHECK(cudaMemcpy(d_As, h_As, stripe_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B[0], h_Bs, stripe_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Cs, h_Cs, stripe_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_B[1], 0, stripe_size));

    // --- Register and exchange buffers via GICC ---
    auto buf_B0 = rt.register_buffer(d_B[0], stripe_size, true);  // index 0
    auto buf_B1 = rt.register_buffer(d_B[1], stripe_size, true);  // index 1
    rt.exchange();

    // Get right neighbor's buffer info
    auto right_B0 = rt.remote_buffer(right_neighbor, buf_B0.index);
    auto right_B1 = rt.remote_buffer(right_neighbor, buf_B1.index);

    // Prepare DeviceCtx for right neighbor QP
    auto* ctx = rt.prepare(right_neighbor, buf_B0.index);

    if (mype == 0) std::cout << "GICC setup complete\n";

    timespec t0, t1;

    // Kernel launch config
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (Ns + blockDim.y - 1) / blockDim.y);

    // Warmup
    {
        int col_offset = mype * Ns;
        matmul_stripe_kernel<<<gridDim, blockDim>>>(d_As, d_B[0], d_Cs, N, Ns, col_offset);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_Cs, 0, stripe_size));

        gpu_rdma_write_and_wait_kernel<<<1, 1>>>(
            ctx, buf_B0.addr, buf_B0.lkey,
            right_B1.addr, right_B1.rkey,
            (uint32_t)stripe_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    rt.barrier();

    // Run 10 iterations, skip first 3
    constexpr int TOTAL_RUNS = 10;
    constexpr int WARMUP_RUNS = 3;
    double times[TOTAL_RUNS];

    for (int run = 0; run < TOTAL_RUNS; run++) {
        CUDA_CHECK(cudaMemset(d_Cs, 0, stripe_size));
        CUDA_CHECK(cudaMemcpy(d_B[0], h_Bs, stripe_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B[1], h_Bs, stripe_size, cudaMemcpyHostToDevice));

        rt.barrier();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

        for (int s = 0; s < npes; s++) {
            const int block_num = (mype + s) % npes;
            const int cur_buf = s % 2;
            const int next_buf = (s + 1) % 2;

            // Pick local and remote buffer info for this iteration
            uint64_t local_addr = (cur_buf == 0) ? buf_B0.addr : buf_B1.addr;
            uint32_t local_lkey = (cur_buf == 0) ? buf_B0.lkey : buf_B1.lkey;
            auto& target = (next_buf == 0) ? right_B0 : right_B1;

            // GPU-triggered RDMA write (non-blocking)
            gpu_rdma_write_kernel<<<1, 1>>>(
                ctx, local_addr, local_lkey,
                target.addr, target.rkey,
                (uint32_t)stripe_size,
                (s == npes - 1));

            // Compute while communication in flight
            int col_offset = block_num * Ns;
            matmul_stripe_kernel<<<gridDim, blockDim>>>(
                d_As, d_B[cur_buf], d_Cs, N, Ns, col_offset);

            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Barrier(MPI_COMM_WORLD);
        }

        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        times[run] = timediff_us(t0, t1);

        if (mype == 0) {
            std::cout << "Run " << run << ": " << times[run] << " us"
                      << (run < WARMUP_RUNS ? " (warmup)" : "") << "\n";
        }
    }

    double sum = 0;
    for (int i = WARMUP_RUNS; i < TOTAL_RUNS; i++) sum += times[i];
    double avg = sum / (TOTAL_RUNS - WARMUP_RUNS);

    if (mype == 0) {
        std::cout << "GICC GPU-triggered RDMA + CUDA average (runs "
                  << WARMUP_RUNS << "-" << (TOTAL_RUNS-1) << "): "
                  << avg << " us\n";
    }

    // Verify (small matrices only)
    CUDA_CHECK(cudaMemcpy(h_Cs, d_Cs, stripe_size, cudaMemcpyDeviceToHost));

    if (N < 32) {
        if (mype == 0) {
            auto C = new float[N * N];
            for (int i = 0; i < Ns * N; i++) C[i] = h_Cs[i];
            for (int r = 1; r < npes; r++)
                MPI_Recv(C + r * Ns * N, Ns * N, MPI_FLOAT, r, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            print_matrix(C, N, N);
            delete[] C;
        } else {
            MPI_Send(h_Cs, Ns * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_B[1]));
    CUDA_CHECK(cudaFree(d_B[0]));
    CUDA_CHECK(cudaFree(d_Cs));
    CUDA_CHECK(cudaFree(d_As));

    delete[] h_Cs;
    delete[] h_Bs;
    delete[] h_As;

    MPI_Finalize();
    return 0;
}
