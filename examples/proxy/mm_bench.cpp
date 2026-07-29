/*
 * mm_bench.cpp - Distributed matmul on a ring, GICC DWQ vs Cray MPI.
 *
 * A CUDA/manual-API sibling of examples/ofi/mm_minimal.cpp, which drives its
 * ring rotation from inside the kernel and therefore needs the LTO pass. Here
 * the rotation is staged from the host and fired by the kernel's trigger
 * store, so it builds with plain nvcc.
 *
 * C = A * B, all N x N, row-block distributed: rank r owns rows
 * [r*bs, (r+1)*bs) of A, of B, and of C.
 *
 *   C_local += A_local[:, k_block(p)] * B_block(p)
 *
 * B blocks rotate around the ring: after each step, rank r ships its current
 * B block to rank r-1 and the block it held moves one hop. After `nranks`
 * steps every rank has seen every B block and C_local is complete.
 *
 * Two B buffers ping-pong so the in-flight write never targets the block the
 * SMs are reading.
 *
 * Run:
 *   srun -N2 -n2 --ntasks-per-node=1 --cpus-per-task=16 --gpus-per-node=1 \
 *        ./mm_bench --n 4096 --mode=gicc
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include <mpi.h>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

#define GPU_CHECK(cmd)                                                        \
    do {                                                                      \
        GpuError _e = (cmd);                                                  \
        if (_e != GPU_SUCCESS) {                                              \
            fprintf(stderr, "GPU error %s at %s:%d\n",                        \
                    gpuGetErrorString(_e), __FILE__, __LINE__);               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

#define TILE 16

// C_local[bs][N] += A_local[bs][N][k_off .. k_off+bs) * Bblk[bs][N]
__global__ void mm_accum(const float* __restrict__ A,
                         const float* __restrict__ Bblk,
                         float* __restrict__ C,
                         int N, int bs, int k_off) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int col = blockIdx.x * TILE + threadIdx.x;   // column of C, 0..N
    int row = blockIdx.y * TILE + threadIdx.y;   // local row of C, 0..bs
    float acc = 0.0f;

    for (int t = 0; t < bs; t += TILE) {
        int a_k = t + threadIdx.x;               // offset within the k block
        sA[threadIdx.y][threadIdx.x] =
            (row < bs && a_k < bs) ? A[(size_t)row * N + k_off + a_k] : 0.0f;
        int b_r = t + threadIdx.y;               // row within the B block
        sB[threadIdx.y][threadIdx.x] =
            (b_r < bs && col < N) ? Bblk[(size_t)b_r * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < bs && col < N) C[(size_t)row * N + col] += acc;
}

// Same, plus the trigger store that ships the current B block onward. The
// flush fires first so the rotation overlaps this step's arithmetic.
__global__ void mm_accum_gicc(const float* __restrict__ A,
                              const float* __restrict__ Bblk,
                              float* __restrict__ C,
                              int N, int bs, int k_off,
                              gicc::DeviceCtx* ctx) {
    gicc::flush(ctx);

    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    float acc = 0.0f;

    for (int t = 0; t < bs; t += TILE) {
        int a_k = t + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < bs && a_k < bs) ? A[(size_t)row * N + k_off + a_k] : 0.0f;
        int b_r = t + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (b_r < bs && col < N) ? Bblk[(size_t)b_r * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < bs && col < N) C[(size_t)row * N + col] += acc;
}

// A[i][j] = (i+j) % 7, B[i][j] = (i*2+j) % 5 -- cheap, exactly representable,
// and enough structure that a wrong rotation changes the checksum.
__global__ void fill_kernel(float* A, float* B, int N, int bs, int row_off) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= bs || col >= N) return;
    int gr = row_off + row;
    A[(size_t)row * N + col] = (float)((gr + col) % 7);
    B[(size_t)row * N + col] = (float)((gr * 2 + col) % 5);
}

__global__ void sum_kernel(const float* C, size_t n, double* out) {
    __shared__ double s[256];
    int t = threadIdx.x;
    double acc = 0.0;
    for (size_t k = t + (size_t)blockIdx.x * blockDim.x; k < n;
         k += (size_t)blockDim.x * gridDim.x) acc += C[k];
    s[t] = acc;
    __syncthreads();
    for (int d = blockDim.x / 2; d > 0; d >>= 1) {
        if (t < d) s[t] += s[t + d];
        __syncthreads();
    }
    if (t == 0) atomicAdd(out, s[0]);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);

    int N = 4096;
    std::string mode = "gicc";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--n" && i + 1 < argc)      N = atoi(argv[++i]);
        else if (a.rfind("--mode=", 0) == 0) mode = a.substr(7);
    }
    if (mode != "gicc" && mode != "mpi") {
        fprintf(stderr, "mm_bench: --mode must be 'gicc' or 'mpi'\n");
        return 2;
    }

    gicc::Runtime rt;
    if (mode == "gicc") rt.enable_host_wait_mode();
    int rank = rt.rank(), nranks = rt.size();

    if (N % nranks != 0 || (N / nranks) % TILE != 0) {
        if (rank == 0)
            fprintf(stderr, "mm_bench: --n (%d) must divide by ranks (%d) "
                    "with a block that is a multiple of %d\n", N, nranks, TILE);
        return 2;
    }
    int bs = N / nranks;
    size_t blk_bytes = (size_t)bs * N * sizeof(float);

    float *d_A = nullptr, *d_C = nullptr, *d_B0 = nullptr, *d_B1 = nullptr;
    GPU_CHECK(gpuMalloc((void**)&d_A, blk_bytes));
    GPU_CHECK(gpuMalloc((void**)&d_C, blk_bytes));
    GPU_CHECK(gpuMalloc((void**)&d_B0, blk_bytes));
    GPU_CHECK(gpuMalloc((void**)&d_B1, blk_bytes));
    GPU_CHECK(gpuMemset(d_C, 0, blk_bytes));

    auto bh_B0 = rt.register_buffer(d_B0, blk_bytes, /*is_device=*/true);
    auto bh_B1 = rt.register_buffer(d_B1, blk_bytes, /*is_device=*/true);
    rt.exchange();

    dim3 tb(16, 16);
    dim3 gf((N + 15) / 16, (bs + 15) / 16);
    gpuLaunchKernel(fill_kernel, gf, tb, 0, 0, d_A, d_B0, N, bs, rank * bs);
    GPU_CHECK(gpuDeviceSynchronize());

    int next = (rank + nranks - 1) % nranks;   // where my B block goes
    dim3 gm((N + TILE - 1) / TILE, (bs + TILE - 1) / TILE);

    if (mode == "gicc") { (void)rt.prepare(); rt.reset(); }
    rt.barrier();

    double t_comm = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int s = 0; s < nranks; ++s) {
        // At step s I hold the B block that originated on rank (rank+s)%nranks,
        // which supplies k in [k_off, k_off+bs).
        int k_off = ((rank + s) % nranks) * bs;
        gicc::Buffer& cur  = (s % 2 == 0) ? bh_B0 : bh_B1;
        gicc::Buffer& nxt  = (s % 2 == 0) ? bh_B1 : bh_B0;
        float* d_cur = (s % 2 == 0) ? d_B0 : d_B1;

        bool rotate = (s + 1 < nranks);

        if (mode == "gicc") {
            // Ship the block I am about to consume to the next hop, landing
            // in that rank's alternate buffer so nobody overwrites live data.
            // The kernel's flush fires it, so the transfer runs underneath the
            // arithmetic. Only the staging cost and whatever is left to drain
            // after the kernel finishes count as exposed communication --
            // timing the whole region would just re-measure the compute.
            double c0 = MPI_Wtime();
            if (rotate) rt.put(cur, next, nxt.index, blk_bytes, 0, 0);
            gicc::DeviceCtx* ctx = rt.prepare();
            t_comm += MPI_Wtime() - c0;

            gpuLaunchKernel(mm_accum_gicc, gm, dim3(TILE, TILE), 0, 0,
                            d_A, d_cur, d_C, N, bs, k_off, ctx);
            GPU_CHECK(gpuDeviceSynchronize());

            double c1 = MPI_Wtime();
            rt.reset();
            rt.barrier();
            t_comm += MPI_Wtime() - c1;
        } else {
            double c0 = MPI_Wtime();
            float* d_nxt = (s % 2 == 0) ? d_B1 : d_B0;
            int prev = (rank + 1) % nranks;    // who feeds me
            if (rotate) {
                MPI_Request req[2];
                MPI_Irecv(d_nxt, (int)blk_bytes, MPI_BYTE, prev, 0,
                          MPI_COMM_WORLD, &req[0]);
                MPI_Isend(d_cur, (int)blk_bytes, MPI_BYTE, next, 0,
                          MPI_COMM_WORLD, &req[1]);
                MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
            }
            t_comm += MPI_Wtime() - c0;

            gpuLaunchKernel(mm_accum, gm, dim3(TILE, TILE), 0, 0,
                            d_A, d_cur, d_C, N, bs, k_off);
            GPU_CHECK(gpuDeviceSynchronize());
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    double* d_sum = nullptr;
    GPU_CHECK(gpuMalloc((void**)&d_sum, sizeof(double)));
    GPU_CHECK(gpuMemset(d_sum, 0, sizeof(double)));
    gpuLaunchKernel(sum_kernel, dim3(256), dim3(256), 0, 0,
                    d_C, (size_t)bs * N, d_sum);
    GPU_CHECK(gpuDeviceSynchronize());
    double h_sum = 0.0;
    GPU_CHECK(gpuMemcpy(&h_sum, d_sum, sizeof(double), gpuMemcpyDeviceToHost));
    double total_sum = 0.0;
    MPI_Reduce(&h_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_comm = 0.0;
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total_s = t1 - t0;
        double gflop = 2.0 * (double)N * N * N / 1e9;
        printf("\n=== mm_bench (mode=%s) ===\n", mode.c_str());
        printf("ranks=%d  N=%d  block=%dx%d  rotation msg=%zu B  steps=%d\n",
               nranks, N, bs, N, blk_bytes, nranks);
        printf("total       %10.3f ms   (%.1f GFLOP/s aggregate)\n",
               total_s * 1e3, gflop / total_s);
        printf("rotation    %10.3f ms   (%.1f%% of total, exposed only -- "
               "gicc hides the rest under compute)\n",
               max_comm * 1e3, 100.0 * max_comm / total_s);
        printf("checksum    %.6e\n", total_sum);
    }

    (void)gpuFree(d_sum);
    (void)gpuFree(d_A);
    (void)gpuFree(d_C);
    (void)gpuFree(d_B0);
    (void)gpuFree(d_B1);
    return 0;
}
