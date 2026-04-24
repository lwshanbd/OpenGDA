/**
 * barrier_bench.cpp - Apples-to-apples latency: single-barrier GICC vs MPI.
 *
 * Each iteration launches the same compute kernel + one barrier. We measure
 * per-iteration wall time.
 *
 *   --gicc  : kernel does compute + gicc::barrier() inside
 *             host loop: setup -> launch -> hipDeviceSync -> reset
 *   --mpi   : kernel does compute only
 *             host loop: launch -> hipDeviceSync -> MPI_Barrier
 *
 * Run: FI_MR_CACHE_MAX_COUNT=0 srun ... ./barrier_bench --gicc [N]
 *      srun ... ./barrier_bench --mpi  [N]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include <hip/hip_runtime.h>
#include <mpi.h>

#include "gicc/bootstrap/bootstrap.hpp"
#include "gicc/platform/ofi/ofi_barrier_device.cuh"
#include "gicc/platform/ofi/internal/gicc_barrier.hpp"
#include "gicc/platform/ofi/internal/hip_device_context.hpp"

// =============================================================================
// Shared tiny compute: each thread bumps one element with a few flops.
// Keeps the kernel short so barrier cost dominates but non-zero.
// =============================================================================
__device__ __forceinline__ void do_compute(float* buf, int tid) {
    float x = buf[tid];
    #pragma unroll
    for (int i = 0; i < 32; i++) x = x * 1.0001f + 0.5f;
    buf[tid] = x;
}

__global__ void compute_only_kernel(float* buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) do_compute(buf, tid);
}

__global__ void compute_then_barrier_kernel(float* buf, int n,
                                            gicc::BarrierCtx* bctx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) do_compute(buf, tid);
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0)
        gicc::barrier(bctx);
}

static double now_us() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec * 1.0e6 + t.tv_nsec / 1.0e3;
}

int main(int argc, char** argv) {
    unset_rocr_visible_devices();

    bool use_gicc = false, use_mpi = false;
    int iters = 1000;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gicc")) use_gicc = true;
        else if (!strcmp(argv[i], "--mpi")) use_mpi = true;
        else iters = atoi(argv[i]);
    }
    if (use_gicc == use_mpi) {
        fprintf(stderr, "Usage: %s --gicc|--mpi [iters]\n", argv[0]);
        return 1;
    }

    gicc::Bootstrap boot;
    gicc::Fabric comm(boot);
    int rank = comm.rank();
    int size = comm.size();

    // Tiny workload: 1 block of 256 threads bumping 256 floats. ~few µs.
    constexpr int N = 256;
    constexpr int BLOCKS = 1;
    constexpr int THREADS = 256;

    float* d_buf = nullptr;
    hipMalloc(&d_buf, N * sizeof(float));
    hipMemset(d_buf, 0, N * sizeof(float));

    std::vector<double> per_iter(iters);

    if (use_gicc) {
        gicc::Barrier barrier(comm);

        if (rank == 0) {
            printf("barrier_bench --gicc: %d ranks, %d iters ... ", size, iters);
            fflush(stdout);
        }

        // Warmup
        for (int w = 0; w < 20; w++) {
            barrier.setup();
            boot.barrier();
            hipLaunchKernelGGL(compute_then_barrier_kernel,
                               dim3(BLOCKS), dim3(THREADS), 0, 0,
                               d_buf, N, barrier.device_ctx());
            hipDeviceSynchronize();
            barrier.reset();
        }
        boot.barrier();

        double total_start = now_us();
        for (int i = 0; i < iters; i++) {
            double t0 = now_us();
            barrier.setup();
            boot.barrier();                 // host-side sync so setup races don't alias iters
            hipLaunchKernelGGL(compute_then_barrier_kernel,
                               dim3(BLOCKS), dim3(THREADS), 0, 0,
                               d_buf, N, barrier.device_ctx());
            hipDeviceSynchronize();
            barrier.reset();
            per_iter[i] = now_us() - t0;
        }
        double total_us = now_us() - total_start;

        if (rank == 0) {
            double avg = total_us / iters;
            double minv = per_iter[0], maxv = per_iter[0];
            for (auto v : per_iter) { if (v < minv) minv = v; if (v > maxv) maxv = v; }
            printf("DONE  avg=%.2f us  min=%.2f us  max=%.2f us\n", avg, minv, maxv);
        }
    } else {
        if (rank == 0) {
            printf("barrier_bench --mpi:  %d ranks, %d iters ... ", size, iters);
            fflush(stdout);
        }

        // Warmup
        for (int w = 0; w < 20; w++) {
            hipLaunchKernelGGL(compute_only_kernel,
                               dim3(BLOCKS), dim3(THREADS), 0, 0, d_buf, N);
            hipDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double total_start = now_us();
        for (int i = 0; i < iters; i++) {
            double t0 = now_us();
            hipLaunchKernelGGL(compute_only_kernel,
                               dim3(BLOCKS), dim3(THREADS), 0, 0, d_buf, N);
            hipDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
            per_iter[i] = now_us() - t0;
        }
        double total_us = now_us() - total_start;

        if (rank == 0) {
            double avg = total_us / iters;
            double minv = per_iter[0], maxv = per_iter[0];
            for (auto v : per_iter) { if (v < minv) minv = v; if (v > maxv) maxv = v; }
            printf("DONE  avg=%.2f us  min=%.2f us  max=%.2f us\n", avg, minv, maxv);
        }
    }

    hipFree(d_buf);
    return 0;
}
