/**
 * jacobi.cpp - Jacobi solver on the unified gicc::launch API.
 *
 * One fused kernel per iter:
 *   1. put_no_db(top neighbor, top halo row)   <-- IPC or DWQ, dispatched in put_no_db
 *   2. put_no_db(bottom neighbor, bottom halo row)
 *   3. compute interior + L2 norm reduction
 *   4. flush(ctx)  <-- ring DWQ trigger for any DWQ-routed puts
 *   5. quiet(ctx)  <-- wait for DWQ completion
 *
 * The two halo puts target DIFFERENT peers (multi-peer in one launch);
 * v1.5 makes peer + dst_buf per-CALL on put_no_db, so this works
 * naturally without launch_multi or two separate launches.
 *
 * Run:
 *   FI_MR_CACHE_MAX_COUNT=0 \
 *     srun -N <nodes> -n <ranks> --ntasks-per-node=8 \
 *          ./jacobi -nx 1024 -ny 1024 -niter 200
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <hip/hip_runtime.h>

#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
#include "gicc/coll.hpp"
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"

#define HIP_CHECK(cmd) do {                                                    \
    hipError_t err = cmd;                                                      \
    if (err != hipSuccess) {                                                   \
        fprintf(stderr, "HIP error: %s at %s:%d\n",                            \
                hipGetErrorString(err), __FILE__, __LINE__);                   \
        exit(1);                                                               \
    }                                                                          \
} while (0)

typedef float real;
constexpr real tol = 1.0e-8f;
const real PI = 2.0f * std::asin(1.0f);

// =============================================================================
// Kernels
// =============================================================================

__global__ void initialize_boundaries_kernel(
    real* __restrict__ a_new, real* __restrict__ a,
    const real pi, const int offset,
    const int nx, const int my_ny, const int ny)
{
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny;
         iy += blockDim.x * gridDim.x) {
        const real y0 = sinf(2.0f * pi * (offset + iy) / (ny - 1));
        a[(iy + 1) * nx + 0]        = y0;
        a[(iy + 1) * nx + (nx - 1)] = y0;
        a_new[(iy + 1) * nx + 0]        = y0;
        a_new[(iy + 1) * nx + (nx - 1)] = y0;
    }
}

void launch_initialize_boundaries(real* a_new, real* a, real pi, int offset,
                                   int nx, int my_ny, int ny) {
    initialize_boundaries_kernel<<<(my_ny + 127) / 128, 128>>>(
        a_new, a, pi, offset, nx, my_ny, ny);
    HIP_CHECK(hipGetLastError());
}

// Fused jacobi step: two halo puts (different peers!) + compute + flush + quiet.
template <int BX, int BY>
__global__ void jacobi_step_kernel(
    gicc::DeviceCtx* ctx,
    real* __restrict__ a_new, const real* __restrict__ a,
    real* __restrict__ l2_norm,
    const int iy_start, const int iy_end, const int nx,
    const bool calculate_norm,
    int top_peer,    int top_dst_buf,
    int bottom_peer, int bottom_dst_buf,
    int src_buf,
    size_t src_off_top,    size_t src_off_bot,
    size_t dst_off_top,    size_t dst_off_bot,
    size_t halo_bytes)
{
    // Two halo puts to two different peers — multi-peer in one launch.
    // put_no_db must be called from all threads of block 0 (IPC route is
    // block-cooperative). Other blocks early-return inside put_no_db.
    gicc::put_no_db(ctx, top_peer,    top_dst_buf,    dst_off_top,
                    src_buf, src_off_top, halo_bytes);
    gicc::put_no_db(ctx, bottom_peer, bottom_dst_buf, dst_off_bot,
                    src_buf, src_off_bot, halo_bytes);

    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    real local_l2 = 0.0f;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25f * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                      a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
        if (calculate_norm) {
            real r = new_val - a[iy * nx + ix];
            local_l2 = r * r;
        }
    }

    if (calculate_norm) {
        for (int off = 16; off > 0; off /= 2)
            local_l2 += __shfl_down(local_l2, off);
        if ((threadIdx.x % 32) == 0 && (threadIdx.y % 32) == 0)
            atomicAdd(l2_norm, local_l2);
    }

    gicc::flush(ctx);   // single-thread MMIO trigger for any DWQ-routed puts
    gicc::quiet(ctx);   // poll DWQ completion
}

// =============================================================================
// CLI helpers
// =============================================================================

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T def) {
    T val = def;
    char** it = std::find(begin, end, arg);
    if (it != end && ++it != end) { std::istringstream(*it) >> val; }
    return val;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    return std::find(begin, end, arg) != end;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
    unset_rocr_visible_devices();

    gicc::Runtime rt;
    int rank = rt.rank();
    int size = rt.size();

    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 200);
    const int nccheck  = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx_cli   = get_argval<int>(argv, argv + argc, "-nx", 1024);
    const int ny_cli   = get_argval<int>(argv, argv + argc, "-ny", 1024);
    const bool csv     = get_arg(argv, argv + argc, "-csv");

    const int nx = nx_cli;
    int chunk_size = (ny_cli - 2) / size;
    if (chunk_size < 2) chunk_size = 2;
    const int ny = chunk_size * size + 2;
    const size_t row_bytes = (size_t)nx * sizeof(real);
    const size_t buf_rows  = (size_t)chunk_size + 2;
    const size_t buf_size  = (size_t)nx * buf_rows * sizeof(real);

    if (rank == 0 && !csv) {
        printf("GICC/OFI Jacobi (unified launch): %d ranks, mesh %d x %d, "
               "chunk %d, buf %zu bytes, %d iters\n",
               size, ny, nx, chunk_size, buf_size, iter_max);
    }

    real* buf[2];
    HIP_CHECK(hipMalloc(&buf[0], buf_size));
    HIP_CHECK(hipMalloc(&buf[1], buf_size));
    HIP_CHECK(hipMemset(buf[0], 0, buf_size));
    HIP_CHECK(hipMemset(buf[1], 0, buf_size));

    const int iy_start_global = rank * chunk_size + 1;
    const int iy_start = 1;
    const int iy_end   = chunk_size + 1;

    launch_initialize_boundaries(buf[0], buf[1], PI, iy_start_global - 1,
                                 nx, chunk_size, ny - 2);
    launch_initialize_boundaries(buf[1], buf[0], PI, iy_start_global - 1,
                                 nx, chunk_size, ny - 2);
    HIP_CHECK(hipDeviceSynchronize());

    auto gbuf0 = rt.register_buffer(buf[0], buf_size, true);
    auto gbuf1 = rt.register_buffer(buf[1], buf_size, true);
    rt.exchange();
    rt.boot().barrier();

    const int top    = (rank > 0) ? rank - 1 : size - 1;
    const int bottom = (rank + 1) % size;

    // Uniform chunks => top neighbor's bottom halo row is at iy_end on their buf.
    const size_t dst_offset_to_top    = (size_t)iy_end * row_bytes;  // their "bottom halo"
    const size_t dst_offset_to_bottom = 0;                           // their "top halo" (row 0)

    const size_t src_offset_top    = (size_t)iy_start * row_bytes;       // my row 1
    const size_t src_offset_bottom = (size_t)(iy_end - 1) * row_bytes;   // my last interior

    real* l2_norm_d;
    real* l2_norm_h;
    HIP_CHECK(hipMalloc(&l2_norm_d, sizeof(real)));
    HIP_CHECK(hipHostMalloc(&l2_norm_h, sizeof(real)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    constexpr int BX = 32, BY = 32;
    dim3 grid((nx + BX - 1) / BX, (chunk_size + BY - 1) / BY);
    dim3 block(BX, BY);

    auto run_step = [&](int cur_buf, int next_buf, bool calc_norm) {
        gicc::Buffer& gnext = (next_buf == 0) ? gbuf0 : gbuf1;

        HIP_CHECK(hipMemsetAsync(l2_norm_d, 0, sizeof(real), stream));

        gicc::launch<jacobi_step_kernel<BX, BY>>(rt, grid, block, (size_t)0, stream,
            buf[next_buf], buf[cur_buf], l2_norm_d,
            iy_start, iy_end, nx, calc_norm,
            top,    next_buf,
            bottom, next_buf,
            (int)gnext.lkey,
            src_offset_top, src_offset_bottom,
            dst_offset_to_top, dst_offset_to_bottom,
            (size_t)row_bytes);

        HIP_CHECK(hipStreamSynchronize(stream));
        rt.reset();
    };

    for (int w = 0; w < 3; w++) {
        run_step(/*cur=*/0, /*next=*/1, /*calc_norm=*/false);
        rt.boot().barrier();
    }

    if (rank == 0 && !csv)
        printf("Starting main loop: %d iters\n", iter_max);

    rt.boot().barrier();
    double t_start = rt.boot().wtime();

    int cur = 0, nxt = 1;
    real l2 = 1.0f;
    int iter = 0;
    for (; iter < iter_max && l2 > tol; iter++) {
        bool calc_norm = (iter % nccheck) == 0 || (!csv && (iter % 50) == 0);
        run_step(cur, nxt, calc_norm);

        if (calc_norm) {
            HIP_CHECK(hipMemcpy(l2_norm_h, l2_norm_d, sizeof(real),
                                hipMemcpyDeviceToHost));
            l2 = gicc::coll::allreduce_sum(rt.boot(), *l2_norm_h);
            l2 = std::sqrt(l2);
            if (rank == 0 && !csv && (iter % 50) == 0)
                printf("  iter %5d  l2=%.6e\n", iter, l2);
        }
        rt.boot().barrier();
        std::swap(cur, nxt);
    }

    HIP_CHECK(hipDeviceSynchronize());
    rt.boot().barrier();
    double t_end = rt.boot().wtime();

    if (rank == 0) {
        if (csv) {
            printf("ofi_jacobi, %d, %d, %d, %d, %d, %f, %d\n",
                   nx, ny, iter, nccheck, size, (t_end - t_start), (int)(l2 <= tol));
        } else {
            printf("Done: %d iters in %.4f s, final l2=%.6e\n",
                   iter, (t_end - t_start), l2);
        }
    }

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipHostFree(l2_norm_h));
    HIP_CHECK(hipFree(l2_norm_d));
    HIP_CHECK(hipFree(buf[1]));
    HIP_CHECK(hipFree(buf[0]));
    return 0;
}
