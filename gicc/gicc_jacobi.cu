/**
 * gicc_jacobi.cu - Jacobi Solver with GPU-Initiated RDMA using GICC API
 *
 * Rewrite of benchmarks/multi-gpu-programming-models/nvgda/jacobi_gpu_init.cu.
 * All IB/QP/MR boilerplate replaced by gicc::Runtime.
 * Fused kernel uses gicc::put() + gicc::quiet() for RDMA.
 *
 * Run: mpirun -np 2 ./gicc_jacobi -nx 8192 -ny 2048 -niter 1000
 */

#include "gicc.hpp"
#include "gicc_device.cuh"

#include <mpi.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (cudaSuccess != err) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(err); \
    } \
} while(0)

#define MPI_CHECK(call) do { \
    int s = call; \
    if (MPI_SUCCESS != s) { \
        fprintf(stderr, "MPI error at line %d\n", __LINE__); \
        exit(s); \
    } \
} while(0)

typedef float real;
constexpr real tol = 1.0e-8;
const real PI = 2.0 * std::asin(1.0);

// Halo info for address exchange
struct HaloInfo {
    uint64_t top_halo_addr[2];
    uint64_t bottom_halo_addr[2];
    uint32_t rkey[2];
};

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
        const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[(iy + 1) * nx + 0] = y0;
        a[(iy + 1) * nx + (nx - 1)] = y0;
        a_new[(iy + 1) * nx + 0] = y0;
        a_new[(iy + 1) * nx + (nx - 1)] = y0;
    }
}

// Basic Jacobi kernel (for single GPU reference)
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(
    real* __restrict__ a_new, const real* __restrict__ a,
    real* __restrict__ l2_norm,
    const int iy_start, const int iy_end, const int nx,
    const bool calculate_norm)
{
    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    real local_l2_norm = 0.0f;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25f * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                      a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
        if (calculate_norm) {
            real r = new_val - a[iy * nx + ix];
            local_l2_norm = r * r;
        }
    }

    if (calculate_norm) {
        real block_norm = BlockReduce(temp_storage).Sum(local_l2_norm);
        if (threadIdx.x == 0 && threadIdx.y == 0)
            atomicAdd(l2_norm, block_norm);
    }
}

/**
 * Fused Jacobi + GPU-initiated RDMA kernel using GICC API.
 * Last block triggers RDMA via gicc::put() + gicc::quiet().
 */
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel_gpu_rdma(
    real* __restrict__ a_new, const real* __restrict__ a,
    real* __restrict__ l2_norm,
    const int iy_start, const int iy_end, const int nx,
    const bool calculate_norm,
    // GICC DeviceCtx per neighbor
    gicc::DeviceCtx* top_ctx,
    gicc::DeviceCtx* bottom_ctx,
    // RDMA parameters
    uint64_t local_top_addr, uint64_t local_bottom_addr,
    uint32_t local_lkey,
    uint64_t remote_top_addr, uint32_t remote_top_rkey,
    uint64_t remote_bottom_addr, uint32_t remote_bottom_rkey,
    uint32_t row_bytes,
    bool send_top, bool send_bottom,
    unsigned int* block_done_counter)
{
    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    real local_l2_norm = 0.0f;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25f * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                      a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
        if (calculate_norm) {
            real r = new_val - a[iy * nx + ix];
            local_l2_norm = r * r;
        }
    }

    if (calculate_norm) {
        real block_norm = BlockReduce(temp_storage).Sum(local_l2_norm);
        if (threadIdx.x == 0 && threadIdx.y == 0)
            atomicAdd(l2_norm, block_norm);
    }

    // Grid-wide sync: last block triggers RDMA
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        unsigned int num_blocks = gridDim.x * gridDim.y;
        unsigned int old = atomicInc(block_done_counter, num_blocks);

        if (old == num_blocks - 1) {
            __threadfence_system();

            if (send_top && top_ctx) {
                gicc::put(top_ctx, local_top_addr, local_lkey,
                          remote_top_addr, remote_top_rkey, row_bytes, true);
            }
            if (send_bottom && bottom_ctx) {
                gicc::put(bottom_ctx, local_bottom_addr, local_lkey,
                          remote_bottom_addr, remote_bottom_rkey, row_bytes, true);
            }
            if (send_top && top_ctx) gicc::quiet(top_ctx);
            if (send_bottom && bottom_ctx) gicc::quiet(bottom_ctx);
        }
    }
}

// =============================================================================
// Host wrappers
// =============================================================================

void launch_initialize_boundaries(real* a_new, real* a, real pi, int offset,
                                   int nx, int my_ny, int ny) {
    initialize_boundaries_kernel<<<(my_ny + 127) / 128, 128>>>(
        a_new, a, pi, offset, nx, my_ny, ny);
}

void launch_jacobi_kernel(real* a_new, const real* a, real* l2_norm,
                          int iy_start, int iy_end, int nx, bool calc_norm,
                          cudaStream_t stream) {
    constexpr int BX = 32, BY = 32;
    dim3 grid((nx + BX - 1) / BX, (iy_end - iy_start + BY - 1) / BY);
    jacobi_kernel<BX, BY><<<grid, dim3(BX, BY), 0, stream>>>(
        a_new, a, l2_norm, iy_start, iy_end, nx, calc_norm);
}

void launch_jacobi_gpu_rdma(
    real* a_new, const real* a, real* l2_norm,
    int iy_start, int iy_end, int nx, bool calc_norm,
    gicc::DeviceCtx* top_ctx, gicc::DeviceCtx* bottom_ctx,
    uint64_t local_top, uint64_t local_bottom, uint32_t local_lkey,
    uint64_t remote_top_addr, uint32_t remote_top_rkey,
    uint64_t remote_bottom_addr, uint32_t remote_bottom_rkey,
    uint32_t row_bytes, bool send_top, bool send_bottom,
    unsigned int* block_done_counter, cudaStream_t stream)
{
    constexpr int BX = 32, BY = 32;
    dim3 grid((nx + BX - 1) / BX, (iy_end - iy_start + BY - 1) / BY);
    jacobi_kernel_gpu_rdma<BX, BY><<<grid, dim3(BX, BY), 0, stream>>>(
        a_new, a, l2_norm, iy_start, iy_end, nx, calc_norm,
        top_ctx, bottom_ctx,
        local_top, local_bottom, local_lkey,
        remote_top_addr, remote_top_rkey,
        remote_bottom_addr, remote_bottom_rkey,
        row_bytes, send_top, send_bottom, block_done_counter);
}

// =============================================================================
// Single GPU reference
// =============================================================================

double single_gpu(int nx, int ny, int iter_max, real* a_ref_h, int nccheck, bool print) {
    real *a, *a_new;
    real *l2_norm_d, *l2_norm_h;
    cudaStream_t stream;

    int iy_start = 1, iy_end = ny - 3;

    CUDA_CHECK(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&a_new, nx * ny * sizeof(real)));
    CUDA_CHECK(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_CHECK(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    launch_initialize_boundaries(a, a_new, PI, 0, nx, ny - 2, ny - 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&l2_norm_d, sizeof(real)));
    CUDA_CHECK(cudaMallocHost(&l2_norm_h, sizeof(real)));

    if (print) printf("Single GPU Jacobi: %d iters on %d x %d\n", iter_max, ny, nx);

    int iter = 0;
    real l2_norm = 1.0f;
    double start = MPI_Wtime();

    while (l2_norm > tol && iter < iter_max) {
        CUDA_CHECK(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), stream));
        bool calc = (iter % nccheck) == 0 || (iter % 100) == 0;
        launch_jacobi_kernel(a_new, a, l2_norm_d, iy_start, iy_end, nx, calc, stream);

        CUDA_CHECK(cudaMemcpyAsync(a_new, a_new + (iy_end - 1) * nx, nx * sizeof(real),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(a_new + iy_end * nx, a_new + iy_start * nx, nx * sizeof(real),
                                   cudaMemcpyDeviceToDevice, stream));

        if (calc) {
            CUDA_CHECK(cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            l2_norm = std::sqrt(*l2_norm_h);
            if (print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }
        std::swap(a_new, a);
        iter++;
    }

    double stop = MPI_Wtime();
    CUDA_CHECK(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(real), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(l2_norm_h));
    CUDA_CHECK(cudaFree(l2_norm_d));
    CUDA_CHECK(cudaFree(a_new));
    CUDA_CHECK(cudaFree(a));
    return stop - start;
}

// =============================================================================
// Argument parsing
// =============================================================================

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T def) {
    T val = def;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) { std::istringstream(*itr) >> val; }
    return val;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    return std::find(begin, end, arg) != end;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    // --- All IB/QP setup in one object ---
    gicc::Runtime rt;

    int rank = rt.rank(), size = rt.size();

    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 8192);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 2048);
    const bool csv = get_arg(argv, argv + argc, "-csv");
    const bool skip_verify = get_arg(argv, argv + argc, "-skip_verify");

    if (!csv && rank == 0) {
        printf("GICC Jacobi with GPU-Initiated RDMA\n");
        printf("  GPU: %s\n", rt.gpu_name());
    }

    // Single GPU reference (rank 0 only)
    real *a_ref_h = nullptr, *a_h = nullptr;
    double runtime_serial = 0.0;
    if (rank == 0) {
        CUDA_CHECK(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
        CUDA_CHECK(cudaMallocHost(&a_h, nx * ny * sizeof(real)));
        runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h, nccheck, !csv);
    }
    MPI_CHECK(MPI_Bcast(&runtime_serial, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    rt.barrier();

    // Domain decomposition
    int chunk_size_low = (ny - 2) / size;
    int chunk_size_high = chunk_size_low + 1;
    int num_ranks_low = size * chunk_size_low + size - (ny - 2);
    int chunk_size = (rank < num_ranks_low) ? chunk_size_low : chunk_size_high;

    // Allocate GPU arrays
    real* buf[2];
    size_t buf_size = nx * (chunk_size + 2) * sizeof(real);
    CUDA_CHECK(cudaMalloc(&buf[0], buf_size));
    CUDA_CHECK(cudaMalloc(&buf[1], buf_size));
    CUDA_CHECK(cudaMemset(buf[0], 0, buf_size));
    CUDA_CHECK(cudaMemset(buf[1], 0, buf_size));

    // Local domain boundaries
    int iy_start_global;
    if (rank < num_ranks_low)
        iy_start_global = rank * chunk_size_low + 1;
    else
        iy_start_global = num_ranks_low * chunk_size_low +
                          (rank - num_ranks_low) * chunk_size_high + 1;
    int iy_end_global = std::min(iy_start_global + chunk_size - 1, ny - 4);
    int iy_start = 1;
    int iy_end = (iy_end_global - iy_start_global + 1) + iy_start;

    // Boundary conditions
    launch_initialize_boundaries(buf[0], buf[1], PI, iy_start_global - 1, nx, chunk_size, ny - 2);
    launch_initialize_boundaries(buf[1], buf[0], PI, iy_start_global - 1, nx, chunk_size, ny - 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Register buffers via GICC ---
    auto gbuf0 = rt.register_buffer(buf[0], buf_size, true);
    auto gbuf1 = rt.register_buffer(buf[1], buf_size, true);
    rt.exchange();

    // Neighbors
    const int top = (rank > 0) ? rank - 1 : (size - 1);
    const int bottom = (rank + 1) % size;
    bool has_top = (size > 1);
    bool has_bottom = (size > 1);

    // Exchange halo addresses via MPI (specific row offsets within buffers)
    size_t halo_bytes = nx * sizeof(real);
    HaloInfo my_info;
    for (int i = 0; i < 2; i++) {
        real* b = buf[i];
        my_info.top_halo_addr[i] = (uint64_t)b;
        my_info.bottom_halo_addr[i] = (uint64_t)b + iy_end * nx * sizeof(real);
        my_info.rkey[i] = (i == 0) ? gbuf0.rkey : gbuf1.rkey;
    }

    HaloInfo top_info = {}, bottom_info = {};
    {
        MPI_Request reqs[4];
        int nreqs = 0;
        if (has_top) {
            int tag = std::min(rank, top);
            MPI_CHECK(MPI_Isend(&my_info, sizeof(HaloInfo), MPI_BYTE, top, tag,
                                MPI_COMM_WORLD, &reqs[nreqs++]));
            MPI_CHECK(MPI_Irecv(&top_info, sizeof(HaloInfo), MPI_BYTE, top, tag,
                                MPI_COMM_WORLD, &reqs[nreqs++]));
        }
        if (has_bottom) {
            int tag = size + std::min(rank, bottom);
            MPI_CHECK(MPI_Isend(&my_info, sizeof(HaloInfo), MPI_BYTE, bottom, tag,
                                MPI_COMM_WORLD, &reqs[nreqs++]));
            MPI_CHECK(MPI_Irecv(&bottom_info, sizeof(HaloInfo), MPI_BYTE, bottom, tag,
                                MPI_COMM_WORLD, &reqs[nreqs++]));
        }
        if (nreqs > 0) MPI_CHECK(MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE));
    }

    // --- Prepare GICC DeviceCtx per neighbor ---
    gicc::DeviceCtx* top_ctx = has_top ? rt.prepare(top, gbuf0.index) : nullptr;
    gicc::DeviceCtx* bottom_ctx = has_bottom ? rt.prepare(bottom, gbuf0.index) : nullptr;

    if (!csv && rank == 0) {
        printf("GICC setup complete: top=%p, bottom=%p\n",
               (void*)top_ctx, (void*)bottom_ctx);
    }

    // Allocate block counter and stream
    unsigned int* d_block_counter;
    CUDA_CHECK(cudaMalloc(&d_block_counter, sizeof(unsigned int)));

    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    real *l2_norm_d, *l2_norm_h;
    CUDA_CHECK(cudaMalloc(&l2_norm_d, sizeof(real)));
    CUDA_CHECK(cudaMallocHost(&l2_norm_h, sizeof(real)));

    // Warmup
    for (int w = 0; w < 10; w++) {
        CUDA_CHECK(cudaMemset(d_block_counter, 0, sizeof(unsigned int)));
        launch_jacobi_gpu_rdma(
            buf[1], buf[0], l2_norm_d, iy_start, iy_end, nx, false,
            top_ctx, bottom_ctx,
            (uint64_t)(buf[1] + iy_start * nx),
            (uint64_t)(buf[1] + (iy_end - 1) * nx),
            gbuf1.lkey,
            has_top ? top_info.bottom_halo_addr[1] : 0,
            has_top ? top_info.rkey[1] : 0,
            has_bottom ? bottom_info.top_halo_addr[1] : 0,
            has_bottom ? bottom_info.rkey[1] : 0,
            halo_bytes, has_top, has_bottom,
            d_block_counter, compute_stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        rt.barrier();
    }

    rt.barrier();
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!csv && rank == 0)
        printf("Jacobi: %d iters on %d x %d mesh\n", iter_max, ny, nx);

    int iter = 0;
    real l2_norm = 1.0f;
    int cur_buf = 0, next_buf = 1;

    rt.barrier();
    double start = MPI_Wtime();

    while (l2_norm > tol && iter < iter_max) {
        CUDA_CHECK(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));
        CUDA_CHECK(cudaMemsetAsync(d_block_counter, 0, sizeof(unsigned int), compute_stream));

        bool calc_norm = (iter % nccheck) == 0 || (!csv && (iter % 100) == 0);

        uint64_t local_top = (uint64_t)(buf[next_buf] + iy_start * nx);
        uint64_t local_bottom = (uint64_t)(buf[next_buf] + (iy_end - 1) * nx);
        uint32_t local_lkey = (next_buf == 0) ? gbuf0.lkey : gbuf1.lkey;

        uint64_t remote_top_addr = has_top ? top_info.bottom_halo_addr[next_buf] : 0;
        uint32_t remote_top_rkey = has_top ? top_info.rkey[next_buf] : 0;
        uint64_t remote_bottom_addr = has_bottom ? bottom_info.top_halo_addr[next_buf] : 0;
        uint32_t remote_bottom_rkey = has_bottom ? bottom_info.rkey[next_buf] : 0;

        launch_jacobi_gpu_rdma(
            buf[next_buf], buf[cur_buf], l2_norm_d, iy_start, iy_end, nx, calc_norm,
            top_ctx, bottom_ctx,
            local_top, local_bottom, local_lkey,
            remote_top_addr, remote_top_rkey,
            remote_bottom_addr, remote_bottom_rkey,
            halo_bytes, has_top, has_bottom,
            d_block_counter, compute_stream);

        if (calc_norm) {
            CUDA_CHECK(cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real),
                                       cudaMemcpyDeviceToHost, compute_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        rt.barrier();

        if (calc_norm) {
            MPI_CHECK(MPI_Allreduce(l2_norm_h, &l2_norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
            l2_norm = std::sqrt(l2_norm);
            if (!csv && rank == 0 && (iter % 100) == 0)
                printf("%5d, %0.6f\n", iter, l2_norm);
        }

        cur_buf = 1 - cur_buf;
        next_buf = 1 - next_buf;
        iter++;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    rt.barrier();
    double stop = MPI_Wtime();

    // Verify
    int result_correct = 1;
    if (!skip_verify) {
        real* local_h;
        CUDA_CHECK(cudaMallocHost(&local_h, chunk_size * nx * sizeof(real)));
        CUDA_CHECK(cudaMemcpy(local_h, buf[cur_buf] + nx, chunk_size * nx * sizeof(real),
                              cudaMemcpyDeviceToHost));

        if (rank == 0) {
            memcpy(a_h + iy_start_global * nx, local_h, chunk_size * nx * sizeof(real));
            for (int r = 1; r < size; r++) {
                int r_cs = (r < num_ranks_low) ? chunk_size_low : chunk_size_high;
                int r_start;
                if (r < num_ranks_low)
                    r_start = r * chunk_size_low + 1;
                else
                    r_start = num_ranks_low * chunk_size_low +
                              (r - num_ranks_low) * chunk_size_high + 1;
                MPI_CHECK(MPI_Recv(a_h + r_start * nx, r_cs * nx, MPI_FLOAT,
                                   r, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
            for (int iy = 1; result_correct && (iy < ny - 1); ++iy)
                for (int ix = 1; result_correct && (ix < nx - 1); ++ix)
                    if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                        fprintf(stderr, "ERROR: a[%d*%d+%d] = %f != %f\n",
                                iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                        result_correct = 0;
                    }
        } else {
            MPI_CHECK(MPI_Send(local_h, chunk_size * nx, MPI_FLOAT, 0, 100, MPI_COMM_WORLD));
        }
        CUDA_CHECK(cudaFreeHost(local_h));
        MPI_CHECK(MPI_Bcast(&result_correct, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    if (rank == 0 && result_correct) {
        if (csv) {
            printf("gicc, %d, %d, %d, %d, %d, 1, %f, %f\n",
                   nx, ny, iter_max, nccheck, size, (stop - start), runtime_serial);
        } else {
            printf("Num GPUs: %d.\n", size);
            printf("%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, efficiency: %8.2f\n",
                   ny, nx, runtime_serial, size, (stop - start),
                   runtime_serial / (stop - start),
                   runtime_serial / (size * (stop - start)) * 100);
        }
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaFreeHost(l2_norm_h));
    CUDA_CHECK(cudaFree(l2_norm_d));
    CUDA_CHECK(cudaFree(d_block_counter));
    CUDA_CHECK(cudaFree(buf[1]));
    CUDA_CHECK(cudaFree(buf[0]));
    if (a_h) CUDA_CHECK(cudaFreeHost(a_h));
    if (a_ref_h) CUDA_CHECK(cudaFreeHost(a_ref_h));

    MPI_CHECK(MPI_Finalize());
    return (result_correct == 1) ? 0 : 1;
}
