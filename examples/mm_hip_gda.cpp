/**
 * mm_hip_gda.cpp - Distributed Matrix Multiplication with HIP + GDA
 *
 * Uses HIP for GPU computation and OpenGDA for GPU-triggered RDMA communication.
 * The entire computation loop runs in a single GPU kernel with GPU-side barriers.
 *
 * Run:
 *   srun -n <npes> ./mm_hip_gda
 */

#include <iostream>
#include <utility>
#include <ctime>
#include <vector>

#include <hip/hip_runtime.h>
#include "gda.h"

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
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
    return (t_end.tv_sec - t_start.tv_sec) * 1.0e6 + (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3;
}

/**
 * Full matrix multiplication kernel with all iterations
 * Runs entirely on GPU without returning to CPU during main computation.
 */
__global__ void mm_full_kernel(
    const float* __restrict__ As,
    float* Bs,
    float* __restrict__ Cs,
    float* Bn,
    int N,
    int Ns,
    int mype,
    int npes,
    gda_gpu_handle_t* put_handles,
    gda_gpu_barrier_t* barrier)
{
    bool is_master = (threadIdx.x == 0 && threadIdx.y == 0 &&
                      blockIdx.x == 0 && blockIdx.y == 0);

    // Current Bs and Bn pointers (will swap each iteration)
    float* cur_Bs = Bs;
    float* cur_Bn = Bn;

    for (int s = 0; s < npes; s++) {
        int block_num = (mype + s) % npes;
        float* Cb = Cs + block_num * Ns;

        // All threads participate in put operation (for efficient IPC parallel copy)
        // For DWQ mode, only thread 0 will actually trigger
        gda_gpu_trigger_all(put_handles[s]);
        __syncthreads();

        // All threads perform matrix multiplication
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (k < N && j < Ns) {
            float b_kj = cur_Bs[k * Ns + j];
            for (int i = 0; i < Ns; i++) {
                atomicAdd(&Cb[i * N + j], As[i * N + k] * b_kj);
            }
        }

        // Wait for put to complete (master only)
        __syncthreads();
        if (is_master) {
            gda_gpu_wait(put_handles[s]);
        }
        __syncthreads();

        // GPU-side barrier to ensure all ranks have completed their puts
        if (is_master) {
            gda_gpu_barrier_wait(barrier);
        }
        __syncthreads();

        // Swap Bs and Bn for next iteration
        float* tmp = cur_Bs;
        cur_Bs = cur_Bn;
        cur_Bn = tmp;
    }
}

/**
 * HIP kernel to initialize matrix data
 */
__global__ void init_matrix_kernel(float* buf, int size, int mype, int pattern)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (pattern == 0) {
            buf[idx] = (idx + mype) % 11 + 7;  // As pattern
        } else if (pattern == 1) {
            buf[idx] = (idx + mype) % 13 + 5;  // Bs pattern
        } else {
            buf[idx] = 0;  // Cs, Bn pattern (zero)
        }
    }
}

int main(int argc, char** argv)
{
    // Initialize GDA
    if (gda_init() != 0) {
        std::cerr << "Failed to initialize OpenGDA" << std::endl;
        return 1;
    }

    int mype = gda_rank();
    int npes = gda_size();


    // Matrix size from command line or default
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    N = (N / npes) * npes;  // Make divisible by npes
    const int Ns = N / npes;
    const size_t stripe_size = N * Ns * sizeof(float);
    const size_t total_gpu_needed = 4 * stripe_size;

    if (mype == 0) {
        std::cout << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size << " bytes\n";
        std::cout << "Total GPU memory needed: " << total_gpu_needed << " bytes\n";
    }

    // Get GPU buffer
    float* gpu_buf = (float*)gda_gpu_buf();
    size_t gpu_buf_size = gda_gpu_buf_size();

    if (!gpu_buf || gpu_buf_size < total_gpu_needed) {
        std::cerr << "Rank " << mype << ": Insufficient GPU buffer. Need "
                  << total_gpu_needed << ", have " << gpu_buf_size << std::endl;
        gda_finalize();
        return 1;
    }

    // Layout GPU buffer
    float* d_As = gpu_buf;
    float* d_Bs = gpu_buf + N * Ns;
    float* d_Cs = gpu_buf + 2 * N * Ns;
    float* d_Bn = gpu_buf + 3 * N * Ns;

    size_t offset_Bn = 3 * stripe_size;

    // Initialize matrices on GPU
    int blockSize = 256;
    int numBlocks = (N * Ns + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_As, N * Ns, mype, 0);
    hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_Bs, N * Ns, mype, 1);
    hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_Cs, N * Ns, mype, 2);
    hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_Bn, N * Ns, mype, 2);
    HIP_CHECK(hipDeviceSynchronize());

    // Allocate GPU barrier with npes iterations (one per loop iteration)
    gda_gpu_barrier_t* barrier = gda_gpu_barrier_alloc(npes);
    if (!barrier) {
        std::cerr << "Rank " << mype << ": Failed to allocate GPU barrier" << std::endl;
        gda_finalize();
        return 1;
    }

    // Pre-create all put handles for all iterations
    int left_neighbor = (mype - 1 + npes) % npes;
    std::vector<gda_handle_t*> put_handles(npes);
    std::vector<gda_gpu_handle_t> gpu_put_handles(npes);

    for (int s = 0; s < npes; s++) {
        // Source alternates between d_Bs and d_Bn due to swapping
        float* src = (s % 2 == 0) ? d_Bs : d_Bn;
        put_handles[s] = gda_put(src, stripe_size, left_neighbor, offset_Bn);
        if (!put_handles[s]) {
            std::cerr << "Rank " << mype << ": Failed to create put handle " << s << std::endl;
            gda_finalize();
            return 1;
        }
        gpu_put_handles[s] = put_handles[s]->gpu;
    }

    // Copy handles and barrier to GPU
    gda_gpu_handle_t* d_put_handles;
    gda_gpu_barrier_t* d_barrier;
    HIP_CHECK(hipMalloc(&d_put_handles, npes * sizeof(gda_gpu_handle_t)));
    HIP_CHECK(hipMalloc(&d_barrier, sizeof(gda_gpu_barrier_t)));
    HIP_CHECK(hipMemcpy(d_put_handles, gpu_put_handles.data(),
                        npes * sizeof(gda_gpu_handle_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_barrier, barrier, sizeof(gda_gpu_barrier_t), hipMemcpyHostToDevice));

    timespec t0, t1;

    // Synchronize before starting
    gda_barrier();
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    // Launch single kernel for entire computation
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (Ns + blockDim.y - 1) / blockDim.y);

    hipLaunchKernelGGL(mm_full_kernel, gridDim, blockDim, 0, 0,
                       d_As, d_Bs, d_Cs, d_Bn, N, Ns, mype, npes,
                       d_put_handles, d_barrier);
    HIP_CHECK(hipDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    // Cleanup
    gda_flush();
    for (int s = 0; s < npes; s++) {
        gda_free(put_handles[s]);
    }
    gda_gpu_barrier_free(barrier);
    HIP_CHECK(hipFree(d_put_handles));
    HIP_CHECK(hipFree(d_barrier));

    gda_barrier();

    if (mype == 0) {
        std::cout << "GDA + HIP (single kernel + GPU barrier): " << timediff_us(t0, t1) << " us\n";
    }

    // For verification
    if (N < 32) {
        auto h_Cs = new float[N * Ns];
        HIP_CHECK(hipMemcpy(h_Cs, d_Cs, stripe_size, hipMemcpyDeviceToHost));
        if (mype == 0) {
            auto C = new float[N * N];
            for (int i = 0; i < Ns * N; i++)
                C[i] = h_Cs[i];
            print_matrix(C, N, N);
            delete[] C;
        }
        delete[] h_Cs;
    }

    gda_barrier();
    gda_finalize();

    std::cout << "Rank " << mype << ": Done!" << std::endl;
    return 0;
}
