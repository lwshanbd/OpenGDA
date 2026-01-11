/**
 * mm_hip_gda_proxy.cpp - Distributed Matrix Multiplication with HIP + GDA Proxy Barrier
 *
 * Uses HIP for GPU computation and OpenGDA proxy barrier for GPU-triggered RDMA communication.
 * The proxy barrier enables unlimited iterations without pre-allocating per-iteration resources.
 *
 * Run:
 *   srun -n <npes> ./mm_hip_gda_proxy [matrix_size]
 *   srun -n 2 ./mm_hip_gda_proxy 4096
 */

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "gda.h"
#include "gda_barrier_proxy.h"

namespace cg = cooperative_groups;

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

/**
 * Reconstruct local As stripe for a given rank from the deterministic pattern.
 * As is Ns x N (Ns rows, N columns).
 * As[local_i][j] = (local_i * N + j + mype) % 11 + 7
 */
void reconstruct_local_As(float* As, int N, int Ns, int mype)
{
    for (int local_i = 0; local_i < Ns; local_i++) {
        for (int j = 0; j < N; j++) {
            int idx = local_i * N + j;
            As[idx] = (idx + mype) % 11 + 7;
        }
    }
}

/**
 * Reconstruct full matrix B from the deterministic initialization pattern.
 * B is N x N, stored as vertical stripes across ranks.
 * B[k][r*Ns + local_j] = (k * Ns + local_j + r) % 13 + 5
 */
void reconstruct_full_B(float* B, int N, int Ns, int npes)
{
    for (int r = 0; r < npes; r++) {
        for (int k = 0; k < N; k++) {
            for (int local_j = 0; local_j < Ns; local_j++) {
                int global_j = r * Ns + local_j;
                int idx = k * Ns + local_j;
                B[k * N + global_j] = (idx + r) % 13 + 5;
            }
        }
    }
}

/**
 * Compute reference for local C stripe: Cs = As * B
 * As is Ns x N, B is N x N, Cs is Ns x N
 */
void cpu_matmul_local_reference(const float* As, const float* B, float* Cs, int N, int Ns)
{
    for (int i = 0; i < Ns; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += As[i * N + k] * B[k * N + j];
            }
            Cs[i * N + j] = sum;
        }
    }
}

/**
 * Compare local C stripe with reference and return max absolute error.
 */
float compare_local_stripe(const float* Cs, const float* Cs_ref, int N, int Ns,
                           int mype, int& num_errors)
{
    float max_error = 0.0f;
    num_errors = 0;
    const float tolerance = 1e-3f;

    for (int i = 0; i < Ns * N; i++) {
        float error = std::abs(Cs[i] - Cs_ref[i]);
        if (error > max_error) {
            max_error = error;
        }
        float rel_error = error / (std::abs(Cs_ref[i]) + 1e-6f);
        if (rel_error > tolerance && error > tolerance) {
            num_errors++;
            if (num_errors <= 3) {
                int row = i / N;
                int col = i % N;
                std::cerr << "  Rank " << mype << " mismatch at [" << row << "][" << col << "]: "
                          << "got " << Cs[i] << ", expected " << Cs_ref[i]
                          << ", error=" << error << std::endl;
            }
        }
    }
    return max_error;
}

float timediff_us(const timespec& t_start, const timespec& t_end)
{
    return (t_end.tv_sec - t_start.tv_sec) * 1.0e6 + (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3;
}

/**
 * Matrix multiplication kernel with GPU-triggered communication
 * Computes C = A * B using cannon-like algorithm with proxy barrier synchronization.
 *
 * Uses HIP Cooperative Groups for efficient grid-wide synchronization.
 * Must be launched with hipLaunchCooperativeKernel.
 */
__global__ void mm_kernel_proxy(
    const float* __restrict__ As,
    float* Bs,
    float* __restrict__ Cs,
    float* Bn,
    int N,
    int Ns,
    int mype,
    int npes,
    gda_gpu_handle_t* put_handles,
    gda_proxy_barrier_dev_t* barrier)
{
    // Get cooperative groups grid handle for efficient grid-wide sync
    cg::grid_group grid = cg::this_grid();

    // Check if this is the master thread (global thread 0)
    bool is_master = (threadIdx.x == 0 && threadIdx.y == 0 &&
                      blockIdx.x == 0 && blockIdx.y == 0);

    // Current Bs and Bn pointers (will swap each iteration)
    float* cur_Bs = Bs;
    float* cur_Bn = Bn;

    for (int s = 0; s < npes; s++) {
        int block_num = (mype + s) % npes;
        float* Cb = Cs + block_num * Ns;

        // Master thread triggers async put to left neighbor
        if (is_master) {
            gda_gpu_trigger(put_handles[s]);
        }

        // All threads perform matrix multiplication: Cb += As * Bs
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (k < N && j < Ns) {
            float b_kj = cur_Bs[k * Ns + j];
            for (int i = 0; i < Ns; i++) {
                atomicAdd(&Cb[i * N + j], As[i * N + k] * b_kj);
            }
        }

        // Grid-wide sync using cooperative groups (hardware-optimized)
        grid.sync();

        // Master thread handles communication synchronization
        if (is_master) {
            // Wait for our put to complete
            gda_gpu_wait(put_handles[s]);

            // Inter-rank barrier
            gda_gpu_proxy_barrier_wait(barrier);
        }

        // Grid-wide sync to ensure all threads see RDMA data
        grid.sync();

        // Memory fence to ensure RDMA-written data is visible to all threads
        __threadfence_system();

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

    // Matrix size from command line or default (same as MPI version)
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    N = (N / npes) * npes;  // Make divisible by npes

    int Ns = N / npes;
    size_t stripe_size = N * Ns * sizeof(float);
    size_t total_gpu_needed = 4 * stripe_size;

    if (mype == 0) {
        std::cout << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size << " bytes\n";
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

    // Define offsets for double-buffering
    // d_Bs is at offset 1 * stripe_size, d_Bn is at offset 3 * stripe_size
    size_t offset_Bs = 1 * stripe_size;
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

    // Allocate proxy barrier
    int window_size = 16;
    gda_proxy_barrier_t* barrier = gda_proxy_barrier_alloc(window_size);
    if (!barrier) {
        std::cerr << "Rank " << mype << ": Failed to allocate proxy GPU barrier" << std::endl;
        gda_finalize();
        return 1;
    }

    // Get device context for GPU kernel use
    gda_proxy_barrier_dev_t* dev_host = gda_proxy_barrier_get_dev(barrier);
    if (!dev_host) {
        std::cerr << "Rank " << mype << ": Failed to get device context" << std::endl;
        gda_proxy_barrier_free(barrier);
        gda_finalize();
        return 1;
    }

    // Pre-create put handles for all iterations
    int left_neighbor = (mype - 1 + npes) % npes;
    std::vector<gda_handle_t*> put_handles(npes);
    std::vector<gda_gpu_handle_t> gpu_put_handles(npes);

    for (int s = 0; s < npes; s++) {
        // Source alternates between d_Bs and d_Bn due to swapping
        float* src = (s % 2 == 0) ? d_Bs : d_Bn;
        // Target offset must alternate to avoid read-write races:
        // - Even iterations: GPU reads from d_Bs, so RDMA writes to d_Bn
        // - Odd iterations: GPU reads from d_Bn, so RDMA writes to d_Bs
        size_t dest_offset = (s % 2 == 0) ? offset_Bn : offset_Bs;
        put_handles[s] = gda_put(src, stripe_size, left_neighbor, dest_offset);
        if (!put_handles[s]) {
            std::cerr << "Rank " << mype << ": Failed to create put handle " << s << std::endl;
            gda_finalize();
            return 1;
        }
        gpu_put_handles[s] = put_handles[s]->gpu;
    }

    // Copy handles to GPU
    gda_gpu_handle_t* d_put_handles;
    HIP_CHECK(hipMalloc(&d_put_handles, npes * sizeof(gda_gpu_handle_t)));
    HIP_CHECK(hipMemcpy(d_put_handles, gpu_put_handles.data(),
                        npes * sizeof(gda_gpu_handle_t), hipMemcpyHostToDevice));

    // Copy device context to GPU memory
    gda_proxy_barrier_dev_t* d_barrier = nullptr;
    HIP_CHECK(hipMalloc(&d_barrier, sizeof(gda_proxy_barrier_dev_t)));
    HIP_CHECK(hipMemcpy(d_barrier, dev_host, sizeof(gda_proxy_barrier_dev_t), hipMemcpyHostToDevice));

    timespec t0, t1;

    // Synchronize before starting
    gda_barrier();

    // Start proxy thread BEFORE launching kernel
    if (gda_proxy_start(barrier) != 0) {
        std::cerr << "Rank " << mype << ": Failed to start proxy thread" << std::endl;
        gda_proxy_barrier_free(barrier);
        gda_finalize();
        return 1;
    }

    gda_barrier();

    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    // Launch cooperative kernel - requires all blocks to run concurrently
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (Ns + blockDim.y - 1) / blockDim.y);

    // Check that cooperative launch is supported and get max blocks
    int num_blocks_per_sm = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm, mm_kernel_proxy, blockDim.x * blockDim.y, 0));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int max_blocks = num_blocks_per_sm * prop.multiProcessorCount;
    int requested_blocks = gridDim.x * gridDim.y;

    if (requested_blocks > max_blocks) {
        // Reduce grid size to fit cooperative launch requirements
        // Scale down gridDim.x proportionally
        float scale = (float)max_blocks / requested_blocks;
        gridDim.x = std::max(1u, (unsigned int)(gridDim.x * scale));
        gridDim.y = std::max(1u, (unsigned int)(gridDim.y * scale));
        if (mype == 0) {
            std::cerr << "Warning: Reduced grid from " << requested_blocks
                      << " to " << (gridDim.x * gridDim.y)
                      << " blocks for cooperative launch" << std::endl;
        }
    }

    // Prepare kernel arguments for cooperative launch
    void* kernel_args[] = {
        &d_As, &d_Bs, &d_Cs, &d_Bn,
        &N, &Ns, &mype, &npes,
        &d_put_handles, &d_barrier
    };

    HIP_CHECK(hipLaunchCooperativeKernel(
        (void*)mm_kernel_proxy,
        gridDim, blockDim,
        kernel_args,
        0,      // shared memory
        0       // stream
    ));
    HIP_CHECK(hipDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    // Stop proxy thread
    gda_proxy_stop(barrier);

    // Get proxy stats before cleanup
    gda_proxy_stats_t stats;
    gda_proxy_barrier_get_stats(barrier, &stats);

    // Cleanup
    gda_flush();
    for (int s = 0; s < npes; s++) {
        gda_free(put_handles[s]);
    }
    gda_proxy_barrier_free(barrier);
    HIP_CHECK(hipFree(d_put_handles));
    HIP_CHECK(hipFree(d_barrier));

    gda_barrier();

    if (mype == 0) {
        std::cout << "GDA + HIP (proxy barrier): " << timediff_us(t0, t1) << " us\n";
        std::cout << "  Barrier iterations: " << npes << "\n";
        std::cout << "  Proxy rearms: " << stats.total_rearms << "\n";
        std::cout << "  Proxy polls: " << stats.queue_polls << "\n";
        std::cout << "  CQ events: " << stats.cq_events_drained << "\n";
    }

    // Local correctness verification - each rank verifies its own C stripe
    const int MAX_VERIFY_N = 512;  // Disabled for performance testing
    bool do_verify = (N <= MAX_VERIFY_N);

    if (do_verify) {
        // Copy computed C stripe to host
        auto h_Cs = new float[N * Ns];
        HIP_CHECK(hipMemcpy(h_Cs, d_Cs, stripe_size, hipMemcpyDeviceToHost));

        // Reconstruct local As and full B on CPU (deterministic, no communication needed)
        auto h_As = new float[N * Ns];
        auto h_B = new float[N * N];
        auto h_Cs_ref = new float[N * Ns];

        reconstruct_local_As(h_As, N, Ns, mype);
        reconstruct_full_B(h_B, N, Ns, npes);

        // Compute local reference: Cs_ref = As * B
        cpu_matmul_local_reference(h_As, h_B, h_Cs_ref, N, Ns);

        // Compare local result with reference
        int num_errors = 0;
        float max_error = compare_local_stripe(h_Cs, h_Cs_ref, N, Ns, mype, num_errors);

        // Report results (serialize output with barriers)
        for (int r = 0; r < npes; r++) {
            if (mype == r) {
                if (num_errors == 0) {
                    std::cout << "Rank " << mype << ": VERIFICATION PASSED (max error: " << max_error << ")" << std::endl;
                } else {
                    std::cout << "Rank " << mype << ": VERIFICATION FAILED (" << num_errors << " errors, max error: " << max_error << ")" << std::endl;
                }
            }
            gda_barrier();
        }

        delete[] h_Cs_ref;
        delete[] h_B;
        delete[] h_As;
        delete[] h_Cs;
    } else if (mype == 0) {
        std::cout << "Skipping verification (N=" << N << " > " << MAX_VERIFY_N << ")" << std::endl;
    }

    gda_barrier();
    gda_finalize();

    if (mype == 0) {
        std::cout << "All ranks completed successfully!" << std::endl;
    }
    return 0;
}
