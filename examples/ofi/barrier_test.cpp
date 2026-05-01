/**
 * barrier_test.cpp - Test GPU-triggered dissemination barrier
 *
 * Tests both single-barrier mode and continuous mode.
 *
 * Run: FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 -t 2 ./barrier_test
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <hip/hip_runtime.h>

#include "gicc/bootstrap/bootstrap.hpp"
#include "gicc/platform/ofi/ofi_barrier_device.cuh"
#include "gicc/platform/ofi/internal/gicc_barrier.hpp"
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"

// =============================================================================
// Test 1: Single barrier per kernel launch
// =============================================================================

__global__ void single_barrier_kernel(gicc::BarrierCtx* bctx,
                                      volatile uint64_t* flag) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    gicc::barrier(bctx);
    *flag = 42;
}

// =============================================================================
// Test 2: Continuous mode — N barriers in one kernel launch
// barrier() advances its own dev_seen; the kernel just calls it in a loop.
// =============================================================================

__global__ void continuous_barrier_kernel(gicc::BarrierCtx* bctx,
                                          int num_barriers,
                                          volatile uint64_t* counter) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < num_barriers; i++) {
        gicc::barrier(bctx);
        *counter = i + 1;
    }
}

int main(int argc, char** argv) {
    unset_rocr_visible_devices();
    const int n_continuous = (argc > 1) ? atoi(argv[1]) : 50;

    int window_size = 8;
    if (const char* env = std::getenv("GICC_WINDOW")) {
        int w = std::atoi(env);
        if (w >= 1 && w <= 32) {
            window_size = w;
        } else {
            fprintf(stderr, "GICC: GICC_WINDOW=%s out of range [1,32], using default %d\n",
                    env, window_size);
        }
    }

    gicc::Bootstrap boot;
    gicc::Fabric comm(boot);
    int rank = comm.rank();
    int nranks = comm.size();

    if (rank == 0)
        printf("barrier_test: %d ranks, GPU %d\n", nranks, comm.gpu_id());

    boot.barrier();

    // =========================================================================
    // Test 1: Single barrier mode (no monitor thread needed)
    // =========================================================================
    {
        gicc::Barrier barrier(comm, window_size);

        volatile uint64_t* h_flag = nullptr;
        hipHostMalloc((void**)&h_flag, sizeof(uint64_t), hipHostMallocDefault);
        *h_flag = 0;

        constexpr int N_SINGLE = 10;
        if (rank == 0) printf("Test 1: %d single barriers ... ", N_SINGLE);
        fflush(stdout);

        for (int i = 0; i < N_SINGLE; i++) {
            *h_flag = 0;
            barrier.setup();

            boot.barrier();

            hipLaunchKernelGGL(single_barrier_kernel, dim3(1), dim3(1), 0, 0,
                               barrier.device_ctx(), (volatile uint64_t*)h_flag);
            hipDeviceSynchronize();
            barrier.reset();

            boot.barrier();

            if (*h_flag != 42) {
                printf("FAILED at iteration %d (flag=%lu)\n", i, (unsigned long)*h_flag);
                hipHostFree((void*)h_flag);
                return 1;
            }
        }

        hipHostFree((void*)h_flag);
        boot.barrier();
        if (rank == 0) printf("PASSED\n");
        fflush(stdout);
    }

    // =========================================================================
    // Test 2: Continuous mode (needs monitor thread)
    // =========================================================================
    {
        gicc::Barrier barrier(comm, window_size);
        barrier.init();

        boot.barrier();

        volatile uint64_t* h_counter = nullptr;
        hipHostMalloc((void**)&h_counter, sizeof(uint64_t), hipHostMallocDefault);
        *h_counter = 0;

        const int N_CONTINUOUS = n_continuous;
        if (rank == 0) printf("Test 2: %d continuous barriers ... ", N_CONTINUOUS);
        fflush(stdout);

        boot.barrier();
        timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

        barrier.start_continuous(N_CONTINUOUS);
        hipLaunchKernelGGL(continuous_barrier_kernel, dim3(1), dim3(1), 0, 0,
                           barrier.device_ctx(), N_CONTINUOUS,
                           (volatile uint64_t*)h_counter);
        barrier.wait_continuous();
        hipDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        double elapsed_us = (t1.tv_sec - t0.tv_sec) * 1.0e6 +
                            (t1.tv_nsec - t0.tv_nsec) / 1.0e3;

        if (*h_counter != (uint64_t)N_CONTINUOUS) {
            printf("FAILED (counter=%lu, expected=%d)\n",
                   (unsigned long)*h_counter, N_CONTINUOUS);
            hipHostFree((void*)h_counter);
            barrier.finalize();
            return 1;
        }

        hipHostFree((void*)h_counter);
        barrier.finalize();
        boot.barrier();
        if (rank == 0) {
            printf("PASSED  (%.1f us total, %.2f us/barrier)\n",
                   elapsed_us, elapsed_us / N_CONTINUOUS);
        }
        fflush(stdout);
    }

    if (rank == 0) printf("All barrier tests PASSED\n");
    return 0;
}
