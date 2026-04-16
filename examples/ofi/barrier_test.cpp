/**
 * barrier_test.cpp - Test GPU-triggered dissemination barrier
 *
 * Tests both single-barrier mode and continuous mode.
 *
 * Run: FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 -t 2 ./barrier_test
 */
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <hip/hip_runtime.h>

#include "gicc/platform/ofi/ofi_barrier_device.cuh"
#include "gicc/platform/ofi/internal/gicc_barrier.hpp"

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
// Increment expected_signal AFTER each barrier (matches host setup ordering).
// =============================================================================

__global__ void continuous_barrier_kernel(gicc::BarrierCtx* bctx,
                                          int num_barriers,
                                          volatile uint64_t* counter) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < num_barriers; i++) {
        gicc::barrier(bctx);
        bctx->expected_signal++;
        *counter = i + 1;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    GdaComm comm;
    int rank = comm.rank();
    int nranks = comm.size();

    if (rank == 0)
        printf("barrier_test: %d ranks, GPU %d\n", nranks, comm.gpu_id());

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Test 1: Single barrier mode (no monitor thread needed)
    // =========================================================================
    {
        gicc::Barrier barrier(comm);

        volatile uint64_t* h_flag = nullptr;
        hipHostMalloc((void**)&h_flag, sizeof(uint64_t), hipHostMallocDefault);
        *h_flag = 0;

        constexpr int N_SINGLE = 10;
        if (rank == 0) printf("Test 1: %d single barriers ... ", N_SINGLE);
        fflush(stdout);

        for (int i = 0; i < N_SINGLE; i++) {
            *h_flag = 0;
            barrier.setup();

            MPI_Barrier(MPI_COMM_WORLD);

            hipLaunchKernelGGL(single_barrier_kernel, dim3(1), dim3(1), 0, 0,
                               barrier.device_ctx(), (volatile uint64_t*)h_flag);
            hipDeviceSynchronize();
            barrier.reset();

            MPI_Barrier(MPI_COMM_WORLD);

            if (*h_flag != 42) {
                printf("FAILED at iteration %d (flag=%lu)\n", i, (unsigned long)*h_flag);
                hipHostFree((void*)h_flag);
                MPI_Finalize();
                return 1;
            }
        }

        hipHostFree((void*)h_flag);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) printf("PASSED\n");
        fflush(stdout);
    }

    // =========================================================================
    // Test 2: Continuous mode (needs monitor thread)
    // =========================================================================
    {
        gicc::Barrier barrier(comm);
        barrier.init();

        MPI_Barrier(MPI_COMM_WORLD);

        volatile uint64_t* h_counter = nullptr;
        hipHostMalloc((void**)&h_counter, sizeof(uint64_t), hipHostMallocDefault);
        *h_counter = 0;

        constexpr int N_CONTINUOUS = 50;
        if (rank == 0) printf("Test 2: %d continuous barriers ... ", N_CONTINUOUS);
        fflush(stdout);

        barrier.start_continuous(N_CONTINUOUS);
        hipLaunchKernelGGL(continuous_barrier_kernel, dim3(1), dim3(1), 0, 0,
                           barrier.device_ctx(), N_CONTINUOUS,
                           (volatile uint64_t*)h_counter);
        barrier.wait_continuous();
        hipDeviceSynchronize();

        if (*h_counter != (uint64_t)N_CONTINUOUS) {
            printf("FAILED (counter=%lu, expected=%d)\n",
                   (unsigned long)*h_counter, N_CONTINUOUS);
            hipHostFree((void*)h_counter);
            barrier.finalize();
            MPI_Finalize();
            return 1;
        }

        hipHostFree((void*)h_counter);
        barrier.finalize();
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) printf("PASSED\n");
        fflush(stdout);
    }

    if (rank == 0) printf("All barrier tests PASSED\n");
    MPI_Finalize();
    return 0;
}
