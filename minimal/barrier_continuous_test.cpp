/**
 * barrier_continuous_test.cpp - Test continuous barrier mode
 *
 * Tests multiple barriers in a single kernel launch.
 * CPU proxy thread handles setup/reset between barriers.
 * GPU notifies CPU via host-visible memory.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <mpi.h>
#include <hip/hip_runtime.h>

#include "gda_barrier_new.hpp"
#include "device_affinity.hpp"

using namespace opengda;

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

static double timediff_us(const timespec& t_start, const timespec& t_end) {
    return (t_end.tv_sec - t_start.tv_sec) * 1.0e6 +
           (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3;
}

/**
 * Kernel that executes N barriers in a loop.
 */
__global__ void multi_barrier_kernel(barrier_new_context_t* ctx, int n_barriers) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int i = 0; i < n_barriers; i++) {
        barrier_new(ctx);
        ctx->expected_signal++;
    }
}

int main(int argc, char** argv) {
    unset_rocr_visible_devices();
    MPI_Init(&argc, &argv);

    GdaComm comm;
    int rank = comm.rank();
    int size = comm.size();

    int n_barriers = (argc > 1) ? atoi(argv[1]) : 10;

    if (rank == 0) {
        printf("Continuous Barrier Test: %d ranks, %d barriers\n", size, n_barriers);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Create and init barrier
    GdaBarrierNew barrier(comm);
    barrier.init();

    MPI_Barrier(MPI_COMM_WORLD);

    // Start continuous mode and launch kernel
    timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t_start);

    barrier.start_continuous(n_barriers);

    hipLaunchKernelGGL(multi_barrier_kernel,
                       dim3(1), dim3(1), 0, 0,
                       barrier.get_device_context(),
                       n_barriers);
    HIP_CHECK(hipDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC_RAW, &t_end);
    double total_time_us = timediff_us(t_start, t_end);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total time: %.2f us, per-barrier: %.2f us\n",
               total_time_us, total_time_us / n_barriers);
        printf("SUCCESS\n");
    }

    barrier.finalize();
    MPI_Finalize();
    return 0;
}
