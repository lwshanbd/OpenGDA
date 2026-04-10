/**
 * barrier_new_test.cpp - Test GPU-triggered dissemination barrier
 *
 * Tests the GdaBarrierNew class which implements a dissemination barrier
 * using DWQ (Deferred Work Queue) operations.
 *
 * Usage:
 *   srun -N 2 -n 2 --ntasks-per-node=1 ./barrier_new_test
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
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

int main(int argc, char** argv) {
    unset_rocr_visible_devices();
    MPI_Init(&argc, &argv);

    GdaComm comm;
    int rank = comm.rank();
    int size = comm.size();

    printf("Rank %d/%d: GPU-triggered Dissemination Barrier Test\n", rank, size);
    fflush(stdout);

    // Parse arguments
    int n_iterations = (argc > 1) ? atoi(argv[1]) : 100;
    int warmup = (argc > 2) ? atoi(argv[2]) : 10;

    if (rank == 0) {
        printf("  Ranks: %d\n", size);
        printf("  Iterations: %d (warmup: %d)\n", n_iterations, warmup);
        printf("  Barrier rounds: %d\n", (int)ceil(log2(size)));
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Create barrier once (reused across iterations via reset)
    GdaBarrierNew barrier(comm);

    // Timing variables
    timespec t_start, t_end;
    double total_time_us = 0;

    // Run iterations
    int total_iterations = warmup + n_iterations;
    for (int iter = 0; iter < total_iterations; iter++) {
        // Setup DWQ operations for this barrier
        barrier.setup();

        MPI_Barrier(MPI_COMM_WORLD);

        // Time the barrier
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_start);

        // Launch barrier kernel
        hipLaunchKernelGGL(barrier_new_kernel,
                           dim3(1), dim3(1), 0, 0,
                           barrier.get_device_context());
        HIP_CHECK(hipDeviceSynchronize());

        clock_gettime(CLOCK_MONOTONIC_RAW, &t_end);

        if (iter >= warmup) {
            total_time_us += timediff_us(t_start, t_end);
        }

        // Reset for next iteration (reuses counters with new threshold)
        barrier.reset();

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Report results
    if (rank == 0) {
        double avg_time_us = total_time_us / n_iterations;

        printf("\n=== Results ===\n");
        printf("Total barriers completed: %d\n", n_iterations);
        printf("Total time: %.2f us\n", total_time_us);
        printf("Average barrier latency: %.2f us\n", avg_time_us);
        printf("Barrier count (verify reset): %lu\n", barrier.get_barrier_count());
        printf("\nSUCCESS\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
