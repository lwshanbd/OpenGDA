/**
 * mpi_barrier_bench.cpp - Pure MPI_Barrier (CPU-triggered) reference for
 * comparing against gicc::Barrier latency at the same scale.
 *
 * Run: srun -N <nodes> -n <ranks> --ntasks-per-node=<ppn> ./mpi_barrier_bench [N]
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = (argc > 1) ? atoi(argv[1]) : 1000;

    if (rank == 0)
        printf("mpi_barrier_bench: %d ranks, %d barriers ... ", size, N);
    fflush(stdout);

    const int warmup = 10;
    for (int i = 0; i < warmup; i++) MPI_Barrier(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    for (int i = 0; i < N; i++) MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    double us = (t1.tv_sec - t0.tv_sec) * 1.0e6 +
                (t1.tv_nsec - t0.tv_nsec) / 1.0e3;
    if (rank == 0) {
        printf("DONE  (%.1f us total, %.2f us/barrier)\n", us, us / N);
    }

    MPI_Finalize();
    return 0;
}
