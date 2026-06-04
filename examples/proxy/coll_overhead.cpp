/*
 * coll_overhead.cpp - decompose GICC per-collective-call FIXED overhead vs the
 * actual communication, to explain the coop-vs-MPI gap. Proxy build.
 * Measures (8 ranks): MPI_Barrier, an empty cooperative launch+sync, a
 * cooperative kernel doing 2(N-1) grid.sync()s, rt.reset() with nothing
 * pending, and MPI_Allreduce@8KB for reference.
 */
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

namespace cg = cooperative_groups;
__global__ void empty_kernel() {}
__global__ void gridsync_kernel(int n) {
    cg::grid_group g = cg::this_grid();
    for (int i = 0; i < n; ++i) g.sync();
}

static int grid_blocks_for(const void* k) {
    int per_sm = 0, n_sm = 0, dev = 0;
    (void)hipGetDevice(&dev);
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, 256, 0);
    (void)hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount, dev);
    return (per_sm > 0 && n_sm > 0) ? per_sm * n_sm : 1;
}

int main(int argc, char** argv) {
    gicc::Runtime rt;
    const int rank = rt.rank();
    const int N    = rt.size();
    const int iters = (argc > 1) ? std::atoi(argv[1]) : 200;
    setvbuf(stdout, nullptr, _IONBF, 0);

    // a tiny registered buffer so prepare()/reset() have a valid runtime
    float* d = nullptr; (void)hipMalloc(&d, 1024);
    auto b = rt.register_buffer(d, 1024, true);
    rt.exchange(); rt.barrier();
    (void)b;

    const int gb_empty = grid_blocks_for((const void*)empty_kernel);
    const int gb_sync  = grid_blocks_for((const void*)gridsync_kernel);
    int nsync = 2 * (N - 1);

    double t0;
    auto sync_all = [](){ MPI_Barrier(MPI_COMM_WORLD); };
    auto report = [&](const char* name, double local) {
        double mx; MPI_Reduce(&local, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("  %-38s : %8.2f us\n", name, mx);
    };
    (void)gb_sync; (void)nsync;   // gridsync probe dropped (occupancy deadlock)
    if (rank == 0) printf("\n=== per-call fixed costs (%d ranks, max over ranks) ===\n", N);

    // 1. MPI_Barrier
    for (int i = 0; i < 20; ++i) MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) MPI_Barrier(MPI_COMM_WORLD);
    report("MPI_Barrier", (MPI_Wtime() - t0) / iters * 1e6);

    // 2. empty cooperative launch + hipDeviceSynchronize
    void* p0[] = {};
    for (int i = 0; i < 20; ++i) { hipLaunchCooperativeKernel((const void*)empty_kernel, dim3(gb_empty), dim3(256), p0, 0, 0); hipDeviceSynchronize(); }
    sync_all(); t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
        hipLaunchCooperativeKernel((const void*)empty_kernel, dim3(gb_empty), dim3(256), p0, 0, 0);
        hipDeviceSynchronize();
    }
    report("coop launch(empty)+devsync", (MPI_Wtime() - t0) / iters * 1e6);

    // 3. rt.reset() with nothing pending (proxy drain + ipc-stream sync)
    sync_all(); t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) { (void)rt.prepare(); rt.reset(); }
    report("rt.prepare()+reset() (idle)", (MPI_Wtime() - t0) / iters * 1e6);

    // 4. MPI_Allreduce @ 8KB (reference)
    float *a_in = nullptr, *a_out = nullptr;
    (void)hipMalloc(&a_in, N * 256 * sizeof(float));
    (void)hipMalloc(&a_out, N * 256 * sizeof(float));
    for (int i = 0; i < 20; ++i) MPI_Allreduce(a_in, a_out, N * 256, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    sync_all(); t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) MPI_Allreduce(a_in, a_out, N * 256, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    report("MPI_Allreduce @8KB (reference)", (MPI_Wtime() - t0) / iters * 1e6);

    rt.barrier();
    (void)hipFree(d); (void)hipFree(a_in); (void)hipFree(a_out);
    return 0;
}
