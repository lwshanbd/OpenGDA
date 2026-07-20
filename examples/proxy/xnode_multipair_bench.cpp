/*
 * xnode_multipair_bench.cpp - multi-rank-pair cross-node CPU-proxy BW bench.
 *
 * Goal: test whether aggregate cross-node bandwidth scales past the single-NIC
 * ~24 GB/s ceiling when N rank-pairs run concurrently, each pair landing on a
 * different GPU-affine CXI NIC (Tioga has 4 NICs/node).
 *
 * Layout: 2 nodes, P ranks per node (P = nranks/2). Rank r in [0,P) is a
 * SENDER on node 0; its peer is r+P on node 1 (a RECEIVER, one-sided target).
 * All P senders fire concurrently inside one timed window. Aggregate BW =
 * P * iters * bytes / max_sender_time.
 *
 * Each rank's GICC Runtime independently selects its GPU-affine NIC via the
 * affinity detector, so P senders on distinct GCDs => distinct NICs.
 *
 * Run: srun -N 2 -n 2P --ntasks-per-node=P ./xnode_multipair_bench
 */
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <mpi.h>

#include <hip/hip_runtime.h>

// Proxy mode: device pushes N puts into the ring, then flush.
__global__ void put_kernel(gicc::DeviceCtx* ctx, int peer, int buf_idx,
                           size_t bytes, int n) {
    for (int i = 0; i < n; ++i) {
        gicc::put(ctx, peer, buf_idx, /*dst_off=*/0,
                       buf_idx, /*src_off=*/0, bytes);
    }
    gicc::flush(ctx);
}

// DWQ mode: host pre-enqueues the deferred works; this kernel only fires the
// MMIO trigger that releases the whole queued batch.
__global__ void dwq_trigger_kernel(gicc::DeviceCtx* ctx) {
    gicc::flush(ctx);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // --mode=proxy (default) | dwq ; --batch=B (puts per flush/trigger boundary)
    std::string mode = "proxy";
    int batch = 0;   // 0 => one big batch of all kIters (coarsest)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--mode=", 0) == 0) mode = a.substr(7);
        else if (a.rfind("--batch=", 0) == 0) batch = std::atoi(a.substr(8).c_str());
    }

    gicc::Runtime rt;
    if (mode == "dwq") rt.enable_host_wait_mode();
    int rank = rt.rank();
    int nranks = rt.size();
    if (nranks < 2 || (nranks % 2) != 0) {
        if (rank == 0) fprintf(stderr, "need even nranks >= 2 (got %d)\n", nranks);
        return 1;
    }
    int P = nranks / 2;                 // pairs = senders on node 0
    bool is_sender = (rank < P);
    int peer = is_sender ? (rank + P) : (rank - P);

    const size_t kMaxBytes = 16 * 1024 * 1024;
    void* d_buf = nullptr;
    hipMalloc(&d_buf, kMaxBytes);
    hipMemset(d_buf, 0xCD, kMaxBytes);
    hipDeviceSynchronize();

    auto bh = rt.register_buffer(d_buf, kMaxBytes, /*is_device=*/true);
    rt.exchange();
    rt.prepare();
    rt.reset();
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t kSizes[] = {8, 256, 4096, 65536, 262144, 1048576, 4194304, 16257024};
    const int kIters = 200;
    // Aggregation granularity: B puts per flush/trigger+reset boundary.
    // batch<=0 or >kIters => one coarse batch of all kIters.
    int B = (batch <= 0 || batch > kIters) ? kIters : batch;

    if (rank == 0) {
        printf("# multipair bench: mode=%s P=%d pairs, %d ranks, iters=%d batch=%d\n",
               mode.c_str(), P, nranks, kIters, B);
        printf("# bytes  per_pair_us  per_pair_GBps  aggregate_GBps\n");
    }

    for (size_t si = 0; si < sizeof(kSizes)/sizeof(kSizes[0]); ++si) {
        size_t bytes = kSizes[si];
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        if (is_sender) {
            // Issue kIters puts in chunks of B; each chunk = one
            // flush/trigger + reset boundary (the aggregation granularity).
            for (int done = 0; done < kIters; done += B) {
                int chunk = (done + B <= kIters) ? B : (kIters - done);
                if (mode == "dwq") {
                    for (int i = 0; i < chunk; ++i)
                        rt.put(bh, peer, bh.index, bytes, /*src_off=*/0, /*dst_off=*/0);
                    gicc::DeviceCtx* d_ctx = rt.prepare();
                    gpuLaunchKernel(dwq_trigger_kernel, dim3(1), dim3(1), 0, 0, d_ctx);
                    hipDeviceSynchronize();
                    rt.reset();
                } else {
                    gicc::DeviceCtx* d_ctx = rt.prepare();
                    gpuLaunchKernel(put_kernel, dim3(1), dim3(1), 0, 0,
                                    d_ctx, peer, bh.index, bytes, chunk);
                    hipDeviceSynchronize();
                    rt.reset();
                }
            }
        }
        double t1 = MPI_Wtime();
        double my_us = (t1 - t0) * 1e6 / kIters;

        // Aggregate: use the SLOWEST sender as the wall-clock for the batch.
        double max_us = 0.0;
        MPI_Reduce(&my_us, &max_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            double per_pair_gbps = bytes / max_us / 1e3;
            double agg_gbps = per_pair_gbps * P;
            printf("%zu %.3f %.2f %.2f\n",
                   bytes, max_us, per_pair_gbps, agg_gbps);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    rt.reset();
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}
