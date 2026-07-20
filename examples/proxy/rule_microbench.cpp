/*
 * rule_microbench.cpp - per-size proxy-vs-dwq, and the rule's achieved envelope.
 *
 * For each size we measure BOTH transports and report which the hand-written
 * rule (gicc_rule.h) picks, then whether the rule's pick equals the per-size
 * optimum (i.e. the rule achieves the lower envelope of {proxy, dwq}).
 *
 * SYMMETRY REQUIREMENT (user): the proxy and DWQ send paths must look the same.
 * Both go through send_batch() with identical structure:
 *   issue N ops (one API call each) -> one trigger kernel -> sync -> reset.
 * The ONLY difference is which op-issuing call + which trigger the transport
 * uses, isolated in issue_one()/fire(). Everything else is shared.
 *
 * Transport is a process-global runtime mode (enable_host_wait_mode), so a
 * single process measures ONE transport; we run the binary twice (--mode=proxy,
 * --mode=dwq) and a driver script merges the two logs + applies the rule. This
 * mirrors reality: the pass picks transport per call-site at compile time, it is
 * not switched mid-run.
 *
 * 2 ranks, rank0 sender. Run: srun -N2 -n2 --ntasks-per-node=1 ./rule_microbench --mode=...
 */
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
#include "gicc_rule.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <hip/hip_runtime.h>

// ---- the two transports, kept structurally identical ----------------------
// PROXY: device pushes each op, device flush fires the ring batch.
__global__ void proxy_send_kernel(gicc::DeviceCtx* ctx, int peer, int buf,
                                  size_t bytes, int n) {
    for (int i = 0; i < n; ++i)
        gicc::put(ctx, peer, buf, 0, buf, 0, bytes);
    gicc::flush(ctx);
}
// DWQ: host pre-enqueues each op, device flush fires the MMIO trigger.
__global__ void dwq_trigger_kernel(gicc::DeviceCtx* ctx) {
    gicc::flush(ctx);
}

struct Bench {
    gicc::Runtime& rt;
    gicc::Buffer&  bh;
    int peer;
    bool is_dwq;

    // One batch of n ops of `bytes`, then fire + wait. SAME shape for both
    // transports; the if(is_dwq) only swaps the op-issue + trigger calls.
    double send_batch(size_t bytes, int n) {
        double t0 = MPI_Wtime();
        if (is_dwq) {
            for (int i = 0; i < n; ++i)
                rt.put(bh, peer, bh.index, bytes, 0, 0);   // host enqueue
            gicc::DeviceCtx* ctx = rt.prepare();
            gpuLaunchKernel(dwq_trigger_kernel, dim3(1), dim3(1), 0, 0, ctx);
        } else {
            gicc::DeviceCtx* ctx = rt.prepare();
            gpuLaunchKernel(proxy_send_kernel, dim3(1), dim3(1), 0, 0,
                            ctx, peer, bh.index, bytes, n);  // device push
        }
        hipDeviceSynchronize();
        rt.reset();
        return (MPI_Wtime() - t0) * 1e6 / n;   // us per op
    }
};

int main(int argc, char** argv) {
    int provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    std::string mode = "proxy";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--mode=", 0) == 0) mode = a.substr(7);
    }
    bool is_dwq = (mode == "dwq");

    gicc::Runtime rt;
    if (is_dwq) rt.enable_host_wait_mode();
    int rank = rt.rank(), nranks = rt.size();
    if (nranks != 2) { if (!rank) fprintf(stderr,"need 2 ranks\n"); return 1; }
    int peer = 1 - rank;

    const size_t kMax = 16*1024*1024;
    void* d = nullptr; hipMalloc(&d, kMax); hipMemset(d, 0xCD, kMax); hipDeviceSynchronize();
    auto bh = rt.register_buffer(d, kMax, true);
    rt.exchange(); rt.prepare(); rt.reset(); MPI_Barrier(MPI_COMM_WORLD);

    const size_t sizes[] = {8,64,256,1024,4096,16384,65536,262144,1048576,4194304};
    const int N = 50;          // coalesced batch (rule: always coalesce)
    const int outer = 30;

    Bench b{rt, bh, peer, is_dwq};
    if (rank == 0) printf("# mode=%s  size  us_per_op\n", mode.c_str());
    for (size_t s : sizes) {
        for (int w = 0; w < 5; ++w) if (rank==0) b.send_batch(s, N);  // warmup
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<double> v;
        for (int o = 0; o < outer; ++o) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) v.push_back(b.send_batch(s, N));
            else rt.reset();
        }
        if (rank == 0) {
            std::sort(v.begin(), v.end());
            printf("%zu %.3f\n", s, v[v.size()/2]);  // median
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    rt.reset(); MPI_Barrier(MPI_COMM_WORLD); MPI_Finalize();
    return 0;
}
