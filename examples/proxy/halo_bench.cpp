/*
 * halo_bench.cpp - a real comm pattern (1-D halo exchange) over proxy vs dwq,
 * plus the rule's achieved time. This is the benchmark (not a microbench): each
 * "step" exchanges a MIX of message sizes with a neighbor, like a stencil app
 * whose halo has faces of different extents.
 *
 * Ring topology: rank r exchanges with r-1 and r+1 (wrap). Each step sends a
 * set of faces (heterogeneous sizes) to each neighbor, then waits. We run many
 * steps and report total time.
 *
 * SYMMETRY (user requirement): proxy and DWQ use the SAME exchange_step()
 * structure -- issue all face-puts, one trigger/flush, sync, reset. The only
 * difference is the op-issue + trigger call, isolated exactly like in
 * rule_microbench.cpp. Transport is process-global (enable_host_wait_mode), so
 * we run the binary once per mode; a driver merges results. The "rule" mode is
 * emulated by the driver: for a per-face transport choice you'd need the pass
 * (compile-time per-call lowering) -- here we report what the rule WOULD pick
 * per face and the achievable time if each face used its optimal transport.
 *
 * Run: srun -N2 -n<P> --ntasks-per-node=... ./halo_bench --mode=proxy|dwq
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

__global__ void proxy_face_kernel(gicc::DeviceCtx* ctx, int peer, int buf,
                                  const size_t* offs, const size_t* lens,
                                  int nfaces) {
    for (int f = 0; f < nfaces; ++f)
        gicc::put(ctx, peer, buf, offs[f], buf, offs[f], lens[f]);
    gicc::flush(ctx);
}
__global__ void dwq_trigger_kernel(gicc::DeviceCtx* ctx) { gicc::flush(ctx); }

// A stencil-like halo: faces of mixed sizes (corner/edge/face of a 3-D tile).
// Sizes chosen to straddle the measured proxy/dwq crossover so transport choice
// actually matters.
static const size_t kFaceBytes[] = {
    64,        // corner-ish tiny
    512,       // edge   (dwq band)
    2048,      // edge   (dwq band)
    32768,     // face   (proxy)
    262144,    // big face (proxy, NIC-bound)
};
static const int kNFaces = sizeof(kFaceBytes)/sizeof(kFaceBytes[0]);

int main(int argc, char** argv) {
    int provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    std::string mode = "proxy";
    int steps = 200;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--mode=",0)==0) mode = a.substr(7);
        else if (a.rfind("--steps=",0)==0) steps = atoi(a.substr(8).c_str());
    }
    bool is_dwq = (mode == "dwq");

    gicc::Runtime rt;
    if (is_dwq) rt.enable_host_wait_mode();
    int rank = rt.rank(), nranks = rt.size();
    if (nranks < 2) { if(!rank) fprintf(stderr,"need >=2 ranks\n"); return 1; }
    int right = (rank + 1) % nranks;        // one-directional ring (send right)

    // Buffer holds all faces contiguously.
    size_t total = 0; std::vector<size_t> offs(kNFaces), lens(kNFaces);
    for (int f = 0; f < kNFaces; ++f) { offs[f]=total; lens[f]=kFaceBytes[f]; total+=kFaceBytes[f]; }
    total = (total + 4095) & ~size_t(4095);

    void* d=nullptr; hipMalloc(&d, total); hipMemset(d,0xCD,total); hipDeviceSynchronize();
    auto bh = rt.register_buffer(d, total, true);
    rt.exchange(); rt.prepare(); rt.reset(); MPI_Barrier(MPI_COMM_WORLD);

    // device-side face tables for the proxy kernel
    size_t *d_offs=nullptr,*d_lens=nullptr;
    hipMalloc(&d_offs,kNFaces*sizeof(size_t)); hipMalloc(&d_lens,kNFaces*sizeof(size_t));
    hipMemcpy(d_offs,offs.data(),kNFaces*sizeof(size_t),hipMemcpyHostToDevice);
    hipMemcpy(d_lens,lens.data(),kNFaces*sizeof(size_t),hipMemcpyHostToDevice);

    // One halo step: send all faces to the right neighbor, fire, wait.
    // SAME structure for both transports.
    bool dbg2 = getenv("HALO_DEBUG") != nullptr;
    auto exchange_step = [&]() {
        if (is_dwq) {
            for (int f = 0; f < kNFaces; ++f)
                rt.put(bh, right, bh.index, lens[f], offs[f], offs[f]);
            if (dbg2){fprintf(stderr,"[r%d] enqueued\n",rank);fflush(stderr);}
            gicc::DeviceCtx* ctx = rt.prepare();
            gpuLaunchKernel(dwq_trigger_kernel, dim3(1),dim3(1),0,0, ctx);
        } else {
            gicc::DeviceCtx* ctx = rt.prepare();
            gpuLaunchKernel(proxy_face_kernel, dim3(1),dim3(1),0,0,
                            ctx, right, bh.index, d_offs, d_lens, kNFaces);
        }
        if (dbg2){fprintf(stderr,"[r%d] launched, syncing\n",rank);fflush(stderr);}
        hipDeviceSynchronize();
        if (dbg2){fprintf(stderr,"[r%d] synced, reset\n",rank);fflush(stderr);}
        rt.reset();
    };

    bool dbg = getenv("HALO_DEBUG") != nullptr;
    for (int w=0; w<10; ++w) {
        if (dbg) { fprintf(stderr,"[r%d] warmup %d begin\n",rank,w); fflush(stderr); }
        exchange_step();
        if (dbg) { fprintf(stderr,"[r%d] warmup %d done\n",rank,w); fflush(stderr); }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int s=0; s<steps; ++s) {
        if (dbg) { fprintf(stderr,"[r%d] step %d begin\n",rank,s); fflush(stderr); }
        exchange_step();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double t1 = MPI_Wtime();

    double my_us = (t1-t0)*1e6/steps;
    double max_us=0; MPI_Reduce(&my_us,&max_us,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (rank==0) {
        size_t bytes_per_step=0; for(int f=0;f<kNFaces;++f) bytes_per_step+=lens[f];
        printf("# halo_bench mode=%s ranks=%d steps=%d faces=%d bytes/step=%zu\n",
               mode.c_str(), nranks, steps, kNFaces, bytes_per_step);
        printf("HALO %s %.3f us/step\n", mode.c_str(), max_us);
    }
    rt.reset(); MPI_Barrier(MPI_COMM_WORLD); MPI_Finalize();
    return 0;
}
