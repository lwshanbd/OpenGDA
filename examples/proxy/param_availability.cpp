/*
 * param_availability.cpp — what parameter availability costs.
 *
 * Three call sites that a communication runtime cannot tell apart. Every
 * one issues 32 puts of the same size to the same peer over the same
 * fabric, with the same amount of independent work behind them. What
 * differs is only where the descriptor's offsets come from, which is a
 * property of the program text:
 *
 *   A  pre-launch known    off = base + i * bytes, from kernel formals
 *                          and the induction variable. The host can
 *                          reconstruct every descriptor before the launch.
 *
 *   B  device-entry known  off = desc[i].off, read out of an array the
 *                          kernel receives. The values exist when the
 *                          kernel starts but live in device memory, so
 *                          the host cannot read them through the ordinary
 *                          API — unless it is told the array has a host
 *                          mirror, which is what GICC_KERNEL_HOST_MIRROR
 *                          declares. The SAME kernel is compiled with and
 *                          without that annotation here.
 *
 *   C  device-dynamic      off = d_off[i], produced by a previous kernel.
 *                          Nothing outside the device knows the value
 *                          before the kernel runs.
 *
 * The point is not that one is faster. It is that the set of LEGAL
 * lowerings differs, and only a compile-time analysis establishes it. A
 * runtime asked to stage a descriptor for site C cannot: the number does
 * not exist yet. Get this wrong in the permissive direction and the
 * transfer silently carries whatever was in the buffer at staging time.
 *
 * So this measures two things:
 *   1. what each site's legal set is  (read off features.json, and the
 *      B-with / B-without pair shows the analysis MOVING that boundary)
 *   2. what the restriction costs     (best legal time vs the best time
 *      an unrestricted site of the same shape achieves)
 *
 * Two ranks, one per node:
 *   GICC_SKIP_DWQ_INIT=1 GICC_PROXY_ENABLED=1 srun -p pci -N 2 -n 2 \
 *       --ntasks-per-node=1 -c 8 --gpu-bind=none -t 3 ./param_availability
 */

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gicc/platform/ofi/annotations.hpp"
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"

static constexpr size_t kBufBytes = 64 * 1024 * 1024;
static constexpr size_t kRecvBase = 32 * 1024 * 1024;

// One descriptor element, mirrored on the host for class B.
struct XferDesc {
    unsigned long long src_off;
    unsigned long long dst_off;
};

//----------------------------------------------------------------------------
// Kernels — one per availability class. They differ ONLY in the expression
// that produces the offsets.
//----------------------------------------------------------------------------

__device__ __forceinline__ void spin_ticks(long long ticks) {
    if (ticks <= 0) return;
    long long t0 = wall_clock64();
    while ((wall_clock64() - t0) < ticks) { }
}

// A — every field is a kernel formal, a literal, or the induction
// variable. Host-knowable by construction.
__global__ void k_prelaunch(gicc::DeviceCtx* ctx, int peer, int buf,
                            unsigned long long src_base,
                            unsigned long long dst_base,
                            unsigned long long bytes, int n,
                            long long ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < n; ++i) {
        gicc::put(ctx, peer, buf, dst_base + (unsigned long long)i * bytes,
                       buf, src_base + (unsigned long long)i * bytes, bytes);
    }
    spin_ticks(ticks);
    gicc::quiet(ctx);
}

// B — offsets come out of an array the kernel is handed. The annotation
// tells the pass a host mirror exists, so the trace function can read the
// same values before the launch. Without it this is indistinguishable
// from class C as far as the analysis is concerned.
// The annotation goes on the FUNCTION and names the formal: Clang emits
// those into @llvm.global.annotations, where the pass can find them.
// Attached to the parameter instead it becomes a local llvm.var.annotation
// that no module-level walk will ever see, and the site silently stays
// proxy-only — which is exactly the failure this benchmark is about, so
// the positional form is given too in case the build strips value names.
__global__ void GICC_KERNEL_HOST_MIRROR(desc) GICC_KERNEL_HOST_MIRROR_PARAM(3)
k_device_entry(gicc::DeviceCtx* ctx, int peer, int buf,
                               const XferDesc* desc,
                               unsigned long long bytes, int n,
                               long long ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < n; ++i) {
        gicc::put(ctx, peer, buf, desc[i].dst_off,
                       buf, desc[i].src_off, bytes);
    }
    spin_ticks(ticks);
    gicc::quiet(ctx);
}

// B' — byte-identical body, no annotation. The pass has no way to reach
// the values, so the site drops to proxy-only.
__global__ void k_device_entry_unmirrored(gicc::DeviceCtx* ctx, int peer,
                                          int buf, const XferDesc* desc,
                                          unsigned long long bytes, int n,
                                          long long ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < n; ++i) {
        gicc::put(ctx, peer, buf, desc[i].dst_off,
                       buf, desc[i].src_off, bytes);
    }
    spin_ticks(ticks);
    gicc::quiet(ctx);
}

// C — the offsets are computed on the device immediately before use.
// No mirror can exist, because the values did not exist at launch.
__global__ void k_device_dynamic(gicc::DeviceCtx* ctx, int peer, int buf,
                                 unsigned long long* scratch,
                                 unsigned long long src_base,
                                 unsigned long long dst_base,
                                 unsigned long long bytes, int n,
                                 long long ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // A device-side permutation: the order is decided here, not before.
    for (int i = 0; i < n; ++i)
        scratch[i] = (unsigned long long)((i * 7 + 3) % n);
    for (int i = 0; i < n; ++i) {
        const unsigned long long slot = scratch[i];
        gicc::put(ctx, peer, buf, dst_base + slot * bytes,
                       buf, src_base + slot * bytes, bytes);
    }
    spin_ticks(ticks);
    gicc::quiet(ctx);
}

// Trigger-path firing kernel: release the staged batch, then do the same
// independent work the proxy kernels do, so the two paths are compared on
// equal terms rather than one of them skipping the phase's compute.
__global__ void k_fire(gicc::DeviceCtx* ctx, long long ticks) {
    gicc::flush(ctx);
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    spin_ticks(ticks);
}

//----------------------------------------------------------------------------

struct Stats { double median = 0, min = 0, max = 0; int n = 0; };

static Stats summarize(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = (int)v.size();
    s.median = (v.size() % 2) ? v[v.size() / 2]
                              : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    s.min = v.front();
    s.max = v.back();
    return s;
}

template <typename F>
static Stats time_it(gicc::Runtime& rt, int rank, int samples, int warmup, F&& body) {
    for (int w = 0; w < warmup; ++w) { rt.barrier(); if (rank == 0) body(); rt.barrier(); }
    std::vector<double> v;
    for (int s = 0; s < samples; ++s) {
        rt.barrier();
        double t0 = MPI_Wtime();
        if (rank == 0) body();
        double t1 = MPI_Wtime();
        rt.barrier();
        if (rank == 0) v.push_back((t1 - t0) * 1e6);
    }
    return summarize(std::move(v));
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    size_t bytes   = 65536;
    int    ops     = 32;
    int    dist_us = 0;
    int    samples = 15, warmup = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--bytes=",   0) == 0) bytes   = (size_t)atol(a.c_str() + 8);
        else if (a.rfind("--ops=",     0) == 0) ops     = atoi(a.c_str() + 6);
        else if (a.rfind("--dist=",    0) == 0) dist_us = atoi(a.c_str() + 7);
        else if (a.rfind("--samples=", 0) == 0) samples = atoi(a.c_str() + 10);
        else if (a.rfind("--warmup=",  0) == 0) warmup  = atoi(a.c_str() + 9);
        else { fprintf(stderr, "unknown arg '%s'\n", a.c_str()); return 2; }
    }

    gicc::Runtime rt;
    // Both paths live in one process: host-wait mode brings up the DWQ so
    // descriptors can be staged, and mixed dispatch keeps the proxy rings
    // alive alongside it. Do NOT set GICC_SKIP_DWQ_INIT for this binary —
    // the trigger BAR has to map, and an empty value still counts as set.
    rt.enable_host_wait_mode();
    rt.enable_mixed_dispatch();
    const int rank = rt.rank(), nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "need exactly 2 ranks (got %d)\n", nranks);
        return 1;
    }
    const int peer = 1 - rank;

    void* d_buf = nullptr;
    if (hipMalloc(&d_buf, kBufBytes) != hipSuccess) return 3;
    (void)hipMemset(d_buf, rank == 0 ? 0xA5 : 0x00, kBufBytes);

    XferDesc*           d_desc = nullptr;
    unsigned long long* d_scr  = nullptr;
    (void)hipMalloc(&d_desc, sizeof(XferDesc) * ops);
    (void)hipMalloc(&d_scr,  sizeof(unsigned long long) * ops);

    // The class-B descriptor table, built on the host and copied down. The
    // mirror registration is what lets the trace function read the same
    // values the kernel will read.
    std::vector<XferDesc> h_desc(ops);
    for (int i = 0; i < ops; ++i) {
        h_desc[i].src_off = (unsigned long long)i * bytes;
        h_desc[i].dst_off = kRecvBase + (unsigned long long)i * bytes;
    }
    (void)hipMemcpy(d_desc, h_desc.data(), sizeof(XferDesc) * ops,
                    hipMemcpyHostToDevice);
    (void)hipDeviceSynchronize();

    auto bh = rt.register_buffer(d_buf, kBufBytes, /*is_device=*/true);
    rt.register_host_mirror(d_desc, h_desc.data(), sizeof(XferDesc), ops);
    rt.exchange();
    (void)rt.prepare();
    rt.reset();
    rt.barrier();

    int wall_khz = 0;
    (void)hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0);
    const double ticks_per_us = wall_khz > 0 ? wall_khz / 1000.0 : 100.0;
    const long long ticks = (long long)(dist_us * ticks_per_us);

    if (rank == 0) {
        printf("=== param_availability bytes=%zu ops=%d dist=%dus ===\n",
               bytes, ops, dist_us);
        printf("CSV,site,availability,path,legal,median_us,min_us,max_us\n");
    }

    auto emit = [&](const char* site, const char* avail, const char* path,
                    bool legal, const Stats& s) {
        if (rank != 0 || s.n == 0) return;
        printf("CSV,%s,%s,%s,%d,%.3f,%.3f,%.3f\n",
               site, avail, path, legal ? 1 : 0, s.median, s.min, s.max);
        printf("  %-3s %-16s %-8s %-7s median=%9.2f us  [%.2f .. %.2f]\n",
               site, avail, path, legal ? "legal" : "ILLEGAL",
               s.median, s.min, s.max);
    };

    // ---- proxy path: legal everywhere, because the worker reads the
    // descriptor at submit time rather than before the launch.
    emit("A", "pre-launch", "proxy", true,
         time_it(rt, rank, samples, warmup, [&] {
             gicc::DeviceCtx* c = rt.prepare();
             hipLaunchKernelGGL(k_prelaunch, dim3(1), dim3(1), 0, 0, c, peer,
                                bh.index, 0ull, (unsigned long long)kRecvBase,
                                (unsigned long long)bytes, ops, ticks);
             (void)hipDeviceSynchronize();
             rt.reset();
         }));
    emit("B", "device-entry", "proxy", true,
         time_it(rt, rank, samples, warmup, [&] {
             gicc::DeviceCtx* c = rt.prepare();
             hipLaunchKernelGGL(k_device_entry, dim3(1), dim3(1), 0, 0, c, peer,
                                bh.index, d_desc, (unsigned long long)bytes,
                                ops, ticks);
             (void)hipDeviceSynchronize();
             rt.reset();
         }));
    emit("B'", "device-entry", "proxy", true,
         time_it(rt, rank, samples, warmup, [&] {
             gicc::DeviceCtx* c = rt.prepare();
             hipLaunchKernelGGL(k_device_entry_unmirrored, dim3(1), dim3(1), 0, 0,
                                c, peer, bh.index, d_desc,
                                (unsigned long long)bytes, ops, ticks);
             (void)hipDeviceSynchronize();
             rt.reset();
         }));
    emit("C", "device-dynamic", "proxy", true,
         time_it(rt, rank, samples, warmup, [&] {
             gicc::DeviceCtx* c = rt.prepare();
             hipLaunchKernelGGL(k_device_dynamic, dim3(1), dim3(1), 0, 0, c, peer,
                                bh.index, d_scr, 0ull,
                                (unsigned long long)kRecvBase,
                                (unsigned long long)bytes, ops, ticks);
             (void)hipDeviceSynchronize();
             rt.reset();
         }));

    // ---- trigger path: the host stages every descriptor before the
    // launch, so it can only be built where the values are reachable.
    // A reconstructs them arithmetically; B reads the mirror; C cannot be
    // written at all, which is the result rather than an omission.
    auto stage_and_fire = [&](const std::vector<XferDesc>& tbl) {
        for (int i = 0; i < ops; ++i)
            rt.put(bh, peer, bh.index, bytes, tbl[i].src_off, tbl[i].dst_off);
        gicc::DeviceCtx* c = rt.prepare_delta(ops, 1);
        hipLaunchKernelGGL(k_fire, dim3(1), dim3(1), 0, 0, c, ticks);
        (void)hipDeviceSynchronize();
        rt.reset();
    };
    std::vector<XferDesc> a_tbl(ops);
    for (int i = 0; i < ops; ++i) {
        a_tbl[i].src_off = (unsigned long long)i * bytes;
        a_tbl[i].dst_off = kRecvBase + (unsigned long long)i * bytes;
    }
    emit("A", "pre-launch", "trigger", true,
         time_it(rt, rank, samples, warmup, [&] { stage_and_fire(a_tbl); }));
    emit("B", "device-entry", "trigger", true,
         time_it(rt, rank, samples, warmup, [&] { stage_and_fire(h_desc); }));
    if (rank == 0) {
        printf("CSV,B',device-entry,trigger,0,,,\n");
        printf("  B'  device-entry     trigger  ILLEGAL   "
               "(same kernel body as B, no host mirror declared)\n");
        printf("CSV,C,device-dynamic,trigger,0,,,\n");
        printf("  C   device-dynamic   trigger  ILLEGAL   "
               "(offsets do not exist until the kernel runs)\n");
    }

    rt.barrier();
    if (rank == 0) printf("=== done ===\n");
    (void)hipFree(d_buf);
    (void)hipFree(d_desc);
    (void)hipFree(d_scr);
    return 0;
}
