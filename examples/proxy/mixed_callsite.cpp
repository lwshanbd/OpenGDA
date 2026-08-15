/*
 * mixed_callsite.cpp - one program, four communication call sites, each with
 * different compiler-visible context. Enumerates every legal per-call-site
 * path assignment and reports what a single global policy costs.
 *
 * The call sites are deliberately chosen so that no one path wins all four:
 *
 *   A  64 x 4KB, then 100 us of independent compute   trigger-favoured
 *      (issue cost dominates once the gap hides the wire time)
 *   B  16 x 4KB, then 200 us of independent compute   trigger-favoured
 *   C  64 x 64KB, result used immediately (no gap)   proxy-favoured
 *      (nothing to hide the transfer behind, so the proxy's ring pushes
 *       pipeline against the wire while the trigger path's staging does not)
 *   D   8 x 16KB at offsets computed ON THE DEVICE    proxy-ONLY
 *      (the descriptor is not host-knowable, so the host cannot pre-stage
 *       it and the trigger path is not merely slower but illegal)
 *
 * Every call site is identical to a communication runtime at the moment it
 * is issued -- same op, same peer, same communicator, same hardware, and
 * A/B/D even share the same message sizes with each other or with C. What
 * differs is trip count, issue-to-first-use distance, and whether the
 * descriptor can be reconstructed before the kernel launches: all of it
 * static program context.
 *
 * Run (2 ranks, 1 per node). Both paths live in the same process, so do NOT
 * set GICC_SKIP_DWQ_INIT -- the trigger BAR has to map:
 *
 *   srun -p pci -N 2 -n 2 --ntasks-per-node=1 -c 8 --gpu-bind=none \
 *     ./mixed_callsite
 */

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"

static constexpr size_t kBufBytes = 32 * 1024 * 1024;

enum Path { PROXY = 0, TRIGGER = 1 };

struct CallSite {
    const char* name;
    size_t      bytes;
    int         ops;
    int         dist_us;      // independent compute between issue and use
    size_t      base_off;     // region of the window this site owns
    bool        host_knowable;  // false => trigger path is illegal
};

// A/B/C/D regions, laid out so every op lands somewhere distinct and the
// receiver can verify all of them.
static CallSite kMixedSites[] = {
    {"A", 4096,  64, 100, 0,                     true },
    {"B", 4096,  16, 200, 64 * 4096,             true },
    {"C", 65536, 64,   0, 64 * 4096 + 16 * 4096, true },
    {"D", 16384,  8,   0, 8 * 1024 * 1024,       false},
};

// Twin call sites: byte-for-byte identical to a communication runtime at the
// moment either is issued -- same operation, same 64KB message, same 64-op
// trip count, same peer, same communicator, same hardware. The ONLY thing
// that differs is how far away the first use of the result is, which is
// static program context the runtime cannot see. If the best path differs
// between E and F, no runtime-visible feature could have told them apart.
static CallSite kTwinSites[] = {
    {"E", 65536, 64,   0, 0,               true },
    {"F", 65536, 64, 400, 8 * 1024 * 1024, true },
};

static CallSite* kSites   = kMixedSites;
static int       kNumSites = 4;

__device__ __forceinline__ void spin_ticks(long long ticks) {
    if (ticks <= 0) return;
    long long t0 = wall_clock64();
    while ((wall_clock64() - t0) < ticks) { }
}

// Proxy issue: K pushes at affine offsets, then the independent compute.
__global__ void k_proxy_site(gicc::DeviceCtx* ctx, int peer, int buf,
                             size_t bytes, int ops, size_t base_off,
                             long long ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < ops; ++i) {
        size_t off = base_off + (size_t)i * bytes;
        gicc::put(ctx, peer, buf, off, buf, off, bytes);
    }
    spin_ticks(ticks);
}

// Proxy issue with DEVICE-COMPUTED offsets. d_off is filled by a prior
// kernel, so the host cannot know the descriptor at launch time and the
// trigger path cannot pre-stage it.
__global__ void k_proxy_site_dynamic(gicc::DeviceCtx* ctx, int peer, int buf,
                                     size_t bytes, int ops,
                                     const size_t* d_off, long long ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < ops; ++i) {
        size_t off = d_off[i];
        gicc::put(ctx, peer, buf, off, buf, off, bytes);
    }
    spin_ticks(ticks);
}

// Trigger issue: the host already staged all K descriptors; one flush
// releases them, then the independent compute.
__global__ void k_trigger_site(gicc::DeviceCtx* ctx, long long ticks) {
    gicc::flush(ctx);
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    spin_ticks(ticks);
}

// Produces the device-side offsets for call site D. Deliberately opaque to
// the host: the offsets come out of a device computation.
__global__ void k_make_offsets(size_t* d_off, int ops, size_t base,
                               size_t bytes, const int* d_perm) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < ops; ++i)
        d_off[i] = base + (size_t)d_perm[i] * bytes;
}

__global__ void k_fill(uint8_t* p, size_t n, uint8_t v) {
    for (size_t i = threadIdx.x + (size_t)blockIdx.x * blockDim.x; i < n;
         i += (size_t)blockDim.x * gridDim.x)
        p[i] = v;
}

struct Stats { double median=0, min=0, max=0; int n=0; };
static Stats summarize(std::vector<double> v) {
    Stats s; if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n=(int)v.size(); s.min=v.front(); s.max=v.back();
    s.median = (v.size()%2) ? v[v.size()/2]
                            : 0.5*(v[v.size()/2-1]+v[v.size()/2]);
    return s;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    int samples = 21, warmup = 10, steps = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--samples=",0)==0) samples = std::atoi(a.c_str()+10);
        else if (a.rfind("--warmup=", 0)==0) warmup  = std::atoi(a.c_str()+9);
        else if (a.rfind("--steps=",  0)==0) steps   = std::atoi(a.c_str()+8);
        else if (a == "--twins") { kSites = kTwinSites; kNumSites = 2; }
        else { fprintf(stderr, "mixed_callsite: unknown arg '%s'\n", a.c_str()); return 2; }
    }

    gicc::Runtime rt;
    rt.enable_host_wait_mode();     // DWQ / trigger path
    rt.enable_mixed_dispatch();     // ...without killing the proxy rings
    int rank = rt.rank(), nranks = rt.size();
    if (nranks != 2) {
        if (rank==0) fprintf(stderr, "mixed_callsite: need 2 ranks (got %d)\n", nranks);
        return 1;
    }
    int peer = 1 - rank;

    void* d_buf = nullptr;
    if (hipMalloc(&d_buf, kBufBytes) != hipSuccess) return 3;
    (void)hipMemset(d_buf, 0, kBufBytes);
    auto bh = rt.register_buffer(d_buf, kBufBytes, /*is_device=*/true);
    rt.exchange();

    // Device-side offsets for call site D.
    int dyn = -1;
    for (int s = 0; s < kNumSites; ++s) if (!kSites[s].host_knowable) dyn = s;
    const int dyn_ops = dyn >= 0 ? kSites[dyn].ops : 1;
    size_t* d_off = nullptr; int* d_perm = nullptr;
    (void)hipMalloc((void**)&d_off,  dyn_ops * sizeof(size_t));
    (void)hipMalloc((void**)&d_perm, dyn_ops * sizeof(int));
    if (dyn >= 0) {
        std::vector<int> perm(dyn_ops);
        for (int i = 0; i < dyn_ops; ++i) perm[i] = (i * 5 + 3) % dyn_ops;
        (void)hipMemcpy(d_perm, perm.data(), perm.size()*sizeof(int),
                        hipMemcpyHostToDevice);
        hipLaunchKernelGGL(k_make_offsets, dim3(1), dim3(1), 0, 0,
                           d_off, dyn_ops, kSites[dyn].base_off,
                           kSites[dyn].bytes, d_perm);
        (void)hipDeviceSynchronize();
    }

    size_t* d_off_scratch = d_off;
    std::vector<size_t> h_off_vec(dyn_ops);
    size_t* h_off_scratch = h_off_vec.data();

    int wall_khz = 0;
    (void)hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0);
    const double ticks_per_us = wall_khz > 0 ? wall_khz/1000.0 : 25.0;

    (void)rt.prepare();
    rt.reset();
    rt.barrier();

    // One timestep: every call site issues, computes, and completes.
    auto timestep = [&](const int* path) {
        for (int s = 0; s < kNumSites; ++s) {
            const CallSite& cs = kSites[s];
            const long long ticks = (long long)(cs.dist_us * ticks_per_us);
            gicc::DeviceCtx* ctx;
            if (path[s] == TRIGGER) {
                if (!cs.host_knowable) {
                    // The only way to stage a descriptor the compiler cannot
                    // reconstruct: stall the host on a device->host readback
                    // of the offsets the device just computed. Legal, and the
                    // cost of forcing one global policy onto every call site.
                    (void)hipMemcpy(h_off_scratch, d_off_scratch,
                                    cs.ops * sizeof(size_t), hipMemcpyDeviceToHost);
                    for (int i = 0; i < cs.ops; ++i)
                        rt.put(bh, peer, bh.index, cs.bytes,
                               h_off_scratch[i], h_off_scratch[i]);
                    ctx = rt.prepare();
                    hipLaunchKernelGGL(k_trigger_site, dim3(1), dim3(1), 0, 0,
                                       ctx, ticks);
                    (void)hipDeviceSynchronize();
                    rt.reset();
                    continue;
                }
                for (int i = 0; i < cs.ops; ++i) {
                    size_t off = cs.base_off + (size_t)i * cs.bytes;
                    rt.put(bh, peer, bh.index, cs.bytes, off, off);
                }
                ctx = rt.prepare();      // delta == cs.ops; one flush fires all
                hipLaunchKernelGGL(k_trigger_site, dim3(1), dim3(1), 0, 0,
                                   ctx, ticks);
            } else {
                ctx = rt.prepare();
                if (cs.host_knowable) {
                    hipLaunchKernelGGL(k_proxy_site, dim3(1), dim3(1), 0, 0,
                                       ctx, peer, bh.index, cs.bytes, cs.ops,
                                       cs.base_off, ticks);
                } else {
                    hipLaunchKernelGGL(k_proxy_site_dynamic, dim3(1), dim3(1),
                                       0, 0, ctx, peer, bh.index, cs.bytes,
                                       cs.ops, d_off, ticks);
                }
            }
            (void)hipDeviceSynchronize();
            rt.reset();
        }
    };

    // Verify a configuration actually delivers every byte of every region.
    auto verify = [&](const int* path) -> bool {
        size_t total = 0;
        for (int s = 0; s < kNumSites; ++s)
            total = std::max(total, kSites[s].base_off +
                                    (size_t)kSites[s].ops * kSites[s].bytes);
        const uint8_t pat = 0xC3;
        if (rank == 0)
            hipLaunchKernelGGL(k_fill, dim3(64), dim3(256), 0, 0,
                               (uint8_t*)d_buf, total, pat);
        else
            hipLaunchKernelGGL(k_fill, dim3(64), dim3(256), 0, 0,
                               (uint8_t*)d_buf, total, (uint8_t)0);
        (void)hipDeviceSynchronize();
        rt.barrier();
        if (rank == 0) timestep(path);
        rt.barrier();
        int ok = 1;
        if (rank == 1) {
            std::vector<uint8_t> h(total);
            (void)hipMemcpy(h.data(), d_buf, total, hipMemcpyDeviceToHost);
            for (int s = 0; s < kNumSites && ok; ++s) {
                const CallSite& cs = kSites[s];
                for (int i = 0; i < cs.ops && ok; ++i) {
                    // Site D permutes its offsets on the device; every slot
                    // in its region is still covered exactly once.
                    size_t off = cs.base_off + (size_t)i * cs.bytes;
                    for (size_t b = 0; b < cs.bytes; b += 512) {
                        if (h[off+b] != pat) {
                            printf("[verify] FAIL site=%s op=%d off=%zu got=0x%02x\n",
                                   cs.name, i, off+b, h[off+b]);
                            ok = 0; break;
                        }
                    }
                }
            }
        }
        MPI_Bcast(&ok, 1, MPI_INT, 1, MPI_COMM_WORLD);
        return ok != 0;
    };

    auto measure = [&](const int* path) {
        for (int w = 0; w < warmup; ++w) {
            rt.barrier(); if (rank==0) timestep(path); rt.barrier();
        }
        std::vector<double> v;
        for (int s = 0; s < samples; ++s) {
            rt.barrier();
            double t0 = MPI_Wtime();
            if (rank == 0) for (int t = 0; t < steps; ++t) timestep(path);
            double t1 = MPI_Wtime();
            rt.barrier();
            if (rank==0) v.push_back((t1-t0)*1e6/steps);
        }
        return summarize(std::move(v));
    };

    if (rank == 0) {
        printf("=== mixed_callsite: %d call sites, %d steps/sample, %d samples ===\n",
               kNumSites, steps, samples);
        for (int s = 0; s < kNumSites; ++s) {
            const CallSite& c = kSites[s];
            printf("  site %s: %d x %zuB, gap=%dus, descriptor %s\n",
                   c.name, c.ops, c.bytes, c.dist_us,
                   c.host_knowable ? "host-knowable (either path legal)"
                                   : "device-computed (PROXY ONLY)");
        }
        char hdr[64]; hdr[0]=0;
        snprintf(hdr, sizeof(hdr), "paths(");
        for (int s = 0; s < kNumSites; ++s)
            snprintf(hdr+strlen(hdr), sizeof(hdr)-strlen(hdr), "%s%s",
                     s?",":"", kSites[s].name);
        snprintf(hdr+strlen(hdr), sizeof(hdr)-strlen(hdr), ")");
        printf("\n%-8s %-22s %14s\n", "assign", hdr, "us/timestep");
        printf("--------------------------------------------------------------\n");
    }

    // Enumerate every LEGAL assignment: site D has no legal trigger form, so
    // it is pinned to proxy and A/B/C are free -> 8 configurations.
    struct Result { int mask; int path[8]; Stats st; };
    std::vector<Result> results;
    const int nmask = 1 << kNumSites;
    for (int mask = 0; mask < nmask; ++mask) {
        Result r; r.mask = mask;
        for (int s = 0; s < kNumSites; ++s) r.path[s] = (mask >> s) & 1;
        if (!verify(r.path)) {
            if (rank==0) printf("  mask=%d FAILED VERIFICATION, skipping\n", mask);
            continue;
        }
        r.st = measure(r.path);
        results.push_back(r);
        if (rank == 0) {
            char pd[64]; pd[0]=0;
            for (int s = 0; s < kNumSites; ++s) {
                bool rb = r.path[s] && !kSites[s].host_knowable;
                snprintf(pd+strlen(pd), sizeof(pd)-strlen(pd), "%s%s",
                         s?",":"", r.path[s] ? (rb?"trig*":"trig") : "prox");
            }
            bool any_rb = false;
            for (int s = 0; s < kNumSites; ++s)
                if (r.path[s] && !kSites[s].host_knowable) any_rb = true;
            printf("%-8d %-22s %14.2f  %s\n", mask, pd, r.st.median,
                   any_rb ? "(needs a D2H readback)" : "");
        }
    }

    if (rank == 0 && !results.empty()) {
        double best = 1e30; int besti = -1;
        for (size_t i = 0; i < results.size(); ++i)
            if (results[i].st.median < best) { best = results[i].st.median; besti=(int)i; }
        // The only global policies that are legal for every call site.
        double all_proxy = -1, all_trigger = -1;
        double best_native = 1e30; int best_native_i = -1;
        for (size_t i = 0; i < results.size(); ++i) {
            auto& r = results[i];
            if (r.mask == 0)          all_proxy   = r.st.median;
            if (r.mask == nmask - 1)  all_trigger = r.st.median;
            if (r.st.median < best_native) {
                best_native = r.st.median; best_native_i = (int)i;
            }
        }
        (void)best_native_i;

        printf("\n--------------------------------------------------------------\n");
        printf("best LEGAL GLOBAL policy (one path for the whole program):\n");
        printf("    all-proxy                 %10.2f us/timestep\n", all_proxy);
        bool needs_rb = false;
        for (int s = 0; s < kNumSites; ++s) if (!kSites[s].host_knowable) needs_rb = true;
        printf("    all-trigger               %10.2f us/timestep%s\n", all_trigger,
               needs_rb ? "  (only legal via a per-timestep D2H readback)" : "");
        printf("best PER-CALL-SITE policy:\n");
        char pd[64]; pd[0]=0;
        for (int s = 0; s < kNumSites; ++s)
            snprintf(pd+strlen(pd), sizeof(pd)-strlen(pd), "%s%s",
                     s?",":"", results[besti].path[s]?"trig":"prox");
        printf("    %-24s  %10.2f us/timestep\n", pd, best);
        double best_global = std::min(all_proxy, all_trigger);
        printf("\n    per-call-site vs best global policy: %.2fx (%.1f%% faster)\n",
               best_global/best, (best_global/best - 1)*100);
        printf("    per-call-site vs all-proxy   : %.2fx\n", all_proxy/best);
        printf("    per-call-site vs all-trigger : %.2fx\n", all_trigger/best);
        for (auto& r : results) {
            char q[64]; q[0]=0;
            for (int s = 0; s < kNumSites; ++s)
                snprintf(q+strlen(q), sizeof(q)-strlen(q), "%s%s",
                         s?",":"", r.path[s]?"trig":"prox");
            printf("CSV,mixed,%d,%s,%.3f,%.3f,%.3f,%d\n",
                   r.mask, q, r.st.median, r.st.min, r.st.max, r.st.n);
        }
    }

    rt.barrier();
    (void)hipFree(d_buf); (void)hipFree(d_off); (void)hipFree(d_perm);
    return 0;
}
