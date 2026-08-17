/*
 * crosssite_merge.cpp — is there a decision here, or just a rewrite rule?
 *
 * Two DIFFERENT put call sites whose destinations happen to be adjacent.
 * Merging them is a transformation across program points, not a knob at
 * one of them, and it has a structure the earlier experiments did not:
 * the more profitable form has the stricter legality requirement.
 *
 *   separate      put A ; compute D1 ; put B ; compute D2 ; wait
 *                 two messages, but A is already on the wire during D1
 *
 *   sink-A        compute D1 ; put A+B ; compute D2 ; wait
 *                 one message. Legal whenever A's payload may be delayed,
 *                 i.e. nothing between the sites reads or rewrites it.
 *                 Pays for the merge by giving up A's overlap with D1.
 *
 *   hoist-B       put A+B ; compute D1 ; compute D2 ; wait
 *                 one message AND full overlap. Legal only if B's payload
 *                 is already final at A's position -- a stronger claim,
 *                 and a dependence question rather than a size one.
 *
 * The point of measuring this BEFORE handing anything to a decider is to
 * find out whether the space has room in it. If one of the three is
 * always best, this is a rewrite rule and there is nothing to decide; a
 * decider that scores well on it would be reporting the absence of a
 * problem. The crossover between `separate` and `sink-A` is the thing to
 * look for: it should exist, because one trades a message against an
 * overlap and the exchange rate depends on the message size and D1.
 *
 * Correctness is checked, not assumed: the receiver verifies both halves
 * arrived and that the bytes on either side of them are untouched.
 *
 *   GICC_SKIP_DWQ_INIT=1 GICC_PROXY_ENABLED=1 srun -p pci -N 2 -n 2 \
 *       --ntasks-per-node=1 -c 8 --gpu-bind=none -t 5 ./crosssite_merge
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

static constexpr size_t   kBufBytes = 128 * 1024 * 1024;
static constexpr size_t   kSendBase = 0;
static constexpr size_t   kRecvBase = 64 * 1024 * 1024;
static constexpr uint8_t  kGuard    = 0x5A;
static constexpr size_t   kGuardLen = 4096;   // bytes checked either side

__device__ __forceinline__ void spin_ticks(long long ticks) {
    if (ticks <= 0) return;
    long long t0 = wall_clock64();
    while ((wall_clock64() - t0) < ticks) { }
}

// Both sites at their original positions. A overlaps the work between.
__global__ void k_separate(gicc::DeviceCtx* ctx, int peer, int buf,
                           unsigned long long bytes,
                           long long t1, long long t2, int blocks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    gicc::put(ctx, peer, buf, kRecvBase,         buf, kSendBase,         bytes);
    spin_ticks(t1);
    gicc::put(ctx, peer, buf, kRecvBase + bytes, buf, kSendBase + bytes, bytes);
    spin_ticks(t2);
    gicc::quiet(ctx);
    (void)blocks;
}

// A sunk to B's position and merged. One message, no overlap for A.
__global__ void k_sink_a(gicc::DeviceCtx* ctx, int peer, int buf,
                         unsigned long long bytes,
                         long long t1, long long t2, int blocks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    spin_ticks(t1);
    gicc::put(ctx, peer, buf, kRecvBase, buf, kSendBase, bytes * 2);
    spin_ticks(t2);
    gicc::quiet(ctx);
    (void)blocks;
}

// B hoisted to A's position and merged. One message, full overlap.
__global__ void k_hoist_b(gicc::DeviceCtx* ctx, int peer, int buf,
                          unsigned long long bytes,
                          long long t1, long long t2, int blocks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    gicc::put(ctx, peer, buf, kRecvBase, buf, kSendBase, bytes * 2);
    spin_ticks(t1);
    spin_ticks(t2);
    gicc::quiet(ctx);
    (void)blocks;
}

struct Stats { double median = 0, min = 0, max = 0; int n = 0; };

static Stats summarize(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = (int)v.size();
    s.median = (v.size() % 2) ? v[v.size() / 2]
                              : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    s.min = v.front(); s.max = v.back();
    return s;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    std::vector<size_t> sizes = {1024, 8192, 65536, 524288};
    std::vector<int>    d1s   = {0, 10, 40, 160, 640};
    int d2_us = 0, samples = 11, warmup = 4;

    auto parse_sizes = [](const std::string& s) {
        std::vector<size_t> o; size_t p = 0;
        while (p < s.size()) {
            size_t c = s.find(',', p); if (c == std::string::npos) c = s.size();
            if (c > p) o.push_back((size_t)atol(s.substr(p, c - p).c_str()));
            p = c + 1;
        } return o;
    };
    auto parse_ints = [](const std::string& s) {
        std::vector<int> o; size_t p = 0;
        while (p < s.size()) {
            size_t c = s.find(',', p); if (c == std::string::npos) c = s.size();
            if (c > p) o.push_back(atoi(s.substr(p, c - p).c_str()));
            p = c + 1;
        } return o;
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--sizes=",   0) == 0) sizes = parse_sizes(a.substr(8));
        else if (a.rfind("--d1=",      0) == 0) d1s   = parse_ints(a.substr(5));
        else if (a.rfind("--d2=",      0) == 0) d2_us = atoi(a.c_str() + 5);
        else if (a.rfind("--samples=", 0) == 0) samples = atoi(a.c_str() + 10);
        else if (a.rfind("--warmup=",  0) == 0) warmup = atoi(a.c_str() + 9);
        else { fprintf(stderr, "unknown arg '%s'\n", a.c_str()); return 2; }
    }

    gicc::Runtime rt;
    const int rank = rt.rank(), nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "need exactly 2 ranks (got %d)\n", nranks);
        return 1;
    }
    const int peer = 1 - rank;

    void* d_buf = nullptr;
    if (hipMalloc(&d_buf, kBufBytes) != hipSuccess) return 3;
    auto bh = rt.register_buffer(d_buf, kBufBytes, /*is_device=*/true);
    rt.exchange();
    (void)rt.prepare();
    rt.reset();

    int wall_khz = 0;
    (void)hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0);
    const double tpu = wall_khz > 0 ? wall_khz / 1000.0 : 100.0;

    if (rank == 0) {
        printf("=== crosssite_merge  d2=%dus  samples=%d ===\n", d2_us, samples);
        printf("CSV,bytes,d1_us,config,median_us,min_us,max_us,verdict\n");
    }

    for (size_t bytes : sizes) {
        const size_t payload = bytes * 2;
        std::vector<uint8_t> h(payload + 2 * kGuardLen);

        for (int d1 : d1s) {
            const long long t1 = (long long)(d1 * tpu);
            const long long t2 = (long long)(d2_us * tpu);

            struct Cfg { const char* name; void (*k)(gicc::DeviceCtx*, int, int,
                                                    unsigned long long,
                                                    long long, long long, int); };
            const Cfg cfgs[] = {
                {"separate", k_separate},
                {"sink-A",   k_sink_a},
                {"hoist-B",  k_hoist_b},
            };

            for (const Cfg& c : cfgs) {
                // Fresh windows: payload on the sender, guard bytes on both
                // sides of the destination so an over-wide transfer shows up.
                if (rank == 0) {
                    std::fill(h.begin(), h.end(), kGuard);
                    for (size_t i = 0; i < payload; ++i)
                        h[kGuardLen + i] = (uint8_t)(0xA5 + ((i / bytes) & 1));
                    (void)hipMemcpy((char*)d_buf + kSendBase, h.data() + kGuardLen,
                                    payload, hipMemcpyHostToDevice);
                }
                std::fill(h.begin(), h.end(), kGuard);
                (void)hipMemcpy((char*)d_buf + kRecvBase - kGuardLen, h.data(),
                                payload + 2 * kGuardLen, hipMemcpyHostToDevice);
                (void)hipDeviceSynchronize();
                rt.barrier();

                std::vector<double> v;
                for (int s = 0; s < samples + warmup; ++s) {
                    rt.barrier();
                    double a = MPI_Wtime();
                    if (rank == 0) {
                        gicc::DeviceCtx* ctx = rt.prepare();
                        hipLaunchKernelGGL(c.k, dim3(1), dim3(1), 0, 0, ctx, peer,
                                           bh.index, (unsigned long long)bytes,
                                           t1, t2, 1);
                        (void)hipDeviceSynchronize();
                        rt.reset();
                    }
                    double b = MPI_Wtime();
                    rt.barrier();
                    if (rank == 0 && s >= warmup) v.push_back((b - a) * 1e6);
                }

                int bad = 0;
                if (rank == 1) {
                    (void)hipMemcpy(h.data(), (char*)d_buf + kRecvBase - kGuardLen,
                                    payload + 2 * kGuardLen, hipMemcpyDeviceToHost);
                    for (size_t i = 0; i < kGuardLen && !bad; ++i)
                        if (h[i] != kGuard) bad = 2;
                    for (size_t i = 0; i < payload && !bad; ++i)
                        if (h[kGuardLen + i] != (uint8_t)(0xA5 + ((i / bytes) & 1)))
                            bad = 1;
                    for (size_t i = 0; i < kGuardLen && !bad; ++i)
                        if (h[kGuardLen + payload + i] != kGuard) bad = 2;
                }
                int allbad = 0;
                MPI_Allreduce(&bad, &allbad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

                const Stats st = summarize(std::move(v));
                if (rank == 0) {
                    const char* verdict = allbad == 0 ? "OK"
                                        : allbad == 1 ? "WRONG-payload"
                                                      : "WRONG-guard";
                    printf("CSV,%zu,%d,%s,%.3f,%.3f,%.3f,%s\n",
                           bytes, d1, c.name, st.median, st.min, st.max, verdict);
                    printf("  %7zuB  d1=%-4d  %-9s median=%9.2f us  %s\n",
                           bytes, d1, c.name, st.median, verdict);
                }
            }
        }
    }

    rt.barrier();
    if (rank == 0) printf("=== done ===\n");
    (void)hipFree(d_buf);
    return 0;
}
