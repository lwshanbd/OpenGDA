/*
 * coalesce_bench.cpp — merging a loop's transfers, and why it needs proof.
 *
 * A loop issues K transfers of `bytes`, iteration i landing at i*stride.
 * When stride == bytes the whole loop is one contiguous region, so any run
 * of m consecutive transfers can be issued as a single transfer of m*bytes.
 * When stride > bytes there are gaps between them and it cannot.
 *
 * Two things are being measured.
 *
 * 1. WHAT MERGING IS WORTH. Fewer, larger messages pay the per-message
 *    cost fewer times, but leave less to pipeline and coarsen completion.
 *    So the best merge factor is not simply "as much as possible", and
 *    where the optimum sits depends on the message size and the trip
 *    count — which makes it a decision rather than a rewrite rule.
 *
 * 2. WHAT AN UNPROVEN MERGE COSTS. --force-illegal merges a strided site
 *    anyway. The result is not a slower program; it is a wrong one, and
 *    the gap check catches it. That is the difference between a knob and
 *    a transformation: a knob set badly loses time, a transformation
 *    applied without a legality proof loses data, and nothing in the
 *    runtime is watching.
 *
 * The window is laid out so the receiver can tell the three cases apart:
 *   payload bytes  0xA5..     written by the sender, must arrive
 *   gap bytes      0x5A       must still be there afterwards
 *
 * Two ranks, one per node:
 *   GICC_PROXY_ENABLED=1 srun -p pci -N 2 -n 2 --ntasks-per-node=1 \
 *       -c 8 --gpu-bind=none -t 3 ./coalesce_bench
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

static constexpr size_t kBufBytes = 128 * 1024 * 1024;
static constexpr size_t kSendBase = 0;
static constexpr size_t kRecvBase = 64 * 1024 * 1024;
static constexpr uint8_t kGapByte = 0x5A;

__device__ __forceinline__ void spin_ticks(long long ticks) {
    if (ticks <= 0) return;
    long long t0 = wall_clock64();
    while ((wall_clock64() - t0) < ticks) { }
}

// n transfers of `bytes`, iteration i at i*stride from each base. Merging
// is expressed by the caller: to merge m transfers it passes n/m, m*bytes
// and m*stride. When stride == bytes that is still exactly the same bytes;
// when stride > bytes it is not, which is the point of --force-illegal.
__global__ void k_xfer(gicc::DeviceCtx* ctx, int peer, int buf,
                       unsigned long long src_base, unsigned long long dst_base,
                       unsigned long long bytes, unsigned long long stride,
                       int n, int blocks, long long ticks) {
    if (threadIdx.x != 0) return;
    const int lane  = blockIdx.x;
    const int base  = n / blocks;
    const int extra = n % blocks;
    const int first = lane * base + (lane < extra ? lane : extra);
    const int mine  = base + (lane < extra ? 1 : 0);
    for (int k = 0; k < mine; ++k) {
        const unsigned long long i = (unsigned long long)(first + k);
        gicc::put(ctx, peer, buf, dst_base + i * stride,
                       buf, src_base + i * stride, bytes, lane);
    }
    spin_ticks(ticks);
    gicc::quiet(ctx, lane);
}

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

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    size_t bytes = 4096;
    int    ops = 64, dist_us = 0, samples = 11, warmup = 4;
    int    stride_mult = 1;          // 1 => contiguous, >1 => gaps
    bool   force_illegal = false;
    std::string site = "S";
    std::vector<int> merges = {1, 2, 4, 8, 16, 32, 64};
    std::vector<int> blocks = {1, 2, 4, 8};

    auto parse_ints = [](const std::string& s) {
        std::vector<int> out; size_t p = 0;
        while (p < s.size()) {
            size_t c = s.find(',', p);
            if (c == std::string::npos) c = s.size();
            if (c > p) out.push_back(atoi(s.substr(p, c - p).c_str()));
            p = c + 1;
        }
        return out;
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--bytes=",   0) == 0) bytes = (size_t)atol(a.c_str() + 8);
        else if (a.rfind("--ops=",     0) == 0) ops = atoi(a.c_str() + 6);
        else if (a.rfind("--dist=",    0) == 0) dist_us = atoi(a.c_str() + 7);
        else if (a.rfind("--stride-mult=", 0) == 0) stride_mult = atoi(a.c_str() + 14);
        else if (a.rfind("--merges=",  0) == 0) merges = parse_ints(a.substr(9));
        else if (a.rfind("--blocks=",  0) == 0) blocks = parse_ints(a.substr(9));
        else if (a.rfind("--samples=", 0) == 0) samples = atoi(a.c_str() + 10);
        else if (a.rfind("--warmup=",  0) == 0) warmup = atoi(a.c_str() + 9);
        else if (a.rfind("--site=", 0) == 0) site = a.substr(7);
        else if (a == "--force-illegal") force_illegal = true;
        else { fprintf(stderr, "unknown arg '%s'\n", a.c_str()); return 2; }
    }
    if (stride_mult < 1) stride_mult = 1;
    const size_t stride = bytes * (size_t)stride_mult;
    const bool   contiguous = (stride_mult == 1);

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
    const double ticks_per_us = wall_khz > 0 ? wall_khz / 1000.0 : 100.0;
    const long long ticks = (long long)(dist_us * ticks_per_us);

    const size_t span = (size_t)ops * stride;
    std::vector<uint8_t> h_send(span), h_recv(span);

    // Sender: payload where the transfers actually read, sentinel in the
    // gaps. Receiver: sentinel everywhere, so an over-wide transfer shows
    // up as a gap byte that changed.
    auto reset_windows = [&] {
        if (rank == 0) {
            std::fill(h_send.begin(), h_send.end(), kGapByte);
            for (int i = 0; i < ops; ++i)
                std::fill(h_send.begin() + i * stride,
                          h_send.begin() + i * stride + bytes,
                          (uint8_t)(0xA5 + (i & 0x0F)));
            (void)hipMemcpy((char*)d_buf + kSendBase, h_send.data(), span,
                            hipMemcpyHostToDevice);
        }
        std::fill(h_recv.begin(), h_recv.end(), kGapByte);
        (void)hipMemcpy((char*)d_buf + kRecvBase, h_recv.data(), span,
                        hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        rt.barrier();
    };

    // Returns 0 = correct, 1 = payload missing, 2 = a gap was overwritten.
    auto verify = [&]() -> int {
        int bad = 0;
        if (rank == 1) {
            (void)hipMemcpy(h_recv.data(), (char*)d_buf + kRecvBase, span,
                            hipMemcpyDeviceToHost);
            for (int i = 0; i < ops && !bad; ++i) {
                const uint8_t want = (uint8_t)(0xA5 + (i & 0x0F));
                for (size_t j = 0; j < bytes; ++j)
                    if (h_recv[i * stride + j] != want) { bad = 1; break; }
                for (size_t j = bytes; j < stride && !bad; ++j)
                    if (h_recv[i * stride + j] != kGapByte) bad = 2;
            }
        }
        int out = 0;
        MPI_Allreduce(&bad, &out, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        return out;
    };

    if (rank == 0) {
        printf("[site %s] ", site.c_str());
        printf("=== coalesce_bench bytes=%zu ops=%d stride=%zu (%s) dist=%dus%s ===\n",
               bytes, ops, stride, contiguous ? "contiguous" : "gaps", dist_us,
               force_illegal ? "  [--force-illegal]" : "");
        printf("CSV,site,bytes,ops,stride_mult,dist_us,coalescable,"
               "merge,blocks,msg_bytes,msgs,median_us,min_us,max_us,verdict\n");
    }

    for (int m : merges) {
        if (m < 1 || m > ops || (ops % m) != 0) continue;
        // A merge is only legal when the loop's transfers are one
        // contiguous region. The compiler proves that (`coalescable` in
        // features.json); here it is the stride that decides.
        if (m > 1 && !contiguous && !force_illegal) {
            if (rank == 0)
                printf("CSV,%s,%zu,%d,%d,%d,%d,%d,-,-,-,,,,SKIPPED-ILLEGAL\n",
                       site.c_str(), bytes, ops, stride_mult, dist_us,
                       contiguous ? 1 : 0, m);
            continue;
        }
        for (int P : blocks) {
            const int    msgs      = ops / m;
            const size_t msg_bytes = bytes * (size_t)m;
            const size_t msg_strid = stride * (size_t)m;
            if (P > msgs) continue;

            reset_windows();
            std::vector<double> v;
            for (int s = 0; s < samples + warmup; ++s) {
                rt.barrier();
                double t0 = MPI_Wtime();
                if (rank == 0) {
                    gicc::DeviceCtx* c = rt.prepare();
                    hipLaunchKernelGGL(k_xfer, dim3(P), dim3(1), 0, 0, c, peer,
                                       bh.index, (unsigned long long)kSendBase,
                                       (unsigned long long)kRecvBase,
                                       (unsigned long long)msg_bytes,
                                       (unsigned long long)msg_strid,
                                       msgs, P, ticks);
                    (void)hipDeviceSynchronize();
                    rt.reset();
                }
                double t1 = MPI_Wtime();
                rt.barrier();
                if (rank == 0 && s >= warmup) v.push_back((t1 - t0) * 1e6);
            }
            const int bad = verify();
            const Stats st = summarize(std::move(v));
            if (rank == 0) {
                const char* verdict = bad == 0 ? "OK"
                                    : bad == 1 ? "WRONG-payload"
                                               : "WRONG-overwrote-gap";
                printf("CSV,%s,%zu,%d,%d,%d,%d,%d,%d,%zu,%d,"
                       "%.3f,%.3f,%.3f,%s\n",
                       site.c_str(), bytes, ops, stride_mult, dist_us,
                       contiguous ? 1 : 0, m, P, msg_bytes, msgs,
                       st.median, st.min, st.max, verdict);
                printf("  merge=%-3d blocks=%-2d  %6zuB x %-4d  median=%9.2f us  %s\n",
                       m, P, msg_bytes, msgs, st.median, verdict);
            }
        }
    }

    rt.barrier();
    if (rank == 0) printf("=== done ===\n");
    (void)hipFree(d_buf);
    return 0;
}
