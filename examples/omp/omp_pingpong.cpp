// omp_pingpong.cpp - p2p microbench for the GICC-from-OpenMP path.
//
// Mirrors examples/proxy/bench_pingpong.cpp, but instead of the HIP device API
// (gicc::put inside a __global__ kernel), it issues ompx_put_dev from inside a
// `#pragma omp target` region -- exactly how the OpenMP minimod halo
// (gicc_halo_issue) drives the proxy. This isolates the cost the omp-target
// kernel-launch layer adds on top of the raw proxy transport.
//
//   --mode=pipelined (default): N puts in ONE omp target region, then one quiet
//                               -> throughput (matches gicc_halo_issue's pattern:
//                               one prepare + one target region + one quiet).
//   --mode=per-msg            : one put per target region + quiet, per message
//                               -> pure per-message latency incl. all per-op cost.
//   --mode=bulk               : N puts in ONE region, ONE quiet at end
//                               -> amortizes omp-target launch + completion overhead.
//
// Build: bash examples/omp/build_omp_pingpong.sh   (clang-21 omp+hip toolchain)
// Run  : HSA_XNACK=1 flux run -N2 -n2 -g1 -o mpibind=off ./omp_pingpong [--mode=...]

#include "gicc/omp.h"
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static const size_t kSizes[] = {
    1, 2, 4, 8, 64, 256, 1024, 4096, 16384, 65536,
    262144, 524288, 1048576, 2097152, 4194304, 16257024
};
static constexpr int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);
static constexpr size_t kBufBytes = 32 * 1024 * 1024;
static constexpr int kWarmup    = 10;
static constexpr int kOuter     = 21;   // median over this many windows
static constexpr int kBatch     = 50;   // puts per timing window

static const char* fmt_size(size_t s, char* b) {
    if (s < 1024)            snprintf(b, 32, "%zuB",  s);
    else if (s < 1024*1024)  snprintf(b, 32, "%zuKB", s / 1024);
    else                     snprintf(b, 32, "%zuMB", s / (1024*1024));
    return b;
}

// --xport selects the cross-node transport. Both forms are hand-written: no
// LTO pass is involved in either.
//
//   proxy : the device pushes a TransferCmd into the mapped ring from inside
//           an omp target region; a host worker thread drains it and issues
//           fi_write.
//   dwq   : the host pre-stages n triggered RMA descriptors, then ONE device
//           MMIO store to the CXI trigger counter fires all of them. The NIC
//           moves the bytes with no host involvement after the store.
enum class Xport { Proxy, Dwq };
static Xport g_xport = Xport::Proxy;

// Issue `n` puts of `bytes` to `peer`. Completion is the caller's job
// (ompx_quiet), so both transports are timed over the same window. The heap is
// symmetric, so our own `buf` address also names the peer's destination.
static void issue_puts(ompx_ctx* d_ctx, int peer, void* buf,
                       size_t bytes, int n) {
    if (g_xport == Xport::Dwq) {
        for (int i = 0; i < n; ++i)
            ompx_dwq_stage_put(peer, buf, buf, bytes);
        ompx_dwq_arm();
        #pragma omp target is_device_ptr(d_ctx)
        { ompx_dwq_fire_dev(d_ctx); }     // one MMIO store fires all n
    } else {
        #pragma omp target is_device_ptr(d_ctx, buf) firstprivate(peer, bytes, n)
        {
            for (int i = 0; i < n; ++i)
                ompx_put_dev(d_ctx, peer, buf, buf, bytes);
        }
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string mode = "pipelined";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--mode=", 0) == 0) mode = a.substr(7);
        else if (a.rfind("--xport=", 0) == 0)
            g_xport = (a.substr(8) == "dwq") ? Xport::Dwq : Xport::Proxy;
    }
    const bool per_msg = (mode == "per-msg");
    const bool bulk    = (mode == "bulk");   // N puts in ONE region, ONE quiet_host at end

    ompx_init();
    void* buf = ompx_alloc(kBufBytes);

    const int rank   = ompx_get_rank_num();
    const int nranks = ompx_get_num_ranks();
    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "omp_pingpong needs exactly 2 ranks (got %d)\n", nranks);
        ompx_free(buf);
        ompx_finalize();
        return 1;
    }
    const int peer = rank ^ 1;
    ompx_ctx* d_ctx = ompx_prepare();

    if (rank == 0) {
        printf("\n=== omp_pingpong (mode=%s xport=%s) ===\n", mode.c_str(),
               g_xport == Xport::Dwq ? "dwq (GPU trigger)" : "proxy");
        printf("ranks=%d  outer=%d  batch=%d  warmup=%d\n", nranks, kOuter, kBatch, kWarmup);
        printf("\n%-10s %14s %14s\n", "size", "mean_us/msg", "median_us/msg");
        printf("---------------------------------------------------\n");
    }

    for (int si = 0; si < kNumSizes; ++si) {
        const size_t bytes = kSizes[si];

        // ---- warmup ----
        for (int w = 0; w < kWarmup; ++w) {
            if (rank == 0) {
                d_ctx = ompx_prepare();
                issue_puts(d_ctx, peer, buf, bytes, 1);
                ompx_quiet();
            }
        }
        ompx_barrier();

        // ---- bulk: N puts in one region, ONE quiet at the very end ----
        // Amortizes BOTH the omp-target launch AND the single quiet completion
        // barrier over N puts -> exposes the sustained proxy injection rate with
        // no per-window completion barrier. If small-msg per-msg collapses here,
        // the per-window quiet barrier (not the omp path) was the cost.
        if (bulk) {
            const int N = 2000;
            double t0 = omp_get_wtime();
            if (rank == 0) {
                d_ctx = ompx_prepare();
                issue_puts(d_ctx, peer, buf, bytes, N);
                ompx_quiet();        // single completion barrier for all N
            }
            double t1 = omp_get_wtime();
            ompx_barrier();
            if (rank == 0) {
                char sb[32];
                printf("%-10s %14.3f %14s\n", fmt_size(bytes, sb),
                       (t1 - t0) * 1e6 / N, "(bulk N=2000)");
            }
            ompx_barrier();
            continue;
        }

        // ---- timed ----
        std::vector<double> samples;
        samples.reserve(kOuter);
        for (int o = 0; o < kOuter; ++o) {
            double t0 = omp_get_wtime();
            if (rank == 0) {
                if (per_msg) {
                    for (int i = 0; i < kBatch; ++i) {
                        d_ctx = ompx_prepare();
                        issue_puts(d_ctx, peer, buf, bytes, 1);
                        ompx_quiet();
                    }
                } else {
                    // pipelined: one prepare, all puts in one target region, one quiet
                    d_ctx = ompx_prepare();
                    issue_puts(d_ctx, peer, buf, bytes, kBatch);
                    ompx_quiet();
                }
            }
            double t1 = omp_get_wtime();
            samples.push_back((t1 - t0) * 1e6 / static_cast<double>(kBatch));
            ompx_barrier();
        }

        if (rank == 0) {
            double sum = 0; for (double v : samples) sum += v;
            double mean = sum / samples.size();
            std::sort(samples.begin(), samples.end());
            double med = samples[samples.size() / 2];
            char sb[32];
            printf("%-10s %14.3f %14.3f\n", fmt_size(bytes, sb), mean, med);
        }
        ompx_barrier();
    }

    ompx_free(buf);
    ompx_finalize();
    return 0;
}
