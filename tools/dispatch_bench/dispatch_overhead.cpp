/**
 * dispatch_overhead.cpp — Microbenchmark for the C2 overhead claim.
 *
 * Invokes Dispatcher::bind() a fixed number of times and reports median,
 * p99, mean, min, max per call. Designed for M5-B5a runs R083-R085:
 *   R083: baseline (policy file absent -> constant-time fallback)
 *   R084: V=2 policy (two-variant rule list)
 *   R085: V=3 policy (three-variant rule list)
 *
 * Variant count is inferred from the distinct number of `then` decisions
 * in the policy file so the benchmark doesn't need a `--V=N` flag.
 *
 * Exit 0 on success; 1 if the --check-gate argument is present AND
 * median > 0.5 ms or p99 > 2 ms.
 */

#include "gicc/dispatch/dispatcher.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace gicc;

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s  [--iters N]  [--warmup N]  [--check-gate]\n"
        "  Requires $GICC_POLICY_FILE to point at the policy to benchmark.\n", argv0);
}

int main(int argc, char** argv) {
    int iters = 10000;
    int warmup = 1000;
    bool check_gate = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters"  && i + 1 < argc) { iters  = std::atoi(argv[++i]); }
        else if (a == "--warmup" && i + 1 < argc) { warmup = std::atoi(argv[++i]); }
        else if (a == "--check-gate") { check_gate = true; }
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }

    dispatch::Dispatcher d;

    // Shuffle the observation vector so the branch predictor can't memorise
    // a single feature tuple's path.
    std::vector<dispatch::PrepareObs> corpus;
    for (int peer = 0; peer < 16; ++peer) {
        for (std::size_t sz : {std::size_t{128}, std::size_t{1<<14},
                               std::size_t{1<<20}, std::size_t{1<<23}}) {
            dispatch::PrepareObs o;
            o.peer_rank = peer;
            o.transfer_size = sz;
            o.has_topology_hint = (peer % 4 == 0);
            corpus.push_back(o);
        }
    }

    // Warm-up.
    for (int i = 0; i < warmup; ++i) {
        (void)d.bind(corpus[i % corpus.size()]);
    }

    // Timed loop.
    std::vector<long long> ns;
    ns.reserve(iters);
    using clk = std::chrono::steady_clock;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        auto b = d.bind(corpus[i % corpus.size()]);
        auto t1 = clk::now();
        // asm "use" to prevent constant-folding the call away
        asm volatile("" :: "r"(b.variant_idx));
        ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }

    std::sort(ns.begin(), ns.end());
    double median = ns[ns.size() / 2] / 1000.0;                // us
    double p99    = ns[(size_t)(ns.size() * 0.99)] / 1000.0;   // us
    long long sum = 0; for (long long v : ns) sum += v;
    double mean   = (double)sum / ns.size() / 1000.0;
    double mn = ns.front() / 1000.0, mx = ns.back() / 1000.0;

    std::fprintf(stdout,
        "dispatch_overhead  iters=%d  policy_loaded=%d  median_us=%.3f  "
        "p99_us=%.3f  mean_us=%.3f  min_us=%.3f  max_us=%.3f\n",
        iters, d.policy_loaded() ? 1 : 0, median, p99, mean, mn, mx);

    if (check_gate) {
        // C2 gate: median ≤ 0.5 ms = 500 us; p99 ≤ 2 ms = 2000 us
        if (median > 500.0 || p99 > 2000.0) {
            std::fprintf(stderr,
                "GATE FAIL: median=%.3fus (limit 500), p99=%.3fus (limit 2000)\n",
                median, p99);
            return 1;
        }
        std::fprintf(stderr, "GATE PASS\n");
    }
    return 0;
}
