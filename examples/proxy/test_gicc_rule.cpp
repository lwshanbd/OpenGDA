/*
 * test_gicc_rule.cpp - unit test for the hand-written decider (host-only, no GPU).
 * Asserts the rule reproduces the measured per-size transport crossovers.
 * Build: g++ -std=c++17 test_gicc_rule.cpp -o test_gicc_rule && ./test_gicc_rule
 */
#include "gicc_rule.h"
#include <cstdio>
#include <cstdlib>

using gicc_rule::Transport;

static int failures = 0;
static void check(bool cond, const char* msg) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", msg);
    if (!cond) ++failures;
}

int main() {
    using gicc_rule::decide;

    printf("transport-by-size rule:\n");
    check(decide(8,      1).transport == Transport::PROXY, "8B   -> proxy");
    check(decide(256,    1).transport == Transport::DWQ,   "256B -> dwq");
    check(decide(4096,   1).transport == Transport::DWQ,   "4KB  -> dwq");
    check(decide(64*1024,1).transport == Transport::PROXY, "64KB -> proxy");
    check(decide(1<<20,  1).transport == Transport::PROXY, "1MB  -> proxy");

    // band edges (measured band is [256, 4096] inclusive)
    check(decide(255,  1).transport == Transport::PROXY, "255B -> proxy (below band)");
    check(decide(4097, 1).transport == Transport::PROXY, "4097B-> proxy (above band)");

    printf("aggregation rule:\n");
    check(decide(256, 50).batch == 50, "coalesce all 50 ops (never per-op)");
    check(decide(256, 1).batch  == 1,  "single op -> batch 1");

    printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
