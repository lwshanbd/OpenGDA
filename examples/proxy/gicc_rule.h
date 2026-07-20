/*
 * gicc_rule.h - hand-written rule-based transport/aggregation decider.
 *
 * This is the "if-else" decider the measurements justify (no ML). Every
 * threshold below is a measured Tioga/CXI crossover, not a fitted parameter.
 * Source data: build_ofi/SUMMARY_zh.md + the CSVs it indexes.
 *
 *   Transport-by-size (per-pair latency crossover, P=4 cross-node):
 *     8B    proxy 6.99us < dwq 8.32us   -> PROXY
 *     256B  dwq   1.69us < proxy 3.02us -> DWQ
 *     4KB   dwq   2.36us < proxy 3.04us -> DWQ
 *     64KB  proxy 3.00us < dwq 4.14us   -> PROXY
 *     >=256KB  tie (~24 GB/s single-NIC cap) -> either; pick PROXY
 *   => DWQ wins the middle band [256B, 4KB] (queue-many-fire-once amortizes
 *      its one MMIO trigger + one cntr wait); PROXY elsewhere.
 *
 *   Aggregation granularity (puts per flush/trigger boundary):
 *     batch1->batchN cuts 256B latency 5.5x(proxy)/9.9x(dwq); 1MB only 1.33x.
 *   => always coalesce maximally. batch=1 (per-op quiet/trigger) is the
 *      anti-pattern; never emit it.
 *
 * Why a rule and not ML: the decision is 1-D in size, monotone, physically
 * explained (NIC round-trip + single-NIC bandwidth + fixed per-op cost), and
 * the axes (concurrency->NIC count, size->transport, aggregation) are
 * orthogonal. A hand-written rule is interpretable and training-free. ML would
 * only earn its place on a workload this rule provably can't handle (e.g.
 * heterogeneous-op grouping under contention) -- not yet demonstrated.
 */
#pragma once
#include <cstddef>

namespace gicc_rule {

enum class Transport { PROXY, DWQ };

struct Decision {
    Transport transport;
    int       batch;       // puts per flush/trigger boundary (op aggregation)
};

// Measured DWQ-win band. Constants are crossover points, not tuned.
inline constexpr size_t kDwqLo = 256;     // bytes, inclusive
inline constexpr size_t kDwqHi = 4096;    // bytes, inclusive

// n_ops = number of coalescible same-peer ops available at this site (the
// pass knows this statically; the runtime API can't see it).
inline Decision decide(size_t bytes, int n_ops) {
    Decision d;
    d.transport = (bytes >= kDwqLo && bytes <= kDwqHi)
                      ? Transport::DWQ : Transport::PROXY;
    d.batch     = n_ops;   // coalesce all; never per-op (the anti-pattern)
    return d;
}

inline const char* name(Transport t) {
    return t == Transport::DWQ ? "dwq" : "proxy";
}

} // namespace gicc_rule
