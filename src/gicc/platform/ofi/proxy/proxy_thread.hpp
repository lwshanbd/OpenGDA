/*
 * proxy_thread.hpp - CPU proxy worker thread.
 *
 * Owns one D2HRing<kProxyRingCapacity> and one ProxyLibfabric. The worker
 * thread loops on:
 *   1. drain the ring (bounded batch for fairness),
 *   2. submit each WRITE via libfabric, tracking in-flight slots,
 *   3. handle QUIET inline (drain pending completions, then ack the slot),
 *   4. poll the CQ, mark_acked + advance_tail_from_mask on each completion.
 *
 * The proxy thread itself is pure host C++; no GPU runtime symbols are
 * referenced from this header. The forward-declared ::gicc::Runtime is
 * defined in ofi_runtime.hpp, which is currently HIP-tied — keep that
 * include out of this header so consumers can link without dragging hip.h.
 */
#pragma once

#include "gicc/proxy/common/d2h_ring.cuh"
#include "gicc/proxy/common/proxy_ring_defs.hpp"
#include "gicc/proxy/common/transfer_cmd.hpp"
#include "proxy_libfabric.hpp"

#include <atomic>
#include <bitset>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

namespace gicc { class Runtime; }

namespace gicc {
namespace proxy {

class ProxyThread {
public:
    // ep_idx selects the (fi_endpoint, fi_cq) this thread submits/polls on.
    // Runtime opens N proxy endpoints in Fabric (Fabric::create_proxy_endpoints)
    // before spawning the fleet, then constructs ProxyThread instances with
    // ep_idx 0..N-1. With one EP per thread, completions never cross threads
    // — fixing the prior shared-CQ bug where fi_cq_read on thread A could
    // consume a completion that thread B was still waiting for.
    ProxyThread(::gicc::Runtime& rt, int ep_idx);
    ~ProxyThread();

    ProxyThread(const ProxyThread&)            = delete;
    ProxyThread& operator=(const ProxyThread&) = delete;

    void start();
    void stop();

    ProxyRing* ring_host()   { return ring_host_; }
    ProxyRing* ring_device() { return ring_device_; }

private:
    void main_loop();
    void handle_quiet(uint64_t quiet_slot);

    // A single pending retry: when submit_write returns -FI_EAGAIN, pop()
    // has already advanced proxy_read_cursor, so the (cmd, slot) must be
    // stashed here and reattempted on the next iteration before any new
    // pop. The proxy is single-threaded, so at most one retry is pending.
    struct PendingRetry {
        TransferCmd cmd;
        uint64_t    slot;
    };

    ::gicc::Runtime&             rt_;
    ProxyRing*                   ring_host_;
    ProxyRing*                   ring_device_;
    std::atomic<bool>            running_;
    std::thread                  thr_;
    ProxyLibfabric               lf_;
    // In-flight slots, indexed by (slot & ring_mask). Bitset is allocation-
    // free and gives O(1) set/clear/test (1 instr each) — replaces the prior
    // std::unordered_set whose insert/erase on hot path triggered allocator
    // calls and pointer chasing. Capacity matches the ring (kProxyRingCapacity)
    // because ring back-pressure guarantees no two in-flight slots collide
    // on the same ring index.
    std::bitset<kProxyRingCapacity> in_flight_;
    // Tracks how many bits are set in in_flight_ — std::bitset::count() is
    // popcount over all words, but we want O(1) emptiness checks in the
    // hot QUIET drain. Updated alongside set/reset.
    size_t                       in_flight_count_ = 0;
    std::optional<PendingRetry>  pending_retry_;

    // --- Optional hot-path profiling (GICC_PROXY_PROFILE=1) -----------------
    // Measures where each op's time goes WITHOUT a GPU profiler (the cost is
    // in the CPU proxy loop + NIC round-trip, which rocprof can't see). For
    // each submitted op we stamp submit_ns_[ring_idx]; on its completion we
    // accumulate (now - submit). Also tracks the max in-flight depth actually
    // achieved — the key signal for "NIC-bound (deep pipeline) vs serialized
    // (depth ~1)". Zero cost when disabled. Dumped to stderr in the dtor.
    bool                         prof_enabled_   = false;
    std::vector<uint64_t>        prof_submit_ns_;        // per ring-idx stamp
    uint64_t                     prof_op_count_  = 0;
    uint64_t                     prof_lat_sum_ns_ = 0;
    uint64_t                     prof_lat_max_ns_ = 0;
    uint64_t                     prof_lat_min_ns_ = ~0ull;
    size_t                       prof_max_inflight_ = 0;
    uint64_t                     prof_submit_calls_ = 0; // fi_write submit count
    uint64_t                     prof_poll_empty_ = 0;   // loop passes, CQ empty
    uint64_t                     prof_poll_hit_   = 0;    // loop passes, CQ had >=1
    void prof_record_completion(size_t bit);
};

} // namespace proxy
} // namespace gicc
