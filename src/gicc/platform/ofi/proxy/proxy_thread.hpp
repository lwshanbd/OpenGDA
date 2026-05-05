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

#include "d2h_ring.cuh"
#include "proxy_libfabric.hpp"
#include "proxy_ring_defs.hpp"
#include "transfer_cmd.hpp"

#include <atomic>
#include <bitset>
#include <cstdint>
#include <optional>
#include <thread>

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
};

} // namespace proxy
} // namespace gicc
