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
#include <cstdint>
#include <optional>
#include <thread>
#include <unordered_set>

namespace gicc { class Runtime; }

namespace gicc {
namespace proxy {

class ProxyThread {
public:
    explicit ProxyThread(::gicc::Runtime& rt);
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
    std::unordered_set<uint64_t> in_flight_;
    std::optional<PendingRetry>  pending_retry_;
};

} // namespace proxy
} // namespace gicc
