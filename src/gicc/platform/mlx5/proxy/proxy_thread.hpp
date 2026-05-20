/*
 * proxy_thread.hpp - CPU proxy worker thread for the MLX5/verbs backend.
 *
 * Mirrors gicc::proxy::ProxyThread (OFI side) one-for-one — same drain/
 * submit/poll/ack flow, same QUIET handling, same bitset-tracked
 * in-flight set — only the submit/poll primitives change (ProxyVerbs
 * uses ibv_post_send / ibv_poll_cq).
 *
 * Owns the D2HRing<kProxyRingCapacity>, the ProxyQpFleet (per-thread
 * CQ + per-peer QPs), and the ProxyVerbs wrapper that drives them.
 */
#pragma once

#include "gicc/proxy/common/d2h_ring.cuh"
#include "gicc/proxy/common/proxy_ring_defs.hpp"
#include "gicc/proxy/common/transfer_cmd.hpp"
#include "proxy_qp_fleet.hpp"
#include "proxy_verbs.hpp"

#include <atomic>
#include <bitset>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>

namespace gicc { class Runtime; }

namespace gicc::mlx5::proxy {

class ProxyThread {
public:
    // thread_idx selects the QP-fleet identity (used in the QPN-exchange
    // tag so multiple parallel proxy threads on the same rank do not
    // collide on the bootstrap tag space).
    ProxyThread(::gicc::Runtime& rt, int thread_idx);
    ~ProxyThread();

    ProxyThread(const ProxyThread&)            = delete;
    ProxyThread& operator=(const ProxyThread&) = delete;

    void start();
    void stop();

    ::gicc::proxy::ProxyRing* ring_host()   { return ring_host_; }
    ::gicc::proxy::ProxyRing* ring_device() { return ring_device_; }

private:
    void main_loop();
    void handle_quiet(uint64_t quiet_slot);

    struct PendingRetry {
        ::gicc::proxy::TransferCmd cmd;
        uint64_t                   slot;
    };

    ::gicc::Runtime&                rt_;
    ::gicc::proxy::ProxyRing*       ring_host_   = nullptr;
    ::gicc::proxy::ProxyRing*       ring_device_ = nullptr;
    std::atomic<bool>               running_{false};
    std::thread                     thr_;

    // Fleet first (it allocates the CQ); then ProxyVerbs binds to it.
    std::unique_ptr<ProxyQpFleet>   fleet_;
    std::unique_ptr<ProxyVerbs>     verbs_;

    // In-flight slot tracking (mirrors OFI ProxyThread). Bitset is
    // alloc-free; in_flight_count_ gives O(1) emptiness for QUIET.
    std::bitset<::gicc::proxy::kProxyRingCapacity> in_flight_;
    size_t                                         in_flight_count_ = 0;
    std::optional<PendingRetry>                    pending_retry_;
};

} // namespace gicc::mlx5::proxy
