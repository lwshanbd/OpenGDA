/*
 * proxy_qp_fleet.hpp - per-thread per-peer RC QP set for the MLX5 CPU proxy.
 *
 * Each ProxyThread owns one ProxyQpFleet (which in turn owns one CQ and
 * one RC QP per remote peer). All QPs in the fleet share:
 *   - the Runtime's ib_context + protection domain (so registered MRs are
 *     valid across the fleet without re-registration), and
 *   - this fleet's send/recv CQ (so completions never cross threads —
 *     mirrors the OFI side's one-EP-per-thread design).
 *
 * The fleet is functionally separate from the device-direct DevX QPs that
 * mlx5_runtime.hpp's Runtime ctor creates. The two QP sets share the PD
 * (and therefore can use the same lkey/rkey for any registered MR) but
 * own no state in common.
 *
 * Used only when GICC_CPU_PROXY is defined.
 */
#pragma once

#include <infiniband/verbs.h>
#include <cstdint>
#include <vector>

// Bootstrap is a using-alias (resolves to detail::BootstrapMPI or
// BootstrapPMI2 depending on GICC_BOOTSTRAP_*), so we can't forward-declare
// it as `class Bootstrap;` — pull in the full header.
#include "gicc/bootstrap/bootstrap.hpp"

namespace gicc::mlx5::proxy {

class ProxyQpFleet {
public:
    // Build N_peers - 1 RC QPs (one per remote peer). All QPs share the
    // given `pd` and post completions to a freshly-allocated send CQ
    // owned by this object. `thread_idx` is informational (used in tag
    // matching during the QPN-exchange so multiple proxy threads on the
    // same rank can do the handshake in parallel without aliasing tags).
    ProxyQpFleet(ibv_context* ctx,
                 ibv_pd*      pd,
                 Bootstrap&   boot,
                 int          thread_idx,
                 uint32_t     sq_depth = 4096,
                 uint32_t     cq_depth = 4096);

    ~ProxyQpFleet();

    ProxyQpFleet(const ProxyQpFleet&)            = delete;
    ProxyQpFleet& operator=(const ProxyQpFleet&) = delete;

    // Per-peer QP for posting RDMA WRITE/ATOMIC sends. nullptr for self.
    ibv_qp* qp(int peer) const noexcept {
        if (peer < 0 || peer >= static_cast<int>(qps_.size())) return nullptr;
        return qps_[peer];
    }

    // Completion queue this fleet posts/polls on.
    ibv_cq* cq() const noexcept { return cq_; }

    int thread_idx() const noexcept { return thread_idx_; }

private:
    ibv_context*           ctx_ = nullptr;
    ibv_pd*                pd_  = nullptr;
    ibv_cq*                cq_  = nullptr;
    int                    thread_idx_ = 0;
    // qps_[peer]; nullptr at the self-rank slot.
    std::vector<ibv_qp*>   qps_;
};

} // namespace gicc::mlx5::proxy
