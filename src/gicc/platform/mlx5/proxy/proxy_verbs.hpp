/*
 * proxy_verbs.hpp - verbs submission/poll wrapper for the MLX5 CPU proxy.
 *
 * One-to-one with one ProxyThread. Owns nothing externally — it borrows
 * the per-thread ProxyQpFleet (peer QPs + CQ) and the Runtime accessors
 * that resolve (rank, buf_idx, offset) → (lkey, rkey, raddr).
 *
 * Mirrors the OFI side's ProxyLibfabric: same return contract for
 * submit_write (0 or -EAGAIN; aborts on hard error) so that
 * ProxyThread's per-thread main loop is one-for-one portable.
 */
#pragma once

#include "gicc/proxy/common/transfer_cmd.hpp"

#include <cstdint>

struct ibv_qp;
struct ibv_cq;
namespace gicc { class Runtime; }

namespace gicc::mlx5::proxy {

class ProxyQpFleet;

struct Completion {
    void* context;   // we use this as the slot index (cast through void*)
};

class ProxyVerbs {
public:
    ProxyVerbs(::gicc::Runtime& rt, ProxyQpFleet& fleet);
    ~ProxyVerbs() = default;

    ProxyVerbs(const ProxyVerbs&)            = delete;
    ProxyVerbs& operator=(const ProxyVerbs&) = delete;

    // Returns 0 on success; -ENOMEM if the SQ is full and caller should
    // retry (mapped from ibv_post_send's documented backpressure path);
    // aborts on hard error. `slot` is opaquely cast to wr_id and surfaces
    // as Completion::context after poll().
    int submit_write(const ::gicc::proxy::TransferCmd& c, uint64_t slot);

    // Issues an RDMA READ via IBV_WR_RDMA_READ. The NIC pulls `c.bytes`
    // from the peer's (c.dst_rank, c.dst_buf, c.dst_offset) into the
    // local landing slice (c.src_buf, c.src_offset). Verbs ordering: an
    // RDMA_READ on the same QP is fenced after prior outbound WRITEs to
    // the same peer per the IB spec, so a put_no_db()→get_no_db() pair
    // pulled through one QP sees its own prior writes. Returns -ENOMEM
    // if SQ full; aborts on hard error.
    int submit_read(const ::gicc::proxy::TransferCmd& c, uint64_t slot);

    // Verbs has no non-fetching FI_SUM-style atomic add: IBV_WR_ATOMIC_FETCH_AND_ADD
    // exists but returns the pre-add value via a local buffer. For now we
    // map ATOMIC to fetch-and-add into a per-Verbs scratch buffer, which
    // gives identical remote-counter semantics (the original value is
    // discarded). Returns -ENOMEM if SQ full; aborts on hard error.
    int submit_atomic_add(const ::gicc::proxy::TransferCmd& c, uint64_t slot);

    // Drains up to `max` completions; returns count read (0..max).
    int poll(Completion* out, int max);

private:
    ::gicc::Runtime& rt_;
    ProxyQpFleet&    fleet_;
};

} // namespace gicc::mlx5::proxy
