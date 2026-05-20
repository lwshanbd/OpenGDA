/*
 * proxy_libfabric.hpp - libfabric submission/poll wrapper for the CPU proxy.
 *
 * One ProxyLibfabric is bound 1:1 to one ProxyThread, which is bound 1:1 to
 * one (fi_endpoint, fi_cq) pair owned by Fabric (proxy_eps_[ep_idx],
 * proxy_cqs_[ep_idx]). With one EP+CQ per thread, completions never cross
 * thread boundaries — fixing the prior shared-CQ design where fi_cq_read
 * on thread A could consume a completion that thread B was still waiting
 * for, hanging quiet/drain at N>1.
 *
 * The peer fi_addr_t and the buffer rkey are still resolved through the
 * shared Fabric AV / MemoryRegion: peers only learn the main EP's address
 * (one allgather), and each MR is fi_mr_bind'd to every proxy EP plus the
 * main EP at register_buffer() time, so one rkey is valid across the fleet.
 *
 * Pure host-only C++ + libfabric. No GPU symbols are referenced from this
 * header so it can be included from a .cpp translation unit under either
 * the HIP or CUDA build target.
 */
#pragma once

#include "gicc/proxy/common/transfer_cmd.hpp"

#include <rdma/fabric.h>      // for fid_ep, fid_cq, fi_addr_t
#include <cstdint>

namespace gicc {

// Forward declarations — pulling in the full ofi_runtime.hpp here would drag
// the HIP runtime header into every TU that touches the proxy interface, and
// Runtime currently still references hip*. Implementation file includes the
// full headers.
class Fabric;
class Runtime;

namespace proxy {

struct Completion {
    void* context;   // we use this as the slot index (cast through void*)
};

class ProxyLibfabric {
public:
    // ep_idx selects the dedicated (fi_endpoint, fi_cq) this instance will
    // submit on / poll. Must be in [0, fab.num_proxy_eps()) — Runtime calls
    // Fabric::create_proxy_endpoints(N) before constructing the fleet.
    ProxyLibfabric(::gicc::Fabric& fab, ::gicc::Runtime& rt, int ep_idx);
    ~ProxyLibfabric();

    ProxyLibfabric(const ProxyLibfabric&)            = delete;
    ProxyLibfabric& operator=(const ProxyLibfabric&) = delete;

    // Returns 0 on success; -FI_EAGAIN if the caller should retry; aborts on
    // any other libfabric error. `slot` is opaquely cast to void* and surfaces
    // as Completion::context after poll().
    int submit_write(const TransferCmd& c, uint64_t slot);

    // Batched write submit — libfabric equivalent of verbs `wr->next` chain.
    // For each item k in [0, n), issues `fi_writemsg` with `FI_MORE` flag set
    // on all but the last (n-1)th. With FI_MORE the provider holds the
    // doorbell until the final message arrives, so a batch of N writes pays
    // ~1 doorbell + ~N user-space WR fills, instead of N doorbells.
    //
    // Returns 0 if the entire batch was accepted. If any submit returns
    // -FI_EAGAIN (flow control), the function returns the index `i` of the
    // first cmd that hit EAGAIN as a *negative* value `-(i + 1)`; the caller
    // should drain CQ then retry the tail starting at `i`. Aborts on any
    // other libfabric error.
    //
    // Caveat: when 0 < accepted < n is reported via the negative return,
    // the provider may or may not have rung the doorbell yet — the caller
    // must keep submitting (FI_MORE-tail then no-FI_MORE on the last) for
    // the partial batch's writes to actually launch.
    int submit_write_batch(const TransferCmd* cmds,
                           const uint64_t*   slots,
                           size_t            n);

    // Issues a non-fetching remote atomic add (FI_SUM, FI_UINT32) on the
    // peer's counter slot identified by (c.dst_rank, c.dst_buf, c.dst_offset)
    // using the local 4-byte source value at (c.src_buf, c.src_offset).
    // `c.bytes` is ignored and treated as 4. Same return contract as
    // submit_write. Provider must advertise FI_ATOMIC; on Slingshot CXI this
    // is true, on a fallback provider lacking atomics this aborts.
    int submit_atomic_add(const TransferCmd& c, uint64_t slot);

    // Drains up to `max` completions into `out`; returns how many were read
    // (0..max). Treats -FI_EAGAIN as "no completions yet" (returns 0).
    int poll(Completion* out, int max);

private:
    ::gicc::Fabric&  fab_;
    ::gicc::Runtime& rt_;
    fid_ep*          ep_;         // points into Fabric::proxy_eps_
    fid_cq*          cq_;         // points into Fabric::proxy_cqs_
    int              ep_idx_;     // for diagnostics
};

} // namespace proxy
} // namespace gicc
