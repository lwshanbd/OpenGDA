/*
 * proxy_libfabric.hpp - libfabric submission/poll wrapper for the CPU proxy.
 *
 * Re-uses Fabric's endpoint, AV, and MR cache; opens its own TX completion
 * queue when the provider supports binding two TX CQs to one endpoint.
 * Falls back to sharing Fabric's existing CQ if dual-CQ binding fails (the
 * cxi provider on Delta is the unknown — see Task 3 probe).
 *
 * Pure host-only C++ + libfabric. No GPU symbols are referenced from this
 * header so it can be included from a .cpp translation unit under either
 * the HIP or CUDA build target.
 */
#pragma once

#include "transfer_cmd.hpp"

#include <rdma/fabric.h>      // for fid_cq, fi_addr_t
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
    // Takes Fabric for endpoint / AV / CQ access and Runtime for
    // local_buf_view() / remote_info() lookups (the latter live on Runtime,
    // not Fabric, in the current codebase).
    ProxyLibfabric(::gicc::Fabric& fab, ::gicc::Runtime& rt);
    ~ProxyLibfabric();

    ProxyLibfabric(const ProxyLibfabric&)            = delete;
    ProxyLibfabric& operator=(const ProxyLibfabric&) = delete;

    // Returns 0 on success; -FI_EAGAIN if the caller should retry; aborts on
    // any other libfabric error. `slot` is opaquely cast to void* and surfaces
    // as Completion::context after poll().
    int submit_write(const TransferCmd& c, uint64_t slot);

    // Drains up to `max` completions into `out`; returns how many were read
    // (0..max). Treats -FI_EAGAIN as "no completions yet" (returns 0).
    int poll(Completion* out, int max);

private:
    ::gicc::Fabric&  fab_;
    ::gicc::Runtime& rt_;
    fid_cq*          proxy_cq_;   // owned iff own_cq_; else == fab_.fabric->cq
    bool             own_cq_;     // true when fi_cq_open + fi_ep_bind succeeded
};

} // namespace proxy
} // namespace gicc
