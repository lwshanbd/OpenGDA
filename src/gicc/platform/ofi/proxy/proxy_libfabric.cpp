/*
 * proxy_libfabric.cpp - host-only libfabric submission/poll wrapper.
 *
 * Each ProxyLibfabric instance owns a 1:1 binding to one (fi_endpoint,
 * fi_cq) in Fabric::proxy_eps_ / proxy_cqs_, selected by ep_idx. The
 * endpoints/CQs themselves are owned by Fabric (it created and will close
 * them); we just hold non-owning pointers.
 */
#include "proxy_libfabric.hpp"

#include "gicc/platform/ofi/internal/fabric.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"

#include <rdma/fi_atomic.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace gicc {
namespace proxy {

ProxyLibfabric::ProxyLibfabric(::gicc::Fabric& fab, ::gicc::Runtime& rt,
                               int ep_idx)
    : fab_(fab), rt_(rt), ep_(nullptr), cq_(nullptr), ep_idx_(ep_idx)
{
    if (ep_idx < 0 || ep_idx >= fab_.num_proxy_eps()) {
        fprintf(stderr,
            "ProxyLibfabric: ep_idx %d out of range [0, %d) — Runtime must "
            "call Fabric::create_proxy_endpoints(N) before constructing the "
            "proxy fleet\n",
            ep_idx, fab_.num_proxy_eps());
        std::abort();
    }
    ep_ = fab_.proxy_ep(ep_idx);
    cq_ = fab_.proxy_cq(ep_idx);
}

ProxyLibfabric::~ProxyLibfabric() = default;

// Helper: build an fi_msg_rma + iovec/rma_iov for one TransferCmd.
// Caller owns iov / rma_iov storage so the batch path can keep them alive
// across the fi_writemsg(FI_MORE) calls in one loop iteration.
static inline void build_write_msg(::gicc::Runtime& rt,
                                   const TransferCmd& c,
                                   uint64_t slot,
                                   void*  desc,
                                   struct iovec* iov,
                                   struct fi_rma_iov* rma_iov,
                                   struct fi_msg_rma* msg)
{
    auto        lb   = rt.local_buf_view(c.src_buf);
    const auto& ri   = rt.remote_info(c.dst_rank, c.dst_buf);
    fi_addr_t   peer = rt.av_addr(c.dst_rank);

    char*    src   = static_cast<char*>(lb.ptr) + c.src_offset;
    uint64_t raddr = rt.is_virt_addr_mode()
                        ? (ri.rma_addr + c.dst_offset)
                        : (ri.rma_addr - ri.base_addr) + c.dst_offset;
    uint64_t rkey  = ri.rma_key;

    iov->iov_base = src;
    iov->iov_len  = c.bytes;

    rma_iov->addr = raddr;
    rma_iov->len  = c.bytes;
    rma_iov->key  = rkey;

    msg->msg_iov       = iov;
    msg->desc          = &desc;
    msg->iov_count     = 1;
    msg->addr          = peer;
    msg->rma_iov       = rma_iov;
    msg->rma_iov_count = 1;
    msg->context       = reinterpret_cast<void*>(slot);
    msg->data          = 0;
}

int ProxyLibfabric::submit_write_batch(const TransferCmd* cmds,
                                       const uint64_t*    slots,
                                       size_t             n)
{
    if (n == 0) return 0;
    // For each cmd: one fi_msg_rma (+ iovec + rma_iov). Reusable scratch
    // pre-sized large enough — typical batch is 32-256 cmds. Static thread
    // local so we don't malloc on every call. ProxyLibfabric is per-thread.
    constexpr size_t kCap = 1024;
    if (n > kCap) {
        fprintf(stderr,
            "ProxyLibfabric[%d]::submit_write_batch: batch n=%zu > cap=%zu; "
            "split caller-side\n", ep_idx_, n, kCap);
        std::abort();
    }
    static thread_local struct iovec       iov_arr[kCap];
    static thread_local struct fi_rma_iov  rma_arr[kCap];
    static thread_local struct fi_msg_rma  msg_arr[kCap];
    static thread_local void*              desc_arr[kCap];

    // Build all messages.
    for (size_t i = 0; i < n; ++i) {
        desc_arr[i] = fab_.proxy_buf_desc(cmds[i].src_buf, ep_idx_);
        build_write_msg(rt_, cmds[i], slots[i], desc_arr[i],
                        &iov_arr[i], &rma_arr[i], &msg_arr[i]);
        // Re-point desc to a stable address inside desc_arr[] so the
        // address survives across builders. msg.desc is `void**` and
        // libfabric reads it during fi_writemsg.
        msg_arr[i].desc = &desc_arr[i];
    }

    // Submit with FI_MORE on all but the last. Last call (no FI_MORE) is
    // what actually rings the CXI doorbell.
    for (size_t i = 0; i < n; ++i) {
        uint64_t flags = (i + 1 < n) ? FI_MORE : 0ULL;
        int ret = (int)fi_writemsg(ep_, &msg_arr[i], flags);
        if (ret == -FI_EAGAIN) {
            // Provider couldn't enqueue. The FI_MORE-prefixed messages
            // already enqueued may be sitting un-rung; we MUST follow up
            // with at least one no-FI_MORE call before returning, but
            // libfabric semantics here are that nothing about the prior
            // FI_MORE cmds is "unwound" — they remain queued, waiting for
            // the next non-FI_MORE submit on this EP to fire. The caller
            // is responsible for retrying [i, n) without any state munging.
            // Return -(i+1) so caller knows how far we got.
            return -static_cast<int>(i + 1);
        }
        if (ret != 0) {
            fprintf(stderr,
                "ProxyLibfabric[%d]::submit_write_batch: fi_writemsg(i=%zu/%zu, "
                "flags=0x%lx) failed: %s\n",
                ep_idx_, i, n, (unsigned long)flags, fi_strerror(-ret));
            std::abort();
        }
    }
    return 0;
}

int ProxyLibfabric::submit_write(const TransferCmd& c, uint64_t slot)
{
    auto        lb   = rt_.local_buf_view(c.src_buf);
    const auto& ri   = rt_.remote_info(c.dst_rank, c.dst_buf);
    fi_addr_t   peer = rt_.av_addr(c.dst_rank);

    char*    src   = static_cast<char*>(lb.ptr) + c.src_offset;
    // Per-proxy-EP local descriptor (CXI requires one MR registration per
    // EP). The main-EP desc on lb.desc is unused on the proxy path.
    void*    desc  = fab_.proxy_buf_desc(c.src_buf, ep_idx_);
    uint64_t raddr = rt_.is_virt_addr_mode()
                       ? (ri.rma_addr + c.dst_offset)
                       : (ri.rma_addr - ri.base_addr) + c.dst_offset;
    uint64_t rkey  = ri.rma_key;

    int ret = fi_write(ep_,
                       src, c.bytes, desc,
                       peer, raddr, rkey,
                       reinterpret_cast<void*>(slot));
    if (ret == -FI_EAGAIN) {
        return -FI_EAGAIN;
    }
    if (ret != 0) {
        fprintf(stderr,
            "ProxyLibfabric[%d]::submit_write: fi_write failed: %s\n",
            ep_idx_, fi_strerror(-ret));
        std::abort();
    }
    return 0;
}


int ProxyLibfabric::submit_atomic_add(const TransferCmd& c, uint64_t slot)
{
    auto        lb   = rt_.local_buf_view(c.src_buf);
    const auto& ri   = rt_.remote_info(c.dst_rank, c.dst_buf);
    fi_addr_t   peer = rt_.av_addr(c.dst_rank);

    // Source is one uint32 living at src_buf[src_offset]. We don't need the
    // bytes field — fi_atomic counts elements, datatype gives element size.
    void*    src   = static_cast<char*>(lb.ptr) + c.src_offset;
    void*    desc  = fab_.proxy_buf_desc(c.src_buf, ep_idx_);
    uint64_t raddr = rt_.is_virt_addr_mode()
                       ? (ri.rma_addr + c.dst_offset)
                       : (ri.rma_addr - ri.base_addr) + c.dst_offset;
    uint64_t rkey  = ri.rma_key;

    int ret = fi_atomic(ep_,
                        src, /*count=*/1, desc,
                        peer, raddr, rkey,
                        FI_UINT32, FI_SUM,
                        reinterpret_cast<void*>(slot));
    if (ret == -FI_EAGAIN) {
        return -FI_EAGAIN;
    }
    if (ret != 0) {
        fprintf(stderr,
            "ProxyLibfabric[%d]::submit_atomic_add: fi_atomic failed: %s\n",
            ep_idx_, fi_strerror(-ret));
        std::abort();
    }
    return 0;
}


int ProxyLibfabric::poll(Completion* out, int max)
{
    // 256 fi_cq_entry slots = 4 KiB on stack. 256 is the minimum bump that
    // absorbs a typical 50-cmd bench burst in one syscall while keeping the
    // on-stack array modest.
    constexpr int kMaxBatch = 256;
    struct fi_cq_entry entries[kMaxBatch];

    int n = (int)fi_cq_read(cq_, entries,
                            std::min(max, kMaxBatch));
    if (n == -FI_EAGAIN) {
        return 0;
    }
    if (n == -FI_EAVAIL) {
        struct fi_cq_err_entry err = {};
        ssize_t r = fi_cq_readerr(cq_, &err, 0);
        if (r > 0) {
            fprintf(stderr,
                "ProxyLibfabric[%d]::poll: CQ error: %s (prov_errno=%d)\n",
                ep_idx_, fi_strerror(err.err), err.prov_errno);
        }
        std::abort();
    }
    if (n < 0) {
        fprintf(stderr,
            "ProxyLibfabric[%d]::poll: fi_cq_read failed: %s\n",
            ep_idx_, fi_strerror(-n));
        std::abort();
    }
    for (int i = 0; i < n; ++i) {
        out[i].context = entries[i].op_context;
    }
    return n;
}

} // namespace proxy
} // namespace gicc
