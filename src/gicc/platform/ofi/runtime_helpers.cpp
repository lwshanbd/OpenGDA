/**
 * runtime_helpers.cpp - Implementations of the C ABI declared in
 * runtime_helpers.h. Built and linked alongside each user binary
 * (examples, minimod) so the LTO-emitted IR resolves these symbols
 * at link time.
 */
#include "gicc/platform/ofi/runtime_helpers.h"
#include "gicc/platform/ofi/ofi_runtime.hpp"

extern "C" {

void *gicc_runtime_peer_ipc_base(gicc::Runtime *rt, int peer, int buf_idx) {
    if (!rt) return nullptr;
    auto &peers = rt->peer_mapped_ptrs_;
    if (peer < 0 || peer >= static_cast<int>(peers.size())) return nullptr;
    if (buf_idx < 0 ||
        buf_idx >= static_cast<int>(peers[peer].size())) return nullptr;
    return peers[peer][buf_idx];
}

void *gicc_runtime_local_buf_base(gicc::Runtime *rt, int buf_idx) {
    if (!rt) return nullptr;
    auto &bufs = rt->local_bufs_;
    if (buf_idx < 0 ||
        buf_idx >= static_cast<int>(bufs.size())) return nullptr;
    return bufs[buf_idx].ptr;
}

GpuStream_t gicc_runtime_ipc_stream(gicc::Runtime *rt) {
    return rt && !rt->ipc_streams_.empty() ? rt->ipc_streams_[0] : nullptr;
}

GpuStream_t gicc_runtime_ipc_stream_indexed(gicc::Runtime *rt, int idx) {
    if (!rt || rt->ipc_streams_.empty()) return nullptr;
    if (idx < 0 || idx >= (int)rt->ipc_streams_.size()) return rt->ipc_streams_[0];
    return rt->ipc_streams_[idx];
}

void gicc_runtime_dwq_enqueue(gicc::Runtime *rt,
                               int          peer,
                               int          dst_buf,
                               std::size_t  dst_off,
                               int          src_buf,
                               std::size_t  src_off,
                               std::size_t  size) {
    if (!rt) return;
    auto &ob = rt->local_bufs_[src_buf];
    auto &ri = rt->remote_info_cache_[
        static_cast<std::size_t>(peer) * static_cast<std::size_t>(rt->n_bufs_) +
        static_cast<std::size_t>(dst_buf)];
    const std::uint64_t remote_addr = rt->comm_->is_virt_addr_mode()
        ? (ri.rma_addr + dst_off)
        : (ri.rma_addr - ri.base_addr) + dst_off;

    ++rt->mono_total_ops_;
    ++rt->my_n_remote_ops_;
    auto *dwq = rt->dwq_get_();
    dwq->queue_rma_write(
        rt->comm_->fabric->domain, rt->comm_->fabric->ep,
        static_cast<char *>(ob.ptr) + src_off, ob.desc_, size,
        rt->comm_->av_addrs[peer], remote_addr, ri.rma_key,
        rt->comm_->fabric->trigger_cntr,
        rt->shared_completion_cntr_,
        /*threshold=*/rt->mono_total_ops_);
    rt->my_pending_.push_back(dwq);
}

// Batched form: queues N RMA writes in one host call. Each descriptor's
// trigger threshold is its 1-based slot within the batch (the kernel's
// single trigger MMIO write at flush-time fires all of them). Saves the
// per-op IR call overhead and combines the mono_total_ops_ /
// my_n_remote_ops_ counter updates. libfabric still gets one
// fi_control(FI_QUEUE_WORK) per descriptor — that's a per-op limit
// inherent to the deferred-work API, no batched FI_QUEUE_WORK exists.
void gicc_runtime_dwq_enqueue_batched(gicc::Runtime    *rt,
                                       int               n_ops,
                                       const int        *peers,
                                       const int        *dst_bufs,
                                       const std::size_t *dst_offs,
                                       const int        *src_bufs,
                                       const std::size_t *src_offs,
                                       const std::size_t *sizes) {
    if (!rt || n_ops <= 0) return;
    rt->mono_total_ops_   += static_cast<std::uint64_t>(n_ops);
    rt->my_n_remote_ops_  += static_cast<std::uint64_t>(n_ops);
    const std::uint64_t batch_top = rt->mono_total_ops_;
    for (int i = 0; i < n_ops; ++i) {
        auto &ob = rt->local_bufs_[src_bufs[i]];
        auto &ri = rt->remote_info_cache_[
            static_cast<std::size_t>(peers[i]) *
                static_cast<std::size_t>(rt->n_bufs_) +
            static_cast<std::size_t>(dst_bufs[i])];
        const std::uint64_t remote_addr = rt->comm_->is_virt_addr_mode()
            ? (ri.rma_addr + dst_offs[i])
            : (ri.rma_addr - ri.base_addr) + dst_offs[i];

        // Each descriptor's threshold is sequential within the batch.
        // The kernel writes trigger_val_ = batch_top, satisfying all.
        const std::uint64_t threshold =
            batch_top - static_cast<std::uint64_t>(n_ops - 1 - i);

        auto *dwq = rt->dwq_get_();
        dwq->queue_rma_write(
            rt->comm_->fabric->domain, rt->comm_->fabric->ep,
            static_cast<char *>(ob.ptr) + src_offs[i], ob.desc_, sizes[i],
            rt->comm_->av_addrs[peers[i]], remote_addr, ri.rma_key,
            rt->comm_->fabric->trigger_cntr,
            rt->shared_completion_cntr_,
            threshold);
        rt->my_pending_.push_back(dwq);
    }
}

volatile std::uint64_t *gicc_runtime_trigger_addr(gicc::Runtime *rt) {
    return rt ? rt->comm_->get_trigger_addr() : nullptr;
}

std::uint64_t gicc_runtime_trigger_val(gicc::Runtime *rt) {
    return rt ? rt->mono_total_ops_ : 0;
}

#ifdef GICC_CPU_PROXY
void* gicc_runtime_proxy_ring_device_ptr(gicc::Runtime *rt) {
    return rt ? rt->ensure_proxy_ring() : nullptr;
}
#endif

const void* gicc_runtime_host_mirror_of(gicc::Runtime *rt, const void* dev_ptr) {
    return rt ? rt->host_mirror_of(dev_ptr) : nullptr;
}

}  // extern "C"
