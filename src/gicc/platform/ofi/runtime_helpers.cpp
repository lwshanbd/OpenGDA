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

hipStream_t gicc_runtime_ipc_stream(gicc::Runtime *rt) {
    return rt ? rt->ipc_stream_ : nullptr;
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

volatile std::uint64_t *gicc_runtime_trigger_addr(gicc::Runtime *rt) {
    return rt ? rt->comm_->get_trigger_addr() : nullptr;
}

std::uint64_t gicc_runtime_trigger_val(gicc::Runtime *rt) {
    return rt ? rt->mono_total_ops_ : 0;
}

}  // extern "C"
