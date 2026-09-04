/**
 * runtime_helpers.cpp - Implementations of the C ABI declared in
 * runtime_helpers.h. Built and linked alongside each user binary
 * (examples, minimod) so the LTO-emitted IR resolves these symbols
 * at link time.
 */
#include "gicc/platform/ofi/runtime_helpers.h"
#include "gicc/platform/ofi/ofi_runtime.hpp"

#include <limits>

namespace {

constexpr bool disjointHalfOpenRanges(std::uintptr_t leftBegin,
                                      std::size_t leftSize,
                                      std::uintptr_t rightBegin,
                                      std::size_t rightSize) {
    constexpr std::uintptr_t limit =
        std::numeric_limits<std::uintptr_t>::max();
    if (leftSize == 0 || rightSize == 0 ||
        leftSize > limit - leftBegin || rightSize > limit - rightBegin)
        return false;
    const std::uintptr_t leftEnd = leftBegin + leftSize;
    const std::uintptr_t rightEnd = rightBegin + rightSize;
    return leftEnd <= rightBegin || rightEnd <= leftBegin;
}

static_assert(disjointHalfOpenRanges(0, 4, 4, 8));
static_assert(disjointHalfOpenRanges(12, 4, 0, 12));
static_assert(!disjointHalfOpenRanges(0, 5, 4, 8));
static_assert(!disjointHalfOpenRanges(0, 16, 4, 4));
static_assert(!disjointHalfOpenRanges(0, 0, 4, 4));
static_assert(!disjointHalfOpenRanges(
    std::numeric_limits<std::uintptr_t>::max() - 1, 2, 0, 1));

__global__ void gicc_set_schedule_phase_kernel(gicc::DeviceCtx *ctx,
                                                std::uint32_t phase) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        ctx->schedule_phase_ = phase;
}

}  // namespace

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
    // Waits on the MMIO trigger counter, so its threshold comes from the
    // MMIO-triggered subsequence rather than the total (which also counts
    // completion-triggered ops staged by put_after).
    ++rt->mono_mmio_ops_;
    ++rt->my_n_remote_ops_;
    auto *dwq = rt->dwq_get_();
    dwq->queue_rma_write(
        rt->comm_->fabric->domain, rt->comm_->fabric->ep,
        static_cast<char *>(ob.ptr) + src_off, ob.desc_, size,
        rt->comm_->av_addrs[peer], remote_addr, ri.rma_key,
        rt->comm_->fabric->trigger_cntr,
        rt->shared_completion_cntr_,
        /*threshold=*/rt->mono_mmio_ops_);
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
    // These descriptors wait on the MMIO trigger counter, so they belong to
    // the MMIO-triggered subsequence that prepare() derives the trigger
    // delta from. Counting them only in mono_total_ops_ would make the
    // kernel's single store fall short of their thresholds and the batch
    // would never fire.
    rt->mono_mmio_ops_    += static_cast<std::uint64_t>(n_ops);
    rt->my_n_remote_ops_  += static_cast<std::uint64_t>(n_ops);
    const std::uint64_t batch_top = rt->mono_mmio_ops_;
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

void gicc_runtime_dwq_enqueue_repeated(gicc::Runtime *rt,
                                        int            n_ops,
                                        int            peer,
                                        int            dst_buf,
                                        std::size_t    dst_off,
                                        int            src_buf,
                                        std::size_t    src_off,
                                        std::size_t    size) {
    if (!rt || n_ops <= 0) return;

    // The compiler has proved every descriptor field loop invariant. Resolve
    // the invariant registration and remote address once, then preserve the
    // original loop's exact number and order of deferred network operations.
    auto &ob = rt->local_bufs_[src_buf];
    auto &ri = rt->remote_info_cache_[
        static_cast<std::size_t>(peer) *
            static_cast<std::size_t>(rt->n_bufs_) +
        static_cast<std::size_t>(dst_buf)];
    const std::uint64_t remote_addr = rt->comm_->is_virt_addr_mode()
        ? (ri.rma_addr + dst_off)
        : (ri.rma_addr - ri.base_addr) + dst_off;

    const auto count = static_cast<std::uint64_t>(n_ops);
    rt->mono_total_ops_  += count;
    rt->mono_mmio_ops_   += count;
    rt->my_n_remote_ops_ += count;
    const std::uint64_t batch_top = rt->mono_mmio_ops_;
    for (int i = 0; i < n_ops; ++i) {
        const std::uint64_t threshold =
            batch_top - static_cast<std::uint64_t>(n_ops - 1 - i);
        auto *dwq = rt->dwq_get_();
        dwq->queue_rma_write(
            rt->comm_->fabric->domain, rt->comm_->fabric->ep,
            static_cast<char *>(ob.ptr) + src_off, ob.desc_, size,
            rt->comm_->av_addrs[peer], remote_addr, ri.rma_key,
            rt->comm_->fabric->trigger_cntr,
            rt->shared_completion_cntr_, threshold);
        rt->my_pending_.push_back(dwq);
    }
}

volatile std::uint64_t *gicc_runtime_trigger_addr(gicc::Runtime *rt) {
    return rt ? rt->comm_->get_trigger_addr() : nullptr;
}

std::uint64_t gicc_runtime_trigger_val(gicc::Runtime *rt) {
    return rt ? rt->mono_total_ops_ : 0;
}

void gicc_runtime_set_schedule_phase_from_kernel_args(
        void *const *kernel_params, std::uint32_t phase,
        GpuStream_t stream) {
    if (!kernel_params || !kernel_params[0]) return;
    auto *ctx = *static_cast<gicc::DeviceCtx *const *>(kernel_params[0]);
    if (!ctx) return;
    gpuLaunchKernel(gicc_set_schedule_phase_kernel, dim3(1), dim3(1),
                    0, stream, ctx, phase);
}

int gicc_runtime_kernel_arg_matches_local_buffer(
        gicc::Runtime *rt, void *const *kernel_params,
        std::uint32_t pointer_param, std::uint32_t buffer_index_param) {
    if (!rt || !kernel_params || !kernel_params[pointer_param] ||
        !kernel_params[buffer_index_param])
        return 0;
    void *pointer =
        *static_cast<void *const *>(kernel_params[pointer_param]);
    const std::int32_t bufferIndex = *static_cast<const std::int32_t *>(
        kernel_params[buffer_index_param]);
    if (!pointer || bufferIndex < 0 ||
        static_cast<std::size_t>(bufferIndex) >= rt->local_bufs_.size())
        return 0;
    return rt->local_bufs_[static_cast<std::size_t>(bufferIndex)].ptr ==
                   pointer
        ? 1
        : 0;
}

int gicc_runtime_local_buffer_contains_interval(
        gicc::Runtime *rt, void *const *kernel_params,
        std::uint32_t buffer_index_param, std::uint64_t offset,
        std::uint64_t size) {
    if (!rt || !kernel_params || !kernel_params[buffer_index_param])
        return 0;
    const std::int32_t bufferIndex = *static_cast<const std::int32_t *>(
        kernel_params[buffer_index_param]);
    if (bufferIndex < 0 ||
        static_cast<std::size_t>(bufferIndex) >= rt->buffers_.size())
        return 0;
    const std::uint64_t registeredSize = static_cast<std::uint64_t>(
        rt->buffers_[static_cast<std::size_t>(bufferIndex)].size);
    return offset <= registeredSize && size <= registeredSize - offset
        ? 1
        : 0;
}

int gicc_runtime_local_buffer_disjoint_from_kernel_arg_allocation(
        gicc::Runtime *rt, void *const *kernel_params,
        std::uint32_t buffer_index_param,
        std::uint32_t write_pointer_param) {
    if (!rt || !kernel_params || !kernel_params[buffer_index_param] ||
        !kernel_params[write_pointer_param])
        return 0;
    const std::int32_t bufferIndex = *static_cast<const std::int32_t *>(
        kernel_params[buffer_index_param]);
    void *writePointer =
        *static_cast<void *const *>(kernel_params[write_pointer_param]);
    if (bufferIndex < 0 || !writePointer ||
        static_cast<std::size_t>(bufferIndex) >= rt->buffers_.size())
        return 0;

    const gicc::Buffer &source =
        rt->buffers_[static_cast<std::size_t>(bufferIndex)];
    if (!source.ptr || source.size == 0) return 0;

    void *writeBase = nullptr;
    std::size_t writeSize = 0;
#if defined(GICC_GPU_HIP)
    hipDeviceptr_t queriedBase = nullptr;
    if (gpuMemGetAddressRange(
            &queriedBase, &writeSize,
            reinterpret_cast<hipDeviceptr_t>(writePointer)) != GPU_SUCCESS)
        return 0;
    writeBase = reinterpret_cast<void *>(queriedBase);
#else
    // The first guarded scheduler is an OFI/HIP candidate. Other GPU APIs
    // retain the original schedule until they provide an audited allocation
    // range query with identical semantics.
    return 0;
#endif
    if (!writeBase || writeSize == 0) return 0;

    const std::uintptr_t sourceBegin =
        reinterpret_cast<std::uintptr_t>(source.ptr);
    const std::uintptr_t writeBegin =
        reinterpret_cast<std::uintptr_t>(writeBase);
    return disjointHalfOpenRanges(
        sourceBegin, source.size, writeBegin, writeSize) ? 1 : 0;
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
