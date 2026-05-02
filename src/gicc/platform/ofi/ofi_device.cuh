/**
 * ofi_device.cuh - libfabric/CXI device-side API
 *
 * Mirrors src/gicc/platform/mlx5/mlx5_device.cuh as closely as the hardware
 * model permits. Both backends expose the same kernel-visible primitives:
 *
 *   gicc::put_no_db(ctx, target_rank,
 *                   dst_buf, dst_offset,
 *                   src_buf, src_offset,
 *                   size, signaled=false)
 *
 *   gicc::get_no_db(ctx, source_rank,
 *                   src_buf, src_offset,
 *                   dst_buf, dst_offset,
 *                   size, signaled=false)
 *
 *   gicc::flush(ctx)
 *   gicc::quiet(ctx)
 *
 * (target_rank, dst_buf, dst_offset) names a slice of the destination
 * peer's registered buffer; (src_buf, src_offset) names a slice of the
 * caller's local buffer.
 *
 * --- LTO-pass-driven dispatch ---
 *
 * The Y' / host-driven model: the LTO host pass synthesizes a trace
 * function per kernel that runs BEFORE the kernel launch. The trace
 * decides per call site:
 *   - Same-node peer with IPC mapping → hipMemcpyAsync on the IPC
 *     stream BEFORE the kernel runs.
 *   - Off-node peer → gicc_runtime_dwq_enqueue queues a deferred RMA
 *     write descriptor with a trigger threshold = mono_total_ops_.
 *
 * The kernel body's gicc::put_no_db / get_no_db calls are erased by
 * GICCDeviceLowering — at LTO time they have nothing to do. The
 * inline bodies below are only for non-LTO builds and are intentionally
 * empty so they can never disagree with the host trace.
 *
 * Only flush() retains a real body: the lead thread writes
 * trigger_val_ → trigger_addr_ at end of kernel. The NIC sees the
 * trigger counter cross the queued descriptors' threshold and
 * processes them all.
 */
#pragma once

#include "internal/gpu_device_context.hpp"
#include <cstdint>

#ifdef GICC_CPU_PROXY
#include "proxy/proxy_ring_defs.hpp"
#include "proxy/transfer_cmd.hpp"
#endif

namespace gicc {

//==============================================================================
// DeviceCtx — GPU-accessible context for the libfabric/CXI backend.
// Shrunk per spec §7.3: only the trigger MMIO addr + threshold value
// survive into the LTO-only world. Everything else (IPC table, local
// buf table, command ring) was Pattern C state, deleted in Phase 6.
//==============================================================================
struct DeviceCtx {
    volatile uint64_t* trigger_addr_;       // MMIO trigger counter
    uint64_t           trigger_val_;        // value to write to trigger_addr_
#ifdef GICC_CPU_PROXY
    // Single-ring back-compat pointer (== proxy_rings_arr[0]). Used by the
    // single-ring put_no_db / quiet device functions below.
    void*              proxy_ring;          // gicc::proxy::ProxyRing*
    // Multi-ring fan-out: device-mapped array of N ring pointers, plus N.
    // Kernels can pick proxy_rings_arr[idx % num_proxy_rings] to dispatch
    // independent submissions through different proxy threads (UCCL-EP
    // model: kNumProxyThs separate workers, each owning its own ring,
    // with the GPU side hashing on warp/expert id).
    void**             proxy_rings_arr;
    int                num_proxy_rings;
#endif
};

//==============================================================================
// flush — lead-thread MMIO write that fires every queued DWQ descriptor.
//
// The host trace function pre-stages all RMA writes against
// shared_completion_cntr_ at threshold = mono_total_ops_. flush()'s
// single store crosses every threshold at once, so the NIC processes
// the entire kernel's worth of DWQ ops with one trigger fire.
//
// IPC-routed ops are NOT visible here — they ran on the host stream
// before the kernel was launched, so by the time we get here they're
// already in flight (and rt.reset() will hipStreamSynchronize them).
//==============================================================================
__device__ inline
void flush(DeviceCtx* ctx) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
        && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // trigger_addr_ is null when the Runtime was constructed with
        // GICC_SKIP_DWQ_INIT (proxy-only mode on platforms whose CUDA
        // runtime cannot map the CXI MMIO BAR, e.g. GH200). Skip the
        // store rather than dereference null and crash; the proxy path
        // does not need a trigger.
        if (ctx->trigger_addr_ != nullptr) {
            *ctx->trigger_addr_ = ctx->trigger_val_;
            __threadfence_system();
        }
    }
}

//==============================================================================
// quiet — completion fence.
//
// LTO-only world (no CPU proxy): NO-OP. Completion is owned by the host;
// rt.reset() (called after hipDeviceSynchronize) busy-polls the shared
// completion counter until every queued op finishes. The kernel doesn't
// need to wait for anything; the LTO pass erases the call entirely.
//
// CPU proxy world (GICC_CPU_PROXY): pushes a QUIET cmd into the proxy
// ring and spins until the host-published tail moves past our slot. This
// is the in-kernel completion fence required by proxy-aware kernels that
// want to read peer-written data without returning to the host first.
// __threadfence_system() after the spin guarantees that subsequent
// device reads observe the proxy's writes.
//==============================================================================
__device__ inline
void quiet(DeviceCtx* ctx) {
#ifdef GICC_CPU_PROXY
    if (!ctx || !ctx->proxy_ring) return;
    auto* ring = reinterpret_cast<gicc::proxy::ProxyRing*>(ctx->proxy_ring);
    gicc::proxy::TransferCmd c{};
    c.cmd_type = gicc::proxy::CmdType::QUIET;
    uint64_t my_slot = ring->atomic_push(c);
    while (ring->device_tail_volatile() <= my_slot) {
#if defined(__CUDA_ARCH__)
        __nanosleep(64);
#elif defined(__HIP_DEVICE_COMPILE__)
        // HIP / AMDGCN: s_sleep takes "cycles / 64" — 1 ~= 64 cycles.
        __builtin_amdgcn_s_sleep(1);
#endif
    }
    __threadfence_system();
#else
    (void)ctx;
#endif
}

//==============================================================================
// put_no_db / get_no_db — NO-OPs in the LTO-only world.
//
// Real work happens in the host-side trace function: the LTO host pass
// emits gicc_runtime_dwq_enqueue (DWQ peers) or hipMemcpyAsync (IPC
// peers) for each call site, BEFORE the kernel launch. The device-side
// body has nothing to do.
//
// Empty bodies so non-LTO builds still link; the GICCDeviceLowering
// pass erases every call site at LTO time.
//
// Under GICC_CPU_PROXY the device-side body publishes a TransferCmd
// into the proxy ring instead. Caller MUST device-sync before
// rt.reset() to ensure all such pushes are published before the host
// drain snapshots the producer head.
//==============================================================================
// Multi-ring variant: pushes the WRITE cmd into proxy_rings_arr[ring_idx %
// num_proxy_rings]. Use this when sharding parallel kernel work across
// multiple proxy threads (UCCL-EP pattern). Falls back to ring 0 if no
// fan-out array is set up. Defined ABOVE single-ring put_no_db so the
// latter can delegate.
__device__ inline
void put_no_db_idx(DeviceCtx* ctx, int ring_idx,
                   int target_rank,
                   int dst_buf, size_t dst_offset,
                   int src_buf, size_t src_offset,
                   size_t size) {
#ifdef GICC_CPU_PROXY
    if (!ctx) return;
    void* ring_ptr = nullptr;
    if (ctx->proxy_rings_arr && ctx->num_proxy_rings > 0) {
        int n = ctx->num_proxy_rings;
        int idx = ring_idx;
        if (idx < 0) idx = 0;
        idx = idx % n;
        ring_ptr = ctx->proxy_rings_arr[idx];
    } else {
        ring_ptr = ctx->proxy_ring;
    }
    if (!ring_ptr) return;
    auto* ring = reinterpret_cast<gicc::proxy::ProxyRing*>(ring_ptr);
    gicc::proxy::TransferCmd c;
    c.cmd_type   = gicc::proxy::CmdType::WRITE;
    c.dst_rank   = static_cast<uint8_t>(target_rank);
    c.src_buf    = static_cast<uint8_t>(src_buf);
    c.dst_buf    = static_cast<uint8_t>(dst_buf);
    c.bytes      = static_cast<uint32_t>(size);
    c.src_offset = src_offset;
    c.dst_offset = dst_offset;
    ring->atomic_push(c);
#else
    (void)ctx; (void)ring_idx; (void)target_rank;
    (void)dst_buf; (void)dst_offset; (void)src_buf; (void)src_offset; (void)size;
#endif
}

// quiet variant for ring `ring_idx`. Pushes QUIET into that specific ring
// and spins on its tail. Use after a series of put_no_db_idx() with the
// same ring_idx to wait for completion of just that lane.
__device__ inline
void quiet_idx(DeviceCtx* ctx, int ring_idx) {
#ifdef GICC_CPU_PROXY
    if (!ctx) return;
    void* ring_ptr = nullptr;
    if (ctx->proxy_rings_arr && ctx->num_proxy_rings > 0) {
        int n = ctx->num_proxy_rings;
        int idx = ring_idx;
        if (idx < 0) idx = 0;
        idx = idx % n;
        ring_ptr = ctx->proxy_rings_arr[idx];
    } else {
        ring_ptr = ctx->proxy_ring;
    }
    if (!ring_ptr) return;
    auto* ring = reinterpret_cast<gicc::proxy::ProxyRing*>(ring_ptr);
    gicc::proxy::TransferCmd c{};
    c.cmd_type = gicc::proxy::CmdType::QUIET;
    uint64_t my_slot = ring->atomic_push(c);
    while (ring->device_tail_volatile() <= my_slot) {
#if defined(__CUDA_ARCH__)
        __nanosleep(64);
#elif defined(__HIP_DEVICE_COMPILE__)
        __builtin_amdgcn_s_sleep(1);
#endif
    }
    __threadfence_system();
#else
    (void)ctx; (void)ring_idx;
#endif
}

__device__ inline
void put_no_db(DeviceCtx* ctx,
               int target_rank,
               int dst_buf, size_t dst_offset,
               int src_buf, size_t src_offset,
               size_t size, bool /*signaled*/ = false) {
#ifdef GICC_CPU_PROXY
    if (!ctx || !ctx->proxy_ring) return;
    auto* ring = reinterpret_cast<gicc::proxy::ProxyRing*>(ctx->proxy_ring);
    gicc::proxy::TransferCmd c;
    c.cmd_type   = gicc::proxy::CmdType::WRITE;
    c.dst_rank   = static_cast<uint8_t>(target_rank);
    c.src_buf    = static_cast<uint8_t>(src_buf);
    c.dst_buf    = static_cast<uint8_t>(dst_buf);
    c.bytes      = static_cast<uint32_t>(size);
    c.src_offset = src_offset;
    c.dst_offset = dst_offset;
    ring->atomic_push(c);
#else
    (void)ctx; (void)target_rank;
    (void)dst_buf; (void)dst_offset;
    (void)src_buf; (void)src_offset;
    (void)size;
#endif
}

__device__ inline
void get_no_db(DeviceCtx* /*ctx*/,
               int /*source_rank*/,
               int /*src_buf*/, size_t /*src_offset*/,
               int /*dst_buf*/, size_t /*dst_offset*/,
               size_t /*size*/, bool /*signaled*/ = false) {}

} // namespace gicc
