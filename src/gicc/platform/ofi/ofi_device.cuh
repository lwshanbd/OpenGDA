/**
 * ofi_device.cuh - libfabric/CXI device-side API
 *
 * Mirrors src/gicc/platform/mlx5/proxy/proxy_device.cuh as closely as the
 * hardware model permits. Both backends expose the same kernel-visible
 * primitives:
 *
 *   gicc::put(ctx, target_rank,
 *             dst_buf, dst_offset,
 *             src_buf, src_offset,
 *             size, lane = 0)
 *
 *   gicc::get(ctx, source_rank,
 *             src_buf, src_offset,
 *             dst_buf, dst_offset,
 *             size, lane = 0)
 *
 *   gicc::quiet(ctx, lane = 0)
 *   gicc::flush(ctx)
 *
 * (target_rank, dst_buf, dst_offset) names a slice of the destination
 * peer's registered buffer; (src_buf, src_offset) names a slice of the
 * caller's local buffer.
 *
 * --- The `lane` parameter ---
 *
 * `lane` is a backend-agnostic logical channel. Same-lane ops are FIFO;
 * different lanes are independent and can be quieted separately. Maps to:
 *   - CPU_PROXY  : `proxy_rings_arr[lane % num_proxy_rings]` (ring fan-out)
 *   - IPC_PUSH   : ipc stream index (LTO pass routes via hint.json)
 *   - DWQ_TRIGGER: ignored in v1; reserved for per-lane completion counter
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
 * The kernel body's gicc::put / get calls are erased by
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
#include "gicc/proxy/common/proxy_ring_defs.hpp"
#include "gicc/proxy/common/transfer_cmd.hpp"
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
    // Single-ring back-compat pointer (== proxy_rings_arr[0]). Used as a
    // fallback when proxy_rings_arr has not been set up.
    void*              proxy_ring;          // gicc::proxy::ProxyRing*
    // Multi-ring fan-out: device-mapped array of N ring pointers, plus N.
    // put/get/quiet pick proxy_rings_arr[lane % num_proxy_rings] to spread
    // submissions across independent proxy threads — N separate workers,
    // each owning its own ring.
    void**             proxy_rings_arr;
    int                num_proxy_rings;
#endif
    // Locality-aware collectives: flat table of peer IPC-mapped buffer base
    // pointers, indexed [peer * ipc_n_bufs + buf_idx]. Entry is non-null only
    // when `peer` is same-node AND that buffer was IPC-mapped, so a kernel can
    // write a same-node peer's buffer directly over xGMI (no NIC). Null entry
    // => not local/mapped => fall back to put/proxy. Set by prepare().
    void**             peer_ipc_base = nullptr;
    int                ipc_n_bufs    = 0;
};

#ifdef GICC_CPU_PROXY
namespace detail {

// Resolve `lane` to a concrete ProxyRing*, with fallback to the legacy
// single-ring pointer when no fan-out array has been set up.
__device__ inline
gicc::proxy::ProxyRing* lane_to_ring(DeviceCtx* ctx, int lane) {
    void* ring_ptr = nullptr;
    if (ctx->proxy_rings_arr && ctx->num_proxy_rings > 0) {
        int n = ctx->num_proxy_rings;
        int idx = lane < 0 ? 0 : (lane % n);
        ring_ptr = ctx->proxy_rings_arr[idx];
    } else {
        ring_ptr = ctx->proxy_ring;
    }
    return reinterpret_cast<gicc::proxy::ProxyRing*>(ring_ptr);
}

}  // namespace detail
#endif  // GICC_CPU_PROXY

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
//
// Phase 2 plan: fold the MMIO trigger into put()'s DWQ lowering so each
// put auto-triggers and `flush` becomes unnecessary. Kept here in Phase 1
// to avoid breaking existing LTO + DWQ examples mid-rename.
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
// put — non-blocking RDMA write. Submission completes asynchronously;
// the caller observes completion via quiet(ctx, lane).
//
// LTO-only world (no CPU proxy): NO-OP. Real work happens in the host-side
// trace function via gicc_runtime_dwq_enqueue (DWQ peers) or hipMemcpyAsync
// (IPC peers), BEFORE the kernel launch. The device-side body has nothing
// to do, and GICCDeviceLowering erases every call site at LTO time.
//
// CPU proxy world (GICC_CPU_PROXY): pushes a WRITE TransferCmd into
// proxy_rings_arr[lane % num_proxy_rings]. The CPU worker resolves
// (rank, buf, off) → (lkey, rkey, raddr) and issues fi_writemsg.
// Caller MUST device-sync before rt.reset() to ensure all such pushes
// are published before the host drain snapshots the producer head.
//==============================================================================
__device__ inline
void put(DeviceCtx* ctx,
         int target_rank,
         int dst_buf, size_t dst_offset,
         int src_buf, size_t src_offset,
         size_t size,
         int lane = 0) {
#ifdef GICC_CPU_PROXY
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;
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
    (void)size; (void)lane;
#endif
}

//==============================================================================
// get — non-blocking RDMA read. The NIC pulls bytes from peer
// (source_rank)'s (src_buf, src_offset) into our local (dst_buf, dst_offset).
//
// Read-ordering guarantee: the proxy ack's the slot only after the CQ
// completion fires, by which point the data has landed in local memory.
// The caller MUST quiet(ctx, lane) before reading the landed data — the
// spin on device_tail_volatile() + __threadfence_system() inside quiet()
// invalidates the GPU's L2 for the destination range and pairs with the
// proxy's release-store of tail.
//==============================================================================
__device__ inline
void get(DeviceCtx* ctx,
         int source_rank,
         int src_buf, size_t src_offset,
         int dst_buf, size_t dst_offset,
         size_t size,
         int lane = 0) {
#ifdef GICC_CPU_PROXY
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;
    // Remap API direction (src=remote, dst=local) onto the cmd-struct
    // convention (src_*=local, dst_*=remote on dst_rank). This lets the
    // host-side accessors stay uniform across WRITE and READ.
    gicc::proxy::TransferCmd c;
    c.cmd_type   = gicc::proxy::CmdType::READ;
    c.dst_rank   = static_cast<uint8_t>(source_rank);
    c.src_buf    = static_cast<uint8_t>(dst_buf);     // LOCAL landing
    c.dst_buf    = static_cast<uint8_t>(src_buf);     // REMOTE source
    c.bytes      = static_cast<uint32_t>(size);
    c.src_offset = dst_offset;                        // LOCAL landing offset
    c.dst_offset = src_offset;                        // REMOTE source offset
    ring->atomic_push(c);
#else
    (void)ctx; (void)source_rank;
    (void)src_buf; (void)src_offset;
    (void)dst_buf; (void)dst_offset;
    (void)size; (void)lane;
#endif
}

//==============================================================================
// quiet — completion fence for `lane`.
//
// LTO-only world (no CPU proxy): NO-OP. Completion is owned by the host;
// rt.reset() (called after hipDeviceSynchronize) busy-polls the shared
// completion counter until every queued op finishes. The kernel doesn't
// need to wait for anything; the LTO pass erases the call entirely.
//
// CPU proxy world (GICC_CPU_PROXY): pushes a QUIET cmd into ring `lane`
// and spins until the host-published tail moves past our slot. Use after
// a sequence of put(..., lane) / get(..., lane) with the same `lane` to
// wait for completion of just that channel. __threadfence_system() after
// the spin guarantees that subsequent device reads observe the proxy's
// writes.
//==============================================================================
__device__ inline
void quiet(DeviceCtx* ctx, int lane = 0) {
#ifdef GICC_CPU_PROXY
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;
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
    (void)ctx; (void)lane;
#endif
}

//==============================================================================
// atomic_add_u32 — non-fetching FI_SUM / FI_UINT32 remote atomic add.
//
// The local 4-byte source value lives at (src_buf, src_offset); the proxy
// issues fi_atomic to add that value into peer's (dst_buf, dst_offset)
// counter slot. Use to publish a per-lane completion fence after a
// sequence of put(..., lane) calls so the receiver can wait on a single
// counter.
//==============================================================================
__device__ inline
void atomic_add_u32(DeviceCtx* ctx,
                    int target_rank,
                    int dst_buf, size_t dst_offset,
                    int src_buf, size_t src_offset,
                    int lane = 0) {
#ifdef GICC_CPU_PROXY
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;
    gicc::proxy::TransferCmd c;
    c.cmd_type   = gicc::proxy::CmdType::ATOMIC;
    c.dst_rank   = static_cast<uint8_t>(target_rank);
    c.src_buf    = static_cast<uint8_t>(src_buf);
    c.dst_buf    = static_cast<uint8_t>(dst_buf);
    c.bytes      = 4;  // implicit; proxy ignores
    c.src_offset = src_offset;
    c.dst_offset = dst_offset;
    ring->atomic_push(c);
#else
    (void)ctx; (void)target_rank;
    (void)dst_buf; (void)dst_offset; (void)src_buf; (void)src_offset;
    (void)lane;
#endif
}

} // namespace gicc
