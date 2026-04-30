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

#include <hip/hip_runtime.h>
#include <cstdint>

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
        *ctx->trigger_addr_ = ctx->trigger_val_;
        __threadfence_system();
    }
}

//==============================================================================
// quiet — NO-OP in the LTO-only world.
//
// Completion is owned by the host: rt.reset() (called immediately after
// hipDeviceSynchronize on the host) busy-polls the shared completion
// counter until every queued op finishes. The kernel doesn't need to
// wait for anything. Empty so non-LTO builds compile; the LTO pass
// erases the call entirely.
//==============================================================================
__device__ inline
void quiet(DeviceCtx* /*ctx*/) {}

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
//==============================================================================
__device__ inline
void put_no_db(DeviceCtx* /*ctx*/,
               int /*target_rank*/,
               int /*dst_buf*/, size_t /*dst_offset*/,
               int /*src_buf*/, size_t /*src_offset*/,
               size_t /*size*/, bool /*signaled*/ = false) {}

__device__ inline
void get_no_db(DeviceCtx* /*ctx*/,
               int /*source_rank*/,
               int /*src_buf*/, size_t /*src_offset*/,
               int /*dst_buf*/, size_t /*dst_offset*/,
               size_t /*size*/, bool /*signaled*/ = false) {}

} // namespace gicc
