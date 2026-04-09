/**
 * cxi_device.cuh - libfabric/CXI device-side API
 *
 * Mirrors src/gicc/platform/mlx5/mlx5_device.cuh as closely as the hardware
 * model permits. The CXI Deferred Work Queue requires that RDMA operations
 * be queued from the HOST (via fi_control(FI_QUEUE_WORK)) before the GPU
 * can trigger them. As a result:
 *
 *   - There is NO __device__ gicc::put_no_db on this backend.
 *     Use Runtime::put_no_db(...) on the host instead.
 *   - gicc::flush(ctx) writes the trigger counter MMIO doorbell, which
 *     causes the NIC to execute every queued operation in one shot.
 *     This is the moral equivalent of "ring the doorbell" on mlx5.
 *   - gicc::quiet(ctx) polls a GPU-resident atomic counter incremented by
 *     the NIC after each queued op completes (via a chained atomic signal).
 *
 * Kernel code is line-for-line identical to the mlx5 backend except that
 * the CXI kernel does not call put_no_db (it was queued from the host).
 */
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace gicc {

//==============================================================================
// DeviceCtx — GPU-accessible context for the libfabric/CXI backend
//==============================================================================
struct DeviceCtx {
    volatile uint64_t* trigger_addr_;   // MMIO trigger counter (mapped to GPU)
    volatile uint64_t* completion_;     // GPU-resident atomic_result
    uint64_t           trigger_val_;    // value to write to trigger_addr_
    uint64_t           n_ops_;          // number of completions to wait for
};

//==============================================================================
// flush — ring the (virtual) doorbell.
// On CXI this writes the trigger counter MMIO; the NIC then executes every
// operation that was previously queued by Runtime::put_no_db on the host.
// Call from exactly one thread (e.g. threadIdx.x == 0).
//==============================================================================
__device__ __forceinline__
void flush(DeviceCtx* ctx) {
    *ctx->trigger_addr_ = ctx->trigger_val_;
    __threadfence_system();
}

//==============================================================================
// quiet — wait for all outstanding RDMA operations to complete.
// May be called from any/all threads; they all poll the same location.
//==============================================================================
__device__ __forceinline__
void quiet(DeviceCtx* ctx) {
    while (*ctx->completion_ < ctx->n_ops_) {}
}

} // namespace gicc
