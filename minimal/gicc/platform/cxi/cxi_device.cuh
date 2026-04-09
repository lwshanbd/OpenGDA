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
 *     causes the NIC to execute every operation that was queued by
 *     Runtime::put_no_db().
 *   - gicc::quiet(ctx) polls per-op atomic_result slots. Each queued put has
 *     its own slot incremented by a chained atomic_signal queued by
 *     Runtime::prepare(). The kernel can be launched with any thread count;
 *     blockDim.x threads cooperatively poll the n_ops_ slots, striding by
 *     blockDim.x. Single-thread launches just iterate sequentially.
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
//
// completion_ points to the BASE of an n_ops_-element array of POINTERS,
// one per queued op. Each pointer references its op's per-stream
// atomic_result GPU memory location (which lives in its own dedicated
// libfabric MR, exactly mirroring benchmark_runner.hpp's per-stream
// layout). quiet() returns once every pointed-to value has reached 1.
//==============================================================================
struct DeviceCtx {
    volatile uint64_t* trigger_addr_;   // MMIO trigger counter (mapped to GPU)
    volatile uint64_t* completion_;     // bit-cast of (uint64_t* const*) — see quiet()
    uint64_t           trigger_val_;    // value to write to trigger_addr_
    uint64_t           n_ops_;          // number of slot pointers to wait on
};

//==============================================================================
// flush — ring the (virtual) doorbell.
// Writes the CXI trigger counter MMIO; the NIC then executes every RMA that
// was queued by Runtime::put_no_db() since the last reset(). Call from
// exactly one thread (e.g. threadIdx.x == 0 with a single-block launch).
//==============================================================================
__device__ __forceinline__
void flush(DeviceCtx* ctx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *ctx->trigger_addr_ = ctx->trigger_val_;
        __threadfence_system();
    }
}

//==============================================================================
// quiet — wait for every queued RMA's chained atomic to fire.
// Each thread polls a stride of slots starting at threadIdx.x. With 1 thread
// the polling is sequential. With n_ops_ threads each thread polls one slot.
//==============================================================================
__device__ __forceinline__
void quiet(DeviceCtx* ctx) {
    if (ctx->completion_ == nullptr) return;   // overlap pattern, no quiet
    // ctx->completion_ is actually a (volatile uint64_t**) — the host laid
    // out an array of per-stream atomic_result pointers. Each thread polls
    // a stride of slots and waits for its slot's atomic_result to reach 1.
    volatile uint64_t* const* slots = (volatile uint64_t* const*)ctx->completion_;
    for (uint64_t i = threadIdx.x; i < ctx->n_ops_; i += blockDim.x) {
        while (*slots[i] < 1) {}
    }
}

} // namespace gicc
