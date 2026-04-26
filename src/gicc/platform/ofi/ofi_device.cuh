/**
 * cxi_device.cuh - libfabric/CXI device-side API
 *
 * Mirrors src/gicc/platform/mlx5/mlx5_device.cuh as closely as the hardware
 * model permits. The CXI Deferred Work Queue requires that RDMA operations
 * be queued from the HOST (via fi_control(FI_QUEUE_WORK)) before the GPU
 * can trigger them. To keep kernel source portable across backends:
 *
 *   - gicc::put_no_db / gicc::get_no_db are provided here as __device__
 *     no-op compile stubs. They exist solely so kernels written in the
 *     MLX5 canonical form (which calls put_no_db inline from the GPU)
 *     compile unchanged on OFI. The stubs emit no instructions and the
 *     compiler folds them away.
 *   - The actual host-side queueing of those put/get operations is
 *     produced by the gicc-clang-plugin's kernel_trace<&K> specialization
 *     (forthcoming work, not yet present in this tree). The plugin walks
 *     the kernel body, lifts each put_no_db call to a host-side
 *     fi_control(FI_QUEUE_WORK) enqueue, and the kernel itself is left
 *     to call only flush() and quiet().
 *   - gicc::flush(ctx) writes the trigger counter MMIO doorbell, which
 *     causes the NIC to execute every operation that the plugin-generated
 *     trace queued on the host.
 *   - gicc::quiet(ctx) polls per-op atomic_result slots. Each queued put has
 *     its own slot incremented by a chained atomic_signal queued by
 *     Runtime::prepare(). The kernel can be launched with any thread count;
 *     blockDim.x threads cooperatively poll the n_ops_ slots, striding by
 *     blockDim.x. Single-thread launches just iterate sequentially.
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
/// Staged local IPC copy. put_no_db() fills one of these per same-node peer
/// during host-side enqueue; put_local() in the kernel executes the copy via
/// direct GPU stores to peer_mapped memory.
struct LocalOp {
    void*    dst;
    void*    src;
    uint64_t size;
};

struct DeviceCtx {
    volatile uint64_t* trigger_addr_;   // MMIO trigger counter (mapped to GPU)
    volatile uint64_t* completion_;     // bit-cast of (uint64_t* const*) — see quiet()
    uint64_t           trigger_val_;    // value to write to trigger_addr_
    uint64_t           n_ops_;          // number of slot pointers to wait on
    LocalOp*           local_ops_;      // device-visible array of local IPC copies
    uint64_t           n_local_ops_;    // length of local_ops_ this batch
};

//==============================================================================
// Block-cooperative device-to-device memcpy for IPC fast path.
// Inspired by nvshmemi_memcpy_threadgroup: progressively tries 16B, 8B, 4B,
// 2B, 1B chunks based on src/dst alignment. Each thread in the block takes
// a stride of chunks. Callers MUST invoke from a single block; typical use
// is the same thread block that triggered the kernel's compute+flush.
//==============================================================================
namespace detail {
__device__ __forceinline__
int tid_in_block() {
    return threadIdx.x
         + threadIdx.y * blockDim.x
         + threadIdx.z * blockDim.x * blockDim.y;
}
__device__ __forceinline__
int block_size() {
    return blockDim.x * blockDim.y * blockDim.z;
}
} // namespace detail

__device__ __forceinline__
void memcpy_block(void* __restrict__ dst_v, const void* __restrict__ src_v,
                  uint64_t len)
{
    int tid   = detail::tid_in_block();
    int nth   = detail::block_size();
    char* dst = (char*)dst_v;
    const char* src = (const char*)src_v;

    if (((uintptr_t)dst % 16 == 0) && ((uintptr_t)src % 16 == 0) && len >= 16) {
        uint64_t n = len / 16;
        int4* d = (int4*)dst; const int4* s = (const int4*)src;
        for (uint64_t i = tid; i < n; i += nth) d[i] = s[i];
        len -= n * 16; dst += n * 16; src += n * 16;
        if (len == 0) return;
    }
    if (((uintptr_t)dst % 8 == 0) && ((uintptr_t)src % 8 == 0) && len >= 8) {
        uint64_t n = len / 8;
        uint64_t* d = (uint64_t*)dst; const uint64_t* s = (const uint64_t*)src;
        for (uint64_t i = tid; i < n; i += nth) d[i] = s[i];
        len -= n * 8; dst += n * 8; src += n * 8;
        if (len == 0) return;
    }
    if (((uintptr_t)dst % 4 == 0) && ((uintptr_t)src % 4 == 0) && len >= 4) {
        uint64_t n = len / 4;
        uint32_t* d = (uint32_t*)dst; const uint32_t* s = (const uint32_t*)src;
        for (uint64_t i = tid; i < n; i += nth) d[i] = s[i];
        len -= n * 4; dst += n * 4; src += n * 4;
        if (len == 0) return;
    }
    if (((uintptr_t)dst % 2 == 0) && ((uintptr_t)src % 2 == 0) && len >= 2) {
        uint64_t n = len / 2;
        uint16_t* d = (uint16_t*)dst; const uint16_t* s = (const uint16_t*)src;
        for (uint64_t i = tid; i < n; i += nth) d[i] = s[i];
        len -= n * 2; dst += n * 2; src += n * 2;
        if (len == 0) return;
    }
    for (uint64_t i = tid; i < len; i += nth) dst[i] = src[i];
}

//==============================================================================
// put_local — execute all IPC copies queued by Runtime::put_no_db().
// Call from exactly one block (all threads cooperate). Does nothing if no
// local ops were queued. Issues a __threadfence_system() at the end so peers
// observe the writes before flush() rings the remote DWQ doorbell.
//==============================================================================
__device__ __forceinline__
void put_local(DeviceCtx* ctx) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) return;
    for (uint64_t i = 0; i < ctx->n_local_ops_; i++) {
        memcpy_block(ctx->local_ops_[i].dst,
                     ctx->local_ops_[i].src,
                     ctx->local_ops_[i].size);
    }
    __syncthreads();
    if (detail::tid_in_block() == 0) __threadfence_system();
}

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
    // ctx->completion_ is the base of a contiguous pool of n_ops_ × uint64_t
    // atomic_result slots. Each thread polls a stride of slots until each
    // reaches 1. With 1 thread, polling is sequential; with N threads each
    // thread polls one slot in its own cacheline.
    for (uint64_t i = threadIdx.x; i < ctx->n_ops_; i += blockDim.x) {
        while (ctx->completion_[i] < 1) {}
    }
}

// ============================================================================
// Device-side stubs — actual queueing happens host-side via the
// gicc-clang-plugin's generated kernel_trace<&K> specialization. These are
// no-ops in the GPU code; the kernel body compiles unchanged from the MLX5
// canonical form.
// ============================================================================
__device__ __forceinline__
void put_no_db(DeviceCtx*, uint64_t /*local_addr*/, uint32_t /*local_lkey*/,
               uint64_t /*remote_addr*/, uint32_t /*remote_rkey*/,
               uint32_t /*size*/, bool /*signaled*/ = false) {}

__device__ __forceinline__
void get_no_db(DeviceCtx*, uint64_t /*local_addr*/, uint32_t /*local_lkey*/,
               uint64_t /*remote_addr*/, uint32_t /*remote_rkey*/,
               uint32_t /*size*/, bool /*signaled*/ = false) {}

} // namespace gicc
