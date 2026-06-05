/*
 * gicc_omp_dwq.hpp - Phase-2 DWQ markers for use inside omp target regions.
 * SEPARATE from gicc::omp::put (Phase-1 proxy) so that path is untouched.
 * `put` is a noinline,used opaque marker that survives the inliner until
 * GICCDeviceLowering erases it; the host trace (synthesized from the
 * KernelTemplate) does the real DWQ enqueue. `flush` lowers to the lead-thread
 * MMIO trigger. The pass keys on these by mangled name.
 */
#pragma once
#include "gicc/platform/ofi/device_ctx.hpp"   // gicc::DeviceCtx (HIP-free)
#include <cstddef>

#pragma omp declare target
namespace gicc { namespace omp_dwq {

__attribute__((noinline, used))
inline void put(gicc::DeviceCtx* ctx, int target_rank,
                int dst_buf, size_t dst_offset,
                int src_buf, size_t src_offset, size_t size) {
    // The asm MUST reference EVERY arg or the optimizer drops the unreferenced
    // ones from the internalized marker signature before discovery reads them.
    asm volatile("" : : "r"((long)ctx), "r"(target_rank),
                        "r"(dst_buf), "r"((long)dst_offset),
                        "r"(src_buf), "r"((long)src_offset),
                        "r"((long)size) : "memory");
}

__attribute__((noinline, used))
inline void flush(gicc::DeviceCtx* ctx) {
    asm volatile("" : : "r"((long)ctx) : "memory");
}

} }  // namespace gicc::omp_dwq
#pragma omp end declare target
