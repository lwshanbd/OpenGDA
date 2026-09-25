/*
 * gicc_omp_dwq.hpp - the unified device-side communication markers behind
 * ompx_dwq_put_dev / ompx_dwq_flush_dev (see src/gicc/omp.h, GIOMP_ENABLE_DWQ).
 *
 * ONE user-visible API, two lowerings:
 *   - Without the GICC pass, these are ordinary device functions: `put`
 *     executes the Proxy-path enqueue (gicc::omp::put) and `flush` is a
 *     no-op, so a single source runs on any OpenMP toolchain.
 *   - Compiled with the LTO pass in omp-dwq mode, the pass keys on these
 *     mangled names (gicc::omp_dwq::put / ::flush): qualifying `put` call
 *     sites are erased and pre-staged by the compiler-synthesized host
 *     trace, and `flush` lowers to the lead-thread MMIO trigger. The Proxy
 *     bodies below then never execute in a lowered kernel.
 *
 * `noinline,used` keeps the call sites visible to the pass until
 * GICCDeviceLowering rewrites them. The trailing `lane` argument serves the
 * Proxy body only; the trace-template builder reads the first seven
 * operands and ignores it.
 */
#pragma once
#include <cstddef>
#if defined(GICC_PLATFORM_MLX5)
// InfiniBand: the GPU posts the put itself, so `put` is the transfer and
// `flush` has nothing to release. No pass is involved.
#include "gicc/platform/mlx5/gicc_omp_device.hpp"  // gicc::omp::put + GdaCtx
namespace gicc { namespace omp_dwq { using Ctx = gicc::mlx5::GdaCtx; } }
#else
#include "gicc/platform/ofi/gicc_omp_device.hpp"   // gicc::omp::put + DeviceCtx
namespace gicc { namespace omp_dwq { using Ctx = gicc::DeviceCtx; } }
#endif

#pragma omp declare target
namespace gicc { namespace omp_dwq {

__attribute__((noinline, used))
inline void put(Ctx* ctx, int target_rank,
                int dst_buf, size_t dst_offset,
                int src_buf, size_t src_offset, size_t size,
                int lane = 0) {
    gicc::omp::put(ctx, target_rank, dst_buf, dst_offset,
                   src_buf, src_offset, size, lane);
}

__attribute__((noinline, used))
inline void flush(Ctx* ctx) {
    // Proxy path: each put publishes its descriptor individually, so the
    // release point has nothing to do. The asm keeps the marker call alive
    // for the pass build, where it is replaced by the trigger sequence.
    asm volatile("" : : "r"((long)ctx) : "memory");
}

} }  // namespace gicc::omp_dwq
#pragma omp end declare target
