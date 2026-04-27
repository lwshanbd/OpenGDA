/**
 * launch.hpp - gicc::launch wrapper for OFI backend.
 *
 * Owns the trace + prepare() + kernel-launch sequence. The trace step
 * runs first so the host-side put_no_db queue is filled BEFORE prepare()
 * snapshots n_ops_/trigger_val_ into the DeviceCtx.
 *
 * v1.5: peer + dst_buf are now PER-CALL on gicc::put_no_db (each put
 * inside the kernel names its own destination), so launch itself no
 * longer carries them. This naturally supports kernels that put to
 * multiple peers (e.g. jacobi halo exchange to top + bottom).
 *
 * detail::kernel_trace<Kernel> primary template is a no-op. The
 * gicc-clang-plugin generates a specialization per kernel that contains
 * liftable RDMA ops. Kernels with no liftable ops fall through to the
 * no-op specialization, which is correct because those kernels do all
 * the work device-side (or there is no work to do).
 *
 * IMPORTANT: the kernel is passed as a NON-TYPE TEMPLATE PARAMETER on
 * launch, e.g. `gicc::launch<k>(rt, grid, block, args...)`. It cannot be
 * a runtime function argument because `kernel_trace<Kernel>` needs
 * `Kernel` to be a constant expression for the per-kernel specialization
 * lookup the plugin emits.
 */
#pragma once

#include <hip/hip_runtime.h>

#include "gicc/platform/ofi/ofi_runtime.hpp"

namespace gicc {
namespace detail {

template <auto Kernel>
struct kernel_trace {
    template <typename... Args>
    static void run(Runtime& /*rt*/,
                    dim3 /*grid*/, dim3 /*block*/,
                    Args... /*args*/) {}
};

} // namespace detail

template <auto Kernel, typename... Args>
inline void launch(Runtime& rt,
                   dim3 grid, dim3 block,
                   Args... args)
{
    detail::kernel_trace<Kernel>::run(rt, grid, block, args...);
    DeviceCtx* ctx = rt.prepare();
    hipLaunchKernelGGL(Kernel, grid, block, 0, 0, ctx, args...);
}

template <auto Kernel, typename... Args>
inline void launch(Runtime& rt,
                   dim3 grid, dim3 block,
                   size_t shmem_bytes, hipStream_t stream,
                   Args... args)
{
    detail::kernel_trace<Kernel>::run(rt, grid, block, args...);
    DeviceCtx* ctx = rt.prepare();
    hipLaunchKernelGGL(Kernel, grid, block, shmem_bytes, stream, ctx, args...);
}

} // namespace gicc
