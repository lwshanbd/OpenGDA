/**
 * launch.hpp - gicc::launch wrapper for MLX5 backend.
 *
 * Owns the build_context() + kernel-launch sequence. The user calls
 *   gicc::launch<kernel>(rt, grid, block, args...)
 * instead of writing rt.build_context() + kernel<<<>>>(gctx, args...) by
 * hand. The kernel receives a GiccContext* and uses the common-form
 * device API (gicc::put_no_db(ctx, rank, dst_buf, dst_off, ...)),
 * matching the cross-backend semantics documented in
 * src/gicc/platform/ofi/ofi_device.cuh.
 *
 * The kernel is passed as a NON-TYPE TEMPLATE PARAMETER (constant
 * expression) so the OFI backend can key its kernel_trace<>
 * specialization on it. We mirror that signature for source-level
 * portability across both backends.
 *
 * NOTE: the underlying GiccContext is heap-allocated by build_context()
 * and held by the launch site. Repeatedly calling launch() will leak
 * GiccContexts; the production refresh of this wrapper should cache
 * the per-Runtime GiccContext (one allocation per Runtime, reused on
 * every launch). That refactor is intentionally not part of this
 * cross-backend parity patch.
 */
#pragma once

#include <cuda_runtime.h>

#include "gicc/launch.hpp"
#include "gicc/platform/mlx5/mlx5_runtime.hpp"

namespace gicc {

template <auto Kernel, typename... Args>
GICC_LAUNCH_SITE
inline void launch(Runtime& rt,
                   dim3 grid, dim3 block,
                   Args... args)
{
    GiccContext* gctx = rt.build_context();
    Kernel<<<grid, block>>>(gctx, args...);
}

template <auto Kernel, typename... Args>
GICC_LAUNCH_SITE
inline void launch(Runtime& rt,
                   dim3 grid, dim3 block,
                   size_t shmem_bytes, cudaStream_t stream,
                   Args... args)
{
    GiccContext* gctx = rt.build_context();
    Kernel<<<grid, block, shmem_bytes, stream>>>(gctx, args...);
}

} // namespace gicc
