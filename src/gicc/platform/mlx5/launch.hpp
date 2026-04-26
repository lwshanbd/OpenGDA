/**
 * launch.hpp - gicc::launch wrapper for MLX5 backend.
 *
 * Owns the prepare() + kernel-launch sequence. The user calls
 *   gicc::launch<kernel>(rt, grid, block, peer, dst_buf_idx, args...)
 * instead of writing rt.prepare() + kernel<<<>>>(ctx, args...) by hand.
 *
 * The kernel is passed as a NON-TYPE TEMPLATE PARAMETER (constant
 * expression) so the OFI backend can key its kernel_trace<> specialization
 * on it. This file mirrors that signature for source-level portability
 * across both backends.
 */
#pragma once

#include <cuda_runtime.h>

#include "gicc/platform/mlx5/mlx5_runtime.hpp"

namespace gicc {

template <auto Kernel, typename... Args>
inline void launch(Runtime& rt,
                   dim3 grid, dim3 block,
                   int peer, int dst_buf_idx,
                   Args... args)
{
    DeviceCtx* ctx = rt.prepare(peer, dst_buf_idx);
    Kernel<<<grid, block>>>(ctx, args...);
}

template <auto Kernel, typename... Args>
inline void launch(Runtime& rt,
                   dim3 grid, dim3 block,
                   int peer, int dst_buf_idx,
                   size_t shmem_bytes, cudaStream_t stream,
                   Args... args)
{
    DeviceCtx* ctx = rt.prepare(peer, dst_buf_idx);
    Kernel<<<grid, block, shmem_bytes, stream>>>(ctx, args...);
}

} // namespace gicc
