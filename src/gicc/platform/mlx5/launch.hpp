/**
 * launch.hpp - gicc::launch wrapper for MLX5 backend.
 *
 * Owns the prepare() + kernel-launch sequence. The user calls
 *   gicc::launch<kernel>(rt, grid, block, args...)
 * instead of writing rt.prepare() + kernel<<<>>>(ctx, args...) by hand.
 *
 * The kernel is passed as a NON-TYPE TEMPLATE PARAMETER (constant
 * expression) so the OFI backend can key its kernel_trace<> specialization
 * on it. This file mirrors that signature for source-level portability
 * across both backends.
 *
 * v1.5: peer + dst_buf are per-CALL on gicc::put_no_db inside the kernel
 * (the MLX5 device-side put_no_db consumes them to build the WQE for the
 * right QP). launch itself no longer carries them.
 */
#pragma once

#include <cuda_runtime.h>

#include "gicc/platform/mlx5/mlx5_runtime.hpp"

namespace gicc {

template <auto Kernel, typename... Args>
inline void launch(Runtime& rt,
                   dim3 grid, dim3 block,
                   Args... args)
{
    DeviceCtx* ctx = rt.prepare();
    Kernel<<<grid, block>>>(ctx, args...);
}

template <auto Kernel, typename... Args>
inline void launch(Runtime& rt,
                   dim3 grid, dim3 block,
                   size_t shmem_bytes, cudaStream_t stream,
                   Args... args)
{
    DeviceCtx* ctx = rt.prepare();
    Kernel<<<grid, block, shmem_bytes, stream>>>(ctx, args...);
}

} // namespace gicc
