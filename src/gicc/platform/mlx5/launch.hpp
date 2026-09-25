/**
 * launch.hpp - gicc::launch wrapper for the InfiniBand backend.
 *
 *   gicc::launch<kernel>(rt, grid, block, args...)
 *
 * launches kernel<<<grid, block>>>(rt.prepare(), args...). The kernel takes
 * the context (gicc::mlx5::GdaCtx*) first and communicates with the device API of
 * gicc/gicc_device.cuh (gicc::put / get / put_signal / quiet), whose calls
 * post work requests directly from GPU threads.
 *
 * The kernel is a NON-TYPE TEMPLATE PARAMETER, as on the libfabric backend,
 * where the LTO pass keys its kernel trace on it; the signature is shared so
 * the same source builds on both.
 */
#pragma once

#include <cuda_runtime.h>

#include "gicc/launch.hpp"
#include "gicc/platform/mlx5/mlx5_runtime.hpp"

namespace gicc {

template <auto Kernel, typename... Args>
GICC_LAUNCH_SITE
inline void launch(Runtime& rt, dim3 grid, dim3 block, Args... args)
{
    Kernel<<<grid, block>>>(rt.prepare(), args...);
}

template <auto Kernel, typename... Args>
GICC_LAUNCH_SITE
inline void launch(Runtime& rt, dim3 grid, dim3 block,
                   size_t shmem_bytes, cudaStream_t stream, Args... args)
{
    Kernel<<<grid, block, shmem_bytes, stream>>>(rt.prepare(), args...);
}

} // namespace gicc
