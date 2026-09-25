/**
 * gicc_device.cuh - GICC Device-Side API
 *
 * The kernel receives the context from gicc::launch (or Runtime::prepare())
 * and names data by (rank, buffer index, offset):
 *
 *   gicc::put(ctx, rank, dst_buf, dst_off, src_buf, src_off, bytes, lane = 0);
 *   gicc::get(ctx, rank, src_buf, src_off, dst_buf, dst_off, bytes, lane = 0);
 *   gicc::quiet(ctx, lane = 0);
 *   gicc::flush(ctx);
 *
 * InfiniBand adds put_signal and signal_wait (mlx5_device.hpp).
 */
#pragma once

#if defined(GICC_PLATFORM_MLX5)
#include "platform/mlx5/mlx5_device.hpp"
#elif defined(GICC_PLATFORM_OFI)
#include "platform/ofi/ofi_device.cuh"
#else
#error "No GICC platform defined. Define GICC_PLATFORM_MLX5 or GICC_PLATFORM_OFI."
#endif
