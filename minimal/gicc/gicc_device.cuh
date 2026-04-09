/**
 * gicc_device.cuh - GICC device-side API dispatch
 *
 * Pulls in the platform-specific gicc::DeviceCtx and the device functions
 * gicc::flush(ctx) / gicc::quiet(ctx).  On mlx5 it also brings in
 * gicc::put_no_db(ctx, dst, src, size, peer); on cxi the equivalent lives
 * on the host as Runtime::put_no_db.
 */
#pragma once

#if defined(GICC_PLATFORM_MLX5)
#include "platform/mlx5/mlx5_device.cuh"
#elif defined(GICC_PLATFORM_CXI)
#include "platform/cxi/cxi_device.cuh"
#else
#error "No GICC platform defined. Define GICC_PLATFORM_MLX5 or GICC_PLATFORM_CXI."
#endif
