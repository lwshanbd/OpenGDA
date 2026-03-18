/**
 * gicc_device.cuh - GICC Device-Side API
 *
 * Platform-neutral GPU-callable functions for RDMA operations.
 * The actual implementation is selected at compile time via platform macros.
 *
 * Usage in GPU kernel:
 *   gicc::put(ctx, local_addr, local_lkey, remote_addr, remote_rkey, size);
 *   gicc::put(ctx, local_addr, local_lkey, size);  // uses default remote
 *   gicc::quiet(ctx);
 */
#pragma once

//==============================================================================
// Include platform-specific DeviceCtx and device functions
//==============================================================================

#if defined(GICC_PLATFORM_MLX5)
#include "platform/mlx5/mlx5_device.cuh"
#elif defined(GICC_PLATFORM_CXI)
#include "platform/cxi/cxi_device.cuh"
#else
#error "No GICC platform defined. Define GICC_PLATFORM_MLX5 or GICC_PLATFORM_CXI."
#endif
