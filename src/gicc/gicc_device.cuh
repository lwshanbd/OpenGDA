/**
 * gicc_device.cuh - GICC Device-Side API
 *
 * Simplified API (use with GiccContext from gicc::context()):
 *   gicc::put(ctx, dst, src, size, peer);     // one-line RDMA
 *   gicc::put_no_db(ctx, dst, src, size, peer);
 *   gicc::flush(ctx, peer);
 *   gicc::quiet(ctx, peer);
 *
 * Legacy API (use with DeviceCtx from rt.prepare()):
 *   gicc::put(ctx, local_addr, local_lkey, remote_addr, remote_rkey, size);
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

// Simplified GiccContext-based API (gicc::put(ctx, dst, src, size, peer))
#include "gicc_context.cuh"
