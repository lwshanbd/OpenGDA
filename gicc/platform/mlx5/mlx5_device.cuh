/**
 * mlx5_device.cuh - MLX5 platform DeviceCtx and device-side RDMA functions
 *
 * Provides gicc::DeviceCtx, gicc::put(), gicc::quiet() for the MLX5 backend.
 * Wraps the optimized WQE building + BlueFlame doorbell from gda_device_opt.cuh.
 */
#pragma once

#include "gda_device_opt.cuh"

namespace gicc {

//==============================================================================
// DeviceCtx - GPU-accessible context for RDMA operations (MLX5 backend)
//
// Contains QP state, WQE buffer, doorbell, CQ, remote peer info, etc.
// Created by Runtime::prepare() and passed to GPU kernels.
//==============================================================================

using DeviceCtx = opengda::GdaDeviceStateOpt;

//==============================================================================
// RDMA PUT - Build WQE + ring BlueFlame doorbell from GPU
//==============================================================================

/**
 * RDMA PUT with explicit local and remote addresses.
 */
__device__ __forceinline__
void put(DeviceCtx* ctx,
         uint64_t local_addr, uint32_t local_lkey,
         uint64_t remote_addr, uint32_t remote_rkey,
         uint32_t size, bool signaled = false)
{
    opengda::gda_rdma_write_opt(
        ctx, local_addr, local_lkey,
        remote_addr, remote_rkey, size, signaled);
}

/**
 * RDMA PUT using default remote address (set by Runtime::prepare()).
 */
__device__ __forceinline__
void put(DeviceCtx* ctx,
         uint64_t local_addr, uint32_t local_lkey,
         uint32_t size, bool signaled = false)
{
    opengda::gda_rdma_write_opt(
        ctx, local_addr, local_lkey,
        ctx->remote_addr, ctx->remote_rkey, size, signaled);
}

//==============================================================================
// Completion
//==============================================================================

/**
 * Wait for all outstanding RDMA operations to complete (poll CQ).
 */
__device__ __forceinline__
void quiet(DeviceCtx* ctx)
{
    opengda::gda_quiet(ctx);
}

} // namespace gicc
