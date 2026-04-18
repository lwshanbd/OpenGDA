/**
 * mlx5_device.cuh - MLX5 platform DeviceCtx and device-side RDMA functions
 *
 * Provides gicc::DeviceCtx, gicc::put(), gicc::quiet() for the MLX5 backend.
 * Wraps the optimized WQE building + BlueFlame doorbell from gda_device_opt.cuh.
 */
#pragma once

#include "gicc/platform/mlx5/device_opt.cuh"

namespace gicc {

//==============================================================================
// DeviceCtx - GPU-accessible context for RDMA operations (MLX5 backend)
//
// Contains QP state, WQE buffer, doorbell, CQ, remote peer info, etc.
// Created by Runtime::prepare() and passed to GPU kernels.
//==============================================================================

using DeviceCtx = gicc::mlx5::DeviceStateOpt;

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
    gicc::mlx5::gda_rdma_write_opt(
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
    gicc::mlx5::gda_rdma_write_opt(
        ctx, local_addr, local_lkey,
        ctx->remote_addr, ctx->remote_rkey, size, signaled);
}

//==============================================================================
// RDMA PUT without doorbell — for batching multiple WQEs per QP
//==============================================================================

/**
 * RDMA PUT without ringing doorbell. Build WQE and advance prod_idx only.
 * Call flush() after posting all WQEs for a QP to ring doorbell once.
 */
__device__ __forceinline__
void put_no_db(DeviceCtx* ctx,
         uint64_t local_addr, uint32_t local_lkey,
         uint64_t remote_addr, uint32_t remote_rkey,
         uint32_t size, bool signaled = false)
{
    gicc::mlx5::gda_rdma_write_no_db(
        ctx, local_addr, local_lkey,
        remote_addr, remote_rkey, size, signaled);
}

/**
 * Flush all pending WQEs to the NIC by ringing the BlueFlame doorbell.
 */
__device__ __forceinline__
void flush(DeviceCtx* ctx)
{
    gicc::mlx5::gda_flush_doorbell(ctx);
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
    gicc::mlx5::gda_quiet(ctx);
}

} // namespace gicc
