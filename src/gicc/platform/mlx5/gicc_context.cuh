/**
 * gicc_context.cuh — Simplified GPU-side GICC context
 *
 * Provides an NVSHMEM-style device API that hides IB details (lkey, rkey,
 * DeviceCtx per peer).  The user just calls:
 *
 *   gicc::put(ctx, remote_ptr, local_ptr, size, peer);
 *   gicc::flush(ctx, peer);
 *   gicc::quiet(ctx, peer);
 *
 * The GiccContext stores per-peer QP state and per-buffer MR info so that
 * put() can look up lkey/rkey automatically from raw pointers.
 */
#pragma once

#include <stdint.h>

// Struct layout (BufEntry, PeerBufEntry, GiccContext, GICC_MAX_*) lives in
// gicc_context_types.hpp; we extend it here with the __device__ helpers
// whose bodies use CUDA atomics / PTX intrinsics and therefore can only
// be parsed by nvcc.
#include "gicc/platform/mlx5/gicc_context_types.hpp"

#if defined(GICC_PLATFORM_MLX5)
#include "gicc/platform/mlx5/device_opt.cuh"
#endif

namespace gicc {

//==============================================================================
// Internal: look up lkey for a local pointer
//==============================================================================

__device__ __forceinline__
uint32_t find_lkey(const GiccContext* ctx, uint64_t addr) {
    for (int i = 0; i < ctx->num_local_bufs; i++) {
        if (addr >= ctx->local_bufs[i].addr &&
            addr < ctx->local_bufs[i].addr + ctx->local_bufs[i].size)
            return ctx->local_bufs[i].lkey;
    }
    return 0;  // should not happen if buffer is registered
}

__device__ __forceinline__
uint32_t find_rkey(const GiccContext* ctx, int peer, uint64_t addr) {
    for (int i = 0; i < ctx->num_local_bufs; i++) {
        uint64_t rbase = ctx->remote_bufs[peer][i].addr;
        uint64_t rsize = ctx->local_bufs[i].size;  // same size on all ranks
        if (addr >= rbase && addr < rbase + rsize)
            return ctx->remote_bufs[peer][i].rkey;
    }
    return 0;
}

//==============================================================================
// Simplified device API
//==============================================================================

/**
 * RDMA PUT — write local data to a remote GPU's memory.
 *
 * @param ctx      GiccContext (GPU-accessible)
 * @param dst      Remote GPU pointer (on peer's address space)
 * @param src      Local GPU pointer (registered with GICC)
 * @param size     Transfer size in bytes
 * @param peer     Target MPI rank
 * @param signaled Whether to generate a CQ entry (default true)
 */
__device__ __forceinline__
void put(GiccContext* ctx, void* dst, const void* src, uint32_t size, int peer,
         bool signaled = true)
{
    uint64_t src_addr = (uint64_t)src;
    uint64_t dst_addr = (uint64_t)dst;
    uint32_t lkey = find_lkey(ctx, src_addr);
    uint32_t rkey = find_rkey(ctx, peer, dst_addr);
    RawDeviceCtx* qp = ctx->peer_ctxs[peer];

    gicc::mlx5::gda_rdma_write_opt(qp, src_addr, lkey, dst_addr, rkey, size, signaled);
}

/**
 * RDMA PUT without doorbell — for batching multiple puts before a single flush.
 */
__device__ __forceinline__
void put_no_db(GiccContext* ctx, void* dst, const void* src, uint32_t size, int peer,
               bool signaled = true)
{
    uint64_t src_addr = (uint64_t)src;
    uint64_t dst_addr = (uint64_t)dst;
    uint32_t lkey = find_lkey(ctx, src_addr);
    uint32_t rkey = find_rkey(ctx, peer, dst_addr);
    RawDeviceCtx* qp = ctx->peer_ctxs[peer];

    gicc::mlx5::gda_rdma_write_no_db(qp, src_addr, lkey, dst_addr, rkey, size, signaled);
}

/**
 * Flush — ring doorbell for a peer, submitting all pending WQEs.
 */
__device__ __forceinline__
void flush(GiccContext* ctx, int peer)
{
    gicc::mlx5::gda_flush_doorbell(ctx->peer_ctxs[peer]);
}

/**
 * Quiet — wait for all outstanding RDMA to a peer to complete.
 */
__device__ __forceinline__
void quiet(GiccContext* ctx, int peer)
{
    gicc::mlx5::gda_quiet(ctx->peer_ctxs[peer]);
}

//==============================================================================
// Common-form RDMA — buffer-index + offset addressing.
//
// Mirrors the unified device API documented in
// src/gicc/platform/ofi/ofi_device.cuh:
//
//   put_no_db(ctx, target_rank,
//             dst_buf, dst_offset,
//             src_buf, src_offset,
//             size, signaled=false)
//
//   get_no_db(ctx, source_rank,
//             src_buf, src_offset,
//             dst_buf, dst_offset,
//             size, signaled=false)
//
// (target_rank, dst_buf, dst_offset) names a slice of the peer's
// registered buffer; (src_buf, src_offset) names a slice of the caller's
// local buffer. Backend resolves lkey/rkey/addresses from the registries
// populated by Runtime::build_context().
//==============================================================================

__device__ __forceinline__
void put_no_db(GiccContext* ctx,
               int target_rank,
               int dst_buf, size_t dst_offset,
               int src_buf, size_t src_offset,
               size_t size, bool signaled = false)
{
    uint64_t  src_addr = ctx->local_bufs[src_buf].addr + src_offset;
    uint32_t  lkey     = ctx->local_bufs[src_buf].lkey;
    uint64_t  dst_addr = ctx->remote_bufs[target_rank][dst_buf].addr + dst_offset;
    uint32_t  rkey     = ctx->remote_bufs[target_rank][dst_buf].rkey;
    RawDeviceCtx* qp   = ctx->peer_ctxs[target_rank];

    gicc::mlx5::gda_rdma_write_no_db(
        qp, src_addr, lkey, dst_addr, rkey,
        static_cast<uint32_t>(size), signaled);
}

__device__ __forceinline__
void get_no_db(GiccContext* ctx,
               int source_rank,
               int src_buf, size_t src_offset,
               int dst_buf, size_t dst_offset,
               size_t size, bool signaled = false)
{
    uint64_t  dst_addr = ctx->local_bufs[dst_buf].addr + dst_offset;
    uint32_t  lkey     = ctx->local_bufs[dst_buf].lkey;
    uint64_t  src_addr = ctx->remote_bufs[source_rank][src_buf].addr + src_offset;
    uint32_t  rkey     = ctx->remote_bufs[source_rank][src_buf].rkey;
    RawDeviceCtx* qp   = ctx->peer_ctxs[source_rank];

    gicc::mlx5::gda_rdma_read_no_db(
        qp, dst_addr, lkey, src_addr, rkey,
        static_cast<uint32_t>(size), signaled);
}

//==============================================================================
// flush / quiet over the common-form context.
//
// MLX5's WQE submission is per-QP. The OFI backend's flush() is a
// single MMIO trigger that releases every queued op across all peers;
// here we fan out one BlueFlame doorbell per peer that still has WQEs
// queued. The kernel pays N peer-iterations of doorbell + fence; this
// is fundamental to the per-QP submission model and matches what
// hand-written MLX5 kernels already do.
//==============================================================================

__device__ __forceinline__
void flush(GiccContext* ctx)
{
    for (int peer = 0; peer < ctx->num_peers; ++peer) {
        if (peer == ctx->my_rank) continue;
        RawDeviceCtx* qp = ctx->peer_ctxs[peer];
        if (qp) gicc::mlx5::gda_flush_doorbell(qp);
    }
}

__device__ __forceinline__
void quiet(GiccContext* ctx)
{
    for (int peer = 0; peer < ctx->num_peers; ++peer) {
        if (peer == ctx->my_rank) continue;
        RawDeviceCtx* qp = ctx->peer_ctxs[peer];
        if (qp) gicc::mlx5::gda_quiet(qp);
    }
}

} // namespace gicc
