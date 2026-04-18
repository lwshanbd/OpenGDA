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

#if defined(GICC_PLATFORM_MLX5)
#include "gicc/platform/mlx5/device_opt.cuh"
namespace gicc { using RawDeviceCtx = gicc::mlx5::DeviceStateOpt; }
#endif

namespace gicc {

/// Maximum number of registered buffers (per rank) and peers.
/// These are compile-time limits for the GPU-side fixed-size arrays.
enum {
    GICC_MAX_BUFS  = 16,
    GICC_MAX_PEERS = 64,
};

/// Memory region entry: maps an address range to its IB lkey/rkey.
struct BufEntry {
    uint64_t addr;   ///< Base GPU virtual address
    uint64_t size;   ///< Size in bytes
    uint32_t lkey;   ///< Local key (for RDMA source)
    uint32_t rkey;   ///< Remote key (for RDMA target when this buf is remote)
};

/// Per-peer remote buffer info.
struct PeerBufEntry {
    uint64_t addr;   ///< Remote GPU virtual address
    uint32_t rkey;   ///< Remote key
};

/// GPU-accessible context for simplified GICC operations.
/// Created by Runtime::build_context(), passed to GPU kernels.
struct GiccContext {
    int my_rank;
    int num_peers;

    /// Per-peer QP state (index = MPI rank, NULL for self)
    RawDeviceCtx* peer_ctxs[GICC_MAX_PEERS];

    /// Local buffer registry
    BufEntry local_bufs[GICC_MAX_BUFS];
    int num_local_bufs;

    /// Remote buffer registry: remote_bufs[peer][buf_idx]
    PeerBufEntry remote_bufs[GICC_MAX_PEERS][GICC_MAX_BUFS];
};

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

} // namespace gicc
