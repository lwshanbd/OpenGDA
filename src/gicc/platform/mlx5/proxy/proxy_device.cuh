/*
 * proxy_device.cuh - kernel-visible CPU-proxy device API for the MLX5 backend.
 *
 * Mirrors the GICC_CPU_PROXY block of src/gicc/platform/ofi/ofi_device.cuh:
 * the device-side put / quiet primitives push commands into a host-pinned,
 * device-mapped D2HRing<kProxyRingCapacity>; the CPU proxy worker
 * (mlx5::proxy::ProxyThread) consumes them and posts the real ibv_post_send.
 *
 * Active only when GICC_CPU_PROXY is defined.
 */
#pragma once

#ifdef GICC_CPU_PROXY

#include "gicc/proxy/common/proxy_ring_defs.hpp"
#include "gicc/proxy/common/transfer_cmd.hpp"

#include <cstdint>

namespace gicc::mlx5::proxy {

// Kernel argument bundle. One per Runtime, populated by
// Runtime::ensure_proxy_ring(s) and passed to user kernels.
//   proxy_ring         single-ring back-compat pointer (== proxy_rings_arr[0])
//   proxy_rings_arr    device-mapped array of N ring pointers for fan-out
//   num_proxy_rings    N
struct ProxyCtx {
    void*  proxy_ring;
    void** proxy_rings_arr;
    int    num_proxy_rings;
};

// Push a WRITE command into proxy_rings_arr[ring_idx % num_proxy_rings].
// Falls back to ring 0 if no fan-out array is set up. No verbs work
// happens device-side — the CPU worker resolves (rank, buf, off) →
// (lkey, rkey, raddr) and issues ibv_post_send.
__device__ inline
void put_no_db_idx(ProxyCtx* ctx, int ring_idx,
                   int target_rank,
                   int dst_buf, size_t dst_offset,
                   int src_buf, size_t src_offset,
                   size_t size)
{
    if (!ctx) return;
    void* ring_ptr = nullptr;
    if (ctx->proxy_rings_arr && ctx->num_proxy_rings > 0) {
        int n   = ctx->num_proxy_rings;
        int idx = ring_idx < 0 ? 0 : (ring_idx % n);
        ring_ptr = ctx->proxy_rings_arr[idx];
    } else {
        ring_ptr = ctx->proxy_ring;
    }
    if (!ring_ptr) return;
    auto* ring = reinterpret_cast<gicc::proxy::ProxyRing*>(ring_ptr);

    gicc::proxy::TransferCmd c{};
    c.cmd_type   = gicc::proxy::CmdType::WRITE;
    c.dst_rank   = static_cast<uint8_t>(target_rank);
    c.src_buf    = static_cast<uint8_t>(src_buf);
    c.dst_buf    = static_cast<uint8_t>(dst_buf);
    c.bytes      = static_cast<uint32_t>(size);
    c.src_offset = src_offset;
    c.dst_offset = dst_offset;
    ring->atomic_push(c);
}

// Single-ring convenience (always pushes to proxy_ring).
__device__ inline
void put_no_db(ProxyCtx* ctx,
               int target_rank,
               int dst_buf, size_t dst_offset,
               int src_buf, size_t src_offset,
               size_t size)
{
    put_no_db_idx(ctx, /*ring_idx=*/0,
                  target_rank, dst_buf, dst_offset,
                  src_buf, src_offset, size);
}

// Push a (non-fetching, FI_SUM-style) 4-byte atomic add command. The
// CPU worker maps it to IBV_WR_ATOMIC_FETCH_AND_ADD into a scratch
// buffer (the original value is discarded) so the remote-side effect
// is identical to the OFI fi_atomic / FI_SUM / FI_UINT32 path.
__device__ inline
void atomic_add_u32_idx(ProxyCtx* ctx, int ring_idx,
                        int target_rank,
                        int dst_buf, size_t dst_offset,
                        int src_buf, size_t src_offset)
{
    if (!ctx) return;
    void* ring_ptr = nullptr;
    if (ctx->proxy_rings_arr && ctx->num_proxy_rings > 0) {
        int n   = ctx->num_proxy_rings;
        int idx = ring_idx < 0 ? 0 : (ring_idx % n);
        ring_ptr = ctx->proxy_rings_arr[idx];
    } else {
        ring_ptr = ctx->proxy_ring;
    }
    if (!ring_ptr) return;
    auto* ring = reinterpret_cast<gicc::proxy::ProxyRing*>(ring_ptr);

    gicc::proxy::TransferCmd c{};
    c.cmd_type   = gicc::proxy::CmdType::ATOMIC;
    c.dst_rank   = static_cast<uint8_t>(target_rank);
    c.src_buf    = static_cast<uint8_t>(src_buf);
    c.dst_buf    = static_cast<uint8_t>(dst_buf);
    c.bytes      = 4;        // implicit; proxy ignores
    c.src_offset = src_offset;
    c.dst_offset = dst_offset;
    ring->atomic_push(c);
}

// Per-ring quiet: push a QUIET cmd into ring `ring_idx` and spin on
// that ring's tail until the proxy has drained every prior in-flight
// op AND acknowledged this QUIET slot. Use after a sequence of
// put_no_db_idx(..., ring_idx, ...) to wait for completion of just
// that lane.
__device__ inline
void quiet_idx(ProxyCtx* ctx, int ring_idx)
{
    if (!ctx) return;
    void* ring_ptr = nullptr;
    if (ctx->proxy_rings_arr && ctx->num_proxy_rings > 0) {
        int n   = ctx->num_proxy_rings;
        int idx = ring_idx < 0 ? 0 : (ring_idx % n);
        ring_ptr = ctx->proxy_rings_arr[idx];
    } else {
        ring_ptr = ctx->proxy_ring;
    }
    if (!ring_ptr) return;
    auto* ring = reinterpret_cast<gicc::proxy::ProxyRing*>(ring_ptr);

    gicc::proxy::TransferCmd c{};
    c.cmd_type = gicc::proxy::CmdType::QUIET;
    uint64_t my_slot = ring->atomic_push(c);
    while (ring->device_tail_volatile() <= my_slot) {
        __nanosleep(64);
    }
    __threadfence_system();
}

__device__ inline
void quiet(ProxyCtx* ctx)
{
    quiet_idx(ctx, 0);
}

} // namespace gicc::mlx5::proxy

#endif  // GICC_CPU_PROXY
