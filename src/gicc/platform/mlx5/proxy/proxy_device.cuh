/*
 * proxy_device.cuh - kernel-visible CPU-proxy device API for the MLX5 backend.
 *
 * Mirrors the GICC_CPU_PROXY block of src/gicc/platform/ofi/ofi_device.cuh:
 * the device-side put / get / quiet primitives push commands into a
 * host-pinned, device-mapped D2HRing<kProxyRingCapacity>; the CPU proxy
 * worker (mlx5::proxy::ProxyThread) consumes them and posts the real
 * ibv_post_send.
 *
 * Active only when GICC_CPU_PROXY is defined.
 *
 * The `lane` parameter selects which proxy ring (and thus which CPU
 * worker / QP) the submission lands on. Same-lane ops are FIFO;
 * different lanes are independent and can be quieted separately.
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

namespace detail {

// Resolve `lane` to a concrete ProxyRing*, with fallback to the legacy
// single-ring pointer when no fan-out array has been set up.
__device__ inline
gicc::proxy::ProxyRing* lane_to_ring(ProxyCtx* ctx, int lane) {
    void* ring_ptr = nullptr;
    if (ctx->proxy_rings_arr && ctx->num_proxy_rings > 0) {
        int n   = ctx->num_proxy_rings;
        int idx = lane < 0 ? 0 : (lane % n);
        ring_ptr = ctx->proxy_rings_arr[idx];
    } else {
        ring_ptr = ctx->proxy_ring;
    }
    return reinterpret_cast<gicc::proxy::ProxyRing*>(ring_ptr);
}

}  // namespace detail

// Push a WRITE command into proxy_rings_arr[lane % num_proxy_rings]. No
// verbs work happens device-side — the CPU worker resolves
// (rank, buf, off) → (lkey, rkey, raddr) and issues ibv_post_send.
__device__ inline
void put(ProxyCtx* ctx,
         int target_rank,
         int dst_buf, size_t dst_offset,
         int src_buf, size_t src_offset,
         size_t size,
         int lane = 0)
{
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;

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

// Push a READ command into proxy_rings_arr[lane % num_proxy_rings].
// The NIC pulls bytes from peer (source_rank)'s (src_buf, src_offset)
// into our local (dst_buf, dst_offset). Read-ordering guarantee mirrors
// the OFI side: the proxy ack's the slot only after the verbs CQE fires,
// by which point the data has landed in local memory. The caller MUST
// quiet(ctx, lane) before reading the landed data so the GPU's L2 is
// invalidated.
__device__ inline
void get(ProxyCtx* ctx,
         int source_rank,
         int src_buf, size_t src_offset,
         int dst_buf, size_t dst_offset,
         size_t size,
         int lane = 0)
{
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;

    // Remap API direction onto cmd-struct convention: src_* = LOCAL,
    // dst_* = REMOTE on dst_rank. Keeps host-side proxy_local_buf /
    // proxy_remote_buf uniform across WRITE and READ.
    gicc::proxy::TransferCmd c{};
    c.cmd_type   = gicc::proxy::CmdType::READ;
    c.dst_rank   = static_cast<uint8_t>(source_rank);
    c.src_buf    = static_cast<uint8_t>(dst_buf);   // LOCAL landing buf
    c.dst_buf    = static_cast<uint8_t>(src_buf);   // REMOTE source buf
    c.bytes      = static_cast<uint32_t>(size);
    c.src_offset = dst_offset;                      // LOCAL landing off
    c.dst_offset = src_offset;                      // REMOTE source off
    ring->atomic_push(c);
}

// Push a (non-fetching, FI_SUM-style) 4-byte atomic add command. The
// CPU worker maps it to IBV_WR_ATOMIC_FETCH_AND_ADD into a scratch
// buffer (the original value is discarded) so the remote-side effect
// is identical to the OFI fi_atomic / FI_SUM / FI_UINT32 path.
__device__ inline
void atomic_add_u32(ProxyCtx* ctx,
                    int target_rank,
                    int dst_buf, size_t dst_offset,
                    int src_buf, size_t src_offset,
                    int lane = 0)
{
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;

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

// Push a QUIET cmd into ring `lane` and spin on that ring's tail until
// the proxy has drained every prior in-flight op AND acknowledged this
// QUIET slot. Use after a sequence of put(..., lane) / get(..., lane)
// with the same `lane` to wait for completion of just that channel.
__device__ inline
void quiet(ProxyCtx* ctx, int lane = 0)
{
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;

    gicc::proxy::TransferCmd c{};
    c.cmd_type = gicc::proxy::CmdType::QUIET;
    uint64_t my_slot = ring->atomic_push(c);
    while (ring->device_tail_volatile() <= my_slot) {
        __nanosleep(64);
    }
    __threadfence_system();
}

} // namespace gicc::mlx5::proxy

#endif  // GICC_CPU_PROXY
