/*
 * gicc_omp_device.hpp - OpenMP-target port of the GICC proxy producer.
 *
 * Same semantics as src/gicc/platform/ofi/ofi_device.cuh (CPU-proxy path),
 * but emitted as AMDGPU device code by the OpenMP offload toolchain instead
 * of HIP. The ring layout (gicc::proxy::ProxyRing / TransferCmd) is shared
 * verbatim, so the CPU proxy thread consumes pushes from an omp target region
 * with no changes. HIP device intrinsics are swapped for C11 __atomic_*
 * builtins, which on AMDGCN default to system scope (CPU-visible through the
 * mapped ring). __hip_atomic_* do NOT compile under -fopenmp.
 *
 * Usage from the application:
 *   #pragma omp target is_device_ptr(ctx)
 *   { gicc::omp::put(ctx, peer, dbuf, doff, sbuf, soff, bytes); }
 *   gicc::omp::quiet(ctx);
 */
#pragma once

#include "gicc/platform/ofi/device_ctx.hpp"   // single-source gicc::DeviceCtx (HIP-free)
#include "gicc/proxy/common/proxy_ring_defs.hpp"   // ProxyRing, TransferCmd, CmdType
#include <cstdint>

#pragma omp declare target

namespace gicc {
namespace omp {

namespace detail {

// Resolve `lane` to a ProxyRing*, mirroring ofi_device.cuh detail::lane_to_ring.
inline gicc::proxy::ProxyRing* lane_to_ring(gicc::DeviceCtx* ctx, int lane) {
    void* ring_ptr = nullptr;
    if (ctx->proxy_rings_arr && ctx->num_proxy_rings > 0) {
        int n = ctx->num_proxy_rings;
        int idx = lane < 0 ? 0 : (lane % n);
        ring_ptr = ctx->proxy_rings_arr[idx];
    } else {
        ring_ptr = ctx->proxy_ring;
    }
    return reinterpret_cast<gicc::proxy::ProxyRing*>(ring_ptr);
}

// System-scope full fence — replaces __threadfence_system().
inline void fence_system() {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// Volatile read of host-published tail — replaces device_tail_volatile().
inline uint64_t tail_volatile(gicc::proxy::ProxyRing* r) {
    return __atomic_load_n(&r->tail, __ATOMIC_RELAXED);
}

// Producer push — reimplements d2h_ring atomic_push with C11 system-scope atomics.
inline uint64_t atomic_push(gicc::proxy::ProxyRing* r, const gicc::proxy::TransferCmd& c) {
    constexpr uint32_t kMask = gicc::proxy::ProxyRing::mask();
    constexpr uint64_t kCap  = gicc::proxy::kProxyRingCapacity;
    uint64_t h, prev;
    do {
        h = __atomic_load_n(&r->head, __ATOMIC_RELAXED);
        uint64_t t = __atomic_load_n(&r->tail, __ATOMIC_RELAXED);
        while (h - t == kCap) {                       // ring full: back off
#ifdef __AMDGCN__
            __builtin_amdgcn_s_sleep(1);
#endif
            t = __atomic_load_n(&r->tail, __ATOMIC_RELAXED);
        }
        uint64_t expected = h;
        // CAS head: h -> h+1. __atomic_compare_exchange_n updates `expected` to
        // the seen value on failure; loop until we win. weak=false (strong CAS).
        bool ok = __atomic_compare_exchange_n(
            &r->head, &expected, h + 1, /*weak=*/false,
            __ATOMIC_RELAXED, __ATOMIC_RELAXED);
        prev = ok ? h : expected;
    } while (prev != h);

    uint32_t idx = static_cast<uint32_t>(h) & kMask;

    // Write everything EXCEPT cmd_type first; cmd_type is the per-slot ready flag.
    r->buf[idx].dst_rank   = c.dst_rank;
    r->buf[idx].src_buf    = c.src_buf;
    r->buf[idx].dst_buf    = c.dst_buf;
    r->buf[idx].bytes      = c.bytes;
    r->buf[idx].src_offset = c.src_offset;
    r->buf[idx].dst_offset = c.dst_offset;

    fence_system();
    r->buf[idx].cmd_type = c.cmd_type;   // publish
    return h;
}

}  // namespace detail

inline void put(gicc::DeviceCtx* ctx, int target_rank,
                int dst_buf, size_t dst_offset,
                int src_buf, size_t src_offset,
                size_t size, int lane = 0) {
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;
    gicc::proxy::TransferCmd c;
    c.cmd_type   = gicc::proxy::CmdType::WRITE;
    c.dst_rank   = static_cast<uint8_t>(target_rank);
    c.src_buf    = static_cast<uint8_t>(src_buf);
    c.dst_buf    = static_cast<uint8_t>(dst_buf);
    c.bytes      = static_cast<uint32_t>(size);
    c.src_offset = src_offset;
    c.dst_offset = dst_offset;
    detail::atomic_push(ring, c);
}

inline void get(gicc::DeviceCtx* ctx, int source_rank,
                int src_buf, size_t src_offset,
                int dst_buf, size_t dst_offset,
                size_t size, int lane = 0) {
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;
    // Same direction remap as ofi_device.cuh get(): src_*=local, dst_*=remote.
    gicc::proxy::TransferCmd c;
    c.cmd_type   = gicc::proxy::CmdType::READ;
    c.dst_rank   = static_cast<uint8_t>(source_rank);
    c.src_buf    = static_cast<uint8_t>(dst_buf);
    c.dst_buf    = static_cast<uint8_t>(src_buf);
    c.bytes      = static_cast<uint32_t>(size);
    c.src_offset = dst_offset;
    c.dst_offset = src_offset;
    detail::atomic_push(ring, c);
}

inline void quiet(gicc::DeviceCtx* ctx, int lane = 0) {
    if (!ctx) return;
    auto* ring = detail::lane_to_ring(ctx, lane);
    if (!ring) return;
    gicc::proxy::TransferCmd c{};
    c.cmd_type = gicc::proxy::CmdType::QUIET;
    uint64_t my_slot = detail::atomic_push(ring, c);
    while (detail::tail_volatile(ring) <= my_slot) {
#ifdef __AMDGCN__
        __builtin_amdgcn_s_sleep(1);
#endif
    }
    detail::fence_system();
}

}  // namespace omp
}  // namespace gicc

#pragma omp end declare target
