// gicc/omp_pipeline.h - compiler-pipelined sends.
//
// Write a kernel the ordinary way and state what it sends at the end of the
// teams region:
//
//     ompx_prepare();
//     #pragma omp target teams is_device_ptr(src)
//     {
//         #pragma omp distribute parallel for [dist_schedule(static, C)]
//         for (i = 0; i < n; ++i) src[i] = ...;
//         ompx_pipelined_put(peer, dst, src, n * sizeof(*src));
//     }
//     ompx_fence();
//
// ompx_pipelined_put means: send these bytes once the kernel's writes to them
// are complete, as a host ompx_put after the kernel would. The gicc-passes
// plugin in GICC_MODE=chunk-lower proves the send can be split along the
// distribute blocks (tools/gicc-passes/src/GICCChunkAnalysis.cpp) and
// rewrites it at the grain GICC_CHUNK_GRAIN selects:
//
//   element (default)  each store to the source is repeated into a same-node
//                      peer's copy, so every word leaves as it is produced;
//                      a peer that is not IPC-mapped gets one
//                      ompx__block_put_one per block instead.
//   block              one ompx__block_put per block, issued as soon as that
//                      block's parallel region joins.
//
// When the pass cannot prove the split, compilation fails with the reason.
// ompx_pipelined_put has no device definition: a build without the pass
// does not link, so the marker can never run untransformed. Build with
// -foffload-lto: otherwise the device runtime is inlined before the pass
// runs and the pass finds no worksharing loop.
//
// Call ompx_prepare() in every translation unit that launches such a kernel:
// the helpers read that unit's device context. A peer that is not IPC-mapped
// is sent to through the CPU proxy (there is no DWQ staging for it).
//
// C++ only.
#pragma once

#include "gicc/omp.h"

#include <ompx.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

// Host fallback of a target region. libomp runs the teams concurrently and
// the marker runs in each, after only its own share of the loop: there is
// no point at which one host put would carry the kernel's final data.
#if !defined(__NVPTX__) && !defined(__AMDGCN__)
extern "C" __attribute__((weak)) void
ompx_pipelined_put(int, void*, const void*, size_t) {
    std::fprintf(stderr, "ompx_pipelined_put: the target region fell back to the host, "
                         "which it cannot run correctly\n");
    std::abort();
}
#endif

#pragma omp declare target
extern "C" int8_t __kmpc_is_spmd_exec_mode();

// Internal linkage with fixed symbol names: the pass inserts calls by these
// names, and an internal definition is one the device link can see into (a
// weak one cannot be internalized, and openmp-opt then treats every call as a
// possible parallel region and keeps the kernel in generic mode).
//
// ompx_spmd_amenable: both helpers stay correct when the device link turns
// the kernel into SPMD mode and every thread runs the code between parallel
// regions -- peer_addr only computes an address, and put_one sends from one
// thread. Without the assumption those calls keep the kernel generic, one
// state-machine round trip per block.
[[omp::assume("ompx_spmd_amenable")]] static void* ompx__peer_addr(int peer, void* addr)
    __asm__("ompx__peer_addr");
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__block_put_one(int peer, void* dst, const void* src, size_t bytes)
    __asm__("ompx__block_put_one");
static void ompx__block_put(int peer, void* dst, const void* src, size_t bytes)
    __asm__("ompx__block_put");

#if defined(__NVPTX__) || defined(__AMDGCN__)
// This unit's device context. A kernel launched before ompx_prepare() ran in
// this unit would otherwise dereference null.
static inline ompx_ctx* ompx__pipeline_ctx() {
    ompx_ctx* c = ompx__ctx;
    if (c == nullptr) {
        printf("ompx_pipelined_put: no device context; call ompx_prepare() in "
               "this translation unit before the kernel\n");
        __builtin_trap();
    }
    return c;
}

// Where `addr`'s object lives in a same-node peer's heap, or null.
static inline char* ompx__peer_base(ompx_ctx* c, int peer, void* addr) {
    if (c->peer_ipc_base == nullptr) return nullptr;
    char* base = static_cast<char*>(c->peer_ipc_base[peer * c->ipc_n_bufs + c->heap_buf]);
    if (base == nullptr) return nullptr;
    return base + (static_cast<char*>(addr) - static_cast<char*>(c->heap_base));
}

// A block put through the CPU proxy. Without a proxy ring the device put
// would return without sending anything.
static inline void ompx__proxy_put(ompx_ctx* c, int peer, void* dst, const void* src,
                                   size_t bytes) {
    if (gicc::omp::detail::lane_to_ring(c, 0) == nullptr) {
        printf("ompx_pipelined_put: peer %d is not IPC-mapped and the CPU proxy is "
               "off; nothing could send the data\n", peer);
        __builtin_trap();
    }
    ompx_put(peer, dst, src, bytes);
}

// The one thread that should act for the team between parallel regions:
// the main thread in generic mode (the only one running there; lane 0 of
// the last warp, not thread 0), hardware thread 0 in SPMD mode, where every
// thread runs that code.
static inline bool ompx__sequential_thread() {
    return !__kmpc_is_spmd_exec_mode() ||
           (ompx_thread_id_x() == 0 && ompx_thread_id_y() == 0 && ompx_thread_id_z() == 0);
}
#endif

// Element-grain lowering: where `addr`'s object lives in a same-node peer's
// heap, or null when the peer is not IPC-mapped (the pass then falls back to
// ompx__block_put_one at run time). Called once per team.
static __attribute__((used)) void* ompx__peer_addr(int peer, void* addr) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    return ompx__peer_base(ompx__pipeline_ctx(), peer, addr);
#else
    (void)peer; (void)addr;
    return nullptr;
#endif
}

// Element-grain fallback for a peer ompx__peer_addr could not map: one proxy
// put per block. No parallel region, so a kernel that only calls this
// between its parallel regions can still be made SPMD by the device link.
static __attribute__((used)) void
ompx__block_put_one(int peer, void* dst, const void* src, size_t bytes) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    if (ompx__sequential_thread()) ompx__proxy_put(ompx__pipeline_ctx(), peer, dst, src, bytes);
#else
    (void)peer; (void)dst; (void)src; (void)bytes;
#endif
}

// Block-grain lowering target, called by a team's main thread right after
// one block's parallel region joins, so every word of [src, src + bytes) is
// written. The team's threads store the block into a same-node peer's copy
// of dst over NVLink / xGMI; for any other peer one thread issues a proxy
// put. Everything the copy loops use is firstprivate: a variable they shared
// by reference would be globalized, a device-heap allocation per call.
// schedule(static, 1) gives neighbouring threads neighbouring words.
static __attribute__((used)) void
ompx__block_put(int peer, void* dst, const void* src, size_t bytes) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    ompx_ctx* c = ompx__pipeline_ctx();
    char* d = ompx__peer_base(c, peer, dst);
    if (d == nullptr) {
        if (ompx__sequential_thread()) ompx__proxy_put(c, peer, dst, src, bytes);
        return;
    }
    const char* s = static_cast<const char*>(src);
    if (((reinterpret_cast<uintptr_t>(d) | reinterpret_cast<uintptr_t>(s)) & 3) == 0) {
        uint32_t* dw = reinterpret_cast<uint32_t*>(d);
        const uint32_t* sw = reinterpret_cast<const uint32_t*>(s);
        const size_t words = bytes / 4;
        #pragma omp parallel for schedule(static, 1) firstprivate(dw, sw, words)
        for (size_t i = 0; i < words; ++i) dw[i] = sw[i];
        if (ompx__sequential_thread())
            for (size_t i = words * 4; i < bytes; ++i) d[i] = s[i];
    } else {
        #pragma omp parallel for schedule(static, 1) firstprivate(d, s, bytes)
        for (size_t i = 0; i < bytes; ++i) d[i] = s[i];
    }
#else
    (void)peer; (void)dst; (void)src; (void)bytes;
#endif
}
#pragma omp end declare target
