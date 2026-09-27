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
// C++ only.
#pragma once

#include "gicc/omp.h"

#include <ompx.h>

#include <cstddef>
#include <cstdint>

#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

// Host fallback of a target region: the region runs on the CPU, the loop
// finishes before this call, so it is exactly the host put it stands for.
#if !defined(__NVPTX__) && !defined(__AMDGCN__)
extern "C" __attribute__((weak)) void
ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes) {
    ompx_put_host(peer, dst, src, bytes);
}
#endif

#pragma omp declare target
// Internal linkage with fixed symbol names: the pass inserts calls by these
// names, and an internal definition is one the device link can see into (a
// weak one cannot be internalized, and openmp-opt then treats every call as a
// possible parallel region and keeps the kernel in generic mode).
//
// ompx_spmd_amenable: both helpers stay correct when the device link turns
// the kernel into SPMD mode and every thread runs the code between parallel
// regions -- peer_addr only computes an address, and put_one sends from
// hardware thread 0 alone. Without the assumption those calls keep the
// kernel generic, one state-machine round trip per block.
[[omp::assume("ompx_spmd_amenable")]] static void* ompx__peer_addr(int peer, void* addr)
    __asm__("ompx__peer_addr");
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__block_put_one(int peer, void* dst, const void* src, size_t bytes)
    __asm__("ompx__block_put_one");
static void ompx__block_put(int peer, void* dst, const void* src, size_t bytes)
    __asm__("ompx__block_put");


// Element-grain lowering: the address of `addr`'s object in a same-node
// peer's heap, or null when the peer is not IPC-mapped (the pass then falls
// back to ompx__block_put_one at run time). Called once per team.
static __attribute__((used)) void* ompx__peer_addr(int peer, void* addr) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    ompx_ctx* c = ompx__ctx;
    if (c == nullptr || c->peer_ipc_base == nullptr) return nullptr;
    char* base = static_cast<char*>(c->peer_ipc_base[peer * c->ipc_n_bufs + c->heap_buf]);
    if (base == nullptr) return nullptr;
    return base + (static_cast<char*>(addr) - static_cast<char*>(c->heap_base));
#else
    (void)peer; (void)addr;
    return nullptr;
#endif
}

// Element-grain fallback for a peer ompx__peer_addr could not map: one
// device ompx_put per block, issued by one thread. No parallel region, so a
// kernel that only calls this between its parallel regions can still be
// made SPMD by the device link. Hardware thread 0 sends: in generic mode it
// is the only caller, in SPMD mode every thread calls. (omp_get_thread_num
// cannot tell them apart outside a parallel region: it is 0 for all.)
static __attribute__((used)) void
ompx__block_put_one(int peer, void* dst, const void* src, size_t bytes) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    if (ompx_thread_id_x() == 0 && ompx_thread_id_y() == 0 && ompx_thread_id_z() == 0)
        ompx_put(peer, dst, src, bytes);
#else
    (void)peer; (void)dst; (void)src; (void)bytes;
#endif
}

// Lowering target, called by a team's main thread right after one block's
// parallel region joins, so every word of [src, src + bytes) is written. The
// team's threads store the block into a same-node peer's copy of dst over
// NVLink / xGMI; for any other peer the main thread issues a device ompx_put.
// `used` keeps it in the module until the pass has inserted its calls.
static __attribute__((used)) void
ompx__block_put(int peer, void* dst, const void* src, size_t bytes) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    ompx_ctx* c = ompx__ctx;
    char* base = c->peer_ipc_base
                     ? static_cast<char*>(c->peer_ipc_base[peer * c->ipc_n_bufs + c->heap_buf])
                     : nullptr;
    if (base == nullptr) {
        ompx_put(peer, dst, src, bytes);
        return;
    }
    char* d = base + (static_cast<char*>(dst) - static_cast<char*>(c->heap_base));
    const char* s = static_cast<const char*>(src);
    // Everything the copy loops use is firstprivate: a variable they shared
    // by reference would be globalized, a device-heap allocation per call.
    // schedule(static, 1) gives neighbouring threads neighbouring words.
    if (((reinterpret_cast<uintptr_t>(d) | reinterpret_cast<uintptr_t>(s)) & 3) == 0) {
        uint32_t* dw = reinterpret_cast<uint32_t*>(d);
        const uint32_t* sw = reinterpret_cast<const uint32_t*>(s);
        const size_t words = bytes / 4;
        #pragma omp parallel for schedule(static, 1) firstprivate(dw, sw, words)
        for (size_t i = 0; i < words; ++i) dw[i] = sw[i];
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
