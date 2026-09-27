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
// The loop may also be a collapse(D) nest writing a D-dimensional box, and
// the put any contiguous range of the written object: part of what the loop
// writes (a halo face of a slab), bytes the kernel does not write at all
// (ghost cells between the rows), or both. Such a put is split at element
// grain whatever GICC_CHUNK_GRAIN says: each store that lands in the range
// is repeated into the peer's copy, and the bytes of the range the kernel
// never writes are sent once, at the start of the kernel. It needs a
// same-node peer.
//
// A negative peer sends nothing, so a kernel can state a put that applies
// to only some ranks (no neighbour, or one that is not on this node).
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
[[omp::assume("ompx_spmd_amenable")]] static void* ompx__peer_addr_mapped(int peer, void* addr)
    __asm__("ompx__peer_addr_mapped");
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__box_residual(void* peer_dst, const void* src, size_t bytes, const void* box, int dims,
                   const int64_t* stride, const int64_t* extent, int64_t elem, int loop_runs)
    __asm__("ompx__box_residual");

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
    if (peer < 0 || c->peer_ipc_base == nullptr) return nullptr;
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
    if (peer >= 0 && ompx__sequential_thread())
        ompx__proxy_put(ompx__pipeline_ctx(), peer, dst, src, bytes);
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
    if (peer < 0) return;
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
// Box element grain: where `addr`'s object lives in a same-node peer's heap.
// Null for a negative peer, which sends nothing; any other peer must be
// IPC-mapped, as a box has no per-block fallback. Called once per team.
static __attribute__((used)) void* ompx__peer_addr_mapped(int peer, void* addr) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    if (peer < 0) return nullptr;
    void* p = ompx__peer_base(ompx__pipeline_ctx(), peer, addr);
    if (p == nullptr) {
        printf("ompx_pipelined_put: peer %d is not IPC-mapped; a put of a "
               "multi-dimensional write needs a same-node peer\n", peer);
        __builtin_trap();
    }
    return p;
#else
    (void)peer; (void)addr;
    return nullptr;
#endif
}

#if defined(__NVPTX__) || defined(__AMDGCN__)
static inline void ompx__copy_range(char* d, const char* s, const char* a, const char* b) {
    char* to = d + (a - s);
    if (((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
          reinterpret_cast<uintptr_t>(to)) & 3) == 0) {
        for (; a < b; a += 4, to += 4)
            *reinterpret_cast<uint32_t*>(to) = *reinterpret_cast<const uint32_t*>(a);
    } else {
        for (; a < b; ++a, ++to) *to = *a;
    }
}
#endif

// Box element grain: the bytes of [src, src + bytes) that the kernel's box
// does not write, copied to peer_dst. The kernel never changes them, so they
// may go at any time, and no mirrored store sends any of them, so the two
// never race. `box` is the address of the point with every digit 0; the dims
// (stride in bytes, extent in points) may come in any order and with either
// sign. loop_runs is 0 when the kernel's loop does not run at all, which
// leaves every byte of the range unwritten. Each team's sequential thread
// takes one slice of the range and walks the box rows that cross it.
static __attribute__((used)) void
ompx__box_residual(void* peer_dst, const void* src, size_t bytes, const void* box, int dims,
                   const int64_t* stride, const int64_t* extent, int64_t elem, int loop_runs) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    if (peer_dst == nullptr || bytes == 0 || !ompx__sequential_thread()) return;
    constexpr int kMaxDims = 8;
    if (dims > kMaxDims) {
        printf("ompx_pipelined_put: a %d-dimensional box has more than %d dims\n", dims,
               kMaxDims);
        __builtin_trap();
    }
    const char* s = static_cast<const char*>(src);
    char* d = static_cast<char*>(peer_dst);
    const char* b0 = static_cast<const char*>(box);
    // Mirrored stores are matched by their first byte: one straddling the
    // range's edge would send bytes outside it or leave some unsent.
    bool aligned = (s - b0) % elem == 0 && static_cast<int64_t>(bytes) % elem == 0;
    for (int k = 0; k < dims; ++k) aligned = aligned && stride[k] % elem == 0;
    if (!aligned) {
        printf("ompx_pipelined_put: the put range and the written box are not aligned to "
               "the %lld-byte stores\n", static_cast<long long>(elem));
        __builtin_trap();
    }
    const int64_t words = static_cast<int64_t>(bytes) / elem;
    const int64_t nt = omp_get_num_teams(), t = omp_get_team_num();
    const char* lo = s + words * t / nt * elem;
    const char* hi = s + words * (t + 1) / nt * elem;
    if (lo >= hi) return;

    // Normalize: positive strides, dims of one point dropped, outermost
    // (largest stride) first.
    int64_t S[kMaxDims], E[kMaxDims];
    int n = 0;
    bool empty = loop_runs == 0;
    for (int k = 0; k < dims; ++k) {
        if (extent[k] <= 0) empty = true;
        if (extent[k] <= 1) continue;
        int64_t st = stride[k];
        if (st < 0) {
            b0 += (extent[k] - 1) * st;
            st = -st;
        }
        int j = n++;
        for (; j > 0 && S[j - 1] < st; --j) {
            S[j] = S[j - 1];
            E[j] = E[j - 1];
        }
        S[j] = st;
        E[j] = extent[k];
    }
    if (empty) {
        ompx__copy_range(d, s, lo, hi);
        return;
    }
    // A dense innermost dim is one contiguous row; otherwise each point is.
    int64_t row = elem;
    if (n > 0 && S[n - 1] == elem) row = E[--n] * elem;
    // The rows must come in address order without overlapping for the gaps
    // between them to be exactly what the box does not write.
    int64_t span = row;
    for (int k = n - 1; k >= 0; --k) {
        if (S[k] < span) {
            printf("ompx_pipelined_put: the rows of the written box overlap; cannot "
                   "tell which bytes of the put it does not write\n");
            __builtin_trap();
        }
        span += (E[k] - 1) * S[k];
    }

    // The last row starting at or before lo (row 0 if none).
    int64_t q[kMaxDims];
    int64_t rel = lo - b0;
    for (int k = 0; k < n; ++k) {
        int64_t qk = rel > 0 ? rel / S[k] : 0;
        if (qk >= E[k]) qk = E[k] - 1;
        q[k] = qk;
        rel -= qk * S[k];
    }
    const char* done = lo;   // everything below is written or already sent
    for (;;) {
        const char* rs = b0;
        for (int k = 0; k < n; ++k) rs += q[k] * S[k];
        if (rs >= hi) break;
        if (rs > done) ompx__copy_range(d, s, done, rs);
        if (rs + row > done) done = rs + row;
        if (done >= hi) break;
        int k = n - 1;
        for (; k >= 0; --k) {
            if (++q[k] < E[k]) break;
            q[k] = 0;
        }
        if (k < 0) break;
    }
    if (done < hi) ompx__copy_range(d, s, done, hi);
#else
    (void)peer_dst; (void)src; (void)bytes; (void)box; (void)dims; (void)stride;
    (void)extent; (void)elem; (void)loop_runs;
#endif
}
#pragma omp end declare target
