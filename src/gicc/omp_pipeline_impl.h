// gicc/omp_pipeline_impl.h - internals of gicc/omp_pipeline.h.
//
// The device functions the gicc-passes plugin calls when it rewrites an
// ompx_pipelined_put (tools/gicc-passes/src/GICCChunkAnalysis.cpp). None of
// them is for application code; include gicc/omp_pipeline.h instead.
#pragma once

#include "gicc/omp.h"

#include <ompx.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

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
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__box_residual(void* peer_dst, const void* src, size_t bytes, const void* box, int dims,
                   const int64_t* stride, const int64_t* extent, int64_t elem, int loop_runs,
                   int team)
    __asm__("ompx__box_residual");
[[omp::assume("ompx_spmd_amenable")]] static void*
ompx__box_peer(int peer, void* dst, const void* src, size_t bytes, int can_count, int after,
               int32_t* counted)
    __asm__("ompx__box_peer");
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__box_hull(const void* src, size_t bytes, const void* box, int dims, const int64_t* stride,
               const int64_t* extent, int64_t elem, int64_t* lo, int64_t* hi)
    __asm__("ompx__box_hull");
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__box_plan(int n, const int32_t* counted, const int64_t* lo, const int64_t* hi,
               int64_t lb0, int64_t iters, int64_t chunk, int64_t* shift, int64_t* due)
    __asm__("ompx__box_plan");
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__after_put(unsigned long long kernel, int i, int src_arg, int32_t* peer, void** dst,
                int64_t* src_rel, int64_t* bytes)
    __asm__("ompx__after_put");
[[omp::assume("ompx_spmd_amenable")]] static int64_t
ompx__box_due(int n, int64_t lb, int64_t* due, const int32_t* peer, void* const* dst,
              const void* const* src, const int64_t* bytes, unsigned* const* counter,
              const int32_t* after)
    __asm__("ompx__box_due");

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

// A put the kernel cannot send, left in the deferred list for the next
// ompx_quiet on the host (ompx_pipe_deferred in gicc/omp.h).
static inline void ompx__pipe_defer(ompx_ctx* c, int peer, void* dst, const void* src,
                                    size_t bytes) {
    ompx_pipe_deferred* q = ompx__pipe_deferred;
    if (q == nullptr) {
        printf("ompx_pipelined_put: no deferred-put list; call ompx_prepare() in this "
               "translation unit before the kernel\n");
        __builtin_trap();
    }
    const unsigned i = __atomic_fetch_add(&q->n, 1u, __ATOMIC_RELAXED);
    if (i >= OMPX_PIPE_DEFERRED_MAX) {
        printf("ompx_pipelined_put: more than %d puts left for the quiet\n",
               OMPX_PIPE_DEFERRED_MAX);
        __builtin_trap();
    }
    q->e[i].peer    = peer;
    q->e[i].dst_off = ompx__off(c, dst);
    q->e[i].src_off = ompx__off(c, src);
    q->e[i].bytes   = bytes;
}

// The doorbell of put `after` of the launch, when the host queued it on DWQ
// as it posted it (ompx_pipe_after_put in gicc/omp.h); null for a put the
// source states (after < 0) or one the host did not queue.
static inline volatile unsigned long long* ompx__after_bell(int after) {
    ompx_pipe_deferred* q = ompx__pipe_deferred;
    if (after < 0 || after >= OMPX_PIPE_AFTER_MAX || q == nullptr) return nullptr;
    return q->after.e[after].bell;
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
#if defined(__NVPTX__) || defined(__AMDGCN__)
// [a, b) of s copied to the same offsets in d, word `id` of every `n` by
// this thread: neighbouring threads take neighbouring words.
static inline void ompx__copy_range(char* d, const char* s, const char* a, const char* b,
                                    int64_t id, int64_t n) {
    char* to = d + (a - s);
    if (((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
          reinterpret_cast<uintptr_t>(to)) & 3) == 0) {
        uint32_t* tw = reinterpret_cast<uint32_t*>(to);
        const uint32_t* aw = reinterpret_cast<const uint32_t*>(a);
        const int64_t words = (b - a) / 4;
        for (int64_t i = id; i < words; i += n) tw[i] = aw[i];
    } else {
        for (int64_t i = id; i < b - a; i += n) to[i] = a[i];
    }
}
#endif

// Box element grain: the bytes of [src, src + bytes) that the kernel's box
// does not write, copied to peer_dst. The kernel never changes them, so they
// may go at any time, and no mirrored store sends any of them, so the two
// never race. `box` is the address of the point with every digit 0; the dims
// (stride in bytes, extent in points) may come in any order and with either
// sign. loop_runs is 0 when the kernel's loop does not run at all, which
// leaves every byte of the range unwritten. Each team takes one slice of the
// range and walks the box rows that cross it. `team` is nonzero when every
// thread of the team calls this, as in an SPMD kernel's code before its
// loop; the threads then share each gap. Otherwise the team's sequential
// thread copies alone, and a large residual holds the whole kernel back.
static __attribute__((used)) void
ompx__box_residual(void* peer_dst, const void* src, size_t bytes, const void* box, int dims,
                   const int64_t* stride, const int64_t* extent, int64_t elem, int loop_runs,
                   int team) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    if (peer_dst == nullptr || bytes == 0) return;
    int64_t id = 0, n_threads = 1;
    if (team && __kmpc_is_spmd_exec_mode()) {
        const int64_t bx = ompx_block_dim_x(), by = ompx_block_dim_y();
        id = ompx_thread_id_x() + bx * (ompx_thread_id_y() + by * ompx_thread_id_z());
        n_threads = bx * by * ompx_block_dim_z();
    } else if (!ompx__sequential_thread()) {
        return;
    }
    constexpr int kMaxDims = 8;
    if (dims > kMaxDims) {
        if (id == 0)
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
        if (id == 0)
            printf("ompx_pipelined_put: the put range and the written box are not aligned "
                   "to the %lld-byte stores\n", static_cast<long long>(elem));
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
        ompx__copy_range(d, s, lo, hi, id, n_threads);
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
            if (id == 0)
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
        if (rs > done) ompx__copy_range(d, s, done, rs, id, n_threads);
        if (rs + row > done) done = rs + row;
        if (done >= hi) break;
        int k = n - 1;
        for (; k >= 0; --k) {
            if (++q[k] < E[k]) break;
            q[k] = 0;
        }
        if (k < 0) break;
    }
    if (done < hi) ompx__copy_range(d, s, done, hi, id, n_threads);
#else
    (void)peer_dst; (void)src; (void)bytes; (void)box; (void)dims; (void)stride;
    (void)extent; (void)elem; (void)loop_runs; (void)team;
#endif
}

// ---- a box put to a peer that is not IPC-mapped ------------------------------
//
// No store can be repeated into such a peer. In an SPMD kernel the kernel
// sends the range once every team has stored its last word of it: through
// the CPU proxy, or under DWQ by ringing the bell of a put the host queued
// when it posted it (a put the host makes after the kernel). To make that
// early, the distribute loop takes its chunks in a rotated order that
// starts with the ones storing into the ranges (ompx__box_plan), and a team
// that has moved past a range's last chunk says so (ompx__box_count). The
// team that says so last sends the range; the kernel goes on with the rest
// of its loop while it travels. Otherwise the put is left for the next
// quiet, or, if queued, for the host to ring after the kernel.

// Where the put's destination lives in a same-node peer's heap, or null,
// with *counted saying who sends to a peer that has none: 1 when the kernel
// does, by count (it can: it is SPMD, and the CPU proxy is on or the put is
// queued), 0 when the host does -- after the kernel if the put is queued,
// else at the next quiet, team 0 leaving it in the deferred list. `after`
// is the put's index among those the host posted, or -1. A negative peer
// sends nothing.
static __attribute__((used)) void*
ompx__box_peer(int peer, void* dst, const void* src, size_t bytes, int can_count, int after,
               int32_t* counted) {
    *counted = 0;
#if defined(__NVPTX__) || defined(__AMDGCN__)
    if (peer < 0) return nullptr;
    ompx_ctx* c = ompx__pipeline_ctx();
    void* p = ompx__peer_base(c, peer, dst);
    if (p != nullptr) return p;
    if (gicc::omp::detail::lane_to_ring(c, 0) != nullptr) {
        if (can_count) *counted = 1;
        else if (omp_get_team_num() == 0 && ompx__sequential_thread())
            ompx__pipe_defer(c, peer, dst, src, bytes);
    } else if (ompx__after_bell(after) != nullptr) {
        *counted = can_count;
    } else if (omp_get_team_num() == 0 && ompx__sequential_thread()) {
        ompx__pipe_defer(c, peer, dst, src, bytes);
    }
    return nullptr;
#else
    (void)peer; (void)dst; (void)src; (void)bytes; (void)can_count; (void)after;
    return nullptr;
#endif
}

#if defined(__NVPTX__) || defined(__AMDGCN__)
static inline int64_t ompx__floor_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
static inline int64_t ompx__ceil_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) == (b < 0))) ? q + 1 : q;
}
#endif

// The iterations [*lo, *hi] of the kernel's collapsed loop that include
// every one storing into [src, src + bytes); *hi < *lo when none does. The
// dims are as the loop nests them, outermost first, so iteration x has the
// digits x = q_0 * R_1 + ... with R_1 the product of the inner extents:
// the interval is the slabs of q_0 whose stores reach the range.
static __attribute__((used)) void
ompx__box_hull(const void* src, size_t bytes, const void* box, int dims, const int64_t* stride,
               const int64_t* extent, int64_t elem, int64_t* lo, int64_t* hi) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    *lo = 0;
    *hi = -1;
    if (dims <= 0 || bytes == 0 || extent[0] <= 0) return;
    int64_t inner = 1, min_rest = 0, max_rest = 0;
    for (int d = 1; d < dims; ++d) {
        if (extent[d] <= 0) return;
        inner *= extent[d];
        const int64_t span = stride[d] * (extent[d] - 1);
        (span < 0 ? min_rest : max_rest) += span;
    }
    // Slab q stores into [S0*q + min_rest, S0*q + max_rest + elem) from the
    // box's origin; it meets the range [s, e) when S0*q lies in [a, b].
    const int64_t s = static_cast<const char*>(src) - static_cast<const char*>(box);
    const int64_t e = s + static_cast<int64_t>(bytes);
    const int64_t a = s - elem - max_rest + 1, b = e - min_rest - 1;
    const int64_t s0 = stride[0];
    int64_t q_lo = 0, q_hi = extent[0] - 1;
    if (s0 > 0) {
        q_lo = ompx__ceil_div(a, s0);
        q_hi = ompx__floor_div(b, s0);
    } else if (s0 < 0) {
        q_lo = ompx__ceil_div(b, s0);
        q_hi = ompx__floor_div(a, s0);
    } else if (a > 0 || b < 0) {
        return;
    }
    if (q_lo < 0) q_lo = 0;
    if (q_hi > extent[0] - 1) q_hi = extent[0] - 1;
    if (q_lo > q_hi) return;
    *lo = q_lo * inner;
    *hi = (q_hi + 1) * inner - 1;
#else
    (void)src; (void)bytes; (void)box; (void)dims; (void)stride; (void)extent; (void)elem;
    *lo = 0;
    *hi = -1;
#endif
}

// The chunk order of a distribute loop over [lb0, lb0 + iters) in chunks of
// `chunk`, for n puts of which those with counted[p] are sent by count. The
// loop runs, where it would run the chunk at lb, the one at
// lb0 + (lb - lb0 + *shift) mod (chunks * chunk): that puts the counted
// puts' chunks first, the rotation starting right after the widest stretch
// of chunks none of them stores in. due[p] is the first lb at which a team
// has run every chunk put p needs -- lb0 when no chunk stores into the
// range, which is then final already -- and INT64_MAX for a put not counted.
static __attribute__((used)) void
ompx__box_plan(int n, const int32_t* counted, const int64_t* lo, const int64_t* hi,
               int64_t lb0, int64_t iters, int64_t chunk, int64_t* shift, int64_t* due) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    constexpr int kMaxPuts = 8;
    const int64_t chunks = (iters + chunk - 1) / chunk;
    // The counted puts' chunk intervals, sorted by start, then merged.
    int64_t a[kMaxPuts], b[kMaxPuts];
    int m = 0;
    for (int p = 0; p < n && p < kMaxPuts; ++p) {
        if (!counted[p] || lo[p] > hi[p]) continue;
        int j = m++;
        for (; j > 0 && a[j - 1] > lo[p] / chunk; --j) {
            a[j] = a[j - 1];
            b[j] = b[j - 1];
        }
        a[j] = lo[p] / chunk;
        b[j] = hi[p] / chunk;
    }
    int merged = 0;
    for (int i = 0; i < m; ++i) {
        if (merged > 0 && a[i] <= b[merged - 1] + 1) {
            if (b[i] > b[merged - 1]) b[merged - 1] = b[i];
        } else {
            a[merged] = a[i];
            b[merged] = b[i];
            ++merged;
        }
    }
    int64_t start = 0, widest = -1;
    for (int i = 0; i < merged; ++i) {
        const int64_t next = i + 1 < merged ? a[i + 1] : a[0] + chunks;
        if (next - b[i] - 1 > widest) {
            widest = next - b[i] - 1;
            start = next % chunks;
        }
    }
    *shift = start * chunk;
    for (int p = 0; p < n; ++p) {
        if (!counted[p]) {
            due[p] = INT64_MAX;
        } else if (lo[p] > hi[p]) {
            due[p] = lb0;
        } else {
            const int64_t last = ((hi[p] / chunk - start) % chunks + chunks) % chunks;
            due[p] = lb0 + (last + 1) * chunk;
        }
    }
#else
    (void)n; (void)counted; (void)lo; (void)hi; (void)lb0; (void)iters; (void)chunk;
    (void)due;
    *shift = 0;
#endif
}

// Called by every thread of a team, once the team has stored the last word
// it stores into [src, src + bytes). The team to call it last sends the
// range: every other team's part of it has reached the device's L2 by
// then, and a system-scope fence writes it back for the NIC. The counter
// is left at zero for the next launch.
static __attribute__((used)) void
ompx__box_count(unsigned* counter, int peer, void* dst, const void* src, size_t bytes,
                int after) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
#if defined(__AMDGCN__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
#else
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
    ompx_sync_block_acq_rel();
    if (ompx_thread_id_x() != 0 || ompx_thread_id_y() != 0 || ompx_thread_id_z() != 0) return;
    const unsigned teams = static_cast<unsigned>(omp_get_num_teams());
    if (__scoped_atomic_fetch_add(counter, 1u, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE) !=
        teams - 1)
        return;
    __scoped_atomic_store_n(counter, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    ompx_ctx* c = ompx__pipeline_ctx();
    volatile unsigned long long* bell;
    if (gicc::omp::detail::lane_to_ring(c, 0) != nullptr) {
        ompx__proxy_put(c, peer, dst, src, bytes);
    } else if ((bell = ompx__after_bell(after)) != nullptr) {
        *bell = 1;
        __atomic_store_n(&ompx__pipe_deferred->after.e[after].fired, 1, __ATOMIC_RELAXED);
    } else {
        ompx__pipe_defer(c, peer, dst, src, bytes);
    }
#else
    (void)counter; (void)peer; (void)dst; (void)src; (void)bytes; (void)after;
#endif
}

// Called by every thread of a team before the chunk at lb (lb near INT64_MAX
// once the loop is done) when some counted put may be due: counts the team
// past every put whose due[p] <= lb, marks those INT64_MAX, and returns the
// smallest due[] left. Not kept out of line: a kernel is given the vector
// registers of the hungriest function it calls, and this one, compiled on
// its own, needed 85 where the minimod stencil needs 78 -- a wave per SIMD
// less. Inlined into its rare branch it costs the loop none.
static __attribute__((used)) int64_t
ompx__box_due(int n, int64_t lb, int64_t* due, const int32_t* peer, void* const* dst,
              const void* const* src, const int64_t* bytes, unsigned* const* counter,
              const int32_t* after) {
    int64_t next = INT64_MAX;
#if defined(__NVPTX__) || defined(__AMDGCN__)
    for (int p = 0; p < n; ++p) {
        if (due[p] <= lb) {
            ompx__box_count(counter[p], peer[p], dst[p], src[p], static_cast<size_t>(bytes[p]),
                            after[p]);
            due[p] = INT64_MAX;
        }
        if (due[p] < next) next = due[p];
    }
#else
    (void)n; (void)lb; (void)due; (void)peer; (void)dst; (void)src; (void)bytes; (void)counter;
    (void)after;
#endif
    return next;
}

// ---- a put the host makes after the kernel ----------------------------------
//
// Put i of a launch the host brackets (ompx_pipe_after in gicc/omp.h), as the
// host posted it: src is the kernel argument src_arg plus *src_rel. The
// kernel sends it as an ompx_pipelined_put after its loop would, and marks it
// handled, so the host does not put it again. A negative peer -- which sends
// nothing -- when the post is for another kernel, another source, or a put
// the host will not reach.
static __attribute__((used)) void
ompx__after_put(unsigned long long kernel, int i, int src_arg, int32_t* peer, void** dst,
                int64_t* src_rel, int64_t* bytes) {
    *peer = -1;
    *dst = nullptr;
    *src_rel = 0;
    *bytes = 0;
#if defined(__NVPTX__) || defined(__AMDGCN__)
    ompx_pipe_deferred* q = ompx__pipe_deferred;
    if (q == nullptr || i < 0 || i >= OMPX_PIPE_AFTER_MAX || q->after.kernel != kernel) return;
    ompx_pipe_after_put* e = &q->after.e[i];
    if (!e->armed || e->src_arg != src_arg) return;
    ompx_ctx* c = ompx__pipeline_ctx();
    *peer = e->peer;
    *dst = static_cast<char*>(c->heap_base) + e->dst_off;
    *src_rel = e->src_rel;
    *bytes = static_cast<int64_t>(e->bytes);
    if (omp_get_team_num() == 0 && ompx__sequential_thread()) e->handled = 1;
#else
    (void)kernel; (void)i; (void)src_arg;
#endif
}
#pragma omp end declare target
