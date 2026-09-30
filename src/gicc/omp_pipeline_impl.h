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
[[omp::assume("ompx_spmd_amenable")]] static int64_t
ompx__box_piece(size_t bytes, int32_t counted, int pieces_max)
    __asm__("ompx__box_piece");
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__box_hull(const void* src, size_t bytes, const void* box, int dims, const int64_t* stride,
               const int64_t* extent, int64_t elem, int64_t* lo, int64_t* hi)
    __asm__("ompx__box_hull");
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__box_plan(int n, const int32_t* counted, const int64_t* lo, const int64_t* hi,
               int64_t lb0, int64_t iters, int64_t chunk, int64_t* shift, int64_t* due,
               int64_t* first)
    __asm__("ompx__box_plan");
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__after_put(unsigned long long kernel, int i, int src_arg, int32_t* peer, void** dst,
                int64_t* src_rel, int64_t* bytes)
    __asm__("ompx__after_put");
[[omp::assume("ompx_spmd_amenable")]] static int64_t
ompx__box_grab(unsigned long long* next, int64_t* slot)
    __asm__("ompx__box_grab");
[[omp::assume("ompx_spmd_amenable")]] static void
ompx__box_grab_done(unsigned long long* next, unsigned* done)
    __asm__("ompx__box_grab_done");
[[omp::assume("ompx_spmd_amenable")]] static int64_t
ompx__box_due(int n, int64_t lb, int64_t* due, const int32_t* peer, void* const* dst,
              const void* const* src, const int64_t* bytes, unsigned* const* counter,
              const int32_t* after)
    __asm__("ompx__box_due");

#if defined(__NVPTX__) || defined(__AMDGCN__)
// No printf here or in anything else a rewritten kernel calls: one gives the
// kernel a dynamic stack, and with it scratch for every wave. What goes wrong
// is recorded for the host to report instead (ompx__pipe_fail).

// This unit's deferred-put list, which also holds everything these helpers
// look up (ompx_pipe_deferred in gicc/omp.h): it is cached by the GPU, the
// device context is not, and every wave of a rewritten kernel reads it.
// Missing if ompx_prepare() was not called in this translation unit, which
// leaves nothing to record the failure in.
static inline ompx_pipe_deferred* ompx__pipe_list() {
    ompx_pipe_deferred* q = ompx__pipe_deferred;
    if (q == nullptr) __builtin_trap();
    return q;
}

// For the host to report at the next quiet, or right after the launch of a
// kernel it bracketed.
static inline void ompx__pipe_fail(ompx_pipe_deferred* q, int code, long long arg) {
    q->fail_arg = arg;
    q->fail = code;
}

// Where `addr`'s object lives in a same-node peer's heap, or null.
static inline char* ompx__peer_base(ompx_pipe_deferred* q, int peer, void* addr) {
    if (peer < 0 || peer >= q->n_ranks) return nullptr;
    char* base = q->peer_heap[peer];
    if (base == nullptr) return nullptr;
    return base + (static_cast<char*>(addr) - q->heap_base);
}

// A block put through the CPU proxy, which must be on: the device put would
// otherwise return without sending anything.
static inline void ompx__proxy_put(ompx_pipe_deferred* q, int peer, void* dst, const void* src,
                                   size_t bytes) {
    if (!q->proxy_on) {
        ompx__pipe_fail(q, OMPX_PIPE_FAIL_NO_PROXY, peer);
        return;
    }
    ompx_put(peer, dst, src, bytes);
}

// A put the kernel cannot send, left in the deferred list for the next
// ompx_quiet on the host. Past the list's end it is dropped, and the quiet
// reports the overflow.
static inline void ompx__pipe_defer(ompx_pipe_deferred* q, int peer, void* dst, const void* src,
                                    size_t bytes) {
    const unsigned i = __atomic_fetch_add(&q->n, 1u, __ATOMIC_RELAXED);
    if (i >= OMPX_PIPE_DEFERRED_MAX) return;
    q->e[i].peer    = peer;
    q->e[i].dst_off = static_cast<const char*>(dst) - q->heap_base;
    q->e[i].src_off = static_cast<const char*>(src) - q->heap_base;
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
    return ompx__peer_base(ompx__pipe_list(), peer, addr);
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
        ompx__proxy_put(ompx__pipe_list(), peer, dst, src, bytes);
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
    ompx_pipe_deferred* q = ompx__pipe_list();
    char* d = ompx__peer_base(q, peer, dst);
    if (d == nullptr) {
        if (ompx__sequential_thread()) ompx__proxy_put(q, peer, dst, src, bytes);
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

// What a residual copies of one team's slice [lo, hi) of the range: the gaps
// between the box's rows, walked by one thread and listed a batch at a time
// for the team to copy. All of it lives in team memory: an array of any
// thread's own would give every wave of the kernel a private segment, which
// the runtime then allocates at each launch.
constexpr int kOmpxGapsMax = 32, kOmpxBoxDimsMax = 8;
// Bytes of the range per team: a short range goes to one team, so the rest
// skip the planning and its barriers altogether.
constexpr int64_t kOmpxResidualSlice = 64 * 1024;

// This team's slice of a residual of `bytes` from src: *t of *nt slices, or
// false if it has none. The slices start at a team picked by the source
// address, so that a kernel's several puts do not all land on team 0 --
// which would plan every one of them, while the others wait for it to end.
// Every thread of every team runs this for every put: no division by
// anything but a constant (a 64-bit one is a hundred instructions).
static inline bool ompx__residual_slice(const void* src, size_t bytes, int64_t* t, int64_t* nt) {
    const int64_t teams = omp_get_num_teams();
    const int64_t want =
        (static_cast<int64_t>(bytes) + kOmpxResidualSlice - 1) / kOmpxResidualSlice;
    *nt = want < teams ? want : teams;
    int64_t first = static_cast<int64_t>((reinterpret_cast<uintptr_t>(src) >> 10) & 63);
    if (first >= teams) first = 0;
    *t = omp_get_team_num() - first;
    if (*t < 0) *t += teams;
    return *t < *nt;
}
struct ompx_gap_list {
    const char* a[kOmpxGapsMax];
    const char* b[kOmpxGapsMax];
    int n;                               // gaps listed
    int more;                            // the walk has more after them
    int ended;                           // no row is left to walk
    int dims;                            // of the normalized box, row dim excluded
    int64_t S[kOmpxBoxDimsMax], E[kOmpxBoxDimsMax];
    int64_t q[kOmpxBoxDimsMax];          // the next row the walk looks at
    int64_t row;                         // bytes of one row
    const char* b0;                      // the normalized box's first point
    const char* done;                    // everything below is written or listed
    const char* lo;
    const char* hi;
};
// Team memory takes no initializer.
[[clang::loader_uninitialized]] static ompx_gap_list ompx__gaps;
#pragma omp allocate(ompx__gaps) allocator(omp_pteam_mem_alloc)

// Lists the walk's next gaps, as many as fit: the bytes before each row not
// already covered, then the bytes after the last one.
static inline void ompx__gaps_fill(ompx_gap_list& L) {
    int m = 0;
    L.more = 0;
    while (!L.ended) {
        const char* rs = L.b0;
        for (int k = 0; k < L.dims; ++k) rs += L.q[k] * L.S[k];
        if (rs >= L.hi) {
            L.ended = 1;
            break;
        }
        if (rs > L.done) {
            if (m == kOmpxGapsMax) {   // resume at this row
                L.more = 1;
                break;
            }
            L.a[m] = L.done;
            L.b[m] = rs;
            ++m;
        }
        if (rs + L.row > L.done) L.done = rs + L.row;
        if (L.done >= L.hi) {
            L.ended = 1;
            break;
        }
        int k = L.dims - 1;
        for (; k >= 0; --k) {
            if (++L.q[k] < L.E[k]) break;
            L.q[k] = 0;
        }
        if (k < 0) L.ended = 1;
    }
    if (L.ended && L.done < L.hi) {
        if (m == kOmpxGapsMax) {
            L.more = 1;
        } else {
            L.a[m] = L.done;
            L.b[m] = L.hi;
            ++m;
            L.done = L.hi;
        }
    }
    L.n = m;
}

// Starts L's walk over this team's slice of the range; false when there is
// nothing to copy (no slice, or a failure recorded for the host).
static inline bool ompx__residual_plan(const char* s, size_t bytes, const void* box, int dims,
                                       const int64_t* stride, const int64_t* extent,
                                       int64_t elem, int loop_runs, int64_t t, int64_t nt,
                                       ompx_gap_list& L) {
    if (dims > kOmpxBoxDimsMax) {
        ompx__pipe_fail(ompx__pipe_list(), OMPX_PIPE_FAIL_DIMS, dims);
        return false;
    }
    const char* b0 = static_cast<const char*>(box);
    // Mirrored stores are matched by their first byte: one straddling the
    // range's edge would send bytes outside it or leave some unsent.
    bool aligned = (s - b0) % elem == 0 && static_cast<int64_t>(bytes) % elem == 0;
    for (int k = 0; k < dims; ++k) aligned = aligned && stride[k] % elem == 0;
    if (!aligned) {
        ompx__pipe_fail(ompx__pipe_list(), OMPX_PIPE_FAIL_ALIGN, elem);
        return false;
    }
    const int64_t words = static_cast<int64_t>(bytes) / elem;
    L.lo = s + words * t / nt * elem;
    L.hi = s + words * (t + 1) / nt * elem;
    if (L.lo >= L.hi) return false;
    L.done = L.lo;
    L.ended = 0;

    // Normalize: positive strides, dims of one point dropped, outermost
    // (largest stride) first.
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
        for (; j > 0 && L.S[j - 1] < st; --j) {
            L.S[j] = L.S[j - 1];
            L.E[j] = L.E[j - 1];
        }
        L.S[j] = st;
        L.E[j] = extent[k];
    }
    if (empty) {   // the box writes nothing: the slice is one gap
        L.ended = 1;
        return true;
    }
    // A dense innermost dim is one contiguous row; otherwise each point is.
    L.row = elem;
    if (n > 0 && L.S[n - 1] == elem) L.row = L.E[--n] * elem;
    // The rows must come in address order without overlapping for the gaps
    // between them to be exactly what the box does not write.
    int64_t span = L.row;
    for (int k = n - 1; k >= 0; --k) {
        if (L.S[k] < span) {
            ompx__pipe_fail(ompx__pipe_list(), OMPX_PIPE_FAIL_OVERLAP, 0);
            return false;
        }
        span += (L.E[k] - 1) * L.S[k];
    }
    L.dims = n;
    L.b0 = b0;
    // The last row starting at or before lo (row 0 if none).
    int64_t rel = L.lo - b0;
    for (int k = 0; k < n; ++k) {
        int64_t qk = rel > 0 ? rel / L.S[k] : 0;
        if (qk >= L.E[k]) qk = L.E[k] - 1;
        L.q[k] = qk;
        rel -= qk * L.S[k];
    }
    return true;
}
#endif

// Box element grain: the bytes of [src, src + bytes) that the kernel's box
// does not write, copied to peer_dst. The kernel never changes them, so they
// may go at any time, and no mirrored store sends any of them, so the two
// never race. `box` is the address of the point with every digit 0; the dims
// (stride in bytes, extent in points) may come in any order and with either
// sign. loop_runs is 0 when the kernel's loop does not run at all, which
// leaves every byte of the range unwritten. A few teams take one slice of the
// range each, of kOmpxResidualSlice bytes or more (ompx__residual_slice); the
// other teams return at once. `team` is nonzero when every thread of the team calls this,
// as in an SPMD kernel's code before its loop: thread 0 then walks the slice
// -- the divisions are not worth doing in every thread -- and the team shares
// each batch of gaps it lists. Otherwise the team's sequential thread does
// both alone, and a large residual holds the whole kernel back.
static __attribute__((used)) void
ompx__box_residual(void* peer_dst, const void* src, size_t bytes, const void* box, int dims,
                   const int64_t* stride, const int64_t* extent, int64_t elem, int loop_runs,
                   int team) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    int64_t t, nt;
    if (peer_dst == nullptr || bytes == 0 || !ompx__residual_slice(src, bytes, &t, &nt)) return;
    const bool shared = team && __kmpc_is_spmd_exec_mode();
    if (!shared && !ompx__sequential_thread()) return;
    int64_t id = 0, n_threads = 1;
    if (shared) {
        const int64_t bx = ompx_block_dim_x(), by = ompx_block_dim_y();
        id = ompx_thread_id_x() + bx * (ompx_thread_id_y() + by * ompx_thread_id_z());
        n_threads = bx * by * ompx_block_dim_z();
    }
    const char* s = static_cast<const char*>(src);
    char* d = static_cast<char*>(peer_dst);
    ompx_gap_list& L = ompx__gaps;
    if (id == 0) {
        if (ompx__residual_plan(s, bytes, box, dims, stride, extent, elem, loop_runs, t, nt, L)) {
            ompx__gaps_fill(L);
        } else {
            L.n = 0;
            L.more = 0;
        }
    }
    for (;;) {
        if (shared) ompx_sync_block_acq_rel();
        const int n = L.n;
        const bool more = L.more;
        for (int g = 0; g < n; ++g) ompx__copy_range(d, s, L.a[g], L.b[g], id, n_threads);
        if (shared) ompx_sync_block_acq_rel();   // before the list is refilled or reused
        if (!more) break;
        if (id == 0) ompx__gaps_fill(L);
    }
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
    // *counted is team memory that every thread stores to: once each, with
    // the same value (a default stored first could land after another
    // thread's answer).
    int32_t by_count = 0;
    void* p = nullptr;
#if defined(__NVPTX__) || defined(__AMDGCN__)
    if (peer >= 0) {
        ompx_pipe_deferred* q = ompx__pipe_list();
        p = ompx__peer_base(q, peer, dst);
        if (p == nullptr) {
            if (q->proxy_on) {
                if (can_count) by_count = 1;
                else if (omp_get_team_num() == 0 && ompx__sequential_thread())
                    ompx__pipe_defer(q, peer, dst, src, bytes);
            } else if (ompx__after_bell(after) != nullptr) {
                by_count = can_count;
            } else if (omp_get_team_num() == 0 && ompx__sequential_thread()) {
                ompx__pipe_defer(q, peer, dst, src, bytes);
            }
        }
    }
#else
    (void)peer; (void)dst; (void)src; (void)bytes; (void)can_count; (void)after;
#endif
    *counted = by_count;
    return p;
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

// A counted put goes out in pieces, each sent once every team is past it
// (ompx__box_count): sent whole, a put over most of the loop would leave
// only when the loop ends, with nothing left to hide it behind. Returns the
// size of the pieces of a put of `bytes`: at least kOmpxPieceMin, 4 KB
// aligned, no more than pieces_max (a power of two) of them; the pass makes
// piece k [k * size, (k + 1) * size) of the range, clipped to it. One piece,
// the whole put, when the put is not counted or the CPU proxy is off: under
// DWQ a counted put is the one descriptor the host queued. No division:
// every thread of every team of a counting launch calls this for every put.
constexpr int kOmpxPieceMinLog2 = 22;   // 4 MB

static __attribute__((used)) int64_t
ompx__box_piece(size_t bytes, int32_t counted, int pieces_max) {
    uint64_t piece = bytes;
#if defined(__NVPTX__) || defined(__AMDGCN__)
    const uint64_t want = static_cast<uint64_t>(bytes) >> kOmpxPieceMinLog2;
    if (counted && want > 1 && pieces_max > 1 && ompx__pipe_list()->proxy_on) {
        const uint64_t most = static_cast<uint64_t>(pieces_max);
        const uint64_t pieces = want >= most ? most : uint64_t(1) << (63 - __builtin_clzll(want));
        const int lg = __builtin_ctzll(pieces);
        piece = (((static_cast<uint64_t>(bytes) + pieces - 1) >> lg) + 4095) & ~uint64_t(4095);
    }
#else
    (void)counted; (void)pieces_max;
#endif
    return static_cast<int64_t>(piece);
}

// ompx__box_hull's interval, as values.
#if defined(__NVPTX__) || defined(__AMDGCN__)
static inline void ompx__hull(const void* src, size_t bytes, const void* box, int dims,
                              const int64_t* stride, const int64_t* extent, int64_t elem,
                              int64_t& lo, int64_t& hi) {
    lo = 0;
    hi = -1;
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
    lo = q_lo * inner;
    hi = (q_hi + 1) * inner - 1;
}
#endif

// The iterations [*lo, *hi] of the kernel's collapsed loop that include
// every one storing into [src, src + bytes); *hi < *lo when none does. The
// dims are as the loop nests them, outermost first, so iteration x has the
// digits x = q_0 * R_1 + ... with R_1 the product of the inner extents:
// the interval is the slabs of q_0 whose stores reach the range. *lo and *hi
// are team memory every thread stores to, once each, with the same value.
static __attribute__((used)) void
ompx__box_hull(const void* src, size_t bytes, const void* box, int dims, const int64_t* stride,
               const int64_t* extent, int64_t elem, int64_t* lo, int64_t* hi) {
    int64_t l = 0, h = -1;
#if defined(__NVPTX__) || defined(__AMDGCN__)
    ompx__hull(src, bytes, box, dims, stride, extent, elem, l, h);
#else
    (void)src; (void)bytes; (void)box; (void)dims; (void)stride; (void)extent; (void)elem;
#endif
    *lo = l;
    *hi = h;
}

// The chunk order of a distribute loop over [lb0, lb0 + iters) in chunks of
// `chunk`, for n puts of which those with counted[p] are sent by count. The
// loop runs, where it would run the chunk at lb, the one at
// lb0 + (lb - lb0 + *shift) mod (chunks * chunk): that puts the counted
// puts' chunks first, the rotation starting right after the widest stretch
// of chunks none of them stores in. due[p] is the first lb at which a team
// has run every chunk put p needs -- lb0 when no chunk stores into the
// range, which is then final already -- and INT64_MAX for a put not counted;
// *first is the smallest due[p], so that the kernel reads one value, not n
// (read together, n of them took a vector register each). shift, due and
// first are team memory: in an SPMD kernel every thread calls this, and
// thread 0 plans for the team, in team memory too (see ompx_gap_list).
#if defined(__NVPTX__) || defined(__AMDGCN__)
// Puts a plan orders: pieces of puts, as the pass lays them out.
constexpr int kOmpxPlanPutsMax = 64;
struct ompx_plan_scratch {
    int64_t a[kOmpxPlanPutsMax], b[kOmpxPlanPutsMax];
};
[[clang::loader_uninitialized]] static ompx_plan_scratch ompx__plan;
#pragma omp allocate(ompx__plan) allocator(omp_pteam_mem_alloc)

// ompx__box_plan's work, done by one thread of the team.
static inline void ompx__plan_team(int n, const int32_t* counted, const int64_t* lo,
                                   const int64_t* hi, int64_t lb0, int64_t iters, int64_t chunk,
                                   int64_t* shift, int64_t* due, int64_t* first) {
    constexpr int kMaxPuts = kOmpxPlanPutsMax;
    const int64_t chunks = (iters + chunk - 1) / chunk;
    // The counted puts' chunk intervals, sorted by start, then merged.
    int64_t* a = ompx__plan.a;
    int64_t* b = ompx__plan.b;
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
    int64_t soonest = INT64_MAX;
    for (int p = 0; p < n; ++p) {
        int64_t d;
        if (!counted[p]) {
            d = INT64_MAX;
        } else if (lo[p] > hi[p]) {
            d = lb0;
        } else {
            const int64_t last = ((hi[p] / chunk - start) % chunks + chunks) % chunks;
            d = lb0 + (last + 1) * chunk;
        }
        due[p] = d;
        if (d < soonest) soonest = d;
    }
    *first = soonest;
}
#endif

static __attribute__((used)) void
ompx__box_plan(int n, const int32_t* counted, const int64_t* lo, const int64_t* hi,
               int64_t lb0, int64_t iters, int64_t chunk, int64_t* shift, int64_t* due,
               int64_t* first) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    // One barrier, which every thread of an SPMD team reaches at this one
    // call: it is an aligned barrier, and the optimizer relies on that.
    const bool spmd = __kmpc_is_spmd_exec_mode();
    if (!spmd || (ompx_thread_id_x() == 0 && ompx_thread_id_y() == 0 && ompx_thread_id_z() == 0))
        ompx__plan_team(n, counted, lo, hi, lb0, iters, chunk, shift, due, first);
    if (spmd) ompx_sync_block_acq_rel();
#else
    (void)n; (void)counted; (void)lo; (void)hi; (void)lb0; (void)iters; (void)chunk;
    (void)due;
    *shift = 0;
    *first = INT64_MAX;
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
    ompx_pipe_deferred* q = ompx__pipe_list();
    volatile unsigned long long* bell;
    if (q->proxy_on) {
        ompx__proxy_put(q, peer, dst, src, bytes);
    } else if ((bell = ompx__after_bell(after)) != nullptr) {
        *bell = 1;
        __atomic_store_n(&ompx__pipe_deferred->after.e[after].fired, 1, __ATOMIC_RELAXED);
    } else {
        ompx__pipe_defer(q, peer, dst, src, bytes);
    }
#else
    (void)counter; (void)peer; (void)dst; (void)src; (void)bytes; (void)after;
#endif
}

// A counting launch hands its chunks out in order, each to the first team
// ready for one, instead of chunk t + j * teams to team t. Waves are
// scheduled oldest first, so the teams that start last fall behind; a range
// goes out only once every team is past it, and with chunks dealt round
// robin the ranges came out slowly at first and in a rush at the end, where
// nothing is left to hide them behind. Each team still runs as many chunks
// as it was dealt, so every chunk is taken once. Returns the number of the
// chunk to run next, in the order the plan's shift rotates. Called by every
// thread of the team: thread 0 takes it, and the team reads it from team
// memory (*slot) between two barriers, the second so that no thread still
// reads it when thread 0 writes the next.
static __attribute__((used)) int64_t
ompx__box_grab(unsigned long long* next, int64_t* slot) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    if (ompx_thread_id_x() == 0 && ompx_thread_id_y() == 0 && ompx_thread_id_z() == 0)
        *slot = static_cast<int64_t>(
            __scoped_atomic_fetch_add(next, 1ull, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE));
    ompx_sync_block_acq_rel();
    const int64_t g = *slot;
    ompx_sync_block_acq_rel();
    return g;
#else
    (void)next; (void)slot;
    return 0;
#endif
}

// Called by every thread of a team of a counting launch once its loop is
// done: the last team to get here leaves the chunk counter at zero for the
// next launch.
static __attribute__((used)) void
ompx__box_grab_done(unsigned long long* next, unsigned* done) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    if (ompx_thread_id_x() != 0 || ompx_thread_id_y() != 0 || ompx_thread_id_z() != 0) return;
    const unsigned teams = static_cast<unsigned>(omp_get_num_teams());
    if (__scoped_atomic_fetch_add(done, 1u, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE) ==
        teams - 1) {
        __scoped_atomic_store_n(next, 0ull, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
        __scoped_atomic_store_n(done, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
#else
    (void)next; (void)done;
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
    // The outs are team memory every thread stores to: once each, with the
    // same value (see ompx__box_peer).
    int32_t pe = -1;
    void* d = nullptr;
    int64_t rel = 0, by = 0;
#if defined(__NVPTX__) || defined(__AMDGCN__)
    ompx_pipe_deferred* q = ompx__pipe_deferred;
    ompx_pipe_after_put* e = q != nullptr && i >= 0 && i < OMPX_PIPE_AFTER_MAX &&
                                     q->after.kernel == kernel
                                 ? &q->after.e[i]
                                 : nullptr;
    if (e != nullptr && e->armed && e->src_arg == src_arg) {
        pe = e->peer;
        d = q->heap_base + e->dst_off;
        rel = e->src_rel;
        by = static_cast<int64_t>(e->bytes);
        if (omp_get_team_num() == 0 && ompx__sequential_thread()) e->handled = 1;
    }
#else
    (void)kernel; (void)i; (void)src_arg;
#endif
    *peer = pe;
    *dst = d;
    *src_rel = rel;
    *bytes = by;
}
#pragma omp end declare target
