// gicc/omp.h - GiOMP public API. GiOMP = DiOMP + GICC: an ompx_* surface that
// lets an OpenMP target-offload application communicate without writing CUDA or
// HIP. ONE include for an app, C and C++ alike.
//
// Memory model: a symmetric heap. ompx_init registers one large device
// allocation with the NIC and exchanges its address book entry once; every
// ompx_alloc carves from that heap, so allocation is a pointer bump and there
// is no per-buffer registration or exchange. Because ranks allocate in the same
// order, an address has the same heap offset on every rank -- so put/get take
// ordinary local addresses (symmetric addressing, as in OpenSHMEM/NVSHMEM) and
// ompx_peer_ptr can hand back a same-node peer's address for the same object.
//
//   Control:    ompx_init/finalize, ompx_get_rank_num/num_ranks
//   Memory:     ompx_alloc, ompx_bind, ompx_free
//   Movement:   ompx_peer_ptr (same-node direct access), ompx_put, ompx_get,
//               ompx_put_no_db (a put whose doorbell rings at the next quiet)
//   Signals:    ompx_put_signal, ompx_stage_put_signal, ompx_signal_wait,
//               ompx_signal_read, ompx_signal_reset
//   Completion: ompx_quiet, ompx_fence, ompx_barrier
//   Device:     ompx_prepare once, then put/get/quiet/trigger and the signal
//               calls work inside #pragma omp target as well as outside
//
// The GPU-compiled runtime lives in libgicc_omp; this header is safe to include
// in a -fopenmp TU AND in a -x hip/-x cuda TU (the device-side inline functions
// are guarded away from the GPU-language compilers, which cannot handle
// `omp declare target`).
#pragma once

#include <stddef.h>

// Backend: GICC_PLATFORM_MLX5 selects InfiniBand, where a GPU thread posts
// its own work requests; anything else is libfabric (CPU proxy / DWQ).
#if defined(__cplusplus) && !defined(__HIPCC__) && !defined(__CUDACC__)
#if defined(GICC_PLATFORM_MLX5)
#include "gicc/platform/mlx5/gicc_omp_device.hpp"  // gicc::omp::put/get/quiet
#else
#include "gicc/platform/ofi/gicc_omp_device.hpp"   // gicc::omp::put/get/quiet
#endif
#elif defined(__cplusplus)
#if defined(GICC_PLATFORM_MLX5)
#include "gicc/platform/mlx5/device_ctx.hpp"        // gicc::DeviceCtx only
#else
#include "gicc/platform/ofi/device_ctx.hpp"        // gicc::DeviceCtx only
#endif
#endif

#ifdef __cplusplus
typedef gicc::DeviceCtx ompx_ctx;
extern "C" {
#else
// Layout-compatible prefix of gicc::DeviceCtx, so a C target region can fire
// the trigger itself. Checked against the real struct on the C++ side.
#include <stdint.h>
typedef struct ompx_ctx {
    volatile uint64_t* trigger_addr_;
    uint64_t           trigger_val_;
} ompx_ctx;
#endif

// ---- control ----------------------------------------------------------------
void ompx_init(void);       // MPI + runtime + device select + heap + exchange
void ompx_finalize(void);
int  ompx_get_rank_num(void);
int  ompx_get_num_ranks(void);

// ---- memory -----------------------------------------------------------------
// Collective and symmetric: every rank must call these in the same order with
// the same sizes, which is what makes an address mean the same object globally.
void* ompx_alloc(size_t bytes);                    // zeroed heap allocation
void* ompx_bind(void* host_ptr, size_t bytes);     // alloc + associate + copy in
void  ompx_free(void* ptr);


// ---- data movement ----------------------------------------------------------
// Addresses are local addresses of symmetric objects. `dst` names the object on
// `peer`; `src` names our own copy.
void* ompx_peer_ptr(int peer, const void* addr);   // NULL unless same-node
void  ompx_put_host(int peer, void* dst, const void* src, size_t bytes);
void  ompx_get_host(int peer, void* dst, const void* src, size_t bytes);

// ---- signals ----------------------------------------------------------------
// A put whose arrival the receiver can observe without a barrier: the payload
// lands, then `value` lands in the peer's slot `sig`. Both writes ride one
// endpoint, which asks the provider for write-after-write ordering, so the
// flag never overtakes the data it announces. Slots are symmetric (slot i
// means the same object on every rank) and there are 64 of them. A slot holds
// the last value written, so give each slot one sender and make its values
// increase -- a sequence number, waited for with `>=`.
//
// ompx_put_signal, ompx_signal_wait and ompx_signal_read work on the host and
// inside a target region alike (see below). On the device, put_signal is how
// a kernel sends data it has just produced:
//   - CPU proxy: it pushes the payload and the signal onto the ring.
//   - DWQ: the NIC can only run descriptors the host queued beforehand, so
//     the host stages the transfer with ompx_stage_put_signal and the device
//     call rings that slot's doorbell. Every slot has its own doorbell, so
//     each team releases only its own chunk, in whatever order teams finish.
//     The n-th device put_signal on a slot releases the n-th one staged there.
//     ompx_stage_put_signal is a no-op under the proxy, so one source serves
//     both transports.
// A device put_signal is issued by one thread; synchronize the threads that
// wrote `src` first (e.g. #pragma omp barrier), as for any put.
void ompx_put_signal_host(int peer, void* dst, const void* src, size_t bytes,
                          int sig, unsigned long long value);
void ompx_stage_put_signal(int peer, void* dst, const void* src, size_t bytes,
                           int sig, unsigned long long value);
unsigned long long ompx_signal_read_host(int sig);
void ompx_signal_wait_host(int sig, unsigned long long ge);
void ompx_signal_reset(int sig);

// ---- completion -------------------------------------------------------------
void ompx_quiet_host(void);
void ompx_barrier(void);
void ompx_fence(void);      // quiet + barrier: call before reading peer writes

// ---- deferred put -----------------------------------------------------------
// ompx_put_no_db posts a put and leaves its doorbell for the next host ompx_quiet
// (or ompx_fence): the source is read when the doorbell rings, not when the
// put is posted. The call means exactly an ompx_put issued at the start of
// that quiet, so the source may still be written in between -- post the halo
// before the kernels that compute it:
//
//     ompx_fence();                               // delivers last step's halo
//     ompx_put_no_db(peer, dst, src, bytes);      // this step's, sent next fence
//     #pragma omp target teams loop ...           // writes src
//     for (...) src[...] = ...;
//
// Built with the gicc-passes plugin in GICC_MODE=chunk-lower, every device
// store into a posted source is also written into a same-node peer's dst as
// it happens, and the quiet sends only what those stores did not cover. The
// data sent is the same as a put at the quiet, given two rules:
//   - dst belongs to the sender from the post until its quiet: the peer
//     neither reads nor writes it in between (the same epoch rule a put
//     issued at the quiet already needs, from the previous fence on).
//   - Between the post and the quiet, the source is written only by device
//     code built with the plugin, in the application's own device image --
//     not by the host, `target update to`, a translation unit or shared
//     library built without it, or a direct MPI transfer. Writes the runtime
//     can see (a get into it, a self-put, a peer write ordered by
//     ompx_barrier or ompx_signal_wait) make the quiet send it whole.
// GICC_NODB_MIRROR=0 turns the mirroring off at run time.
#define OMPX_NODB_MAX 8
typedef struct ompx_nodb_entry {    // one mirrored post
    char*              src;         // device address of the source
    unsigned long long bytes;
    char*              peer;        // the peer's dst, IPC-mapped
    unsigned char*     map;         // one byte per 4-byte word: == epoch if sent
} ompx_nodb_entry;
typedef struct ompx_nodb_state {    // read by the plugin's instrumentation
    char*           lo;             // [lo, hi) bounds every entry's source
    char*           hi;
    unsigned        n;
    unsigned        epoch;          // 1..255, advanced by every quiet
    int             poison;         // a write the plugin could not mirror
    int             reserved;
    ompx_nodb_entry e[OMPX_NODB_MAX];
} ompx_nodb_state;
// `state` is the device address of ompx__nodb in the application's image.
void ompx_put_no_db_host(int peer, void* dst, const void* src, size_t bytes,
                         void* state);

// ---- explicit batched DWQ ---------------------------------------------------
// ompx_put stages; nothing moves until a trigger is pulled. ompx_quiet pulls it
// for you, so an application only calls ompx_trigger when it wants the transfer
// to start earlier than that -- typically before a kernel it wants to overlap.
// No-op under the CPU proxy, which is already in flight on issue.
void ompx_trigger_host(void);

// ---- device-side ------------------------------------------------------------
ompx_ctx* ompx_prepare_ctx(void);

// Internal: pipelined puts a kernel could not send itself (its peer is not
// IPC-mapped and the CPU proxy is off, or the kernel is not SPMD), left for
// the next ompx_quiet / ompx_fence to put. Host memory the device writes; the
// offsets are into the symmetric heap.
#define OMPX_PIPE_DEFERRED_MAX 64
typedef struct ompx_pipe_deferred_put {
    int                peer;
    int                reserved;
    unsigned long long dst_off, src_off, bytes;
} ompx_pipe_deferred_put;

// Internal: the ompx_put calls that follow a kernel launch, as the
// gicc-passes plugin found them in a first compile (GICC_MODE=put-discover).
// Before the launch the host posts each one here; the kernel sends the ones
// it can itself, as an ompx_pipelined_put after its loop would, and says
// so; after it, each put is made only if the kernel did not.
//
// Under DWQ a kernel cannot start a transfer the host has not queued, so a
// put to a peer that is not IPC-mapped is queued when it is posted, on a
// trigger of its own (one per i), and `bell` rings it: the kernel does once
// the source is written, and says so in `fired`; otherwise the host does
// after the kernel. The NIC reads the source when the bell rings.
#define OMPX_PIPE_AFTER_MAX 8
typedef struct ompx_pipe_after_put {
    int                armed;     // posted, and the host will reach the put
    int                handled;   // the kernel sent it
    int                peer;
    int                src_arg;   // the kernel argument src is an offset from
    unsigned long long dst_off;   // into the symmetric heap
    long long          src_rel;   // src minus that argument
    unsigned long long bytes;
    volatile unsigned long long* bell;   // device doorbell; null if not queued
    int                fired;     // the kernel rang it
    int                reserved;
} ompx_pipe_after_put;
typedef struct ompx_pipe_after {
    unsigned long long  kernel;   // FNV-1a of the kernel's name
    ompx_pipe_after_put e[OMPX_PIPE_AFTER_MAX];
} ompx_pipe_after;

// What a pipelined put the device could not make reports, for the host to
// print at the next quiet (ompx_pipe_deferred::fail). A device printf would
// give every kernel with such a put a dynamic stack.
#define OMPX_PIPE_FAIL_NO_PROXY 1   // peer not IPC-mapped, no CPU proxy (arg: peer)
#define OMPX_PIPE_FAIL_DIMS     2   // a box with too many dims (arg: dims)
#define OMPX_PIPE_FAIL_ALIGN    3   // range and box not aligned to the stores (arg: bytes)
#define OMPX_PIPE_FAIL_OVERLAP  4   // the box's rows overlap

typedef struct ompx_pipe_deferred {
    unsigned               n;
    unsigned               reserved;
    ompx_pipe_deferred_put e[OMPX_PIPE_DEFERRED_MAX];
    ompx_pipe_after        after;   // the one launch posted at a time
    // What a kernel's pipelined puts look up, copied here by ompx_prepare:
    // this struct is cached by the GPU and the device context is not, and
    // every wave of such a kernel reads them.
    char*                  heap_base;   // this rank's symmetric heap
    char* const*           peer_heap;   // [rank]: its heap as mapped here, or null
    int                    n_ranks;
    int                    proxy_on;    // the CPU proxy takes device puts
    int                    fail;        // OMPX_PIPE_FAIL_*, 0 if none
    int                    fail_reserved;
    long long              fail_arg;
} ompx_pipe_deferred;
ompx_pipe_deferred* ompx__pipe_deferred_list(void);   // its device address
void ompx__after_post(unsigned long long kernel, int i, int src_arg, int armed, int peer,
                      void* dst, const void* src, long long src_rel, size_t bytes);
void ompx__after_done(unsigned long long kernel, int i, int peer, void* dst,
                      const void* src, size_t bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#include <omp.h>

// ---- one name, host or device ----------------------------------------------
// These are `declare target`, so the same call works in ordinary code and
// inside `#pragma omp target`. On the device they use the context published by
// ompx_prepare; on the host they call the runtime. ompx_trigger works from C
// and C++ alike; ompx_put/get/quiet have no C device body, so a C application
// issues those from the host and only pulls the trigger on the device.
#pragma omp declare target
static ompx_ctx* ompx__ctx = 0;
static ompx_pipe_deferred* ompx__pipe_deferred = 0;   // ompx__pipe_deferred_list
// Weak, so every translation unit shares one copy in host and device image
// alike; the plugin's instrumentation in any unit reads it by name.
__attribute__((weak)) ompx_nodb_state ompx__nodb;
#pragma omp end declare target

static inline void ompx_put_no_db(int peer, void* dst, const void* src,
                                  size_t bytes) {
    ompx_put_no_db_host(peer, dst, src, bytes,
                        omp_get_mapped_ptr(&ompx__nodb, omp_get_default_device()));
}

// The runtime's DeviceCtx is host-pinned and mapped and prepare() returns the
// same device pointer for the life of the runtime, so this publishes it (and
// the deferred-put list) once; later calls only update contents the device
// already sees.
static inline ompx_ctx* ompx_prepare(void) {
    ompx_ctx* c = ompx_prepare_ctx();
    ompx_pipe_deferred* q = ompx__pipe_deferred_list();
    if (ompx__ctx != c || ompx__pipe_deferred != q) {
        ompx__ctx = c;
        ompx__pipe_deferred = q;
        #pragma omp target is_device_ptr(c, q)
        { ompx__ctx = c; ompx__pipe_deferred = q; }
    }
    return c;
}

#pragma omp declare target
static inline void ompx_trigger(void) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
    // No trigger mapped means the CPU proxy, which is already in flight.
    if (ompx__ctx && ompx__ctx->trigger_addr_)
        *(ompx__ctx->trigger_addr_) = ompx__ctx->trigger_val_;
#else
    ompx_trigger_host();
#endif
}
#pragma omp end declare target

#if defined(__cplusplus) && !defined(__HIPCC__) && !defined(__CUDACC__)
#pragma omp declare target
static inline size_t ompx__off(ompx_ctx* c, const void* a) {
    return (size_t)((const char*)a - (const char*)c->heap_base);
}

// `lane` picks one of the proxy rings, each drained by its own worker, so two
// transfers issued on different lanes progress independently. It is ignored on
// the host, which dispatches per call.
inline void ompx_put(int peer, void* dst, const void* src, size_t bytes,
                     int lane = 0) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
    ompx_ctx* c = ompx__ctx;
    gicc::omp::put(c, peer, c->heap_buf, ompx__off(c, dst),
                   c->heap_buf, ompx__off(c, src), bytes, lane);
#else
    (void)lane; ompx_put_host(peer, dst, src, bytes);
#endif
}

// A get writes dst through the proxy, which no mirrored store sees: if dst
// meets a posted source, the next quiet sends the sources whole.
static inline void ompx__nodb_touch(const void* dst, size_t bytes) {
    const char* p = (const char*)dst;
    if (p < ompx__nodb.hi && p + bytes > ompx__nodb.lo) ompx__nodb.poison = 1;
}

inline void ompx_get(int peer, void* dst, const void* src, size_t bytes,
                     int lane = 0) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
    ompx_ctx* c = ompx__ctx;
    ompx__nodb_touch(dst, bytes);
    gicc::omp::get(c, peer, c->heap_buf, ompx__off(c, src),
                   c->heap_buf, ompx__off(c, dst), bytes, lane);
#else
    (void)lane; ompx_get_host(peer, dst, src, bytes);
#endif
}

// Single-issuer form: the caller guarantees one work-item enqueues on this lane
// until the matching quiet, which skips the contended ring CAS. Device only.
inline void ompx_get_single(int peer, void* dst, const void* src, size_t bytes,
                            int lane = 0) {
    ompx_ctx* c = ompx__ctx;
    ompx__nodb_touch(dst, bytes);
    gicc::omp::get_single(c, peer, c->heap_buf, ompx__off(c, src),
                          c->heap_buf, ompx__off(c, dst), bytes, lane);
}

inline void ompx_quiet(int lane = 0) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
    gicc::omp::quiet(ompx__ctx, lane);
#else
    (void)lane; ompx_quiet_host();
#endif
}

// `lane` picks the proxy ring, as for ompx_put; the signal always rides the
// same ring as its payload. DWQ ignores it.
inline void ompx_put_signal(int peer, void* dst, const void* src, size_t bytes,
                            int sig, unsigned long long value, int lane = 0) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
    ompx_ctx* c = ompx__ctx;
    if (c->sig_trigger) {
        // DWQ. The NIC reads `src` once the doorbell rings, so the payload
        // must be out of this device's caches first.
        volatile uint64_t* bell = c->sig_trigger[sig];
        if (!bell) __builtin_trap();      // nothing was ever staged on `sig`
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        *bell = 1;
    } else {
        gicc::omp::put_signal(c, peer, c->heap_buf, ompx__off(c, dst),
                              c->heap_buf, ompx__off(c, src), bytes,
                              (size_t)sig * sizeof(uint64_t), value, lane);
    }
#else
    (void)lane; ompx_put_signal_host(peer, dst, src, bytes, sig, value);
#endif
}

inline unsigned long long ompx_signal_read(int sig) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
    return __atomic_load_n(&ompx__ctx->sig_base[sig], __ATOMIC_ACQUIRE);
#else
    return ompx_signal_read_host(sig);
#endif
}

// Returns once slot `sig` holds a value >= `ge`; the payload that value
// announces is then visible to the caller.
inline void ompx_signal_wait(int sig, unsigned long long ge) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
    const uint64_t* slot = &ompx__ctx->sig_base[sig];
    while (__atomic_load_n(slot, __ATOMIC_ACQUIRE) < ge) {
#ifdef __AMDGCN__
        __builtin_amdgcn_s_sleep(1);
#endif
    }
#else
    ompx_signal_wait_host(sig, ge);
#endif
}
#pragma omp end declare target
#else
static inline void ompx_put(int peer, void* dst, const void* src, size_t n) {
    ompx_put_host(peer, dst, src, n);
}
static inline void ompx_put_signal(int peer, void* dst, const void* src,
                                   size_t n, int sig, unsigned long long value) {
    ompx_put_signal_host(peer, dst, src, n, sig, value);
}
static inline unsigned long long ompx_signal_read(int sig) {
    return ompx_signal_read_host(sig);
}
static inline void ompx_signal_wait(int sig, unsigned long long ge) {
    ompx_signal_wait_host(sig, ge);
}
static inline void ompx_get(int peer, void* dst, const void* src, size_t n) {
    ompx_get_host(peer, dst, src, n);
}
static inline void ompx_quiet(void) { ompx_quiet_host(); }
#endif
#endif  // !__HIPCC__ && !__CUDACC__
