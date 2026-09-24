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
//   Movement:   ompx_peer_ptr (same-node direct access), ompx_put, ompx_get
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

#if defined(__cplusplus) && !defined(__HIPCC__) && !defined(__CUDACC__)
#include "gicc/platform/ofi/gicc_omp_device.hpp"   // gicc::omp::put/get/quiet
#elif defined(__cplusplus)
#include "gicc/platform/ofi/device_ctx.hpp"        // gicc::DeviceCtx only
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

// ---- explicit batched DWQ ---------------------------------------------------
// ompx_put stages; nothing moves until a trigger is pulled. ompx_quiet pulls it
// for you, so an application only calls ompx_trigger when it wants the transfer
// to start earlier than that -- typically before a kernel it wants to overlap.
// No-op under the CPU proxy, which is already in flight on issue.
void ompx_trigger_host(void);

// ---- device-side ------------------------------------------------------------
ompx_ctx* ompx_prepare_ctx(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#if !defined(__HIPCC__) && !defined(__CUDACC__)
// ---- one name, host or device ----------------------------------------------
// These are `declare target`, so the same call works in ordinary code and
// inside `#pragma omp target`. On the device they use the context published by
// ompx_prepare; on the host they call the runtime. ompx_trigger works from C
// and C++ alike; ompx_put/get/quiet have no C device body, so a C application
// issues those from the host and only pulls the trigger on the device.
#pragma omp declare target
static ompx_ctx* ompx__ctx = 0;
#pragma omp end declare target

// The runtime's DeviceCtx is host-pinned and mapped and prepare() returns the
// same device pointer for the life of the runtime, so this publishes it once;
// later calls only update contents the device already sees.
static inline ompx_ctx* ompx_prepare(void) {
    ompx_ctx* c = ompx_prepare_ctx();
    if (ompx__ctx != c) {
        ompx__ctx = c;
        #pragma omp target is_device_ptr(c)
        { ompx__ctx = c; }
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

inline void ompx_get(int peer, void* dst, const void* src, size_t bytes,
                     int lane = 0) {
#if defined(__AMDGCN__) || defined(__NVPTX__)
    ompx_ctx* c = ompx__ctx;
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
