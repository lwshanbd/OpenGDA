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
//   Completion: ompx_quiet, ompx_fence, ompx_barrier
//   Device-side (call INSIDE your own #pragma omp target):
//               ompx_prepare, ompx_put_dev, ompx_quiet_dev
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

// ---- cross-node transport ---------------------------------------------------
// IPC is always preferred for a same-node peer; this selects what a cross-node
// put uses. OMPX_AUTO honours the GICC_HALO_DWQ environment variable.
typedef enum { OMPX_AUTO = 0, OMPX_PROXY = 1, OMPX_DWQ = 2 } ompx_xport;

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
size_t ompx_heap_size(void);                       // configured heap bytes

// Escape hatch for the compiler path: the DWQ markers below take the buffer
// index and heap offset the LTO pass analyses, so an application using them
// needs to name an allocation that way. Ordinary code never calls these.
int    ompx_heap_index(void);                      // address-book index of the heap
size_t ompx_heap_offset_of(const void* addr);      // heap offset of an address

// ---- data movement ----------------------------------------------------------
// Addresses are local addresses of symmetric objects. `dst` names the object on
// `peer`; `src` names our own copy.
void* ompx_peer_ptr(int peer, const void* addr);   // NULL unless same-node
void  ompx_put(int peer, void* dst, const void* src, size_t bytes);
void  ompx_get(int peer, void* dst, const void* src, size_t bytes);
void  ompx_set_transport(ompx_xport xport);

// ---- completion -------------------------------------------------------------
void ompx_quiet(void);      // drain the transfers this rank issued
void ompx_barrier(void);
void ompx_fence(void);      // quiet + barrier: call before reading peer writes

// ---- explicit batched DWQ ---------------------------------------------------
// ompx_put already picks the DWQ when that transport is selected. These calls
// expose the two halves separately: stage any number of operations, then fire
// them all with one GPU MMIO write. Requires GICC_HALO_DWQ=1 before ompx_init.
int  ompx_dwq_enabled(void);
void ompx_dwq_stage_put(int peer, void* dst, const void* src, size_t bytes);
void ompx_dwq_stage_get(int peer, void* dst, const void* src, size_t bytes);
void ompx_dwq_trigger(void);    // arm + fire the staged batch
// Arm without firing, for a kernel that fires the trigger itself as part of the
// work it is already doing: stage, arm, then write *(ctx->trigger_addr_) =
// ctx->trigger_val_ from one lead thread inside your own target region.
void ompx_dwq_arm(void);

// ---- device-side ------------------------------------------------------------
ompx_ctx* ompx_prepare(void);   // per-step device context (host call)

#ifdef __cplusplus
}  // extern "C"
#endif

#if !defined(__cplusplus)
// Fire the armed DWQ batch from inside the application's own target region.
// Pair with ompx_dwq_arm() on the host; one thread should call it.
#pragma omp declare target
static inline void ompx_dwq_fire_dev(ompx_ctx* ctx) {
    *(ctx->trigger_addr_) = ctx->trigger_val_;
}
#pragma omp end declare target
#endif

#if defined(__cplusplus) && !defined(__HIPCC__) && !defined(__CUDACC__)
// Device-side RMA: call inside your own `#pragma omp target is_device_ptr(ctx)`.
// Addresses are translated to heap offsets in-kernel, so no handle is needed.
#pragma omp declare target
inline size_t ompx_heap_offset(ompx_ctx* ctx, const void* addr) {
    return (size_t)((const char*)addr - (const char*)ctx->heap_base);
}

// `lane` picks one of the proxy rings, each drained by its own worker, so two
// transfers issued on different lanes progress independently.
inline void ompx_put_dev(ompx_ctx* ctx, int peer,
                         void* dst, const void* src, size_t bytes,
                         int lane = 0) {
    gicc::omp::put(ctx, peer,
                   ctx->heap_buf, ompx_heap_offset(ctx, dst),
                   ctx->heap_buf, ompx_heap_offset(ctx, src), bytes, lane);
}

inline void ompx_get_dev(ompx_ctx* ctx, int peer,
                         void* dst, const void* src, size_t bytes,
                         int lane = 0) {
    gicc::omp::get(ctx, peer,
                   ctx->heap_buf, ompx_heap_offset(ctx, src),
                   ctx->heap_buf, ompx_heap_offset(ctx, dst), bytes, lane);
}

// Lead-thread form: the caller guarantees a single work-item enqueues on this
// lane until the matching quiet, which skips the contended ring CAS.
inline void ompx_get_dev_single(ompx_ctx* ctx, int peer,
                                void* dst, const void* src, size_t bytes,
                                int lane = 0) {
    gicc::omp::get_single(ctx, peer,
                          ctx->heap_buf, ompx_heap_offset(ctx, src),
                          ctx->heap_buf, ompx_heap_offset(ctx, dst), bytes, lane);
}

inline void ompx_quiet_dev(ompx_ctx* ctx, int lane = 0) {
    gicc::omp::quiet(ctx, lane);
}

// Fire the armed DWQ batch from inside a region the application is launching
// anyway: one lead thread writes the NIC's trigger counter. Pair it with
// ompx_dwq_arm() on the host. ompx_dwq_trigger() is the convenience form that
// launches its own kernel; this one costs no extra launch, which matters when
// the trigger sits inside a measured phase.
inline void ompx_dwq_fire_dev(ompx_ctx* ctx) {
    *(ctx->trigger_addr_) = ctx->trigger_val_;
}
#pragma omp end declare target

// ---- DWQ marker path (opt-in) -----------------------------------------------
// Alternative to the runtime DWQ above: the LTO pass recognizes these markers
// and synthesizes the host trace, so the app TU must be compiled with
// -fpass-plugin and -foffload-lto. The markers keep the buffer-index form the
// pass analyses; they are a compiler interface, not the user-facing API.
#ifdef GIOMP_ENABLE_DWQ
#include "examples/omp/gicc_omp_dwq.hpp"
#pragma omp declare target
inline void ompx_dwq_put_dev(ompx_ctx* ctx, int node,
                             int dst_buf, size_t dst_off,
                             int src_buf, size_t src_off, size_t bytes) {
    gicc::omp_dwq::put(ctx, node, dst_buf, dst_off, src_buf, src_off, bytes);
}
inline void ompx_dwq_flush_dev(ompx_ctx* ctx) { gicc::omp_dwq::flush(ctx); }
#pragma omp end declare target
#endif
#endif
