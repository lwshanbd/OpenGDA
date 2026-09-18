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
typedef struct ompx_ctx ompx_ctx;                  // opaque handle in C
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

// ---- device-side ------------------------------------------------------------
ompx_ctx* ompx_prepare(void);   // per-step device context (host call)

#ifdef __cplusplus
}  // extern "C"
#endif

#if defined(__cplusplus) && !defined(__HIPCC__) && !defined(__CUDACC__)
// Device-side RMA: call inside your own `#pragma omp target is_device_ptr(ctx)`.
// Addresses are translated to heap offsets in-kernel, so no handle is needed.
#pragma omp declare target
inline size_t ompx_heap_offset(ompx_ctx* ctx, const void* addr) {
    return (size_t)((const char*)addr - (const char*)ctx->heap_base);
}

inline void ompx_put_dev(ompx_ctx* ctx, int peer,
                         void* dst, const void* src, size_t bytes) {
    gicc::omp::put(ctx, peer,
                   ctx->heap_buf, ompx_heap_offset(ctx, dst),
                   ctx->heap_buf, ompx_heap_offset(ctx, src), bytes);
}

inline void ompx_get_dev(ompx_ctx* ctx, int peer,
                         void* dst, const void* src, size_t bytes) {
    gicc::omp::get(ctx, peer,
                   ctx->heap_buf, ompx_heap_offset(ctx, src),
                   ctx->heap_buf, ompx_heap_offset(ctx, dst), bytes);
}

inline void ompx_quiet_dev(ompx_ctx* ctx) { gicc::omp::quiet(ctx); }
#pragma omp end declare target
#endif
