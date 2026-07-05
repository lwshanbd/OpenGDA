// gicc/omp.h - GiOMP public API. GiOMP = DiOMP + GICC: a DiOMP-aligned ompx_*
// surface backed by the GICC proxy/IPC/DWQ transport. ONE include for an app.
//
//   Host (ordinary code):   ompx_init/finalize, omp_get_rank_num/num_ranks,
//                           ompx_alloc/register/free/exchange, ompx_prepare,
//                           ompx_barrier, ompx_quiet_host.
//   Device (#pragma omp target): ompx_put/ompx_get/ompx_quiet.
//   DWQ (needs -fpass-plugin, guarded): ompx_dwq_put/ompx_dwq_flush.
//
// The HIP-compiled runtime lives in libgicc_omp; this header is safe to include
// in a -fopenmp TU AND in a -x hip host-only TU (the device inline functions are
// guarded away from the HIP compiler, which cannot handle omp declare target).
#pragma once
#include <cstddef>
#ifndef __HIPCC__
#include "gicc/platform/ofi/gicc_omp_device.hpp"   // device-side gicc::omp::put/get/quiet + put_auto + DeviceCtx
#else
#include "gicc/platform/ofi/device_ctx.hpp"         // just gicc::DeviceCtx (HIP-free), enough for host API
#endif

// ---- host-side buffer handle: bridges DiOMP pointer model + GICC index model
struct ompx_buffer { void* ptr; int index; size_t bytes; };

// ---- host-side control API (defined in libgicc_omp / ompx_host.cpp) ----------
void   ompx_init();                                  // MPI + Runtime + device select + omp_set_default_device
void   ompx_finalize();
int    omp_get_rank_num();                           // reused DiOMP name
int    omp_get_num_ranks();                          // reused DiOMP name
ompx_buffer ompx_alloc(size_t bytes);                // IPC-capable alloc + auto-register
int    ompx_register(void* dev_ptr, size_t bytes);   // register an already-owned buffer -> index
void   ompx_free(ompx_buffer b);
void   ompx_exchange();                              // exchange RMA address book (after all registrations)
gicc::DeviceCtx* ompx_prepare();                     // per-iteration device ctx (stream-like)
void   ompx_barrier();                               // reused DiOMP name
void   ompx_quiet_host();                            // host-side drain (== runtime reset())

// ---- device-side RMA (call INSIDE #pragma omp target) ------------------------
// Not visible to the HIP compiler (-x hip host TU): HIP ignores omp declare target
// and cannot compile the gicc::omp::* device functions that gicc_omp_device.hpp
// defines. Only the OpenMP offload toolchain needs these wrappers.
#ifndef __HIPCC__
#pragma omp declare target
inline void ompx_put(gicc::DeviceCtx* ctx, int node,
                     int dst_buf, size_t dst_off,
                     int src_buf, size_t src_off, size_t bytes) {
    gicc::omp::put(ctx, node, dst_buf, dst_off, src_buf, src_off, bytes);
}
inline void ompx_get(gicc::DeviceCtx* ctx, int node,
                     int src_buf, size_t src_off,
                     int dst_buf, size_t dst_off, size_t bytes) {
    gicc::omp::get(ctx, node, src_buf, src_off, dst_buf, dst_off, bytes);
}
inline void ompx_quiet(gicc::DeviceCtx* ctx) { gicc::omp::quiet(ctx); }
#pragma omp end declare target

// ---- DWQ path (opt-in; app TU must compile with -fpass-plugin -foffload-lto) --
#ifdef GIOMP_ENABLE_DWQ
#include "examples/omp/gicc_omp_dwq.hpp"             // gicc::omp_dwq::put/flush markers
#pragma omp declare target
inline void ompx_dwq_put(gicc::DeviceCtx* ctx, int node,
                         int dst_buf, size_t dst_off,
                         int src_buf, size_t src_off, size_t bytes) {
    gicc::omp_dwq::put(ctx, node, dst_buf, dst_off, src_buf, src_off, bytes);
}
inline void ompx_dwq_flush(gicc::DeviceCtx* ctx) { gicc::omp_dwq::flush(ctx); }
#pragma omp end declare target
#endif
#endif  // !__HIPCC__
