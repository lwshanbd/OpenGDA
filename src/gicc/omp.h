// gicc/omp.h - GiOMP public API. GiOMP = DiOMP + GICC: a DiOMP-aligned ompx_*
// surface backed by the GICC IPC / proxy / DWQ transports. ONE include for an app.
//
//   Host (ordinary code):   ompx_init/finalize, omp_get_rank_num/num_ranks,
//                           ompx_alloc/register/free/exchange, ompx_prepare,
//                           ompx_barrier, ompx_quiet_host.
//   Smart PUT (host-side):  ompx_put(ctx, peer, ...)  -- picks IPC if the peer is
//                           same-node reachable, else the cross-node transport
//                           (proxy or DWQ, selected by the xport argument). It
//                           issues the omp target region(s) for you.
//   Advanced device-side (call INSIDE your own #pragma omp target):
//                           ompx_put_proxy / ompx_get / ompx_quiet.
//
// The HIP-compiled runtime lives in libgicc_omp; this header is safe to include
// in a -fopenmp TU AND in a -x hip host-only TU (the device inline functions are
// guarded away from the HIP compiler, which cannot handle omp declare target).
#pragma once
#include <cstddef>
#ifndef __HIPCC__
#include "gicc/platform/ofi/gicc_omp_device.hpp"   // device-side gicc::omp::put/get/quiet + DeviceCtx
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
void   ompx_quiet_host();                            // host-side drain (IPC sync + proxy/DWQ completion)

// ---- cross-node transport selector for the smart ompx_put ---------------------
// (IPC is always preferred when the peer is same-node reachable; this only picks
//  what to do for a NON-IPC / cross-node peer.)
enum ompx_xport { OMPX_PROXY = 0, OMPX_DWQ = 1 };
bool ompx_dwq_enabled();

// Host helpers used by the inline smart ompx_put and the explicit batched-DWQ
// API below. Defined in libgicc_omp (ompx_host.cpp); visible to both the
// -fopenmp app TU (which inlines the public wrappers) and the -x hip library TU
// (which defines them).
extern "C" int   ompx_ipc_reachable(int peer, int buf);   // 1 if same-node IPC-mapped
extern "C" void* ompx_peer_ipc_base(int peer, int buf);   // peer's xGMI-mapped buffer base
extern "C" void* ompx_local_base(int buf);                // our own buffer device base
extern "C" void  ompx_dwq_stage(int peer, int dst_buf, size_t dst_off,
                                int src_buf, size_t src_off, size_t bytes);
extern "C" void  ompx_dwq_stage_get_impl(int peer, int src_buf, size_t src_off,
                                         int dst_buf, size_t dst_off, size_t bytes);
extern "C" void  ompx_dwq_arm();

#ifndef __HIPCC__

// ---- advanced device-side RMA (call INSIDE your own #pragma omp target) -------
#pragma omp declare target
inline void ompx_put_proxy(gicc::DeviceCtx* ctx, int node,
                           int dst_buf, size_t dst_off,
                           int src_buf, size_t src_off, size_t bytes,
                           int lane = 0) {
    gicc::omp::put(ctx, node, dst_buf, dst_off, src_buf, src_off, bytes, lane);
}
inline void ompx_get(gicc::DeviceCtx* ctx, int node,
                     int src_buf, size_t src_off,
                     int dst_buf, size_t dst_off, size_t bytes,
                     int lane = 0) {
    gicc::omp::get(ctx, node, src_buf, src_off, dst_buf, dst_off, bytes, lane);
}
// Faster lead-thread form.  Caller guarantees only one work-item enqueues to
// this lane until the corresponding host/device quiet completes.
inline void ompx_get_single(gicc::DeviceCtx* ctx, int node,
                            int src_buf, size_t src_off,
                            int dst_buf, size_t dst_off, size_t bytes,
                            int lane = 0) {
    gicc::omp::get_single(
        ctx, node, src_buf, src_off, dst_buf, dst_off, bytes, lane);
}
inline void ompx_quiet(gicc::DeviceCtx* ctx, int lane = 0) {
    gicc::omp::quiet(ctx, lane);
}
#pragma omp end declare target

// ---- smart PUT (host-side): IPC-first, else cross-node proxy/DWQ --------------
// Issues the omp target region(s) internally, so the caller does NOT write a
// #pragma omp target. Preference order:
//   1. same-node & IPC-mapped  -> in-kernel xGMI store straight into the peer's
//      buffer (no NIC, cheapest);
//   2. otherwise               -> the cross-node transport `xport`:
//        OMPX_PROXY (default)  -> device pushes a TransferCmd; CPU proxy fi_write;
//        OMPX_DWQ              -> host pre-stages a triggered RMA descriptor and a
//                                 lead-thread device MMIO write fires the NIC.
// Completion is separate: call ompx_quiet_host() before reading the delivered data.
inline void ompx_put(gicc::DeviceCtx* ctx, int peer,
                     int dst_buf, size_t dst_off,
                     int src_buf, size_t src_off, size_t bytes,
                     ompx_xport xport = OMPX_PROXY) {
    if (ompx_ipc_reachable(peer, dst_buf)) {
        float*       d = reinterpret_cast<float*>(
                             static_cast<char*>(ompx_peer_ipc_base(peer, dst_buf)) + dst_off);
        const float* s = reinterpret_cast<const float*>(
                             static_cast<char*>(ompx_local_base(src_buf)) + src_off);
        const size_t n = bytes / sizeof(float);
        #pragma omp target teams distribute parallel for is_device_ptr(d, s) firstprivate(n)
        for (size_t i = 0; i < n; ++i) d[i] = s[i];
    } else if (xport == OMPX_DWQ) {
        ompx_dwq_stage(peer, dst_buf, dst_off, src_buf, src_off, bytes);
        ompx_dwq_arm();
        #pragma omp target is_device_ptr(ctx)
        { *(ctx->trigger_addr_) = ctx->trigger_val_; }   // lead-thread MMIO trigger
    } else {
        #pragma omp target is_device_ptr(ctx) \
                firstprivate(peer, dst_buf, dst_off, src_buf, src_off, bytes)
        { gicc::omp::put(ctx, peer, dst_buf, dst_off, src_buf, src_off, bytes); }
    }
}

// ---- explicit batched DWQ (host-side) ---------------------------------------
// These calls expose the GPU-triggered transport without requiring the LTO
// marker pass. Stage one or more operations, then call ompx_dwq_trigger once;
// ompx_quiet_host completes the batch. This is useful when an application has
// host-known transfer descriptors and wants one GPU MMIO trigger for the batch.
// GICC_HALO_DWQ=1 must be set before ompx_init().
inline void ompx_dwq_stage_put(int peer,
                               int dst_buf, size_t dst_off,
                               int src_buf, size_t src_off, size_t bytes) {
    ompx_dwq_stage(peer, dst_buf, dst_off, src_buf, src_off, bytes);
}

inline void ompx_dwq_stage_get(int peer,
                               int src_buf, size_t src_off,
                               int dst_buf, size_t dst_off, size_t bytes) {
    ompx_dwq_stage_get_impl(peer, src_buf, src_off,
                            dst_buf, dst_off, bytes);
}

inline void ompx_dwq_trigger(gicc::DeviceCtx* ctx) {
    ompx_dwq_arm();
    #pragma omp target is_device_ptr(ctx)
    { *(ctx->trigger_addr_) = ctx->trigger_val_; }
}

// ---- DWQ marker path (opt-in; app TU must compile with -fpass-plugin -foffload-lto)
// Alternative to the runtime DWQ above: the pass synthesizes the host trace.
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
