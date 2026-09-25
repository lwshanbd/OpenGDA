/*
 * gda_device.hpp - GPU threads post InfiniBand work requests themselves.
 *
 * A put is: reserve send-queue slots, write the WQEs, ring the doorbell. No
 * host thread and no trigger is involved, so the NIC starts the transfer
 * as soon as the issuing thread rings. Completion is read back from a
 * collapsed CQE (see gda_types.hpp).
 *
 * One source serves two compilers:
 *   - CUDA (nvcc, clang -x cuda): the functions are __device__, usable from
 *     any kernel;
 *   - OpenMP offload (clang -fopenmp --offload-arch=sm_XX): the functions
 *     are `declare target`, usable inside `#pragma omp target`.
 * Everything that touches the NIC is PTX inline assembly, which both accept.
 * The OpenMP host pass also compiles declare-target bodies; there the
 * primitives trap, because nothing on the host may call them.
 */
#pragma once

#include "gicc/platform/mlx5/gda_types.hpp"

#include <cstdint>
#include <cstdio>

#if defined(__CUDACC__)
  #define GICC_GDA_FN __device__ __forceinline__
#else
  #define GICC_GDA_FN inline
#endif

#if defined(__CUDA_ARCH__) || defined(__NVPTX__)
  #define GICC_GDA_PTX 1
#endif

#if !defined(__CUDACC__) && defined(_OPENMP)
#pragma omp declare target
#endif

namespace gicc::mlx5::gda {

// mlx5 PRM encodings used below.
constexpr uint32_t kOpRdmaWrite = 0x08;
constexpr uint32_t kOpRdmaRead  = 0x10;
constexpr uint32_t kCtrlCqUpdate = 0x08;         // fm_ce_se: CQE on completion
constexpr uint32_t kInlineSeg    = 0x80000000u;  // data segment is inline
constexpr uint32_t kCqeInvalid   = 0xF;          // opcode of a never-written CQE
constexpr uint32_t kCqeReqErr    = 0xD;
constexpr uint32_t kCqeRespErr   = 0xE;

// A QP that makes no progress for this long is reported and the kernel
// traps, instead of hanging a job silently.
constexpr uint64_t kStallNs = 20ull * 1000 * 1000 * 1000;

//==============================================================================
// PTX primitives
//==============================================================================

GICC_GDA_FN uint32_t be32(uint32_t x) {
#ifdef GICC_GDA_PTX
    uint32_t r;
    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(r) : "r"(x));
    return r;
#else
    return __builtin_bswap32(x);
#endif
}

GICC_GDA_FN void trap() {
#ifdef GICC_GDA_PTX
    asm volatile("trap;");
#elif !defined(__CUDACC__)
    __builtin_trap();
#endif
}

GICC_GDA_FN void st_v4(uint32_t* p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
#ifdef GICC_GDA_PTX
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
#else
    (void)p; (void)a; (void)b; (void)c; (void)d; trap();
#endif
}

GICC_GDA_FN void fence_sys() {
#ifdef GICC_GDA_PTX
    asm volatile("fence.sc.sys;" ::: "memory");
#endif
}

GICC_GDA_FN void st_sys_u32(volatile uint32_t* p, uint32_t v) {
#ifdef GICC_GDA_PTX
    asm volatile("st.relaxed.sys.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
#else
    (void)p; (void)v; trap();
#endif
}

// Doorbell register: an MMIO store, never merged or cached.
GICC_GDA_FN void st_mmio_u64(uint64_t* p, uint64_t v) {
#ifdef GICC_GDA_PTX
    asm volatile("st.relaxed.sys.global.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
#else
    (void)p; (void)v; trap();
#endif
}

// Written by the NIC, so read at system scope (never from a stale L1 line).
GICC_GDA_FN uint32_t ld_sys_u32(const uint32_t* p) {
#ifdef GICC_GDA_PTX
    uint32_t r;
    asm volatile("ld.relaxed.sys.global.b32 %0, [%1];" : "=r"(r) : "l"(p) : "memory");
    return r;
#else
    (void)p; trap(); return 0;
#endif
}

GICC_GDA_FN uint64_t ld_acquire_u64(const uint64_t* p) {
#ifdef GICC_GDA_PTX
    uint64_t r;
    asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(r) : "l"(p) : "memory");
    return r;
#else
    (void)p; trap(); return 0;
#endif
}

GICC_GDA_FN void st_release_u64(uint64_t* p, uint64_t v) {
#ifdef GICC_GDA_PTX
    asm volatile("st.release.gpu.global.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
#else
    (void)p; (void)v; trap();
#endif
}

GICC_GDA_FN uint64_t atomic_add_u64(uint64_t* p, uint64_t v) {
#ifdef GICC_GDA_PTX
    uint64_t r;
    asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], %2;"
                 : "=l"(r) : "l"(p), "l"(v) : "memory");
    return r;
#else
    (void)p; (void)v; trap(); return 0;
#endif
}

GICC_GDA_FN void backoff() {
#ifdef GICC_GDA_PTX
    asm volatile("nanosleep.u32 32;");
#endif
}

GICC_GDA_FN uint64_t now_ns() {
#ifdef GICC_GDA_PTX
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#else
    return 0;
#endif
}

//==============================================================================
// Send queue
//==============================================================================

GICC_GDA_FN uint32_t* wqe_at(const GdaQp* qp, uint64_t slot) {
    return reinterpret_cast<uint32_t*>(
        qp->wq + ((slot & (uint64_t)(qp->nwqes - 1)) << 6));
}

// WQEs completed so far, from the collapsed CQE and the rung count `ready`.
// The CQE's wqe_counter is the 16-bit index of the newest completed WQE; its
// signed distance from ready-1 recovers the full index (gda_types.hpp).
GICC_GDA_FN uint64_t completed(const GdaQp* qp, uint64_t ready, uint32_t tail) {
    const uint32_t opcode = tail >> 28;
    if (opcode == kCqeInvalid) return 0;                 // nothing completed yet
    if (opcode == kCqeReqErr || opcode == kCqeRespErr) {
        // Bytes 60..63 hold only the counter; the syndrome sits at byte 55.
        const uint32_t synd = ld_sys_u32(qp->cqe_tail - 2) >> 24;
        printf("[gicc] IB completion error on QP 0x%x: opcode 0x%x syndrome 0x%x "
               "wqe %u\n", qp->qpn, opcode, synd,
               (unsigned)((be32(tail) >> 16) & 0xffff));
        trap();
    }
    const uint16_t counter = (uint16_t)(be32(tail) >> 16);
    const int16_t  ahead   = (int16_t)(uint16_t)(counter - (uint16_t)(ready - 1));
    return (uint64_t)((int64_t)ready + ahead);
}

// Spin until at least `target` WQEs on this QP have completed.
GICC_GDA_FN void wait_completed(const GdaQp* qp, uint64_t target) {
    uint64_t start = 0;
    for (;;) {
        // CQE first, then ready: a CQE never names a WQE the poster has not
        // reserved, and `ahead` absorbs one rung but not yet published.
        const uint32_t tail  = ld_sys_u32(qp->cqe_tail);
        const uint64_t ready = ld_acquire_u64(qp->ready);
        if (ready >= target && completed(qp, ready, tail) >= target) return;
        backoff();
        if (start == 0) {
            start = now_ns();
        } else if (now_ns() - start > kStallNs) {
            printf("[gicc] QP 0x%x stalled: waiting for %llu completions, "
                   "%llu rung\n", qp->qpn, (unsigned long long)target,
                   (unsigned long long)ready);
            trap();
        }
    }
}

// Reserve `n` consecutive slots and wait until the ring has room for them.
GICC_GDA_FN uint64_t reserve(GdaQp* qp, uint32_t n) {
    const uint64_t first = atomic_add_u64(qp->resv, n);
    const uint64_t end   = first + n;
    if (end > qp->nwqes) wait_completed(qp, end - qp->nwqes);
    return first;
}

// Ring the doorbell for slots [first, first+n). Posters ring in slot order:
// each waits for the previous one to publish `ready`, so the doorbell record
// only ever moves forward.
GICC_GDA_FN void ring(GdaQp* qp, uint64_t first, uint32_t n) {
    while (ld_acquire_u64(qp->ready) != first) backoff();
    const uint32_t end = (uint32_t)(first + n) & 0xffff;
    fence_sys();                               // WQEs before the doorbell record
    st_sys_u32(qp->dbrec, be32(end));
    fence_sys();                               // record before the doorbell
    const uint64_t db = ((uint64_t)be32(qp->qpn << 8) << 32) | be32(end << 8);
    st_mmio_u64(qp->uar, db);
    st_release_u64(qp->ready, first + n);
}

// ctrl | raddr | data, 48 bytes = 3 data segments, completion requested.
GICC_GDA_FN void write_rdma(const GdaQp* qp, uint64_t slot, uint32_t opcode,
                            uint64_t laddr, uint32_t lkey,
                            uint64_t raddr, uint32_t rkey, uint32_t bytes) {
    uint32_t* w = wqe_at(qp, slot);
    st_v4(w + 0, be32(((uint32_t)(slot & 0xffff) << 8) | opcode),
                 be32((qp->qpn << 8) | 3), kCtrlCqUpdate << 24, 0);
    st_v4(w + 4, be32((uint32_t)(raddr >> 32)), be32((uint32_t)raddr), be32(rkey), 0);
    st_v4(w + 8, be32(bytes), be32(lkey),
                 be32((uint32_t)(laddr >> 32)), be32((uint32_t)laddr));
}

// RDMA WRITE of one 8-byte value carried inside the WQE: ctrl | raddr |
// inline header + value = 44 bytes, 3 data segments. Used for signals, so a
// signal needs no source buffer.
GICC_GDA_FN void write_inline_u64(const GdaQp* qp, uint64_t slot,
                                  uint64_t raddr, uint32_t rkey, uint64_t value) {
    uint32_t* w = wqe_at(qp, slot);
    st_v4(w + 0, be32(((uint32_t)(slot & 0xffff) << 8) | kOpRdmaWrite),
                 be32((qp->qpn << 8) | 3), kCtrlCqUpdate << 24, 0);
    st_v4(w + 4, be32((uint32_t)(raddr >> 32)), be32((uint32_t)raddr), be32(rkey), 0);
    // The value goes out in memory order, so the peer reads it as the same
    // little-endian integer.
    st_v4(w + 8, be32(kInlineSeg | 8), (uint32_t)value, (uint32_t)(value >> 32), 0);
}

GICC_GDA_FN uint32_t chunks(size_t bytes) {
    return (uint32_t)((bytes + kGdaMaxMsg - 1) / kGdaMaxMsg);
}

// Write the WQEs of one transfer, split at kGdaMaxMsg, starting at `slot`.
GICC_GDA_FN void write_transfer(const GdaQp* qp, uint64_t slot, uint32_t opcode,
                                uint64_t laddr, uint32_t lkey,
                                uint64_t raddr, uint32_t rkey, size_t bytes) {
    for (size_t done = 0; done < bytes; done += kGdaMaxMsg, ++slot) {
        const size_t len = bytes - done < kGdaMaxMsg ? bytes - done : kGdaMaxMsg;
        write_rdma(qp, slot, opcode, laddr + done, lkey, raddr + done, rkey,
                   (uint32_t)len);
    }
}

//==============================================================================
// Context-level operations
//==============================================================================

GICC_GDA_FN GdaQp* qp_of(const GdaCtx* c, int peer, int lane) {
    const int l = lane <= 0 ? 0 : lane % c->nlanes;
    return &c->qps[peer * c->nlanes + l];
}

// Local (src_buf, src_off) -> peer's (dst_buf, dst_off).
GICC_GDA_FN void put(const GdaCtx* c, int peer, int dst_buf, size_t dst_off,
                     int src_buf, size_t src_off, size_t bytes, int lane = 0) {
    if (bytes == 0) return;
    GdaQp* qp = qp_of(c, peer, lane);
    const int r = peer * c->nbufs + dst_buf;
    const uint32_t n = chunks(bytes);
    const uint64_t first = reserve(qp, n);
    write_transfer(qp, first, kOpRdmaWrite,
                   c->lbuf_addr[src_buf] + src_off, c->lbuf_lkey[src_buf],
                   c->rbuf_addr[r] + dst_off, c->rbuf_rkey[r], bytes);
    ring(qp, first, n);
}

// Peer's (src_buf, src_off) -> local (dst_buf, dst_off).
GICC_GDA_FN void get(const GdaCtx* c, int peer, int src_buf, size_t src_off,
                     int dst_buf, size_t dst_off, size_t bytes, int lane = 0) {
    if (bytes == 0) return;
    GdaQp* qp = qp_of(c, peer, lane);
    const int r = peer * c->nbufs + src_buf;
    const uint32_t n = chunks(bytes);
    const uint64_t first = reserve(qp, n);
    write_transfer(qp, first, kOpRdmaRead,
                   c->lbuf_addr[dst_buf] + dst_off, c->lbuf_lkey[dst_buf],
                   c->rbuf_addr[r] + src_off, c->rbuf_rkey[r], bytes);
    ring(qp, first, n);
}

// Payload, then `value` into the peer's signal slot at byte offset sig_off,
// posted together on one QP behind one doorbell. An RC QP executes RDMA
// writes in order, so the signal cannot land before the payload.
GICC_GDA_FN void put_signal(const GdaCtx* c, int peer, int dst_buf, size_t dst_off,
                            int src_buf, size_t src_off, size_t bytes,
                            size_t sig_off, uint64_t value, int lane = 0) {
    GdaQp* qp = qp_of(c, peer, lane);
    const int r = peer * c->nbufs + dst_buf;
    const int s = peer * c->nbufs + c->sig_buf;
    const uint32_t n = chunks(bytes) + 1;
    const uint64_t first = reserve(qp, n);
    write_transfer(qp, first, kOpRdmaWrite,
                   c->lbuf_addr[src_buf] + src_off, c->lbuf_lkey[src_buf],
                   c->rbuf_addr[r] + dst_off, c->rbuf_rkey[r], bytes);
    write_inline_u64(qp, first + n - 1, c->rbuf_addr[s] + sig_off,
                     c->rbuf_rkey[s], value);
    ring(qp, first, n);
}

// Wait for everything rung on `lane` (to every peer) to complete. Local
// buffers are then reusable and data fetched by get() is visible.
GICC_GDA_FN void quiet(const GdaCtx* c, int lane = 0) {
    for (int peer = 0; peer < c->nranks; ++peer) {
        const GdaQp* qp = qp_of(c, peer, lane);
        const uint64_t target = ld_acquire_u64(qp->ready);
        if (target) wait_completed(qp, target);
    }
    fence_sys();
}

GICC_GDA_FN void quiet_all(const GdaCtx* c) {
    for (int lane = 0; lane < c->nlanes; ++lane) quiet(c, lane);
}

GICC_GDA_FN uint64_t signal_read(const GdaCtx* c, int sig) {
#ifdef GICC_GDA_PTX
    uint64_t v;
    asm volatile("ld.acquire.sys.global.b64 %0, [%1];"
                 : "=l"(v) : "l"(c->sig_base + sig) : "memory");
    return v;
#else
    (void)c; (void)sig; trap(); return 0;
#endif
}

// Returns once slot `sig` holds a value >= ge; the payload that value
// announces is then visible to the caller.
GICC_GDA_FN void signal_wait(const GdaCtx* c, int sig, uint64_t ge) {
    while (signal_read(c, sig) < ge) backoff();
}

} // namespace gicc::mlx5::gda

#if !defined(__CUDACC__) && defined(_OPENMP)
#pragma omp end declare target
#endif

#if defined(__CUDACC__)
//==============================================================================
// Kernel-facing API: the same buffer-index form as the libfabric backend's
// gicc::put / get / quiet (ofi_device.cuh), taking the context returned by
// Runtime::prepare(). There is nothing to trigger, so flush() is empty.
//==============================================================================
namespace gicc {

__device__ __forceinline__
void put(mlx5::GdaCtx* ctx, int target_rank, int dst_buf, size_t dst_offset,
         int src_buf, size_t src_offset, size_t size, int lane = 0) {
    mlx5::gda::put(ctx, target_rank, dst_buf, dst_offset, src_buf, src_offset,
                   size, lane);
}

__device__ __forceinline__
void get(mlx5::GdaCtx* ctx, int source_rank, int src_buf, size_t src_offset,
         int dst_buf, size_t dst_offset, size_t size, int lane = 0) {
    mlx5::gda::get(ctx, source_rank, src_buf, src_offset, dst_buf, dst_offset,
                   size, lane);
}

// Payload, then `value` into the peer's signal slot `sig`.
__device__ __forceinline__
void put_signal(mlx5::GdaCtx* ctx, int target_rank, int dst_buf, size_t dst_offset,
                int src_buf, size_t src_offset, size_t size,
                int sig, uint64_t value, int lane = 0) {
    mlx5::gda::put_signal(ctx, target_rank, dst_buf, dst_offset, src_buf, src_offset,
                          size, (size_t)sig * sizeof(uint64_t), value, lane);
}

__device__ __forceinline__
void quiet(mlx5::GdaCtx* ctx, int lane = 0) { mlx5::gda::quiet(ctx, lane); }

__device__ __forceinline__
void flush(mlx5::GdaCtx*) {}

__device__ __forceinline__
uint64_t signal_read(mlx5::GdaCtx* ctx, int sig) {
    return mlx5::gda::signal_read(ctx, sig);
}

__device__ __forceinline__
void signal_wait(mlx5::GdaCtx* ctx, int sig, uint64_t ge) {
    mlx5::gda::signal_wait(ctx, sig, ge);
}

} // namespace gicc
#endif
