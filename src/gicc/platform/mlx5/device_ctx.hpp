/*
 * device_ctx.hpp - POD layout of the GPU-driven InfiniBand context.
 *
 * On InfiniBand the GPU does not need a trigger or a proxy: a GPU thread
 * writes the work-queue entry (WQE) into the send queue and rings the NIC
 * doorbell itself, so the transfer starts the moment the thread issues it.
 * This header describes what that thread needs -- one QpView per
 * (peer, lane) and the address book of registered buffers -- and nothing
 * else, so it can be included from host C++, CUDA and OpenMP TUs alike.
 *
 * The device-side functions that operate on these types live in
 * mlx5_device.hpp; the host-side setup lives in gpu_qp.hpp / transport.hpp.
 */
#pragma once

#include <cstddef>
#include <cstdint>

// The constants below are read by device code; under OpenMP offload that
// needs them in a declare-target region.
#if !defined(__CUDACC__) && defined(_OPENMP)
#pragma omp declare target
#endif

namespace gicc::mlx5 {

// One GPU-owned RC queue pair. Every field is a device address.
//
// The send queue is a ring of `nwqes` 64-byte basic blocks. A thread
// reserves slots with an atomic add on *resv, fills them, then waits until
// *ready reaches its first slot and publishes *ready = its last slot + 1.
// Posters publish in slot order, and only a poster inside that ordered step
// rings the doorbell (and advances *rung), which is what keeps the 16-bit
// doorbell record monotonic. A poster may publish without ringing
// (put_nbi); the next doorbell -- any later put, a flush, or the automatic
// one once half the ring is waiting -- carries its WQEs too.
//
// Only the last WQE behind each doorbell asks for a completion, and the
// completion queue is collapsed: the NIC overwrites a single CQE whose
// wqe_counter is the index of the newest completed signaled WQE, and an RC
// queue completes in order, so everything before it has completed too. With
// at most nwqes WQEs in flight that 16-bit counter identifies a unique 64-bit
// slot, so "how many WQEs have completed" is a pure function of *ready and
// that one CQE. The counter is read as a signed 16-bit distance from *ready,
// which also covers a WQE that completes before its poster has published
// *ready, so nwqes stays <= 16384.
struct QpView {
    uint8_t*           wq;        // send queue ring (host memory, mapped)
    volatile uint32_t* dbrec;     // send doorbell record (host memory, mapped)
    uint64_t*          uar;       // NIC doorbell register (MMIO, mapped)
    const uint32_t*    cqe_tail;  // bytes 60..63 of the collapsed CQE (GPU mem)
    uint64_t*          resv;      // next WQE slot to hand out   (GPU memory)
    uint64_t*          ready;     // WQE slots published so far  (GPU memory)
    uint64_t*          rung;      // WQE slots behind a doorbell (GPU memory)
    uint32_t           qpn;
    uint32_t           nwqes;     // power of two
};

// Largest single RDMA message; bigger transfers are split. The QPs are
// created with log_msg_max = 30.
constexpr uint64_t kMaxMsg = 1ull << 30;

// Send queue depth ceiling; see the completion arithmetic above.
constexpr uint32_t kMaxWqes = 16384;

} // namespace gicc::mlx5

namespace gicc {

// What a kernel or a target region needs to communicate: gicc::DeviceCtx on
// InfiniBand, the counterpart of the libfabric one in ofi/device_ctx.hpp. The
// first two fields match it, so the C view of ompx_ctx in gicc/omp.h stays
// one struct; on InfiniBand there is no trigger and both stay zero.
struct DeviceCtx {
    volatile uint64_t*  trigger_addr_ = nullptr;
    uint64_t            trigger_val_  = 0;

    // Symmetric heap (GiOMP): base of this rank's heap and its buffer index.
    void*               heap_base   = nullptr;
    int                 heap_buf    = -1;

    // Signal inbox: device address of this rank's slots and their buffer
    // index. sig_trigger is the DWQ doorbell table on libfabric; posting a
    // WQE needs none, so it is always null here.
    uint64_t*           sig_base    = nullptr;
    int                 sig_buf     = -1;
    volatile uint64_t** sig_trigger = nullptr;

    int                 my_rank = 0;
    int                 nranks  = 0;
    int                 nlanes  = 0;
    int                 nbufs   = 0;

    mlx5::QpView*       qps       = nullptr;  // [peer * nlanes + lane]
    const uint64_t*     lbuf_addr = nullptr;  // [buf]
    const uint32_t*     lbuf_lkey = nullptr;  // [buf]
    const uint64_t*     rbuf_addr = nullptr;  // [peer * nbufs + buf]
    const uint32_t*     rbuf_rkey = nullptr;  // [peer * nbufs + buf]
};

} // namespace gicc

#if !defined(__CUDACC__) && defined(_OPENMP)
#pragma omp end declare target
#endif
