/*
 * gda_types.hpp - POD layout of the GPU-driven InfiniBand context.
 *
 * On InfiniBand the GPU does not need a trigger or a proxy: a GPU thread
 * writes the work-queue entry (WQE) into the send queue and rings the NIC
 * doorbell itself, so the transfer starts the moment the thread issues it.
 * This header describes what that thread needs -- one GdaQp per
 * (peer, lane) and the address book of registered buffers -- and nothing
 * else, so it can be included from host C++, CUDA and OpenMP TUs alike.
 *
 * The device-side functions that operate on these types live in
 * gda_device.hpp; the host-side setup lives in gda_qp.hpp / gda_engine.hpp.
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
// *ready reaches its first slot, rings the doorbell and publishes
// *ready = its last slot + 1. Doorbells are therefore rung in slot order,
// which is what keeps the 16-bit doorbell record monotonic.
//
// Every WQE asks for a completion, and the completion queue is collapsed:
// the NIC overwrites a single CQE whose wqe_counter is the index of the
// newest completed WQE. With at most nwqes (<= 32768) WQEs in flight that
// 16-bit counter identifies a unique 64-bit slot, so "how many WQEs have
// completed" is a pure function of *ready and that one CQE. The counter is
// read as a signed 16-bit distance from *ready, which also covers a WQE that
// completes before its poster has published *ready, so nwqes stays <= 16384.
struct GdaQp {
    uint8_t*           wq;        // send queue ring (host memory, mapped)
    volatile uint32_t* dbrec;     // send doorbell record (host memory, mapped)
    uint64_t*          uar;       // NIC doorbell register (MMIO, mapped)
    const uint32_t*    cqe_tail;  // bytes 60..63 of the collapsed CQE (GPU mem)
    uint64_t*          resv;      // next WQE slot to hand out   (GPU memory)
    uint64_t*          ready;     // WQE slots rung so far       (GPU memory)
    uint32_t           qpn;
    uint32_t           nwqes;     // power of two
};

// What a kernel or a target region needs to communicate. The first two
// fields mirror gicc::DeviceCtx on the libfabric backend so the C view of
// ompx_ctx in gicc/omp.h stays one struct; on InfiniBand there is no trigger
// and both stay zero.
struct GdaCtx {
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

    GdaQp*              qps       = nullptr;  // [peer * nlanes + lane]
    const uint64_t*     lbuf_addr = nullptr;  // [buf]
    const uint32_t*     lbuf_lkey = nullptr;  // [buf]
    const uint64_t*     rbuf_addr = nullptr;  // [peer * nbufs + buf]
    const uint32_t*     rbuf_rkey = nullptr;  // [peer * nbufs + buf]
};

// Largest single RDMA message; bigger transfers are split. The QPs are
// created with log_msg_max = 30.
constexpr uint64_t kGdaMaxMsg = 1ull << 30;

// Send queue depth ceiling; see the completion arithmetic above.
constexpr uint32_t kGdaMaxWqes = 16384;

} // namespace gicc::mlx5

#if !defined(__CUDACC__) && defined(_OPENMP)
#pragma omp end declare target
#endif
