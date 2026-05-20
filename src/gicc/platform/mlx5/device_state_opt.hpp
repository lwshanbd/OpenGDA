/*
 * device_state_opt.hpp - POD layout of the per-QP device-side state shared
 * between GPU kernels (which build WQEs into it) and the host Runtime
 * (which populates the descriptors at prepare()-time).
 *
 * Split out from device_opt.cuh so the runtime header can describe the
 * type without dragging in every __device__ helper / __global__ kernel
 * — those still live in device_opt.cuh and are only meaningful in TUs
 * compiled by nvcc/hipcc.
 *
 * No CUDA intrinsics here on purpose: this header must be safe to
 * include from a pure host C++ TU (e.g. mlx5/proxy/proxy_thread.cpp
 * built by g++).
 */
#pragma once

#include <cstdint>

namespace gicc::mlx5 {

struct Cqe64Opt {
    uint8_t  rsvd0[46];
    uint16_t wqe_counter;
    uint8_t  signature;
    uint8_t  op_own;
} __attribute__((packed));

struct DeviceStateOpt {
    // QP info
    uint32_t qpn;
    uint16_t nwqes;
    uint16_t nwqes_mask;

    // WQE buffer (GPU-accessible)
    void*    wqe_buf;
    uint32_t wqe_lkey;

    // Doorbell record (GPU-writable)
    volatile uint32_t* dbrec;

    // BlueFlame register (GPU-writable, 64-bit)
    volatile uint64_t* bf_reg;

    // Producer indices (nvshmem-style separated indices)
    volatile uint64_t* resv_head;
    volatile uint64_t* ready_head;
    volatile uint64_t* prod_idx;

    // CQ for completion
    volatile Cqe64Opt* cqe;
    uint32_t           ncqes;
    uint32_t           ncqes_mask;
    volatile uint64_t* cq_cons_idx;
    volatile uint32_t* cq_dbrec;

    // Remote peer info (set by Runtime::prepare(); also reachable per-call
    // via the device-side put/get overloads that take explicit raddr/rkey).
    uint64_t remote_addr;
    uint32_t remote_rkey;

    // Completion tracking
    volatile uint64_t* num_completions;

    // Batching configuration
    uint32_t batch_size;
    uint32_t batch_mask;
};

} // namespace gicc::mlx5
