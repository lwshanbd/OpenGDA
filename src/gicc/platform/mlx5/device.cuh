/**
 * gda_device.cuh - GPU device functions for GPU-triggered RDMA
 *
 * Provides device functions that allow GPU kernels to:
 *   - Build RDMA WQEs directly in GPU-accessible memory
 *   - Ring doorbells to trigger NIC operations
 *   - Poll CQ for completions
 *
 * This is the core of GPU-initiated RDMA - no CPU involvement needed.
 */
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace gicc::mlx5 {

// MLX5 constants
#define MLX5_SEND_WQE_BB 64
#define MLX5_SEND_WQE_SHIFT 6
#define MLX5_CQE_SIZE 64

// WQE opcodes
#define MLX5_OPCODE_RDMA_WRITE 0x08
#define MLX5_OPCODE_RDMA_READ  0x10
#define MLX5_OPCODE_NOP        0x00

// WQE control flags
#define MLX5_WQE_CTRL_CQ_UPDATE (1 << 2)
#define MLX5_WQE_CTRL_FENCE     (1 << 5)

// CQE ownership bit
#define MLX5_CQE_OWNER_MASK 1

// Byte swap for big-endian wire format
__device__ __forceinline__ uint32_t gda_htobe32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

__device__ __forceinline__ uint64_t gda_htobe64(uint64_t x) {
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t lo = (uint32_t)x;
    return ((uint64_t)gda_htobe32(lo) << 32) | gda_htobe32(hi);
}

__device__ __forceinline__ uint16_t gda_htobe16(uint16_t x) {
    return (x >> 8) | (x << 8);
}

__device__ __forceinline__ uint16_t gda_betoh16(uint16_t x) {
    return gda_htobe16(x);
}

// MLX5 WQE Control Segment (16 bytes)
struct GdaCtrlSeg {
    uint32_t opmod_idx_opcode;
    uint32_t qpn_ds;
    uint8_t  signature;
    uint8_t  rsvd[2];
    uint8_t  fm_ce_se;
    uint32_t imm;
} __attribute__((packed));

// MLX5 Remote Address Segment (16 bytes)
struct GdaRaddrSeg {
    uint64_t raddr;
    uint32_t rkey;
    uint32_t reserved;
} __attribute__((packed));

// MLX5 Data Segment (16 bytes)
struct GdaDataSeg {
    uint32_t byte_count;
    uint32_t lkey;
    uint64_t addr;
} __attribute__((packed));

// MLX5 CQE structure (64 bytes) - simplified
struct GdaCqe64 {
    uint8_t  rsvd0[46];
    uint16_t wqe_counter;
    uint8_t  signature;
    uint8_t  op_own;
} __attribute__((packed));

// GPU-accessible QP state (passed from host)
struct GdaDeviceState {
    uint32_t qpn;                    // QP number
    uint16_t nwqes;                  // Number of WQEs in queue
    uint16_t nwqes_mask;             // nwqes - 1 for fast modulo

    // WQE buffer (GPU-accessible)
    void* wqe_buf;                   // WQE buffer base address
    uint32_t wqe_lkey;               // lkey for WQE buffer

    // Doorbell (GPU-writable, mapped from NIC)
    volatile uint32_t* dbrec;        // Doorbell record address

    // Producer index (GPU-managed)
    volatile uint64_t* prod_idx;     // Current producer index

    // CQ for completion
    volatile GdaCqe64* cqe;          // CQ entry buffer
    uint32_t ncqes;                  // Number of CQ entries
    uint32_t ncqes_mask;             // ncqes - 1
    volatile uint64_t* cq_cons_idx;  // CQ consumer index (GPU-managed)
    volatile uint32_t* cq_dbrec;     // CQ doorbell

    // Remote peer info
    uint64_t remote_addr;            // Remote buffer address
    uint32_t remote_rkey;            // Remote rkey

    // Completion tracking
    volatile uint64_t* num_completions;  // Number of completions seen
};

// Memory fence for NIC visibility
__device__ __forceinline__ void gda_membar() {
    __threadfence_system();
}

// Get pointer to WQE at given index
__device__ __forceinline__ void* gda_get_wqe_ptr(GdaDeviceState* state, uint16_t wqe_idx) {
    uint16_t idx = wqe_idx & state->nwqes_mask;
    return (void*)((uintptr_t)state->wqe_buf + (idx << MLX5_SEND_WQE_SHIFT));
}

/**
 * Build an RDMA WRITE WQE directly from GPU
 *
 * @param state      Device state with QP info
 * @param local_addr Local buffer address to send from
 * @param local_lkey Local memory key
 * @param remote_addr Remote buffer address to write to
 * @param remote_rkey Remote memory key
 * @param size       Transfer size in bytes
 * @param wqe_idx    WQE index in the queue
 * @param signaled   Whether to request completion notification
 */
__device__ __forceinline__ void gda_build_rdma_write_wqe(
    GdaDeviceState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    uint16_t wqe_idx,
    bool signaled)
{
    // Get WQE pointer
    void* wqe_ptr = gda_get_wqe_ptr(state, wqe_idx);

    // Layout: Control Seg (16B) + Raddr Seg (16B) + Data Seg (16B) = 48B = 3 DS
    GdaCtrlSeg* ctrl = (GdaCtrlSeg*)wqe_ptr;
    GdaRaddrSeg* raddr = (GdaRaddrSeg*)((uintptr_t)wqe_ptr + 16);
    GdaDataSeg* data = (GdaDataSeg*)((uintptr_t)wqe_ptr + 32);

    // Build control segment
    // opmod_idx_opcode: opmod(8) | wqe_idx(16) | opcode(8)
    ctrl->opmod_idx_opcode = gda_htobe32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);
    // qpn_ds: qpn(24) | ds(8) - ds=3 for RDMA WRITE (ctrl + raddr + data)
    ctrl->qpn_ds = gda_htobe32((state->qpn << 8) | 3);
    ctrl->signature = 0;
    ctrl->rsvd[0] = 0;
    ctrl->rsvd[1] = 0;
    ctrl->fm_ce_se = signaled ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
    ctrl->imm = 0;

    // Build remote address segment
    raddr->raddr = gda_htobe64(remote_addr);
    raddr->rkey = gda_htobe32(remote_rkey);
    raddr->reserved = 0;

    // Build data segment
    data->byte_count = gda_htobe32(size);
    data->lkey = gda_htobe32(local_lkey);
    data->addr = gda_htobe64(local_addr);
}

/**
 * Ring the doorbell to submit WQE to NIC
 * This is the key operation that triggers the RDMA transfer from GPU
 *
 * @param state   Device state
 * @param wqe_idx Index of the WQE to post (producer index + 1)
 */
__device__ __forceinline__ void gda_ring_doorbell(GdaDeviceState* state, uint16_t wqe_idx) {
    // Ensure WQE writes are visible to NIC before ringing doorbell
    gda_membar();

    // Write to doorbell record
    // The doorbell value is the new producer index (wqe_idx)
    // NIC uses this to know how many WQEs to process
    *state->dbrec = gda_htobe32(wqe_idx & 0xFFFF);

    // Update producer index
    *state->prod_idx = wqe_idx;
}

/**
 * Poll CQ for completion
 *
 * @param state   Device state
 * @param expected_completions Expected number of completions after this op
 * @return 0 on success, -1 on timeout, -2 on error
 */
__device__ __forceinline__ int gda_poll_cq(GdaDeviceState* state, uint64_t expected_completions) {
    if (!state->cqe) return 0;  // No CQ access, assume success

    // Get current completion count
    uint64_t start_completions = state->num_completions ? *state->num_completions : 0;
    uint64_t cqe_idx = start_completions & state->ncqes_mask;

    // Calculate expected ownership bit for the next CQE
    // Owner toggles each time we wrap around the CQ
    uint8_t expected_owner = (start_completions / state->ncqes) & 1;

    volatile GdaCqe64* cqe = &state->cqe[cqe_idx];

    // Timeout counter
    uint64_t timeout = 10000000ULL;  // ~10 million iterations

    while (timeout-- > 0) {
        // Memory barrier to ensure we see the latest CQE
        __threadfence_system();

        uint8_t op_own = cqe->op_own;
        uint8_t owner = op_own & MLX5_CQE_OWNER_MASK;

        // Check if CQE is valid (owner bit matches)
        if (owner == expected_owner) {
            // Got a completion
            // Check opcode for completion type (top 4 bits)
            uint8_t opcode = (op_own >> 4) & 0x0F;

            // Update completion count
            if (state->num_completions) {
                (*state->num_completions)++;
            }

            if (opcode == 0x00) {
                // Success completion (RDMA_WRITE, etc.)
                return 0;
            } else if (opcode == 0x0D) {
                // Error completion
                return -2;
            }
            return 0;  // Assume success for other opcodes
        }
    }

    return -1;  // Timeout
}

/**
 * GPU kernel to perform RDMA WRITE from GPU
 *
 * Each thread block handles one RDMA operation
 */
__global__ void gda_rdma_write_kernel(
    GdaDeviceState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    bool signaled,
    int* result)
{
    // Only thread 0 in block 0 does the work
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Get current producer index
    uint64_t prod = *state->prod_idx;
    uint16_t wqe_idx = (uint16_t)((prod + 1) & 0xFFFF);

    // Build the WQE
    gda_build_rdma_write_wqe(state, local_addr, local_lkey,
                             remote_addr, remote_rkey, size, wqe_idx, signaled);

    // Ring doorbell to trigger NIC
    gda_ring_doorbell(state, wqe_idx);

    // If signaled, poll for completion
    if (signaled) {
        *result = gda_poll_cq(state, wqe_idx);
    } else {
        *result = 0;
    }
}

/**
 * Simple ping-pong kernel - one thread triggers RDMA and waits
 */
__global__ void gda_pingpong_kernel(
    GdaDeviceState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    int iterations,
    uint64_t* cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t start = clock64();

    for (int i = 0; i < iterations; i++) {
        // Get WQE index
        uint64_t prod = *state->prod_idx;
        uint16_t wqe_idx = (uint16_t)((prod + 1) & 0xFFFF);

        // Build and post RDMA WRITE
        gda_build_rdma_write_wqe(state, local_addr, local_lkey,
                                 remote_addr, remote_rkey, size, wqe_idx, true);

        gda_ring_doorbell(state, wqe_idx);

        // Poll for completion
        gda_poll_cq(state, wqe_idx);
    }

    uint64_t end = clock64();
    *cycles = end - start;
}

}  // namespace gicc::mlx5
