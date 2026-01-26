/**
 * gda_persistent.cuh - Persistent kernel for GPU-triggered RDMA
 *
 * Key optimizations:
 * 1. Persistent kernel - no kernel launch overhead per operation
 * 2. GPU-side command queue - CPU posts commands, GPU executes
 * 3. GPU-side CQ polling - true completion detection
 * 4. BlueFlame doorbell - low latency doorbell mechanism
 */
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace opengda {

// Command types for persistent kernel
enum GdaCommandType : uint32_t {
    GDA_CMD_NOP = 0,
    GDA_CMD_RDMA_WRITE = 1,
    GDA_CMD_RDMA_READ = 2,
    GDA_CMD_FENCE = 3,
    GDA_CMD_EXIT = 0xFF
};

// Command structure (64 bytes to match WQE size)
struct alignas(64) GdaCommand {
    uint32_t type;           // Command type
    uint32_t flags;          // Flags (signaled, etc.)
    uint64_t local_addr;     // Local buffer address
    uint32_t local_lkey;     // Local memory key
    uint32_t size;           // Transfer size
    uint64_t remote_addr;    // Remote buffer address
    uint32_t remote_rkey;    // Remote memory key
    uint32_t reserved;       // Padding
    volatile uint64_t* completion;  // Where to signal completion (optional)
    uint64_t user_data;      // User data for completion
};

// Ring buffer for commands (GPU polls this)
struct GdaCommandRing {
    volatile uint64_t head;    // CPU writes (producer)
    volatile uint64_t tail;    // GPU writes (consumer)
    uint32_t size;             // Number of entries
    uint32_t mask;             // size - 1
    GdaCommand* commands;      // Command buffer (GPU-accessible)
};

// Extended device state for persistent kernel
struct GdaPersistentState {
    // QP info
    uint32_t qpn;
    uint16_t nwqes;
    uint16_t nwqes_mask;

    // WQE buffer
    void* wqe_buf;

    // Doorbell
    volatile uint32_t* dbrec;
    volatile uint64_t* bf_reg;   // BlueFlame register for fast doorbell
    uint32_t bf_size;

    // Producer index
    volatile uint64_t* prod_idx;

    // CQ
    volatile void* cqe;
    uint32_t ncqes;
    uint32_t ncqes_mask;
    volatile uint64_t* cq_cons_idx;
    volatile uint32_t* cq_dbrec;
    uint32_t cq_arm_sn;      // CQ arm sequence number

    // Command ring
    GdaCommandRing cmd_ring;

    // Statistics
    volatile uint64_t* ops_completed;
    volatile uint64_t* ops_posted;
};

// MLX5 constants
#define MLX5_SEND_WQE_BB 64
#define MLX5_SEND_WQE_SHIFT 6
#define MLX5_OPCODE_RDMA_WRITE 0x08
#define MLX5_WQE_CTRL_CQ_UPDATE (1 << 2)
#define MLX5_CQE_OWNER_MASK 1

// MLX5 WQE segments
struct PersistentCtrlSeg {
    uint32_t opmod_idx_opcode;
    uint32_t qpn_ds;
    uint8_t  signature;
    uint8_t  rsvd[2];
    uint8_t  fm_ce_se;
    uint32_t imm;
} __attribute__((packed));

struct PersistentRaddrSeg {
    uint64_t raddr;
    uint32_t rkey;
    uint32_t reserved;
} __attribute__((packed));

struct PersistentDataSeg {
    uint32_t byte_count;
    uint32_t lkey;
    uint64_t addr;
} __attribute__((packed));

// Byte swap for big-endian wire format
__device__ __forceinline__ uint32_t p_htobe32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

__device__ __forceinline__ uint64_t p_htobe64(uint64_t x) {
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t lo = (uint32_t)x;
    return ((uint64_t)p_htobe32(lo) << 32) | p_htobe32(hi);
}

// Memory fence
__device__ __forceinline__ void p_membar() {
    __threadfence_system();
}

// Get WQE pointer
__device__ __forceinline__ void* p_get_wqe(GdaPersistentState* s, uint16_t idx) {
    return (void*)((uintptr_t)s->wqe_buf + ((idx & s->nwqes_mask) << MLX5_SEND_WQE_SHIFT));
}

// Build RDMA WRITE WQE
__device__ __forceinline__ void p_build_write_wqe(
    GdaPersistentState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    uint16_t wqe_idx,
    bool signaled)
{
    void* wqe = p_get_wqe(state, wqe_idx);

    PersistentCtrlSeg* ctrl = (PersistentCtrlSeg*)wqe;
    PersistentRaddrSeg* raddr = (PersistentRaddrSeg*)((uintptr_t)wqe + 16);
    PersistentDataSeg* data = (PersistentDataSeg*)((uintptr_t)wqe + 32);

    // Control segment
    ctrl->opmod_idx_opcode = p_htobe32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);
    ctrl->qpn_ds = p_htobe32((state->qpn << 8) | 3);
    ctrl->signature = 0;
    ctrl->rsvd[0] = 0;
    ctrl->rsvd[1] = 0;
    ctrl->fm_ce_se = signaled ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
    ctrl->imm = 0;

    // Remote address segment
    raddr->raddr = p_htobe64(remote_addr);
    raddr->rkey = p_htobe32(remote_rkey);
    raddr->reserved = 0;

    // Data segment
    data->byte_count = p_htobe32(size);
    data->lkey = p_htobe32(local_lkey);
    data->addr = p_htobe64(local_addr);
}

// Ring doorbell (following nvshmem sequence)
// @param new_prod_idx The new producer index (current + 1)
__device__ __forceinline__ void p_ring_doorbell(GdaPersistentState* state, uint16_t new_prod_idx) {
    // Update producer index first
    *state->prod_idx = new_prod_idx;

    // Memory fence
    p_membar();

    // Update doorbell record
    *state->dbrec = p_htobe32(new_prod_idx & 0xFFFF);

    // Memory fence before BlueFlame
    p_membar();

    // BlueFlame doorbell (8-byte write per nvshmem)
    if (state->bf_reg) {
        // Format: opmod_idx_opcode (4B) + qpn_ds (4B)
        uint32_t opmod_idx_opcode = p_htobe32((new_prod_idx & 0xFFFF) << 8);
        uint32_t qpn_ds = p_htobe32(state->qpn << 8);
        uint64_t bf_val = ((uint64_t)opmod_idx_opcode) | ((uint64_t)qpn_ds << 32);

        volatile uint64_t* bf = state->bf_reg;
        *bf = bf_val;

        p_membar();
    }
}

// Poll CQ for completion (non-blocking)
__device__ __forceinline__ bool p_poll_cq_once(GdaPersistentState* state) {
    if (!state->cqe || !state->cq_cons_idx) return true;  // No CQ, assume done

    uint64_t cons = *state->cq_cons_idx;
    uint32_t cqe_idx = cons & state->ncqes_mask;
    uint8_t expected_owner = (cons / state->ncqes) & 1;

    volatile uint8_t* cqe = (volatile uint8_t*)state->cqe + (cqe_idx * 64);
    uint8_t op_own = cqe[63];  // op_own is at offset 63
    uint8_t owner = op_own & MLX5_CQE_OWNER_MASK;

    if (owner == expected_owner) {
        // Got completion
        (*state->cq_cons_idx)++;
        if (state->ops_completed) {
            (*state->ops_completed)++;
        }
        return true;
    }
    return false;
}

// Poll CQ with timeout
__device__ __forceinline__ bool p_poll_cq(GdaPersistentState* state, int max_polls) {
    for (int i = 0; i < max_polls; i++) {
        if (p_poll_cq_once(state)) return true;
    }
    return false;
}

/**
 * Persistent kernel - runs continuously, processes commands from ring
 *
 * Design:
 * - Single thread executes RDMA operations
 * - Polls command ring for new work
 * - Posts WQEs and rings doorbell
 * - Optionally waits for completion
 */
__global__ void gda_persistent_kernel(GdaPersistentState* state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    GdaCommandRing* ring = &state->cmd_ring;
    uint64_t local_tail = ring->tail;

    while (true) {
        // Check for new commands
        __threadfence_system();
        uint64_t head = ring->head;

        if (local_tail == head) {
            // No work, spin briefly
            continue;
        }

        // Process command
        uint32_t cmd_idx = local_tail & ring->mask;
        GdaCommand* cmd = &ring->commands[cmd_idx];

        // Fence to ensure we see the full command
        __threadfence_system();

        uint32_t type = cmd->type;

        if (type == GDA_CMD_EXIT) {
            break;
        }

        if (type == GDA_CMD_RDMA_WRITE) {
            // Get current producer index
            uint64_t prod = *state->prod_idx;
            uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);
            uint16_t new_prod_idx = (uint16_t)((prod + 1) & 0xFFFF);

            bool signaled = (cmd->flags & 1) != 0;

            // Build WQE at current slot
            p_build_write_wqe(
                state,
                cmd->local_addr,
                cmd->local_lkey,
                cmd->remote_addr,
                cmd->remote_rkey,
                cmd->size,
                wqe_slot,
                signaled
            );

            // Ring doorbell with new producer index
            p_ring_doorbell(state, new_prod_idx);

            if (state->ops_posted) {
                (*state->ops_posted)++;
            }

            // Wait for completion if signaled
            if (signaled && cmd->completion) {
                // Poll CQ until complete
                while (!p_poll_cq_once(state)) {
                    // Spin
                }
                // Signal completion
                *(cmd->completion) = cmd->user_data;
            }
        }

        // Advance tail
        local_tail++;
        ring->tail = local_tail;
        __threadfence_system();
    }
}

/**
 * Simple ping-pong kernel for latency testing
 * GPU directly triggers RDMA and waits for completion
 */
__global__ void gda_simple_pingpong(
    GdaPersistentState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    int iterations,
    volatile int* done_flag,
    uint64_t* result_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Wait for start signal
    while (*done_flag == 0) {
        __threadfence_system();
    }

    uint64_t start = clock64();

    for (int i = 0; i < iterations; i++) {
        // Get current producer index
        uint64_t prod = *state->prod_idx;
        uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);
        uint16_t new_prod_idx = (uint16_t)((prod + 1) & 0xFFFF);

        // Build WQE at current slot (not signaled for max speed)
        p_build_write_wqe(
            state,
            local_addr, local_lkey,
            remote_addr, remote_rkey,
            size, wqe_slot,
            false  // Not signaled
        );

        // Ring doorbell with new producer index
        p_ring_doorbell(state, new_prod_idx);
    }

    uint64_t end = clock64();
    *result_cycles = end - start;

    // Signal done
    *done_flag = 2;
    __threadfence_system();
}

/**
 * Batched RDMA kernel - one kernel launch for multiple operations
 */
__global__ void gda_batched_write(
    GdaPersistentState* state,
    uint64_t* local_addrs,
    uint32_t* local_lkeys,
    uint64_t* remote_addrs,
    uint32_t* remote_rkeys,
    uint32_t* sizes,
    int count,
    volatile uint64_t* completions)  // Optional per-op completion
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int i = 0; i < count; i++) {
        uint64_t prod = *state->prod_idx;
        uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);
        uint16_t new_prod_idx = (uint16_t)((prod + 1) & 0xFFFF);

        bool signaled = (completions != nullptr);

        p_build_write_wqe(
            state,
            local_addrs[i], local_lkeys[i],
            remote_addrs[i], remote_rkeys[i],
            sizes[i], wqe_slot,
            signaled
        );

        p_ring_doorbell(state, new_prod_idx);

        // If signaled, wait for this one to complete
        if (signaled) {
            while (!p_poll_cq_once(state)) {
                // Spin
            }
            completions[i] = 1;
        }
    }
}

}  // namespace opengda
