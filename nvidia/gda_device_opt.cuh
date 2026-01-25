/**
 * gda_device_opt.cuh - Optimized GPU device functions for GPU-triggered RDMA
 *
 * Optimizations based on nvshmem ibgda implementation:
 *   1. L1 cache bypass PTX for NIC buffer access
 *   2. BlueFlame doorbell (64-bit write)
 *   3. Batched doorbell (multiple WQEs per doorbell)
 *   4. Per-32bit atomic WQE writes
 *   5. Proper memory ordering (relaxed vs release)
 *   6. Lock-free WQE slot reservation
 */
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace opengda {

// Re-use constants from gda_device.cuh if already defined
#ifndef MLX5_SEND_WQE_BB
#define MLX5_SEND_WQE_BB 64
#endif
#ifndef MLX5_SEND_WQE_SHIFT
#define MLX5_SEND_WQE_SHIFT 6
#endif
#ifndef MLX5_CQE_SIZE
#define MLX5_CQE_SIZE 64
#endif
#ifndef MLX5_OPCODE_RDMA_WRITE
#define MLX5_OPCODE_RDMA_WRITE 0x08
#endif
#ifndef MLX5_OPCODE_RDMA_READ
#define MLX5_OPCODE_RDMA_READ  0x10
#endif
#ifndef MLX5_OPCODE_NOP
#define MLX5_OPCODE_NOP        0x00
#endif
#ifndef MLX5_WQE_CTRL_CQ_UPDATE
#define MLX5_WQE_CTRL_CQ_UPDATE (1 << 2)
#endif
#ifndef MLX5_WQE_CTRL_FENCE
#define MLX5_WQE_CTRL_FENCE     (1 << 5)
#endif
#ifndef MLX5_CQE_OWNER_MASK
#define MLX5_CQE_OWNER_MASK 1
#endif

// Default batch size (number of WQEs per doorbell)
#define GDA_DEFAULT_BATCH_SIZE 32

//==============================================================================
// OPTIMIZED BYTE SWAP - using CUDA intrinsics (use prefix to avoid conflicts)
//==============================================================================

__device__ __forceinline__ uint32_t gda_opt_htobe32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

__device__ __forceinline__ uint64_t gda_opt_htobe64(uint64_t x) {
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t lo = (uint32_t)x;
    return ((uint64_t)gda_opt_htobe32(lo) << 32) | gda_opt_htobe32(hi);
}

__device__ __forceinline__ uint16_t gda_opt_htobe16(uint16_t x) {
    return (x >> 8) | (x << 8);
}

//==============================================================================
// OPTIMIZED MEMORY ACCESS - L1 cache bypass PTX instructions
// These reduce cache pollution and provide more predictable latency
//==============================================================================

// Relaxed store with L1 cache bypass - for WQE data
__device__ __forceinline__ void gda_store_relaxed_u32(volatile uint32_t* ptr, uint32_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;"
                 : : "l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void gda_store_relaxed_u64(volatile uint64_t* ptr, uint64_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b64 [%0], %1;"
                 : : "l"(ptr), "l"(val) : "memory");
}

// Release store with L1 cache bypass - for doorbell and synchronization
__device__ __forceinline__ void gda_store_release_u32(volatile uint32_t* ptr, uint32_t val) {
    asm volatile("st.release.sys.global.L1::no_allocate.b32 [%0], %1;"
                 : : "l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void gda_store_release_u64(volatile uint64_t* ptr, uint64_t val) {
    asm volatile("st.release.sys.global.L1::no_allocate.b64 [%0], %1;"
                 : : "l"(ptr), "l"(val) : "memory");
}

// Relaxed load with L1 cache bypass
__device__ __forceinline__ uint32_t gda_load_relaxed_u32(volatile uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];"
                 : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

__device__ __forceinline__ uint64_t gda_load_relaxed_u64(volatile uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];"
                 : "=l"(ret) : "l"(ptr) : "memory");
    return ret;
}

// Acquire load with L1 cache bypass
__device__ __forceinline__ uint64_t gda_load_acquire_u64(volatile uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.L1::no_allocate.b64 %0, [%1];"
                 : "=l"(ret) : "l"(ptr) : "memory");
    return ret;
}

// Memory fence - ensure GPU writes visible to NIC
__device__ __forceinline__ void gda_membar_gpu() {
    asm volatile("membar.gpu;" ::: "memory");
}

__device__ __forceinline__ void gda_membar_sys() {
    asm volatile("membar.sys;" ::: "memory");
}

// Global timer for precise timing
__device__ __forceinline__ uint64_t gda_globaltimer() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret) :: "memory");
    return ret;
}

//==============================================================================
// OPTIMIZED DEVICE STATE - includes BlueFlame and batching support
//==============================================================================

struct GdaCqe64Opt {
    uint8_t  rsvd0[46];
    uint16_t wqe_counter;
    uint8_t  signature;
    uint8_t  op_own;
} __attribute__((packed));

// Extended device state with optimization support
struct GdaDeviceStateOpt {
    // QP info
    uint32_t qpn;
    uint16_t nwqes;
    uint16_t nwqes_mask;

    // WQE buffer (GPU-accessible)
    void* wqe_buf;
    uint32_t wqe_lkey;

    // Doorbell record (GPU-writable)
    volatile uint32_t* dbrec;

    // BlueFlame register (GPU-writable, 64-bit)
    volatile uint64_t* bf_reg;

    // Producer indices (nvshmem-style separated indices)
    volatile uint64_t* resv_head;    // Reserved slots (atomically incremented)
    volatile uint64_t* ready_head;   // Ready to post (after WQE written)
    volatile uint64_t* prod_idx;     // Posted to hardware

    // CQ for completion
    volatile GdaCqe64Opt* cqe;
    uint32_t ncqes;
    uint32_t ncqes_mask;
    volatile uint64_t* cq_cons_idx;
    volatile uint32_t* cq_dbrec;

    // Remote peer info
    uint64_t remote_addr;
    uint32_t remote_rkey;

    // Completion tracking
    volatile uint64_t* num_completions;

    // Batching configuration
    uint32_t batch_size;          // Number of WQEs per doorbell
    uint32_t batch_mask;          // batch_size - 1 for fast modulo
};

//==============================================================================
// OPTIMIZED WQE BUILDING - Per-32bit writes with L1 bypass
//==============================================================================

__device__ __forceinline__ void* gda_get_wqe_ptr(GdaDeviceStateOpt* state, uint16_t wqe_idx) {
    uint16_t idx = wqe_idx & state->nwqes_mask;
    return (void*)((uintptr_t)state->wqe_buf + (idx << MLX5_SEND_WQE_SHIFT));
}

/**
 * Build RDMA WRITE WQE with optimized per-32bit writes
 *
 * WQE Layout (48 bytes = 3 data segments):
 *   [0-15]  Control Segment
 *   [16-31] Remote Address Segment
 *   [32-47] Data Segment
 */
__device__ __forceinline__ void gda_build_rdma_write_wqe_opt(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    uint16_t wqe_idx,
    bool signaled)
{
    // Get WQE pointer as uint32_t array for per-word writes
    volatile uint32_t* wqe = (volatile uint32_t*)gda_get_wqe_ptr(state, wqe_idx);

    // Control Segment (16 bytes = 4 words)
    // Word 0: opmod_idx_opcode = opmod(8) | wqe_idx(16) | opcode(8)
    gda_store_relaxed_u32(&wqe[0], gda_opt_htobe32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE));

    // Word 1: qpn_ds = qpn(24) | ds(8)
    gda_store_relaxed_u32(&wqe[1], gda_opt_htobe32((state->qpn << 8) | 3));

    // Word 2: signature(8) | rsvd(16) | fm_ce_se(8)
    uint32_t flags = signaled ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
    gda_store_relaxed_u32(&wqe[2], flags << 24);  // fm_ce_se in MSB

    // Word 3: imm (0 for RDMA WRITE)
    gda_store_relaxed_u32(&wqe[3], 0);

    // Remote Address Segment (16 bytes = 4 words)
    // Word 4-5: raddr (64-bit)
    // After htobe64, lower 32 bits go to lower address on little-endian
    uint64_t raddr_be = gda_opt_htobe64(remote_addr);
    gda_store_relaxed_u32(&wqe[4], (uint32_t)raddr_be);
    gda_store_relaxed_u32(&wqe[5], (uint32_t)(raddr_be >> 32));

    // Word 6: rkey
    gda_store_relaxed_u32(&wqe[6], gda_opt_htobe32(remote_rkey));

    // Word 7: reserved
    gda_store_relaxed_u32(&wqe[7], 0);

    // Data Segment (16 bytes = 4 words)
    // Word 8: byte_count
    gda_store_relaxed_u32(&wqe[8], gda_opt_htobe32(size));

    // Word 9: lkey
    gda_store_relaxed_u32(&wqe[9], gda_opt_htobe32(local_lkey));

    // Word 10-11: addr (64-bit)
    // After htobe64, lower 32 bits go to lower address on little-endian
    uint64_t laddr_be = gda_opt_htobe64(local_addr);
    gda_store_relaxed_u32(&wqe[10], (uint32_t)laddr_be);
    gda_store_relaxed_u32(&wqe[11], (uint32_t)(laddr_be >> 32));
}

//==============================================================================
// OPTIMIZED DOORBELL - BlueFlame 64-bit write
//==============================================================================

/**
 * Ring doorbell using BlueFlame (64-bit write)
 *
 * BlueFlame writes the first 8 bytes of the control segment directly
 * to the NIC, bypassing the doorbell record for lower latency.
 *
 * Memory layout of control segment (first 8 bytes):
 *   Bytes 0-3: opmod_idx_opcode (opmod:8 | wqe_idx:16 | opcode:8)
 *   Bytes 4-7: qpn_ds (qpn:24 | ds:8)
 *
 * On little-endian, when storing a 64-bit value:
 *   Lower 32 bits go to bytes 0-3
 *   Upper 32 bits go to bytes 4-7
 */
__device__ __forceinline__ void gda_ring_doorbell_bf(
    GdaDeviceStateOpt* state,
    uint16_t wqe_idx)
{
    // Build BlueFlame value matching control segment layout
    // nvshmem uses just index and QPN (no opcode/ds) for the BlueFlame notification
    uint32_t opmod_idx_opcode = gda_opt_htobe32(wqe_idx << 8);
    uint32_t qpn_ds = gda_opt_htobe32(state->qpn << 8);
    // On little-endian: lower 32 bits go to bytes 0-3 (opmod_idx_opcode)
    //                   upper 32 bits go to bytes 4-7 (qpn_ds)
    uint64_t bf_val = ((uint64_t)qpn_ds << 32) | opmod_idx_opcode;

    // Memory fence to ensure WQE writes are visible before doorbell
    gda_membar_sys();

    // Write doorbell record first (required before BlueFlame)
    gda_store_release_u32(state->dbrec, gda_opt_htobe32(wqe_idx & 0xFFFF));

    // BlueFlame write - 64-bit atomic write to UAR
    if (state->bf_reg) {
        gda_store_release_u64(state->bf_reg, bf_val);
    }

    // Update producer index
    gda_store_relaxed_u64(state->prod_idx, wqe_idx);
}

/**
 * Simple doorbell (no BlueFlame) - fallback path
 */
__device__ __forceinline__ void gda_ring_doorbell_simple(
    GdaDeviceStateOpt* state,
    uint16_t wqe_idx)
{
    // Memory fence to ensure WQE writes are visible
    gda_membar_sys();

    // Write doorbell record
    gda_store_release_u32(state->dbrec, gda_opt_htobe32(wqe_idx & 0xFFFF));

    // Update producer index
    gda_store_relaxed_u64(state->prod_idx, wqe_idx);
}

//==============================================================================
// BATCHED OPERATIONS - Multiple WQEs per doorbell
//==============================================================================

/**
 * Reserve WQE slots atomically (lock-free)
 * Returns the base WQE index for the reservation
 */
__device__ __forceinline__ uint64_t gda_reserve_wqe_slots(
    GdaDeviceStateOpt* state,
    uint32_t num_slots)
{
    return atomicAdd((unsigned long long*)state->resv_head, num_slots);
}

/**
 * Mark WQEs as ready (after building)
 * Uses CAS to ensure ordering with other threads
 */
__device__ __forceinline__ void gda_mark_wqes_ready(
    GdaDeviceStateOpt* state,
    uint64_t base_idx,
    uint32_t num_wqes)
{
    uint64_t expected = base_idx;
    uint64_t new_val = base_idx + num_wqes;

    // Spin until our slot is ready
    while (atomicCAS((unsigned long long*)state->ready_head,
                     expected, new_val) != expected) {
        // Wait for previous slots to complete
        expected = gda_load_relaxed_u64(state->ready_head);
        if (expected >= new_val) break;  // Already advanced past us
    }
}

/**
 * Check if we should post doorbell based on batch boundary
 * Uses mask-based boundary detection (O(1))
 */
__device__ __forceinline__ bool gda_should_post_doorbell(
    GdaDeviceStateOpt* state,
    uint64_t base_idx,
    uint64_t new_idx)
{
    // Cross batch boundary?
    uint64_t mask = ~((uint64_t)(state->batch_size - 1));
    return (base_idx & mask) != (new_idx & mask);
}

/**
 * Post multiple WQEs with single doorbell
 */
__device__ __forceinline__ void gda_post_wqes_batched(
    GdaDeviceStateOpt* state,
    uint64_t new_prod_idx)
{
    // Ensure all WQE writes are visible
    gda_membar_sys();

    // Write doorbell
    if (state->bf_reg) {
        // BlueFlame path - match nvshmem's ibgda_ring_db format
        uint16_t wqe_idx = (uint16_t)(new_prod_idx & 0xFFFF);
        uint32_t opmod_idx_opcode = gda_opt_htobe32(wqe_idx << 8);
        uint32_t qpn_ds = gda_opt_htobe32(state->qpn << 8);
        // On little-endian: lower 32 bits -> bytes 0-3, upper -> bytes 4-7
        uint64_t bf_val = ((uint64_t)qpn_ds << 32) | opmod_idx_opcode;

        gda_store_release_u32(state->dbrec, gda_opt_htobe32(wqe_idx));
        gda_store_release_u64(state->bf_reg, bf_val);
    } else {
        // Simple doorbell
        gda_store_release_u32(state->dbrec, gda_opt_htobe32((uint32_t)(new_prod_idx & 0xFFFF)));
    }

    // Update producer index
    gda_store_relaxed_u64(state->prod_idx, new_prod_idx);
}

//==============================================================================
// CQ POLLING - Optimized with L1 bypass
//==============================================================================

__device__ __forceinline__ int gda_poll_cq_opt(
    GdaDeviceStateOpt* state,
    uint64_t expected_completions,
    uint64_t timeout_ns = 50000)  // 50us default timeout
{
    if (!state->cqe) return 0;

    uint64_t start_completions = state->num_completions ?
                                 gda_load_relaxed_u64(state->num_completions) : 0;
    uint64_t cqe_idx = start_completions & state->ncqes_mask;
    uint8_t expected_owner = (start_completions / state->ncqes) & 1;

    volatile GdaCqe64Opt* cqe = &state->cqe[cqe_idx];

    uint64_t start_time = gda_globaltimer();
    uint64_t timeout_cycles = timeout_ns;  // globaltimer is ~1ns resolution

    while ((gda_globaltimer() - start_time) < timeout_cycles) {
        // Acquire load to get latest CQE
        uint8_t op_own = ((volatile uint8_t*)cqe)[63];  // op_own is last byte
        uint8_t owner = op_own & MLX5_CQE_OWNER_MASK;

        if (owner == expected_owner) {
            // Got completion
            if (state->num_completions) {
                atomicAdd((unsigned long long*)state->num_completions, 1);
            }

            uint8_t opcode = (op_own >> 4) & 0x0F;
            return (opcode == 0x00 || opcode == 0x02) ? 0 : -2;  // 0=success, 2=flush
        }
    }

    return -1;  // Timeout
}

//==============================================================================
// HIGH-PERFORMANCE RDMA WRITE - Combined optimizations
//==============================================================================

/**
 * Optimized single RDMA WRITE with BlueFlame doorbell
 *
 * WQE indexing (matching nvshmem):
 *   - prod_idx points to the NEXT slot to use
 *   - WQE is built at slot prod_idx
 *   - Doorbell contains prod_idx + 1 (the new producer index)
 */
__device__ __forceinline__ void gda_rdma_write_opt(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    bool signaled)
{
    // Get current producer index (this is where we write the WQE)
    uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
    uint16_t wqe_idx = (uint16_t)(prod & 0xFFFF);
    uint16_t new_prod = (uint16_t)((prod + 1) & 0xFFFF);

    // Build WQE at slot wqe_idx
    gda_build_rdma_write_wqe_opt(state, local_addr, local_lkey,
                                  remote_addr, remote_rkey, size,
                                  wqe_idx, signaled);

    // Ring doorbell with new producer index
    gda_ring_doorbell_bf(state, new_prod);
}

/**
 * Batched RDMA WRITE - builds WQE and optionally rings doorbell
 *
 * Returns true if doorbell was rung (caller should wait if needed)
 */
__device__ __forceinline__ bool gda_rdma_write_batched(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    bool signaled,
    bool force_post)
{
    // Get current slot and calculate new producer index
    uint64_t wqe_slot = gda_load_relaxed_u64(state->prod_idx);
    uint64_t new_prod = wqe_slot + 1;

    // Build WQE at wqe_slot
    gda_build_rdma_write_wqe_opt(state, local_addr, local_lkey,
                                  remote_addr, remote_rkey, size,
                                  (uint16_t)(wqe_slot & 0xFFFF), signaled);

    // Check if we should post doorbell
    bool do_post = force_post || signaled ||
                   gda_should_post_doorbell(state, wqe_slot, new_prod);

    if (do_post) {
        gda_post_wqes_batched(state, new_prod);
        return true;
    }

    // Just update local tracking
    gda_store_relaxed_u64(state->prod_idx, new_prod);
    return false;
}

//==============================================================================
// OPTIMIZED KERNELS
//==============================================================================

/**
 * Optimized burst kernel - single thread, maximum throughput
 *
 * WQE indexing:
 *   - WQE i is written to slot (prod + i)
 *   - Doorbell contains (prod + i + 1) = next empty slot
 */
__global__ void gda_burst_kernel_opt(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size,
    int count,
    int batch_size,
    uint64_t* gpu_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t start = clock64();

    uint64_t base_prod = gda_load_relaxed_u64(state->prod_idx);

    for (int i = 0; i < count; i++) {
        // WQE slot is base_prod + i
        uint16_t wqe_slot = (uint16_t)((base_prod + i) & 0xFFFF);

        // Build WQE at wqe_slot
        gda_build_rdma_write_wqe_opt(
            state, local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, wqe_slot, (i == count - 1)  // Signal last
        );

        // Ring doorbell every batch_size operations or on last
        // Doorbell value = next empty slot = wqe_slot + 1
        if ((i + 1) % batch_size == 0 || i == count - 1) {
            uint16_t new_prod = (uint16_t)((wqe_slot + 1) & 0xFFFF);
            gda_ring_doorbell_bf(state, new_prod);
        }
    }

    // Final update - producer now points to base_prod + count
    gda_store_relaxed_u64(state->prod_idx, base_prod + count);

    uint64_t end = clock64();
    *gpu_cycles = end - start;
}

/**
 * Optimized ping-pong kernel with cycle counting
 */
__global__ void gda_pingpong_kernel_opt(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size,
    int iterations,
    uint64_t* cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t start = clock64();

    for (int i = 0; i < iterations; i++) {
        gda_rdma_write_opt(
            state, local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, false  // unsignaled
        );
    }

    uint64_t end = clock64();
    *cycles = end - start;
}

/**
 * Multi-thread cooperative kernel - each warp handles one operation
 * For maximum parallelism when building many WQEs
 */
__global__ void gda_multi_wqe_kernel(
    GdaDeviceStateOpt* state,
    uint64_t* local_addrs,   // Array of source addresses
    uint32_t local_lkey,
    uint64_t* remote_addrs,  // Array of dest addresses
    uint32_t remote_rkey,
    uint32_t* sizes,         // Array of sizes
    int num_ops,
    uint64_t* gpu_cycles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Only lane 0 of each warp does the work
    if (lane_id != 0) return;
    if (warp_id >= num_ops) return;

    // Each warp reserves its own WQE slot
    uint64_t wqe_idx;

    if (warp_id == 0) {
        // First warp records start time
        if (gpu_cycles) *gpu_cycles = clock64();
    }

    // Atomic reservation
    wqe_idx = atomicAdd((unsigned long long*)state->resv_head, 1);
    uint16_t slot = (uint16_t)((wqe_idx + 1) & 0xFFFF);

    // Build WQE
    bool signaled = (warp_id == num_ops - 1);
    gda_build_rdma_write_wqe_opt(
        state,
        local_addrs[warp_id], local_lkey,
        remote_addrs[warp_id], remote_rkey,
        sizes[warp_id], slot, signaled
    );

    // Synchronize to ensure all WQEs built before doorbell
    __syncthreads();

    // Last warp rings doorbell for all
    if (warp_id == num_ops - 1) {
        gda_ring_doorbell_bf(state, slot);

        if (gpu_cycles) {
            uint64_t end = clock64();
            *gpu_cycles = end - *gpu_cycles;
        }
    }
}

}  // namespace opengda
