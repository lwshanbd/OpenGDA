/**
 * gda_device_nvshmem.cuh - Optimized GPU device functions based on nvshmem ibgda
 *
 * Key optimizations from nvshmem:
 *   1. Three-tier index management (resv_head, ready_head, prod_idx)
 *   2. Batch boundary detection for doorbell coalescing
 *   3. Lock-free FIFO using atomicCAS for ready_head
 *   4. atomicMax for prod_idx to prevent redundant doorbells
 *   5. Inline data for small messages (up to 12 bytes)
 *   6. Proper memory ordering (MEMBAR placement)
 */
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace gicc::mlx5 {

//==============================================================================
// Constants
//==============================================================================

#define MLX5_SEND_WQE_BB 64
#define MLX5_SEND_WQE_SHIFT 6
#define MLX5_OPCODE_NOP 0x00
#define MLX5_OPCODE_RDMA_WRITE 0x08
#define MLX5_OPCODE_RDMA_READ 0x10
#define MLX5_WQE_CTRL_CQ_UPDATE (1 << 2)
#define MLX5_INLINE_SEG 0x80000000

#define GDA_DEFAULT_BATCH_SIZE 32

//==============================================================================
// Byte swap using CUDA intrinsics
//==============================================================================

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

//==============================================================================
// Memory access with L1 cache bypass (from nvshmem)
//==============================================================================

// Relaxed store - for WQE data (no ordering guarantee)
__device__ __forceinline__ void gda_store_relaxed_u32(volatile uint32_t* ptr, uint32_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;"
                 : : "l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void gda_store_relaxed_u64(volatile uint64_t* ptr, uint64_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b64 [%0], %1;"
                 : : "l"(ptr), "l"(val) : "memory");
}

// Release store - for doorbell and synchronization
__device__ __forceinline__ void gda_store_release_u32(volatile uint32_t* ptr, uint32_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;"
                 : : "l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void gda_store_release_u64(volatile uint64_t* ptr, uint64_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;"
                 : : "l"(ptr), "l"(val) : "memory");
}

// Relaxed load with L1 bypass
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

// Memory barriers (from nvshmem)
__device__ __forceinline__ void gda_mfence() {
    asm volatile("fence.acq_rel.cta;" ::: "memory");
}

__device__ __forceinline__ void gda_membar() {
    __threadfence();  // GPU scope barrier
}

__device__ __forceinline__ void gda_membar_system() {
    __threadfence_system();  // System scope barrier (GPU + NIC)
}

// Global timer for precise timing
__device__ __forceinline__ uint64_t gda_globaltimer() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret) :: "memory");
    return ret;
}

//==============================================================================
// Device state structure (nvshmem-style)
//==============================================================================

struct GdaDeviceQpState {
    // QP info
    uint32_t qpn;
    uint16_t nwqes;           // Number of WQEs (power of 2)
    uint16_t nwqes_mask;      // nwqes - 1

    // WQE buffer (GPU-accessible)
    void* wqe_buf;
    uint32_t wqe_lkey;        // Not used for current implementation

    // Doorbell record (SQ doorbell at offset 4)
    volatile uint32_t* dbrec;

    // BlueFlame register
    volatile uint64_t* bf_reg;

    // Three-tier index management (nvshmem-style)
    volatile uint64_t* resv_head;    // Reserved WQE idx + 1 (for slot allocation)
    volatile uint64_t* ready_head;   // Ready WQE idx + 1 (WQE written)
    volatile uint64_t* prod_idx;     // Posted WQE idx + 1 (doorbell rung)

    // Post-send lock
    volatile int* post_send_lock;

    // Remote peer info
    uint64_t remote_addr;
    uint32_t remote_rkey;

    // Batching configuration
    uint32_t batch_size;
    uint32_t batch_mask;    // ~(batch_size - 1)
};

//==============================================================================
// Lock functions (from nvshmem)
//==============================================================================

__device__ __forceinline__ void gda_lock_acquire(volatile int* lock) {
    while (atomicCAS((int*)lock, 0, 1) == 1) {
        // Spin until lock acquired
    }
    gda_mfence();  // Prevent reordering after lock
}

__device__ __forceinline__ void gda_lock_release(volatile int* lock) {
    gda_mfence();  // Prevent reordering before unlock
    atomicExch((int*)lock, 0);
}

//==============================================================================
// WQE pointer calculation
//==============================================================================

__device__ __forceinline__ void* gda_get_wqe_ptr(GdaDeviceQpState* qp, uint16_t wqe_idx) {
    uint16_t idx = wqe_idx & qp->nwqes_mask;
    return (void*)((uintptr_t)qp->wqe_buf + (idx << MLX5_SEND_WQE_SHIFT));
}

//==============================================================================
// WQE Building (per-32bit writes with L1 bypass)
//==============================================================================

/**
 * Build RDMA WRITE WQE for RC QP
 * WQE Layout (48 bytes = 3 DS):
 *   [0-15]  Control Segment
 *   [16-31] Remote Address Segment
 *   [32-47] Data Segment
 */
__device__ __forceinline__ void gda_build_rdma_write_wqe(
    GdaDeviceQpState* qp,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    uint16_t wqe_idx,
    uint8_t fm_ce_se)
{
    volatile uint32_t* wqe = (volatile uint32_t*)gda_get_wqe_ptr(qp, wqe_idx);

    // Control Segment (16 bytes)
    gda_store_relaxed_u32(&wqe[0], gda_htobe32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE));
    gda_store_relaxed_u32(&wqe[1], gda_htobe32((qp->qpn << 8) | 3));  // ds=3 for RC
    gda_store_relaxed_u32(&wqe[2], fm_ce_se << 24);
    gda_store_relaxed_u32(&wqe[3], 0);

    // Remote Address Segment (16 bytes)
    uint64_t raddr_be = gda_htobe64(remote_addr);
    gda_store_relaxed_u32(&wqe[4], (uint32_t)raddr_be);
    gda_store_relaxed_u32(&wqe[5], (uint32_t)(raddr_be >> 32));
    gda_store_relaxed_u32(&wqe[6], gda_htobe32(remote_rkey));
    gda_store_relaxed_u32(&wqe[7], 0);

    // Data Segment (16 bytes)
    gda_store_relaxed_u32(&wqe[8], gda_htobe32(size));
    gda_store_relaxed_u32(&wqe[9], gda_htobe32(local_lkey));
    uint64_t laddr_be = gda_htobe64(local_addr);
    gda_store_relaxed_u32(&wqe[10], (uint32_t)laddr_be);
    gda_store_relaxed_u32(&wqe[11], (uint32_t)(laddr_be >> 32));
}

/**
 * Build RDMA WRITE WQE with inline data (up to 12 bytes)
 * Avoids separate data fetch for small messages
 */
__device__ __forceinline__ void gda_build_rdma_write_inline_wqe(
    GdaDeviceQpState* qp,
    const void* data,
    uint32_t size,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint16_t wqe_idx,
    uint8_t fm_ce_se)
{
    volatile uint32_t* wqe = (volatile uint32_t*)gda_get_wqe_ptr(qp, wqe_idx);

    // Control Segment
    gda_store_relaxed_u32(&wqe[0], gda_htobe32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE));
    gda_store_relaxed_u32(&wqe[1], gda_htobe32((qp->qpn << 8) | 3));
    gda_store_relaxed_u32(&wqe[2], fm_ce_se << 24);
    gda_store_relaxed_u32(&wqe[3], 0);

    // Remote Address Segment
    uint64_t raddr_be = gda_htobe64(remote_addr);
    gda_store_relaxed_u32(&wqe[4], (uint32_t)raddr_be);
    gda_store_relaxed_u32(&wqe[5], (uint32_t)(raddr_be >> 32));
    gda_store_relaxed_u32(&wqe[6], gda_htobe32(remote_rkey));
    gda_store_relaxed_u32(&wqe[7], 0);

    // Inline Data Segment (byte_count with inline flag)
    gda_store_relaxed_u32(&wqe[8], gda_htobe32(size | MLX5_INLINE_SEG));

    // Copy inline data (up to 12 bytes fit in remaining space)
    const uint32_t* src = (const uint32_t*)data;
    if (size >= 4) gda_store_relaxed_u32(&wqe[9], src[0]);
    if (size >= 8) gda_store_relaxed_u32(&wqe[10], src[1]);
    if (size == 12) gda_store_relaxed_u32(&wqe[11], src[2]);
}

//==============================================================================
// Doorbell functions (from nvshmem)
//==============================================================================

/**
 * Update doorbell record
 * DBREC contains the index of the next empty WQEBB
 */
__device__ __forceinline__ void gda_update_dbr(GdaDeviceQpState* qp, uint32_t dbrec_head) {
    // Optimized: mask to 16-bit, byte-swap, and store
    uint32_t dbrec_val;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask1;\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 mask2;\n\t"
        "mov.b32 mask1, 0xffff;\n\t"
        "mov.b32 mask2, 0x0123;\n\t"
        "and.b32 dbrec_head_16b, %1, mask1;\n\t"
        "prmt.b32 %0, dbrec_head_16b, ign, mask2;\n\t"
        "}"
        : "=r"(dbrec_val)
        : "r"(dbrec_head));
    gda_store_release_u32(qp->dbrec, dbrec_val);
}

/**
 * Ring BlueFlame doorbell
 * Single 64-bit write containing prod_idx and qpn
 */
__device__ __forceinline__ void gda_ring_db(GdaDeviceQpState* qp, uint16_t prod_idx) {
    // Control segment format for BlueFlame
    uint32_t opmod_idx_opcode = gda_htobe32(prod_idx << 8);
    uint32_t qpn_ds = gda_htobe32(qp->qpn << 8);
    uint64_t bf_val = ((uint64_t)qpn_ds << 32) | opmod_idx_opcode;

    gda_store_release_u64(qp->bf_reg, bf_val);
}

//==============================================================================
// Post-send functions (from nvshmem)
//==============================================================================

/**
 * Post WQEs with locking and atomicMax
 * - Acquires post_send_lock
 * - Uses atomicMax to only ring doorbell if new_prod_idx > old
 * - Sequences: MEMBAR -> update_dbr -> MEMBAR -> ring_db
 */
template <bool need_strong_flush = false>
__device__ __forceinline__ void gda_post_send(GdaDeviceQpState* qp, uint64_t new_prod_idx) {
    gda_lock_acquire(qp->post_send_lock);

    uint64_t old_prod_idx;
    if (need_strong_flush) {
        old_prod_idx = atomicMax((unsigned long long*)qp->prod_idx,
                                  (unsigned long long)new_prod_idx);
    } else {
        old_prod_idx = atomicMax_block((unsigned long long*)qp->prod_idx,
                                        (unsigned long long)new_prod_idx);
    }

    if (new_prod_idx > old_prod_idx) {
        gda_membar();
        gda_update_dbr(qp, new_prod_idx);
        gda_membar();
        gda_ring_db(qp, (uint16_t)new_prod_idx);
    }

    gda_lock_release(qp->post_send_lock);
}

/**
 * Submit requests with batching and FIFO ordering
 *
 * Three triggers for posting doorbell:
 *   1. No concurrent submissions (new_wqe_idx == resv_head)
 *   2. Crossed batch boundary ((base & mask) != (new & mask))
 *   3. Large batch (num_wqes >= batch_size)
 */
template <bool need_strong_flush = false>
__device__ __forceinline__ void gda_submit_requests(
    GdaDeviceQpState* qp,
    uint64_t base_wqe_idx,
    uint16_t num_wqes)
{
    uint64_t mask = qp->batch_mask;
    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

    unsigned long long* ready_idx = (unsigned long long*)qp->ready_head;

    // Wait for prior WQE slots to be filled (FIFO ordering)
    if (need_strong_flush) {
        gda_membar_system();
        while (atomicCAS(ready_idx, (unsigned long long)base_wqe_idx,
                         (unsigned long long)new_wqe_idx) != base_wqe_idx)
            ;
        gda_mfence();
    } else {
        gda_mfence();
        while (atomicCAS_block(ready_idx, (unsigned long long)base_wqe_idx,
                               (unsigned long long)new_wqe_idx) != base_wqe_idx)
            ;
        gda_mfence();
    }

    // Decide whether to post doorbell
    bool do_post_send =
        (new_wqe_idx == gda_load_relaxed_u64(qp->resv_head))  // No concurrent submissions
        || ((base_wqe_idx & mask) != (new_wqe_idx & mask))    // Crossed batch boundary
        || (num_wqes >= qp->batch_size);                      // Batch threshold

    if (do_post_send) {
        gda_post_send<need_strong_flush>(qp, new_wqe_idx);
    }
}

/**
 * Reserve WQE slots atomically
 * Returns the base WQE index for the reservation
 */
template <bool is_shared_among_ctas = false>
__device__ __forceinline__ uint64_t gda_reserve_wqe_slots(
    GdaDeviceQpState* qp,
    uint32_t num_wqes)
{
    uint64_t wqe_idx;
    if (is_shared_among_ctas) {
        wqe_idx = atomicAdd((unsigned long long*)qp->resv_head,
                            (unsigned long long)num_wqes);
    } else {
        wqe_idx = atomicAdd_block((unsigned long long*)qp->resv_head,
                                  (unsigned long long)num_wqes);
    }
    return wqe_idx;
}

//==============================================================================
// High-level operations
//==============================================================================

/**
 * Single RDMA WRITE with immediate doorbell
 */
__device__ __forceinline__ void gda_rdma_write(
    GdaDeviceQpState* qp,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    bool signaled = false)
{
    // Reserve slot
    uint64_t base_idx = gda_reserve_wqe_slots(qp, 1);
    uint16_t wqe_idx = (uint16_t)(base_idx & 0xFFFF);

    // Build WQE
    uint8_t fm_ce_se = signaled ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
    gda_build_rdma_write_wqe(qp, local_addr, local_lkey,
                              remote_addr, remote_rkey,
                              size, wqe_idx, fm_ce_se);

    // Submit
    gda_submit_requests(qp, base_idx, 1);
}

/**
 * Batched RDMA WRITE - builds WQE without immediate doorbell
 * Call gda_flush() to ring doorbell after batch
 */
__device__ __forceinline__ uint64_t gda_rdma_write_batched(
    GdaDeviceQpState* qp,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    uint64_t base_wqe_idx,
    uint16_t wqe_offset,
    bool signaled = false)
{
    uint16_t wqe_idx = (uint16_t)((base_wqe_idx + wqe_offset) & 0xFFFF);

    uint8_t fm_ce_se = signaled ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
    gda_build_rdma_write_wqe(qp, local_addr, local_lkey,
                              remote_addr, remote_rkey,
                              size, wqe_idx, fm_ce_se);

    return base_wqe_idx + wqe_offset + 1;
}

/**
 * Flush pending WQEs (ring doorbell)
 */
__device__ __forceinline__ void gda_flush(
    GdaDeviceQpState* qp,
    uint64_t base_wqe_idx,
    uint16_t num_wqes)
{
    gda_submit_requests(qp, base_wqe_idx, num_wqes);
}

//==============================================================================
// Optimized kernels using nvshmem-style batching
//==============================================================================

/**
 * Concurrent RDMA writes with single doorbell
 * Each thread builds one WQE, thread 0 rings doorbell
 */
__global__ void gda_concurrent_write_kernel(
    GdaDeviceQpState* qp,
    uint64_t* local_addrs,
    uint32_t* local_lkeys,
    uint64_t* remote_addrs,
    uint32_t remote_rkey,
    uint32_t size,
    int num_writes,
    uint64_t* start_ns,
    uint64_t* end_ns)
{
    int tid = threadIdx.x;

    __shared__ uint64_t base_wqe_idx;

    // Thread 0 reserves all slots and records start time
    if (tid == 0) {
        *start_ns = gda_globaltimer();
        base_wqe_idx = gda_reserve_wqe_slots(qp, num_writes);
    }
    __syncthreads();

    // Each thread builds one WQE
    if (tid < num_writes) {
        uint16_t wqe_idx = (uint16_t)((base_wqe_idx + tid) & 0xFFFF);
        bool signaled = (tid == num_writes - 1);  // Signal last WQE

        gda_build_rdma_write_wqe(
            qp, local_addrs[tid], local_lkeys[tid],
            remote_addrs[tid], remote_rkey,
            size, wqe_idx,
            signaled ? MLX5_WQE_CTRL_CQ_UPDATE : 0
        );
    }

    __syncthreads();

    // Thread 0 submits all WQEs with single doorbell
    if (tid == 0) {
        gda_submit_requests(qp, base_wqe_idx, num_writes);
        *end_ns = gda_globaltimer();
    }
}

}  // namespace gicc::mlx5
