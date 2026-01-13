/**
 * gda_barrier_proxy.h - CPU Proxy Barrier API for Unlimited GPU Barrier Iterations
 *
 * This module provides a GPU-driven barrier implementation using the Dissemination
 * algorithm with O(log P) rounds. The CPU proxy thread continuously rearms DWQ
 * operations, enabling unlimited barrier iterations from a persistent GPU kernel.
 *
 * Key Features:
 * - Dissemination barrier: O(log P) network operations per barrier
 * - Threshold-based triggering: Round-by-round triggering using counter thresholds
 * - Slot-based windowing: W reusable slots for unlimited iterations
 * - Slot safety: d_slot_done mechanism ensures safe slot reuse
 * - CPU proxy thread: Continuously rearms completed slots
 */

#ifndef OPENGDA_BARRIER_PROXY_H
#define OPENGDA_BARRIER_PROXY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration Constants
// ============================================================================

#define GDA_PROXY_DEFAULT_WINDOW_SIZE 16
#define GDA_PROXY_MAX_WINDOW_SIZE 64
#define GDA_PROXY_MAX_ROUNDS 6  // log2(64) = 6 for up to 64 ranks
#define GDA_PROXY_RING_SIZE 256

// Slot states for GPU/CPU coordination
#define GDA_SLOT_NEED_QUEUE 0
#define GDA_SLOT_ARMED 1

// ============================================================================
// Device-Side Structures (GPU-accessible)
// ============================================================================

/**
 * GPU-accessible barrier context for Dissemination barrier with proxy support.
 * This structure is designed for efficient GPU access and must remain stable
 * in memory while the GPU kernel is running.
 */
typedef struct gda_proxy_barrier_dev {
    // Basic info
    int mype;                                // My rank
    int npes;                                // Total number of ranks
    int num_rounds;                          // Number of dissemination rounds (ceil(log2(npes)))
    int window_size;                         // Number of reusable slots

    // Per-round targets for dissemination (computed once, constant)
    int round_targets[GDA_PROXY_MAX_ROUNDS]; // send_to[k] = (mype + 2^k) % npes
    int round_sources[GDA_PROXY_MAX_ROUNDS]; // recv_from[k] = (mype - 2^k + npes) % npes

    // GPU-visible slot management (pinned memory)
    volatile int* slot_state;                // [window_size]: NEED_QUEUE(0) or ARMED(1)

    // Per-slot trigger MMIO addresses (GPU writes to trigger DWQ work)
    volatile uint64_t** slot_trigger_addrs;  // [window_size]: trigger doorbell per slot

    // Per-round receive counters (GPU memory, remote peers atomic-add to these)
    volatile uint64_t* d_round_recv;         // [num_rounds]: arrival counter per round

    // Per-slot completion signals (GPU memory, NIC writes when slot ops complete)
    volatile uint64_t* d_slot_done;          // [window_size]: completion signal per slot

    // Epoch tracking (GPU memory)
    volatile uint64_t* d_epoch;              // Current barrier epoch (shared)

    // Statistics (GPU memory)
    volatile uint64_t* d_spin_cycles;        // Total cycles spent spinning
} gda_proxy_barrier_dev_t;

// ============================================================================
// Host-Side Opaque Handle
// ============================================================================

/**
 * Opaque handle for host-side proxy barrier management.
 * Contains fabric resources, proxy thread state, and device context.
 */
typedef struct gda_proxy_barrier gda_proxy_barrier_t;

// ============================================================================
// Host API - Lifecycle Management
// ============================================================================

/**
 * Allocate and initialize a proxy-enabled GPU barrier.
 * This creates all necessary DWQ resources and prepares for unlimited iterations.
 *
 * @param window_size Number of reusable slots (0 for default=16, max=64)
 * @return Barrier handle, or NULL on failure
 *
 * Usage:
 *   gda_proxy_barrier_t* pb = gda_proxy_barrier_alloc(16);
 *   gda_proxy_barrier_dev_t* dev = gda_proxy_barrier_get_dev(pb);
 *   // Copy dev to GPU, launch kernel
 *   gda_proxy_start(pb);
 *   // GPU runs barriers
 *   gda_proxy_stop(pb);
 *   gda_proxy_barrier_free(pb);
 */
gda_proxy_barrier_t* gda_proxy_barrier_alloc(int window_size);

/**
 * Free proxy barrier resources.
 * Proxy thread must be stopped first (call gda_proxy_stop()).
 *
 * @param barrier Barrier handle to free
 */
void gda_proxy_barrier_free(gda_proxy_barrier_t* barrier);

/**
 * Get the device-side context for GPU kernel use.
 * The returned pointer can be copied to GPU memory.
 *
 * @param barrier Barrier handle
 * @return Device context, or NULL if invalid
 */
gda_proxy_barrier_dev_t* gda_proxy_barrier_get_dev(gda_proxy_barrier_t* barrier);

// ============================================================================
// Host API - Proxy Thread Control
// ============================================================================

/**
 * Start the CPU proxy thread.
 * Must be called BEFORE launching the GPU kernel.
 * The proxy thread will continuously rearm completed slots.
 *
 * @param barrier Barrier handle
 * @return 0 on success, negative error code on failure
 */
int gda_proxy_start(gda_proxy_barrier_t* barrier);

/**
 * Stop the CPU proxy thread.
 * Call AFTER the GPU kernel has completed.
 *
 * @param barrier Barrier handle
 * @return 0 on success, negative error code on failure
 */
int gda_proxy_stop(gda_proxy_barrier_t* barrier);

/**
 * Check if proxy thread is running.
 *
 * @param barrier Barrier handle
 * @return 1 if running, 0 if stopped, negative on error
 */
int gda_proxy_is_running(gda_proxy_barrier_t* barrier);

// ============================================================================
// Host API - Statistics
// ============================================================================

/**
 * Proxy barrier statistics.
 */
typedef struct gda_proxy_stats {
    uint64_t total_rearms;         // Total slot rearms performed
    uint64_t queue_polls;          // Number of polling iterations
    uint64_t cq_events_drained;    // CQ events drained to prevent overflow
    uint64_t gpu_spin_cycles;      // GPU cycles spent spinning (from d_spin_cycles)
} gda_proxy_stats_t;

/**
 * Get barrier statistics.
 *
 * @param barrier Barrier handle
 * @param stats Output: statistics structure
 * @return 0 on success, negative error code on failure
 */
int gda_proxy_barrier_get_stats(gda_proxy_barrier_t* barrier, gda_proxy_stats_t* stats);

// ============================================================================
// GPU Kernel Macros (for use in HIP/CUDA kernels)
// ============================================================================

#if defined(__HIPCC__) || defined(__CUDACC__)

/**
 * GPU-side Dissemination barrier wait with proxy support.
 *
 * Algorithm:
 * 1. Compute slot = epoch % window_size
 * 2. Compute expected_done = (epoch / window_size) + 1 (monotonic counter)
 * 3. Wait for slot to be armed (slot_state[slot] == ARMED)
 * 4. For each round k = 0..num_rounds-1:
 *    a. Write (k+1) to trigger doorbell to fire round k's DWQ work
 *    b. Wait for d_round_recv[k] >= (epoch+1) (peer's atomic arrived)
 * 5. Wait for d_slot_done[slot] >= expected_done (all outbound RDMA ops complete)
 *    - NIC increments d_slot_done[slot] when completion_cntr reaches num_rounds
 *    - Using monotonic counter avoids hipMemcpy clearing in proxy hot path
 * 6. Mark slot_state[slot] = NEED_QUEUE (tells CPU proxy to rearm)
 * 7. Increment epoch
 *
 * CRITICAL: Step 5 prevents race conditions where a slot is re-queued
 * while outbound RDMA operations are still in-flight.
 *
 * This macro should be called by a single thread (typically thread 0).
 */
#define gda_gpu_proxy_barrier_wait(dev) do { \
    /* Get current epoch and compute slot */ \
    uint64_t _epoch = __atomic_load_n((unsigned long long*)(dev)->d_epoch, __ATOMIC_ACQUIRE); \
    int _slot = (int)(_epoch % (dev)->window_size); \
    int _num_rounds = (dev)->num_rounds; \
    \
    /* Compute expected d_slot_done value (monotonic counter) */ \
    /* Each time this slot completes, d_slot_done[slot] increments by 1 */ \
    uint64_t _expected_done = (_epoch / (dev)->window_size) + 1; \
    \
    /* Wait for slot to be armed by CPU proxy */ \
    while (__atomic_load_n(&(dev)->slot_state[_slot], __ATOMIC_ACQUIRE) != GDA_SLOT_ARMED) { \
        /* spin */ \
    } \
    \
    /* Dissemination rounds with threshold-based triggering */ \
    uint64_t _next_epoch = _epoch + 1; \
    for (int _k = 0; _k < _num_rounds; _k++) { \
        /* Write threshold (k+1) to trigger round k's DWQ work */ \
        *(dev)->slot_trigger_addrs[_slot] = (uint64_t)(_k + 1); \
        __threadfence_system(); \
        \
        /* Wait for arrival signal from round k's source peer */ \
        while (__atomic_load_n((unsigned long long*)&(dev)->d_round_recv[_k], __ATOMIC_ACQUIRE) < _next_epoch) { \
            /* spin */ \
        } \
    } \
    \
    __threadfence_system(); \
    \
    /* Wait for d_slot_done[slot] >= expected_done, indicating all outbound */ \
    /* RDMA operations have completed. Uses monotonic counter to avoid */ \
    /* hipMemcpy clearing in proxy hot path. */ \
    while (__atomic_load_n((unsigned long long*)&(dev)->d_slot_done[_slot], __ATOMIC_ACQUIRE) < _expected_done) { \
        /* spin */ \
    } \
    \
    /* Mark slot for CPU proxy to rearm */ \
    __atomic_store_n(&(dev)->slot_state[_slot], GDA_SLOT_NEED_QUEUE, __ATOMIC_RELEASE); \
    \
    /* Increment epoch for next barrier */ \
    __atomic_store_n((unsigned long long*)(dev)->d_epoch, _next_epoch, __ATOMIC_RELEASE); \
} while(0)

/**
 * Initialize device-side barrier state (call once at kernel start).
 * This should be called before the first barrier.
 */
#define gda_gpu_proxy_barrier_init(dev) do { \
    /* Ensure epoch starts at 0 */ \
    if (__atomic_load_n((unsigned long long*)(dev)->d_epoch, __ATOMIC_ACQUIRE) != 0) { \
        __atomic_store_n((unsigned long long*)(dev)->d_epoch, 0ULL, __ATOMIC_RELEASE); \
    } \
    __threadfence_system(); \
} while(0)

#endif /* __HIPCC__ || __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OPENGDA_BARRIER_PROXY_H */
