/**
 * gda_barrier_proxy.h - CPU Proxy Barrier API for Unlimited GPU Barrier Iterations
 *
 * Based on P1 prototype: All-to-all barrier with single threshold triggering.
 * Each rank atomically adds +1 to all other ranks' d_arrive counter.
 * GPU waits for d_arrive >= (epoch+1) * (npes-1).
 *
 * Key Features:
 * - All-to-all barrier: O(P) network operations per barrier (simpler than dissemination)
 * - Single threshold triggering: All atomics fire at once when GPU writes to doorbell
 * - Slot-based windowing: W reusable slots for unlimited iterations
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

// Slot states for GPU/CPU coordination
#define GDA_SLOT_NEED_QUEUE 0
#define GDA_SLOT_ARMED 1

// ============================================================================
// Device-Side Structures (GPU-accessible)
// ============================================================================

/**
 * GPU-accessible barrier context for all-to-all barrier with proxy support.
 * This structure is designed for efficient GPU access and must remain stable
 * in memory while the GPU kernel is running.
 */
typedef struct gda_proxy_barrier_dev {
    // Basic info
    int mype;                                // My rank
    int npes;                                // Total number of ranks
    int num_peers;                           // npes - 1
    int window_size;                         // Number of reusable slots

    // GPU-visible slot management (pinned memory)
    volatile int* slot_state;                // [window_size]: NEED_QUEUE(0) or ARMED(1)

    // Per-slot trigger MMIO addresses (GPU writes to trigger DWQ work)
    volatile uint64_t** slot_trigger_addrs;  // [window_size]: trigger doorbell per slot

    // Arrival counter (GPU memory, all peers atomic-add to this)
    volatile uint64_t* d_arrive;             // Arrival counter for this rank

    // Current epoch (GPU memory)
    volatile uint64_t* d_epoch;              // Current barrier epoch
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
    uint64_t gpu_spin_cycles;      // Not used in all-to-all variant
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
 * GPU-side all-to-all barrier wait with proxy support.
 *
 * Algorithm (from P1 prototype):
 * 1. Compute slot = epoch % window_size
 * 2. Wait for slot to be armed (slot_state[slot] == ARMED)
 * 3. Write 1 to trigger doorbell (fires all atomics to all peers)
 * 4. Wait for d_arrive >= (epoch+1) * num_peers
 * 5. Mark slot_state[slot] = NEED_QUEUE
 * 6. Increment epoch
 *
 * This macro should be called by a single thread (typically thread 0).
 */
#define gda_gpu_proxy_barrier_wait(dev) do { \
    /* Get current epoch and compute slot */ \
    uint64_t _epoch = __atomic_load_n((unsigned long long*)(dev)->d_epoch, __ATOMIC_ACQUIRE); \
    int _slot = (int)(_epoch % (dev)->window_size); \
    int _num_peers = (dev)->num_peers; \
    \
    /* Wait for slot to be armed by CPU proxy */ \
    while (__atomic_load_n(&(dev)->slot_state[_slot], __ATOMIC_ACQUIRE) != GDA_SLOT_ARMED) { \
        /* spin */ \
    } \
    \
    /* Trigger all atomics by writing to doorbell */ \
    __threadfence_system(); \
    *(dev)->slot_trigger_addrs[_slot] = 1; \
    __threadfence_system(); \
    \
    /* Wait for arrivals from all peers */ \
    uint64_t _expected = (_epoch + 1) * _num_peers; \
    while (__atomic_load_n((unsigned long long*)(dev)->d_arrive, __ATOMIC_ACQUIRE) < _expected) { \
        /* spin */ \
    } \
    \
    /* Mark slot for CPU proxy to rearm */ \
    __atomic_store_n(&(dev)->slot_state[_slot], GDA_SLOT_NEED_QUEUE, __ATOMIC_RELEASE); \
    \
    /* Increment epoch for next barrier */ \
    __atomic_store_n((unsigned long long*)(dev)->d_epoch, _epoch + 1, __ATOMIC_RELEASE); \
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
