/**
 * gda_barrier.h - GPU Barrier with CPU Proxy for Unlimited Iterations
 *
 * This module provides a GPU-initiated dissemination barrier that can
 * execute unlimited iterations via CPU proxy thread slot reuse.
 *
 * Key Features:
 * - Dissemination barrier: O(log P) phases per barrier
 * - Windowed slots: Reusable DWQ counter pairs
 * - CPU proxy thread: Continuously rearms completed slots
 * - GPU-driven: All barrier logic runs on GPU
 *
 * Usage:
 *   // Host code
 *   gda_barrier_t* b = gda_barrier_alloc(16);  // window_size=16
 *   gda_barrier_dev_t* dev = gda_barrier_get_dev(b);
 *   // Copy dev pointer to GPU
 *   gda_barrier_start(b);
 *   // Launch GPU kernel that calls gda_gpu_barrier(dev)
 *   hipDeviceSynchronize();
 *   gda_barrier_stop(b);
 *   gda_barrier_free(b);
 */

#ifndef GDA_BARRIER_H
#define GDA_BARRIER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration
// ============================================================================

#define GDA_BARRIER_DEFAULT_WINDOW 16
#define GDA_BARRIER_MAX_WINDOW 64
#define GDA_BARRIER_MAX_PHASES 6  // log2(64) for up to 64 ranks

// Slot states
#define GDA_SLOT_NEED_REARM 0
#define GDA_SLOT_READY 1

// ============================================================================
// GPU-Accessible Device Context
// ============================================================================

/**
 * GPU-accessible barrier device context.
 * Passed to GPU kernel for barrier operations.
 */
typedef struct gda_barrier_dev {
    int rank;
    int size;
    int num_phases;
    int window_size;
    int total_slots;  // window_size * num_phases

    // Slot states (pinned memory, GPU-visible)
    volatile int* slot_state;

    // Trigger MMIO addresses per slot (GPU writes to trigger DWQ)
    volatile uint64_t** trigger_addrs;

    // Per-phase arrival counters (GPU memory, peers atomic-add to these)
    volatile uint64_t* phase_counters;
} gda_barrier_dev_t;

// ============================================================================
// Host-Side Opaque Handle
// ============================================================================

typedef struct gda_barrier gda_barrier_t;

// ============================================================================
// Host API
// ============================================================================

/**
 * Allocate barrier resources.
 * @param window_size Number of reusable slots (0 for default=16)
 * @return Barrier handle, or NULL on failure
 */
gda_barrier_t* gda_barrier_alloc(int window_size);

/**
 * Free barrier resources.
 * Must call gda_barrier_stop() first if proxy is running.
 */
void gda_barrier_free(gda_barrier_t* barrier);

/**
 * Get device context for GPU kernel.
 * The returned pointer can be passed to GPU kernel.
 */
gda_barrier_dev_t* gda_barrier_get_dev(gda_barrier_t* barrier);

/**
 * Start CPU proxy thread.
 * Call BEFORE launching GPU kernel.
 */
int gda_barrier_start(gda_barrier_t* barrier);

/**
 * Stop CPU proxy thread.
 * Call AFTER GPU kernel completes.
 */
int gda_barrier_stop(gda_barrier_t* barrier);

/**
 * Get statistics.
 */
typedef struct gda_barrier_stats {
    uint64_t total_rearms;
    uint64_t polls;
} gda_barrier_stats_t;

int gda_barrier_get_stats(gda_barrier_t* barrier, gda_barrier_stats_t* stats);

// ============================================================================
// GPU Kernel Macro
// ============================================================================

#if defined(__HIPCC__) || defined(__CUDACC__)

/**
 * GPU-side dissemination barrier.
 * Call from a single thread (typically thread 0).
 *
 * Algorithm:
 * - For each phase p (0 to num_phases-1):
 *   1. Compute slot = epoch % window_size * num_phases + phase
 *   2. Wait for slot_state[slot] == READY
 *   3. Trigger atomic add to partner at distance 2^p
 *   4. Wait for phase_counters[p] >= epoch+1
 *   5. Mark slot for rearm
 * - Increment epoch
 */
#define gda_gpu_barrier(dev, epoch_ptr) do { \
    uint64_t _epoch = *(epoch_ptr); \
    int _num_phases = (dev)->num_phases; \
    int _window_size = (dev)->window_size; \
    uint64_t _next = _epoch + 1; \
    \
    for (int _p = 0; _p < _num_phases; _p++) { \
        int _slot = (int)((_epoch % _window_size) * _num_phases + _p); \
        \
        /* Wait for slot to be ready */ \
        while (__atomic_load_n(&(dev)->slot_state[_slot], __ATOMIC_ACQUIRE) != GDA_SLOT_READY) { \
        } \
        \
        /* Trigger atomic to partner */ \
        __threadfence_system(); \
        *(dev)->trigger_addrs[_slot] = 1; \
        __threadfence_system(); \
        \
        /* Wait for arrival from partner */ \
        while (__atomic_load_n((unsigned long long*)&(dev)->phase_counters[_p], __ATOMIC_ACQUIRE) < _next) { \
        } \
        \
        /* Mark slot for rearm */ \
        __atomic_store_n(&(dev)->slot_state[_slot], GDA_SLOT_NEED_REARM, __ATOMIC_RELEASE); \
    } \
    \
    *(epoch_ptr) = _next; \
} while(0)

#endif /* __HIPCC__ || __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* GDA_BARRIER_H */
