/**
 * gda.h - OpenGDA: GPU-Direct Async
 *
 * This is the main public header for the OpenGDA library.
 * OpenGDA (GPU-Direct Async) provides asynchronous GPU-direct RDMA
 * communication using Libfabric and ROCm.
 *
 * Simple Usage:
 *   gda_init();
 *   gda_handle_t h = gda_put(local_gpu_buf, size, dest_rank, dest_offset);
 *   // In GPU kernel:
 *   //   gda_gpu_trigger(h->gpu);
 *   //   gda_gpu_wait(h->gpu);
 *   gda_wait(h);
 *   gda_free(h);
 *   gda_finalize();
 */

#ifndef OPENGDA_H
#define OPENGDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version information
#define OPENGDA_VERSION_MAJOR 0
#define OPENGDA_VERSION_MINOR 2
#define OPENGDA_VERSION_PATCH 0

// Forward declarations
typedef struct gda_context gda_context_t;
typedef struct gda_mr gda_mr_t;

// ============================================================================
// GPU Handle - passed to GPU kernel for trigger/wait operations
// ============================================================================

/**
 * GPU-accessible handle for triggering and waiting on operations.
 * This structure is designed to be passed to GPU kernels.
 *
 * Two modes are supported:
 * - DWQ mode (is_ipc=0): GPU writes to trigger_addr to initiate RDMA via NIC
 * - IPC mode (is_ipc=1): GPU can directly copy to/from ipc_dest_addr for same-node peers
 *
 * Completion uses monotonic counter (same as proxy barrier's d_slot_done):
 * - NIC atomically adds +1 to completion_addr after each operation
 * - GPU waits for completion_addr >= completion_threshold
 * - This avoids hipMemset in hot path which causes jitter
 */
typedef struct gda_gpu_handle {
    volatile uint64_t* trigger_addr;      // Write threshold here to trigger (DWQ mode)
    volatile uint64_t* completion_addr;   // Poll this for completion (GPU memory, NIC atomic adds)
    uint64_t trigger_threshold;           // Value to write to trigger (DWQ mode)
    uint64_t completion_threshold;        // Wait until completion_addr >= this value (monotonic)

    // IPC mode fields (for same-node GPU communication)
    int is_ipc;                           // 1 = IPC mode (direct copy), 0 = DWQ mode (RDMA)
    void* ipc_dest_addr;                  // Destination address in peer's GPU memory (IPC mode)
    void* ipc_src_addr;                   // Source address (local GPU memory)
    size_t ipc_size;                      // Transfer size in bytes
} gda_gpu_handle_t;

/**
 * Opaque operation handle (CPU side)
 */
typedef struct gda_op gda_op_t;

/**
 * Complete handle returned by gda_put/gda_get
 * Contains both CPU handle and GPU-accessible info
 */
typedef struct gda_handle {
    gda_op_t* op;                         // Opaque operation handle (CPU use)
    gda_gpu_handle_t gpu;                 // GPU-accessible handle
} gda_handle_t;

/**
 * Initialize OpenGDA library
 * @return 0 on success, negative error code on failure
 */
int gda_init(void);

/**
 * Finalize OpenGDA library and cleanup resources
 * @return 0 on success, negative error code on failure
 */
int gda_finalize(void);

/**
 * Get library version string
 * @return Version string in format "major.minor.patch"
 */
const char* gda_get_version(void);

// ============================================================================
// Query Functions
// ============================================================================

/**
 * Get current process rank
 * @return Rank (0 to size-1), or -1 if not initialized
 */
int gda_rank(void);

/**
 * Get total number of processes
 * @return Size, or -1 if not initialized
 */
int gda_size(void);

/**
 * Get default GPU buffer address (pre-registered)
 * Use this buffer for zero-copy RDMA operations
 * @return GPU buffer pointer, or NULL if not available
 */
void* gda_gpu_buf(void);

/**
 * Get default GPU buffer size
 * @return Size in bytes, or 0 if not available
 */
size_t gda_gpu_buf_size(void);

// ============================================================================
// Synchronization
// ============================================================================

/**
 * Barrier across all processes
 * @return 0 on success, negative error code on failure
 */
int gda_barrier(void);

// ============================================================================
// Simplified RDMA Operations (GPU-triggered)
// ============================================================================

/**
 * Create a GPU-triggered put (write) operation
 *
 * The local buffer must be within the default GPU buffer (gda_gpu_buf()).
 * The remote offset is relative to the remote rank's default GPU buffer.
 *
 * @param local_buf    Local GPU buffer (must be within gda_gpu_buf())
 * @param size         Number of bytes to transfer
 * @param dest_rank    Destination rank
 * @param dest_offset  Offset into destination's GPU buffer
 * @return Handle for the operation, or NULL on failure
 *
 * Usage:
 *   gda_handle_t* h = gda_put(local_buf, 4096, 1, 0);
 *   // Copy h->gpu to GPU memory, then in kernel:
 *   //   *gpu_handle.trigger_addr = gpu_handle.trigger_threshold;
 *   //   __threadfence_system();
 *   //   while (*gpu_handle.completion_addr == 0) {}
 *   gda_wait(h);
 *   gda_free(h);
 */
gda_handle_t* gda_put(void* local_buf, size_t size, int dest_rank, size_t dest_offset);

/**
 * Create a GPU-triggered get (read) operation
 *
 * @param local_buf    Local GPU buffer to receive data (must be within gda_gpu_buf())
 * @param size         Number of bytes to transfer
 * @param src_rank     Source rank
 * @param src_offset   Offset into source's GPU buffer
 * @return Handle for the operation, or NULL on failure
 */
gda_handle_t* gda_get(void* local_buf, size_t size, int src_rank, size_t src_offset);

/**
 * Wait for an operation to complete (CPU-side)
 * @param handle Operation handle
 * @return 0 on success, negative error code on failure/timeout
 */
int gda_wait(gda_handle_t* handle);

/**
 * Wait for an operation with timeout
 * @param handle Operation handle
 * @param timeout_ms Timeout in milliseconds
 * @return 0 on success, -ETIMEDOUT on timeout, negative on error
 */
int gda_wait_timeout(gda_handle_t* handle, int timeout_ms);

/**
 * Test if an operation has completed (non-blocking)
 * @param handle Operation handle
 * @return 1 if completed, 0 if still pending, negative on error
 */
int gda_test(gda_handle_t* handle);

/**
 * Test if any operation in an array has completed (non-blocking)
 * @param handles Array of operation handles
 * @param count Number of handles in the array
 * @param completed_idx Output: index of the first completed operation (if any)
 * @return 1 if at least one completed, 0 if none completed, negative on error
 */
int gda_test_any(gda_handle_t** handles, int count, int* completed_idx);

/**
 * Test if all operations in an array have completed (non-blocking)
 * @param handles Array of operation handles
 * @param count Number of handles in the array
 * @return 1 if all completed, 0 if any still pending, negative on error
 */
int gda_test_all(gda_handle_t** handles, int count);

/**
 * Reset operation for reuse (avoids reallocation overhead)
 * After reset, call gda_put/gda_get to prepare a new operation
 * @param handle Operation handle to reset
 * @return 0 on success, negative error code on failure
 */
int gda_reset(gda_handle_t* handle);

/**
 * Free operation handle and release resources
 * @param handle Operation handle to free
 */
void gda_free(gda_handle_t* handle);

/**
 * Flush work queue (call after each batch of operations)
 * Required to release DWQ slots for subsequent operations
 */
void gda_flush(void);

// ============================================================================
// GPU Kernel Helper Macros (for use in HIP/CUDA kernels)
// ============================================================================

#ifdef __HIPCC__
/**
 * Trigger the operation - single thread version
 * - DWQ mode: Writes to trigger_addr to initiate RDMA via NIC (efficient)
 * - IPC mode: Sequential copy (slow for large data, use gda_gpu_trigger_all instead)
 *
 * Use this for small transfers or when only master thread should trigger.
 */
#define gda_gpu_trigger(gpu_handle) do { \
    if ((gpu_handle).is_ipc) { \
        /* IPC mode: sequential 64-bit copy */ \
        volatile uint64_t* dst = (volatile uint64_t*)(gpu_handle).ipc_dest_addr; \
        volatile uint64_t* src = (volatile uint64_t*)(gpu_handle).ipc_src_addr; \
        size_t count = (gpu_handle).ipc_size / sizeof(uint64_t); \
        for (size_t i = 0; i < count; i++) { \
            dst[i] = src[i]; \
        } \
        __threadfence_system(); \
        *(gpu_handle).completion_addr = 1; \
    } else { \
        /* DWQ mode: trigger RDMA via NIC */ \
        *(gpu_handle).trigger_addr = (gpu_handle).trigger_threshold; \
        __threadfence_system(); \
    } \
} while(0)

/**
 * Parallel trigger - ALL threads in grid should call this
 * - IPC mode: Parallel copy using all threads (efficient for large data)
 * - DWQ mode: Only thread 0 triggers, others do nothing
 *
 * Use this for large data transfers when all threads can participate.
 */
#define gda_gpu_trigger_all(gpu_handle) do { \
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x + \
                 (threadIdx.y + blockIdx.y * blockDim.y) * (blockDim.x * gridDim.x); \
    size_t total_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y; \
    if ((gpu_handle).is_ipc) { \
        /* IPC mode: parallel copy using all threads */ \
        volatile uint64_t* dst = (volatile uint64_t*)(gpu_handle).ipc_dest_addr; \
        volatile uint64_t* src = (volatile uint64_t*)(gpu_handle).ipc_src_addr; \
        size_t count = (gpu_handle).ipc_size / sizeof(uint64_t); \
        for (size_t i = tid; i < count; i += total_threads) { \
            dst[i] = src[i]; \
        } \
        __threadfence_system(); \
        /* Only thread 0 signals completion after all threads sync */ \
        __syncthreads(); \
        if (tid == 0) { \
            *(gpu_handle).completion_addr = 1; \
        } \
    } else { \
        /* DWQ mode: only thread 0 triggers */ \
        if (tid == 0) { \
            *(gpu_handle).trigger_addr = (gpu_handle).trigger_threshold; \
            __threadfence_system(); \
        } \
    } \
} while(0)

/**
 * Wait for DWQ operation to complete.
 *
 * Uses monotonic counter pattern (same as proxy barrier's d_slot_done):
 * - NIC atomically adds +1 to completion_addr after operation completes
 * - GPU waits until completion_addr >= completion_threshold
 * - This avoids hipMemset in hot path which causes jitter
 *
 * Uses cache-bypassing loads to avoid L1/L2 cache coherency issues when
 * polling memory written by NIC. Similar to NVSHMEM's poll_cq approach.
 *
 * The inline asm uses glc (globally coherent) and slc (system level coherent)
 * flags to bypass L1/L2 caches and read directly from memory/LLC.
 */
#define gda_gpu_wait(gpu_handle) do { \
    __threadfence_system(); /* Ensure trigger is visible to NIC before polling */ \
    uint64_t _completion_val; \
    volatile uint64_t* _completion_ptr = (gpu_handle).completion_addr; \
    do { \
        /* Cache-bypassing load for AMD GPUs (MI200/MI300) */ \
        asm volatile( \
            "global_load_dwordx2 %0, %1, off glc slc\n" \
            "s_waitcnt vmcnt(0)" \
            : "=v"(_completion_val) \
            : "v"(_completion_ptr) \
            : "memory" \
        ); \
    } while (_completion_val < (gpu_handle).completion_threshold); \
    __threadfence();  /* GPU memory barrier after completion */ \
} while(0)

#endif

#ifdef __CUDACC__
/**
 * Trigger the operation - single thread version
 */
#define gda_gpu_trigger(gpu_handle) do { \
    if ((gpu_handle).is_ipc) { \
        volatile uint64_t* dst = (volatile uint64_t*)(gpu_handle).ipc_dest_addr; \
        volatile uint64_t* src = (volatile uint64_t*)(gpu_handle).ipc_src_addr; \
        size_t count = (gpu_handle).ipc_size / sizeof(uint64_t); \
        for (size_t i = 0; i < count; i++) { \
            dst[i] = src[i]; \
        } \
        __threadfence_system(); \
        *(gpu_handle).completion_addr = 1; \
    } else { \
        *(gpu_handle).trigger_addr = (gpu_handle).trigger_threshold; \
        __threadfence_system(); \
    } \
} while(0)

/**
 * Parallel trigger - ALL threads in grid should call this
 */
#define gda_gpu_trigger_all(gpu_handle) do { \
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x + \
                 (threadIdx.y + blockIdx.y * blockDim.y) * (blockDim.x * gridDim.x); \
    size_t total_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y; \
    if ((gpu_handle).is_ipc) { \
        volatile uint64_t* dst = (volatile uint64_t*)(gpu_handle).ipc_dest_addr; \
        volatile uint64_t* src = (volatile uint64_t*)(gpu_handle).ipc_src_addr; \
        size_t count = (gpu_handle).ipc_size / sizeof(uint64_t); \
        for (size_t i = tid; i < count; i += total_threads) { \
            dst[i] = src[i]; \
        } \
        __threadfence_system(); \
        __syncthreads(); \
        if (tid == 0) { \
            *(gpu_handle).completion_addr = 1; \
        } \
    } else { \
        if (tid == 0) { \
            *(gpu_handle).trigger_addr = (gpu_handle).trigger_threshold; \
            __threadfence_system(); \
        } \
    } \
} while(0)

/**
 * Wait for DWQ operation to complete.
 * Uses monotonic counter pattern (same as proxy barrier's d_slot_done).
 * Uses cache-bypassing loads for proper coherency with NIC writes.
 */
#define gda_gpu_wait(gpu_handle) do { \
    __threadfence_system(); /* Ensure trigger is visible to NIC before polling */ \
    uint64_t _completion_val; \
    volatile uint64_t* _completion_ptr = (gpu_handle).completion_addr; \
    do { \
        /* Cache-bypassing load for NVIDIA GPUs (Volta+) */ \
        asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" \
                     : "=l"(_completion_val) \
                     : "l"(_completion_ptr) \
                     : "memory"); \
    } while (_completion_val < (gpu_handle).completion_threshold); \
    __threadfence();  /* GPU memory barrier after completion */ \
} while(0)

#endif

// ============================================================================
// GPU-side Barrier API
// ============================================================================

/**
 * Maximum number of ranks supported for GPU barrier
 */
#define GDA_BARRIER_MAX_RANKS 64

/**
 * Maximum number of phases in dissemination algorithm (log2(MAX_RANKS))
 */
#define GDA_BARRIER_MAX_PHASES 6

/**
 * Maximum number of barrier iterations supported
 * (DWQ operations are one-shot, so we pre-allocate for all iterations)
 */
#define GDA_BARRIER_MAX_ITERS 128

/**
 * GPU-accessible barrier structure
 * This structure is designed to be used within GPU kernels.
 */
typedef struct gda_gpu_barrier {
    volatile uint64_t* sync_arr;          // Sync array: one slot per rank
    volatile uint64_t* sync_counter;      // Current barrier counter (also iteration index)
    int mype;                             // My rank
    int npes;                             // Total number of ranks
    int num_phases;                       // Number of phases in dissemination
    int max_iters;                        // Max iterations allocated
    // Handles for each phase and iteration: [iter * num_phases + phase]
    gda_gpu_handle_t* phase_handles;      // Dynamically allocated array
    int phase_targets[GDA_BARRIER_MAX_PHASES];   // Target ranks for each phase
    int phase_sources[GDA_BARRIER_MAX_PHASES];   // Source ranks for each phase
} gda_gpu_barrier_t;

/**
 * Allocate and initialize GPU barrier resources
 * Must be called by all ranks before using the barrier.
 *
 * @param max_iters Maximum number of barrier iterations (DWQ ops are one-shot)
 *                  Default: GDA_BARRIER_MAX_ITERS if 0
 * @return Pointer to GPU-accessible barrier structure, or NULL on failure
 *
 * Usage:
 *   gda_gpu_barrier_t* barrier = gda_gpu_barrier_alloc(10); // 10 iterations max
 *   // Copy to GPU, then in kernel:
 *   //   gda_gpu_barrier_wait(barrier);
 *   gda_gpu_barrier_free(barrier);
 */
gda_gpu_barrier_t* gda_gpu_barrier_alloc(int max_iters);

/**
 * Free GPU barrier resources
 * @param barrier Barrier handle to free
 */
void gda_gpu_barrier_free(gda_gpu_barrier_t* barrier);

/**
 * Reset barrier for reuse (call between barrier uses if needed)
 * @param barrier Barrier handle
 * @return 0 on success, negative on error
 */
int gda_gpu_barrier_reset(gda_gpu_barrier_t* barrier);

// ============================================================================
// GPU Barrier Macros (for use in HIP/CUDA kernels)
// ============================================================================

#if defined(__HIPCC__) || defined(__CUDACC__)

/**
 * GPU-side barrier wait (dissemination algorithm)
 * Should be called by a single thread (typically thread 0)
 */
#define gda_gpu_barrier_wait(barrier) do { \
    volatile uint64_t* sync_arr = (barrier)->sync_arr; \
    volatile uint64_t* counter = (barrier)->sync_counter; \
    uint64_t cur_iter = *counter; \
    uint64_t next_iter = cur_iter + 1; \
    int mype = (barrier)->mype; \
    int num_phases = (barrier)->num_phases; \
    \
    /* Update our sync slot with next iteration value */ \
    sync_arr[mype] = next_iter; \
    __threadfence_system(); \
    \
    /* Dissemination algorithm (similar to NVSHMEM) */ \
    /* Use iteration-indexed handles since DWQ ops are one-shot */ \
    int handle_base = (int)(cur_iter * num_phases); \
    for (int phase = 0; phase < num_phases; phase++) { \
        /* Signal to neighbor: trigger put to write our sync value */ \
        gda_gpu_trigger((barrier)->phase_handles[handle_base + phase]); \
        \
        /* Wait for signal from neighbor (their sync value arriving) */ \
        int src = (barrier)->phase_sources[phase]; \
        while (sync_arr[src] < next_iter) { \
            /* spin */ \
        } \
        __threadfence_system(); \
    } \
    \
    /* Update counter for next barrier iteration */ \
    *counter = next_iter; \
    __threadfence_system(); \
} while(0)

#endif /* __HIPCC__ || __CUDACC__ */

// ============================================================================
// Legacy Memory Registration API (for advanced use)
// ============================================================================

/**
 * Register memory region for RDMA operations
 * @param buf Pointer to memory buffer (host or device memory)
 * @param size Size of the memory region in bytes
 * @param is_device_mem 1 if GPU memory, 0 if host memory
 * @return Memory region handle on success, NULL on failure
 */
gda_mr_t* gda_register_memory(void* buf, size_t size, int is_device_mem);

/**
 * Deregister memory region
 * @param mr Memory region handle returned by gda_register_memory
 */
void gda_deregister_memory(gda_mr_t* mr);

/**
 * Get the remote key for a registered memory region
 * @param mr Memory region handle
 * @return Remote key value (used for RDMA operations)
 */
uint64_t gda_mr_get_key(gda_mr_t* mr);

/**
 * Get the local descriptor for a registered memory region
 * @param mr Memory region handle
 * @return Local descriptor pointer (used for local RDMA operations)
 */
void* gda_mr_get_desc(gda_mr_t* mr);

#ifdef __cplusplus
}
#endif

#endif // OPENGDA_H
