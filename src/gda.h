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
 */
typedef struct gda_gpu_handle {
    volatile uint64_t* trigger_addr;      // Write threshold here to trigger
    volatile uint64_t* completion_addr;   // Poll this until non-zero
    uint64_t trigger_threshold;           // Value to write to trigger
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
#define gda_gpu_trigger(gpu_handle) do { \
    *(gpu_handle).trigger_addr = (gpu_handle).trigger_threshold; \
    __threadfence_system(); \
} while(0)

#define gda_gpu_wait(gpu_handle) do { \
    while (*(gpu_handle).completion_addr == 0) {} \
    __threadfence_system(); \
} while(0)
#endif

#ifdef __CUDACC__
#define gda_gpu_trigger(gpu_handle) do { \
    *(gpu_handle).trigger_addr = (gpu_handle).trigger_threshold; \
    __threadfence_system(); \
} while(0)

#define gda_gpu_wait(gpu_handle) do { \
    while (*(gpu_handle).completion_addr == 0) {} \
    __threadfence_system(); \
} while(0)
#endif

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
