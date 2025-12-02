/**
 * gda.h - OpenGDA: GPU-Direct Async
 *
 * This is the main public header for the OpenGDA library.
 * OpenGDA (GPU-Direct Async) provides asynchronous GPU-direct RDMA
 * communication using Libfabric and ROCm.
 *
 * This header provides the C API.
 * For C++ API, include <opengda/bootstrap/common.hpp> and <opengda/network/ofi.hpp>
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
#define OPENGDA_VERSION_MINOR 1
#define OPENGDA_VERSION_PATCH 0

// Forward declarations
typedef struct gda_context gda_context_t;
typedef struct gda_mr gda_mr_t;

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
