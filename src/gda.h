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

#ifdef __cplusplus
extern "C" {
#endif

// Version information
#define OPENGDA_VERSION_MAJOR 0
#define OPENGDA_VERSION_MINOR 1
#define OPENGDA_VERSION_PATCH 0

// Forward declarations
typedef struct gda_context gda_context_t;

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

#ifdef __cplusplus
}
#endif

#endif // OPENGDA_H
