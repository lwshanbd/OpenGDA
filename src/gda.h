/**
 * gda.h - OpenGDA: GPU-Direct Async
 *
 * This is the main public header for the OpenGDA library.
 * OpenGDA (GPU-Direct Async) provides asynchronous GPU-direct RDMA
 * communication using Libfabric and ROCm.
 */

#ifndef OPENGDA_H
#define OPENGDA_H

// C++ Interface
#include "bootstrap/common.hpp"
#include "network/ofi.hpp"
#ifdef BOOTSTRAP_PMI2
#include "bootstrap/pmi2.hpp"
#endif

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



#endif // __cplusplus

#endif // OPENGDA_H
