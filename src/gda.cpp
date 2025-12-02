/**
 * gda.cpp - Implementation of OpenGDA C API
 */

#include "gda.h"
#include "bootstrap/common.hpp"
#include "network/ofi.hpp"
#include <cstdio>
#include <memory>

// Version string (compile-time construction)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
static const char* VERSION_STRING =
    TOSTRING(OPENGDA_VERSION_MAJOR) "."
    TOSTRING(OPENGDA_VERSION_MINOR) "."
    TOSTRING(OPENGDA_VERSION_PATCH);

// Global initialization flag
static bool initialized = false;
std::unique_ptr<Bootstrap> bootstrap;
std::unique_ptr<OFI> ofi;
extern "C" {

int gda_init(void) {
    if (initialized) {
        fprintf(stderr, "OpenGDA: Already initialized\n");
        return -1;
    }

    #ifdef BOOTSTRAP_PMI2
    bootstrap = Bootstrap::create_bootstrap("pmi2");
    #elif defined(BOOTSTRAP_PMIX)
    bootstrap = Bootstrap::create_bootstrap("pmix");
    #else
    bootstrap = nullptr;
    fprintf(stderr, "OpenGDA: No bootstrap type defined\n");
    return -1;
    #endif

    if (bootstrap == nullptr) {
        fprintf(stderr, "OpenGDA: Failed to create bootstrap\n");
        return -1;
    }

    // Initialize bootstrap
    if (!bootstrap->bootstrap_initialize()) {
        fprintf(stderr, "OpenGDA: Failed to initialize bootstrap\n");
        bootstrap = nullptr;
        return -1;
    }

    int rank = bootstrap->get_rank();

    ofi = std::make_unique<OFI>(rank);
    if (!ofi->ofi_initialize()) {
        fprintf(stderr, "OpenGDA: Failed to initialize OFI\n");
        ofi = nullptr;
        return -1;
    }

    initialized = true;
    return 0;
}

int gda_finalize(void) {
    if (!initialized) {
        fprintf(stderr, "OpenGDA: Not initialized\n");
        return -1;
    }

    bootstrap->bootstrap_finalize();

    bootstrap = nullptr;
    initialized = false;
    return 0;
}

const char* gda_get_version(void) {
    return VERSION_STRING;
}

// ============================================================================
// Memory Registration API
// ============================================================================

gda_mr_t* gda_register_memory(void* buf, size_t size, int is_device_mem) {
    if (!initialized) {
        fprintf(stderr, "OpenGDA: Not initialized (call gda_init first)\n");
        return nullptr;
    }

    if (!ofi) {
        fprintf(stderr, "OpenGDA: OFI not initialized\n");
        return nullptr;
    }

    struct fid_mr* mr = ofi->register_memory(buf, size, is_device_mem != 0);
    return reinterpret_cast<gda_mr_t*>(mr);
}

void gda_deregister_memory(gda_mr_t* mr) {
    if (!initialized) {
        fprintf(stderr, "OpenGDA: Not initialized\n");
        return;
    }

    if (!ofi) {
        fprintf(stderr, "OpenGDA: OFI not initialized\n");
        return;
    }

    if (!mr) {
        fprintf(stderr, "OpenGDA: Cannot deregister null MR\n");
        return;
    }

    struct fid_mr* fid_mr = reinterpret_cast<struct fid_mr*>(mr);
    ofi->deregister_memory(fid_mr);
}

uint64_t gda_mr_get_key(gda_mr_t* mr) {
    if (!mr) {
        fprintf(stderr, "OpenGDA: Cannot get key from null MR\n");
        return 0;
    }

    struct fid_mr* fid_mr = reinterpret_cast<struct fid_mr*>(mr);
    return fi_mr_key(fid_mr);
}

void* gda_mr_get_desc(gda_mr_t* mr) {
    if (!mr) {
        fprintf(stderr, "OpenGDA: Cannot get descriptor from null MR\n");
        return nullptr;
    }

    struct fid_mr* fid_mr = reinterpret_cast<struct fid_mr*>(mr);
    return fi_mr_desc(fid_mr);
}

} // extern "C"
