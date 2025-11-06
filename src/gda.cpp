/**
 * gda.cpp - Implementation of OpenGDA C API
 */

#include "gda.h"
#include <cstdio>

// Version string (compile-time construction)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
static const char* VERSION_STRING =
    TOSTRING(OPENGDA_VERSION_MAJOR) "."
    TOSTRING(OPENGDA_VERSION_MINOR) "."
    TOSTRING(OPENGDA_VERSION_PATCH);

// Global initialization flag
static bool initialized = false;

extern "C" {

int gda_init(void) {
    if (initialized) {
        fprintf(stderr, "OpenGDA: Already initialized\n");
        return -1;
    }

    #ifdef BOOTSTRAP_PMI2
    std::unique_ptr<Bootstrap> bootstrap = Bootstrap::create_bootstrap("pmi2");
    #else
    std::unique_ptr<Bootstrap> bootstrap = Bootstrap::create_bootstrap("NONE");
    #endif
    if (bootstrap == nullptr) {
        fprintf(stderr, "OpenGDA: Failed to create bootstrap\n");
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

    // TODO: Cleanup subsystems

    initialized = false;
    return 0;
}

const char* gda_get_version(void) {
    return VERSION_STRING;
}

} // extern "C"
