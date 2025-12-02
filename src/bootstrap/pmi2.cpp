#include "pmi2.hpp"

PMI2::PMI2() : Bootstrap() {
    // Constructor - don't initialize here, use bootstrap_initialize()
}

PMI2::~PMI2() {
    // Cleanup if still initialized
    if (bootstrap_initialized) {
        bootstrap_finalize();
    }
}

bool PMI2::bootstrap_initialize() {
    if (bootstrap_initialized) {
        return true;  // Already initialized
    }
    int rc = PMI2_Init(&spawned, &size, &rank, &appnum);
    if (rc != PMI2_SUCCESS) {
        fprintf(stderr, "PMI2_Init failed (%d)\n", rc);
        return false;
    }

    bootstrap_initialized = true;
    return true;
}

bool PMI2::bootstrap_finalize() {
    if (!bootstrap_initialized) {
        return true;  // Already finalized
    }

    int rc = PMI2_Finalize();
    if (rc != PMI2_SUCCESS) {
        fprintf(stderr, "PMI2_Finalize failed (%d)\n", rc);
        return false;
    }

    bootstrap_initialized = false;
    return true;
}

int PMI2::get_rank() const {
    return rank;
}

std::string PMI2::get_bootstrap_name() const {
    return "pmi2";
}