#include "pmix.hpp"

PMIX::PMIX() : Bootstrap() {
    // Constructor - don't initialize here, use bootstrap_initialize()
}

PMIX::~PMIX() {
    // Cleanup if still initialized
    if (bootstrap_initialized) {
        bootstrap_finalize();
    }
}

bool PMIX::bootstrap_initialize() {
    if (bootstrap_initialized) {
        return true;  // Already initialized
    }

    pmix_value_t *val;

    // Initialize PMIx client
    int rc = PMIx_Init(&myproc, NULL, 0);
    if (rc != PMIX_SUCCESS) {
        fprintf(stderr, "PMIx_Init failed (%d)\n", rc);
        return false;
    }

    // Get rank
    PMIX_PROC_LOAD(&myproc, myproc.nspace, PMIX_RANK_WILDCARD);
    rc = PMIx_Get(&myproc, PMIX_RANK, NULL, 0, &val);
    if (rc == PMIX_SUCCESS) {
        rank = val->data.rank;
        PMIX_VALUE_RELEASE(val);
    }

    // Get size
    rc = PMIx_Get(&myproc, PMIX_JOB_SIZE, NULL, 0, &val);
    if (rc == PMIX_SUCCESS) {
        size = val->data.uint32;
        PMIX_VALUE_RELEASE(val);
    }

    bootstrap_initialized = true;
    return true;
}

bool PMIX::bootstrap_finalize() {
    if (!bootstrap_initialized) {
        return true;  // Already finalized
    }

    PMIx_Finalize(NULL, 0);
    bootstrap_initialized = false;
    return true;
}

std::string PMIX::get_bootstrap_name() const {
    return "pmix";
}
