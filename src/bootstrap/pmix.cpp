#include "pmix.hpp"

PMIX::PMIX() : Bootstrap() {
    // TODO: Initialize PMIx
    pmix_proc_t myproc;
    pmix_value_t *val;

    // Initialize PMIx client
    int rc = PMIx_Init(&myproc, NULL, 0);
    if (rc != PMIX_SUCCESS) {
        fprintf(stderr, "PMIx_Init failed (%d)\n", rc);
        exit(1);
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
}

PMIX::~PMIX() {
    PMIx_Finalize(NULL, 0);
}
