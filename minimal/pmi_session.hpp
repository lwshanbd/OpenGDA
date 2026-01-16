/*
 * pmi_session.hpp - Thin PMI2 wrapper for process management
 */
#pragma once

#include <pmi2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

class PmiSession {
public:
    int rank;
    int size;

    PmiSession() : rank(-1), size(0) {
        int spawned, appnum;
        PMI2_Init(&spawned, &size, &rank, &appnum);
    }

    ~PmiSession() {
        PMI2_Finalize();
    }

    // No copy/move
    PmiSession(const PmiSession&) = delete;
    PmiSession& operator=(const PmiSession&) = delete;

    void barrier() {
        int rc = PMI2_KVS_Fence();
        if (rc != PMI2_SUCCESS) {
            fprintf(stderr, "Rank %d: PMI2_KVS_Fence failed: %d\n", rank, rc);
            exit(1);
        }
    }

    void kvs_put(const char* key, const char* value) {
        int rc = PMI2_KVS_Put(key, value);
        if (rc != PMI2_SUCCESS) {
            fprintf(stderr, "Rank %d: PMI2_KVS_Put(%s) failed: %d\n", rank, key, rc);
            exit(1);
        }
    }

    void kvs_get(const char* key, char* value, int maxlen) {
        int vallen;
        int rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key, value, maxlen, &vallen);
        if (rc != PMI2_SUCCESS) {
            fprintf(stderr, "Rank %d: PMI2_KVS_Get(%s) failed: %d\n", rank, key, rc);
            exit(1);
        }
    }
};
