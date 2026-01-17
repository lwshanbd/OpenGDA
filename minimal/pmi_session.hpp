/*
 * pmi_session.hpp - Thin PMI2 wrapper for process management
 */
#pragma once

#include <pmi2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

class PmiSession {
public:
    int rank;
    int size;
    int local_rank;
    int local_size;
    char hostname[256];

    PmiSession() : rank(-1), size(0), local_rank(-1), local_size(0) {
        int spawned, appnum;
        PMI2_Init(&spawned, &size, &rank, &appnum);

        // Get hostname
        gethostname(hostname, sizeof(hostname));

        // Try to get local rank from environment
        const char* env_vars[] = {"SLURM_LOCALID", "FLUX_TASK_LOCAL_ID",
                                   "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", NULL};
        for (int i = 0; env_vars[i] && local_rank < 0; i++) {
            const char* val = getenv(env_vars[i]);
            if (val) local_rank = atoi(val);
        }

        // Try to get local size from environment
        const char* size_vars[] = {"SLURM_NTASKS_PER_NODE", "FLUX_LOCAL_RANKS", NULL};
        for (int i = 0; size_vars[i] && local_size <= 0; i++) {
            const char* val = getenv(size_vars[i]);
            if (val) local_size = atoi(val);
        }
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

    // Exchange hostnames and check if peer is on same node
    // Call after barrier to ensure all hostnames are published
    bool is_same_node(int peer_rank) {
        // Publish my hostname
        char key[64];
        snprintf(key, sizeof(key), "hostname-%d", rank);
        kvs_put(key, hostname);
        barrier();

        // Get peer's hostname
        char peer_hostname[256];
        snprintf(key, sizeof(key), "hostname-%d", peer_rank);
        kvs_get(key, peer_hostname, sizeof(peer_hostname));

        return strcmp(hostname, peer_hostname) == 0;
    }

    // Build a map of which ranks are on same node (call once after init)
    // Returns array[size] where array[i] = true if rank i is on same node
    bool* build_locality_map() {
        // Publish my hostname
        char key[64];
        snprintf(key, sizeof(key), "hostname-%d", rank);
        kvs_put(key, hostname);
        barrier();

        // Get all hostnames and compare
        bool* same_node = new bool[size];
        for (int i = 0; i < size; i++) {
            char peer_hostname[256];
            snprintf(key, sizeof(key), "hostname-%d", i);
            kvs_get(key, peer_hostname, sizeof(peer_hostname));
            same_node[i] = (strcmp(hostname, peer_hostname) == 0);
        }
        return same_node;
    }
};
