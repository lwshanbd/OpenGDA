#include "pmi2.hpp"
#include <cstring>
#include <unistd.h>
#include <cstdint>

PMI2::PMI2() : Bootstrap(), rank_(0), size_(0), spawned_(0), appnum_(0),
               local_rank_(-1), local_size_(-1), node_id_(-1) {
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

    int rc = PMI2_Init(&spawned_, &size_, &rank_, &appnum_);
    if (rc != PMI2_SUCCESS) {
        fprintf(stderr, "PMI2_Init failed (%d)\n", rc);
        return false;
    }

    // Query node-local information
    if (!query_node_info()) {
        fprintf(stderr, "PMI2: Warning: Could not query node info, IPC optimization disabled\n");
        // Not fatal - continue without node info
    }

    bootstrap_initialized = true;
    return true;
}

bool PMI2::query_node_info() {
    char value[PMI2_MAX_VALLEN];
    int found = 0;

    // Try to get local rank (waitfor=0 means don't wait)
    int rc = PMI2_Info_GetNodeAttr("localRank", value, sizeof(value), &found, 0);
    if (rc == PMI2_SUCCESS && found) {
        local_rank_ = atoi(value);
    } else {
        // Fallback: assume single process per node
        local_rank_ = 0;
    }

    // Try to get local size
    found = 0;
    rc = PMI2_Info_GetNodeAttr("localRankCount", value, sizeof(value), &found, 0);
    if (rc == PMI2_SUCCESS && found) {
        local_size_ = atoi(value);
    } else {
        // Fallback: assume single process per node
        local_size_ = 1;
    }

    // Calculate node_id based on ranks
    // node_id = global_rank - local_rank (for rank 0 on each node)
    // This gives us a unique but not necessarily contiguous node ID
    // Better approach: exchange node info and compute unique IDs
    if (local_rank_ >= 0 && local_size_ > 0) {
        // Simple approach: derive from global rank
        // Ranks on same node have consecutive global ranks, so:
        // node_id = rank / local_size (approximately, assumes balanced distribution)
        // But this is not reliable. Instead, use a hash of hostname.
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) == 0) {
            // Simple hash of hostname
            uint64_t hash = 0;
            for (char* p = hostname; *p; p++) {
                hash = hash * 31 + (unsigned char)*p;
            }
            node_id_ = (int)(hash % 1000000);  // Use lower bits as node ID
        } else {
            node_id_ = rank_;  // Fallback
        }
    }

    return (local_rank_ >= 0 && local_size_ > 0);
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
    return rank_;
}

int PMI2::get_size() const {
    return size_;
}

int PMI2::get_local_rank() const {
    return local_rank_;
}

int PMI2::get_local_size() const {
    return local_size_;
}

int PMI2::get_node_id() const {
    return node_id_;
}

std::string PMI2::get_bootstrap_name() const {
    return "pmi2";
}

bool PMI2::bootstrap_barrier() {
    if (!bootstrap_initialized) {
        fprintf(stderr, "PMI2: bootstrap_barrier called before initialization\n");
        return false;
    }

    int rc = PMI2_KVS_Fence();
    if (rc != PMI2_SUCCESS) {
        fprintf(stderr, "PMI2_KVS_Fence failed (%d)\n", rc);
        return false;
    }

    return true;
}

bool PMI2::bootstrap_kvs_put(const char* key, const char* value) {
    if (!bootstrap_initialized) {
        fprintf(stderr, "PMI2: bootstrap_kvs_put called before initialization\n");
        return false;
    }

    int rc = PMI2_KVS_Put(key, value);
    if (rc != PMI2_SUCCESS) {
        fprintf(stderr, "PMI2_KVS_Put failed for key '%s' (%d)\n", key, rc);
        return false;
    }

    return true;
}

bool PMI2::bootstrap_kvs_get(const char* key, char* value, int* value_len) {
    if (!bootstrap_initialized) {
        fprintf(stderr, "PMI2: bootstrap_kvs_get called before initialization\n");
        return false;
    }

    int rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key, value, PMI2_MAX_VALLEN, value_len);
    if (rc != PMI2_SUCCESS) {
        fprintf(stderr, "PMI2_KVS_Get failed for key '%s' (%d)\n", key, rc);
        return false;
    }

    return true;
}

bool PMI2::bootstrap_exchange(const char* my_data, size_t data_len, char* all_data) {
    if (!bootstrap_initialized) {
        fprintf(stderr, "PMI2: bootstrap_exchange called before initialization\n");
        return false;
    }

    // Put my data with rank-based key
    char key[PMI2_MAX_KEYLEN];
    snprintf(key, sizeof(key), "addr-%d", rank_);

    int rc = PMI2_KVS_Put(key, my_data);
    if (rc != PMI2_SUCCESS) {
        fprintf(stderr, "PMI2_KVS_Put failed in exchange (%d)\n", rc);
        return false;
    }

    // Barrier to ensure all ranks have put their data
    rc = PMI2_KVS_Fence();
    if (rc != PMI2_SUCCESS) {
        fprintf(stderr, "PMI2_KVS_Fence failed in exchange (%d)\n", rc);
        return false;
    }

    // Get data from all ranks
    for (int i = 0; i < size_; i++) {
        snprintf(key, sizeof(key), "addr-%d", i);

        char val[PMI2_MAX_VALLEN];
        int vallen;
        rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key, val, sizeof(val), &vallen);
        if (rc != PMI2_SUCCESS) {
            fprintf(stderr, "PMI2_KVS_Get failed for rank %d in exchange (%d)\n", i, rc);
            return false;
        }

        // Copy to output buffer (each slot is data_len + 1 for null terminator)
        char* slot = all_data + i * (data_len + 1);
        strncpy(slot, val, data_len);
        slot[data_len] = '\0';
    }

    return true;
}
