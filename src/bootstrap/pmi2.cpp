#include "pmi2.hpp"
#include <cstring>

PMI2::PMI2() : Bootstrap(), rank_(0), size_(0), spawned_(0), appnum_(0) {
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
    return rank_;
}

int PMI2::get_size() const {
    return size_;
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
