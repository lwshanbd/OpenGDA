#include "pmix.hpp"
#include <cstring>
#include <unistd.h>
#include <cstdint>

PMIX::PMIX() : Bootstrap(), rank_(0), size_(0), local_rank_(-1), local_size_(-1), node_id_(-1) {
    memset(&myproc_, 0, sizeof(myproc_));
}

PMIX::~PMIX() {
    if (bootstrap_initialized) {
        bootstrap_finalize();
    }
}

bool PMIX::bootstrap_initialize() {
    if (bootstrap_initialized) {
        return true;
    }

    pmix_status_t rc = PMIx_Init(&myproc_, NULL, 0);
    if (rc != PMIX_SUCCESS) {
        fprintf(stderr, "PMIx_Init failed: %s\n", PMIx_Error_string(rc));
        return false;
    }

    // Rank is directly available from myproc_
    rank_ = myproc_.rank;

    // Get job size
    pmix_proc_t wildcard;
    pmix_value_t *val;
    PMIX_PROC_LOAD(&wildcard, myproc_.nspace, PMIX_RANK_WILDCARD);

    rc = PMIx_Get(&wildcard, PMIX_JOB_SIZE, NULL, 0, &val);
    if (rc != PMIX_SUCCESS) {
        fprintf(stderr, "PMIx_Get(PMIX_JOB_SIZE) failed: %s\n", PMIx_Error_string(rc));
        PMIx_Finalize(NULL, 0);
        return false;
    }
    size_ = val->data.uint32;
    PMIX_VALUE_RELEASE(val);

    // Query node-local information
    if (!query_node_info()) {
        fprintf(stderr, "PMIX: Warning: Could not query node info, IPC optimization disabled\n");
        // Not fatal - continue without node info
    }

    bootstrap_initialized = true;
    return true;
}

bool PMIX::query_node_info() {
    pmix_value_t *val;
    pmix_status_t rc;

    // Get local rank (rank within node)
    rc = PMIx_Get(&myproc_, PMIX_LOCAL_RANK, NULL, 0, &val);
    if (rc == PMIX_SUCCESS) {
        local_rank_ = val->data.uint16;
        PMIX_VALUE_RELEASE(val);
    } else {
        local_rank_ = 0;  // Fallback
    }

    // Get local size (number of processes on this node)
    rc = PMIx_Get(&myproc_, PMIX_LOCAL_SIZE, NULL, 0, &val);
    if (rc == PMIX_SUCCESS) {
        local_size_ = val->data.uint32;
        PMIX_VALUE_RELEASE(val);
    } else {
        local_size_ = 1;  // Fallback
    }

    // Try to get node ID
    rc = PMIx_Get(&myproc_, PMIX_NODEID, NULL, 0, &val);
    if (rc == PMIX_SUCCESS) {
        node_id_ = val->data.uint32;
        PMIX_VALUE_RELEASE(val);
    } else {
        // Fallback: use hostname hash
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) == 0) {
            uint64_t hash = 0;
            for (char* p = hostname; *p; p++) {
                hash = hash * 31 + (unsigned char)*p;
            }
            node_id_ = (int)(hash % 1000000);
        } else {
            node_id_ = rank_;
        }
    }

    return (local_rank_ >= 0 && local_size_ > 0);
}

bool PMIX::bootstrap_finalize() {
    if (!bootstrap_initialized) {
        return true;
    }

    PMIx_Finalize(NULL, 0);
    bootstrap_initialized = false;
    return true;
}

int PMIX::get_rank() const {
    return rank_;
}

int PMIX::get_size() const {
    return size_;
}

int PMIX::get_local_rank() const {
    return local_rank_;
}

int PMIX::get_local_size() const {
    return local_size_;
}

int PMIX::get_node_id() const {
    return node_id_;
}

std::string PMIX::get_bootstrap_name() const {
    return "pmix";
}

bool PMIX::bootstrap_barrier() {
    if (!bootstrap_initialized) {
        fprintf(stderr, "PMIX: bootstrap_barrier called before initialization\n");
        return false;
    }

    // Fence with all procs in my namespace - this is the collective barrier
    pmix_proc_t wildcard;
    PMIX_PROC_LOAD(&wildcard, myproc_.nspace, PMIX_RANK_WILDCARD);

    pmix_status_t rc = PMIx_Fence(&wildcard, 1, NULL, 0);
    if (rc != PMIX_SUCCESS) {
        fprintf(stderr, "PMIx_Fence failed: %s\n", PMIx_Error_string(rc));
        return false;
    }

    return true;
}

bool PMIX::bootstrap_kvs_put(const char* key, const char* value) {
    if (!bootstrap_initialized) {
        fprintf(stderr, "PMIX: bootstrap_kvs_put called before initialization\n");
        return false;
    }

    // Just stage the data locally - no commit needed
    // The data will be exchanged during the next Fence call
    pmix_value_t val;
    PMIX_VALUE_CONSTRUCT(&val);
    val.type = PMIX_STRING;
    val.data.string = strdup(value);

    pmix_status_t rc = PMIx_Put(PMIX_GLOBAL, key, &val);
    PMIX_VALUE_DESTRUCT(&val);

    if (rc != PMIX_SUCCESS) {
        fprintf(stderr, "PMIx_Put failed for key '%s': %s\n", key, PMIx_Error_string(rc));
        return false;
    }

    return true;
}

bool PMIX::bootstrap_kvs_get(const char* key, char* value, int* value_len) {
    if (!bootstrap_initialized) {
        fprintf(stderr, "PMIX: bootstrap_kvs_get called before initialization\n");
        return false;
    }

    // After Fence, this is a pure local memory access - no network!
    pmix_proc_t wildcard;
    PMIX_PROC_LOAD(&wildcard, myproc_.nspace, PMIX_RANK_WILDCARD);

    pmix_value_t *val;
    pmix_status_t rc = PMIx_Get(&wildcard, key, NULL, 0, &val);
    if (rc != PMIX_SUCCESS) {
        fprintf(stderr, "PMIx_Get failed for key '%s': %s\n", key, PMIx_Error_string(rc));
        return false;
    }

    if (val->type == PMIX_STRING && val->data.string != NULL) {
        size_t len = strlen(val->data.string);
        strncpy(value, val->data.string, len);
        value[len] = '\0';
        if (value_len) {
            *value_len = (int)len;
        }
    }

    PMIX_VALUE_RELEASE(val);
    return true;
}

bool PMIX::bootstrap_exchange(const char* my_data, size_t data_len, char* all_data) {
    if (!bootstrap_initialized) {
        fprintf(stderr, "PMIX: bootstrap_exchange called before initialization\n");
        return false;
    }

    // Step 1: Put my data with rank-based key (local staging only)
    char key[256];
    snprintf(key, sizeof(key), "addr-%d", rank_);

    pmix_value_t val;
    PMIX_VALUE_CONSTRUCT(&val);
    val.type = PMIX_STRING;
    val.data.string = strdup(my_data);

    pmix_status_t rc = PMIx_Put(PMIX_GLOBAL, key, &val);
    PMIX_VALUE_DESTRUCT(&val);

    if (rc != PMIX_SUCCESS) {
        fprintf(stderr, "PMIx_Put failed in exchange: %s\n", PMIx_Error_string(rc));
        return false;
    }

    // Step 2: Fence with DATA COLLECTION (Modex)
    // Tell PMIx to collect all Put data and distribute to all ranks
    pmix_info_t info;
    bool collect = true;
    PMIX_INFO_LOAD(&info, PMIX_COLLECT_DATA, &collect, PMIX_BOOL);

    pmix_proc_t wildcard;
    PMIX_PROC_LOAD(&wildcard, myproc_.nspace, PMIX_RANK_WILDCARD);

    // Pass info to enable collective data exchange (tree/ring O(log N))
    rc = PMIx_Fence(&wildcard, 1, &info, 1);
    PMIX_INFO_DESTRUCT(&info);

    if (rc != PMIX_SUCCESS) {
        fprintf(stderr, "PMIx_Fence failed in exchange: %s\n", PMIx_Error_string(rc));
        return false;
    }

    // Step 3: Get all ranks' data - pure local memory access, no network!
    for (int i = 0; i < size_; i++) {
        snprintf(key, sizeof(key), "addr-%d", i);

        pmix_value_t *get_val;
        rc = PMIx_Get(&wildcard, key, NULL, 0, &get_val);
        if (rc != PMIX_SUCCESS) {
            fprintf(stderr, "PMIx_Get failed for rank %d: %s\n", i, PMIx_Error_string(rc));
            return false;
        }

        // Copy to output buffer
        char* slot = all_data + i * (data_len + 1);
        if (get_val->type == PMIX_STRING && get_val->data.string != NULL) {
            size_t src_len = strlen(get_val->data.string);
            size_t copy_len = (src_len < data_len) ? src_len : data_len;
            memcpy(slot, get_val->data.string, copy_len);
            slot[copy_len] = '\0';
        } else {
            slot[0] = '\0';
        }

        PMIX_VALUE_RELEASE(get_val);
    }

    return true;
}
