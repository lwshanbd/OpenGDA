/**
 * gda.cpp - Implementation of OpenGDA C API
 */

#include "gda.h"
#include "bootstrap/common.hpp"
#include "network/ofi.hpp"
#include <cstdio>
#include <cerrno>
#include <memory>
#include <vector>

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

// ============================================================================
// Internal: Operation wrapper for simplified API
// ============================================================================

struct gda_op {
    std::unique_ptr<DWQOperation> dwq_op;
    int dest_rank;
    bool is_put;  // true = put, false = get
};

// Pool of reusable handles (to avoid frequent allocation)
static std::vector<gda_handle_t*> handle_pool;
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
    int size = bootstrap->get_size();

    // Create OFI with bootstrap for address exchange
    ofi = std::make_unique<OFI>(rank, size, bootstrap.get());

    // Note: OFI constructor now handles initialization including address exchange

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

// ============================================================================
// Query Functions
// ============================================================================

int gda_rank(void) {
    if (!initialized || !ofi) {
        return -1;
    }
    return ofi->get_rank();
}

int gda_size(void) {
    if (!initialized || !ofi) {
        return -1;
    }
    return ofi->get_size();
}

void* gda_gpu_buf(void) {
    if (!initialized || !ofi) {
        return nullptr;
    }
    const DefaultMRInfo* gpu_mr = ofi->get_default_gpu_mr();
    if (!gpu_mr || !gpu_mr->allocated) {
        return nullptr;
    }
    return gpu_mr->buffer;
}

size_t gda_gpu_buf_size(void) {
    if (!initialized || !ofi) {
        return 0;
    }
    const DefaultMRInfo* gpu_mr = ofi->get_default_gpu_mr();
    if (!gpu_mr || !gpu_mr->allocated) {
        return 0;
    }
    return gpu_mr->size;
}

// ============================================================================
// Synchronization
// ============================================================================

int gda_barrier(void) {
    if (!initialized || !bootstrap) {
        fprintf(stderr, "OpenGDA: Not initialized\n");
        return -1;
    }
    bootstrap->bootstrap_barrier();
    return 0;
}

// ============================================================================
// Simplified RDMA Operations
// ============================================================================

gda_handle_t* gda_put(void* local_buf, size_t size, int dest_rank, size_t dest_offset) {
    if (!initialized || !ofi) {
        fprintf(stderr, "OpenGDA: Not initialized\n");
        return nullptr;
    }

    // Get peer info
    const PeerInfo* peer = ofi->get_peer_info(dest_rank);
    if (!peer || !peer->valid) {
        fprintf(stderr, "OpenGDA: Invalid destination rank %d\n", dest_rank);
        return nullptr;
    }

    // Create DWQ operation
    auto dwq_op = ofi->create_dwq_operation();
    if (!dwq_op) {
        fprintf(stderr, "OpenGDA: Failed to create DWQ operation\n");
        return nullptr;
    }

    // Prepare the write operation
    bool prepared = dwq_op->prepare_write_explicit(
        local_buf,
        size,
        peer->fi_addr,
        dest_offset,
        peer->mr_info.gpu_mr_key
    );

    if (!prepared) {
        fprintf(stderr, "OpenGDA: Failed to prepare put operation\n");
        return nullptr;
    }

    // Allocate handle
    gda_handle_t* handle = new gda_handle_t;
    handle->op = new gda_op;
    handle->op->dwq_op = std::move(dwq_op);
    handle->op->dest_rank = dest_rank;
    handle->op->is_put = true;

    // Fill GPU handle
    handle->gpu.trigger_addr = handle->op->dwq_op->get_trigger_addr();
    handle->gpu.completion_addr = handle->op->dwq_op->get_completion_signal();
    handle->gpu.trigger_threshold = handle->op->dwq_op->get_trigger_threshold();

    return handle;
}

gda_handle_t* gda_get(void* local_buf, size_t size, int src_rank, size_t src_offset) {
    if (!initialized || !ofi) {
        fprintf(stderr, "OpenGDA: Not initialized\n");
        return nullptr;
    }

    // Get peer info
    const PeerInfo* peer = ofi->get_peer_info(src_rank);
    if (!peer || !peer->valid) {
        fprintf(stderr, "OpenGDA: Invalid source rank %d\n", src_rank);
        return nullptr;
    }

    // Create DWQ operation
    auto dwq_op = ofi->create_dwq_operation();
    if (!dwq_op) {
        fprintf(stderr, "OpenGDA: Failed to create DWQ operation\n");
        return nullptr;
    }

    // Prepare the read operation
    bool prepared = dwq_op->prepare_read_explicit(
        local_buf,
        size,
        peer->fi_addr,
        src_offset,
        peer->mr_info.gpu_mr_key
    );

    if (!prepared) {
        fprintf(stderr, "OpenGDA: Failed to prepare get operation\n");
        return nullptr;
    }

    // Allocate handle
    gda_handle_t* handle = new gda_handle_t;
    handle->op = new gda_op;
    handle->op->dwq_op = std::move(dwq_op);
    handle->op->dest_rank = src_rank;
    handle->op->is_put = false;

    // Fill GPU handle
    handle->gpu.trigger_addr = handle->op->dwq_op->get_trigger_addr();
    handle->gpu.completion_addr = handle->op->dwq_op->get_completion_signal();
    handle->gpu.trigger_threshold = handle->op->dwq_op->get_trigger_threshold();

    return handle;
}

int gda_wait(gda_handle_t* handle) {
    return gda_wait_timeout(handle, -1);
}

int gda_wait_timeout(gda_handle_t* handle, int timeout_ms) {
    if (!handle || !handle->op || !handle->op->dwq_op) {
        return -EINVAL;
    }

    CompletionQueue* cq = ofi->get_completion_queue();
    if (!cq) {
        return -EINVAL;
    }

    uint64_t seq_num = handle->op->dwq_op->get_seq_num();
    if (seq_num == CompletionQueue::INVALID_SEQ) {
        return -EINVAL;
    }

    bool completed = cq->wait_one(seq_num, timeout_ms);
    if (!completed) {
        return -ETIMEDOUT;
    }

    return 0;
}

int gda_reset(gda_handle_t* handle) {
    if (!handle || !handle->op || !handle->op->dwq_op) {
        return -EINVAL;
    }

    if (!handle->op->dwq_op->reset()) {
        return -1;
    }

    // Clear GPU handle (will be refilled on next put/get)
    handle->gpu.trigger_addr = nullptr;
    handle->gpu.completion_addr = nullptr;
    handle->gpu.trigger_threshold = 0;

    return 0;
}

void gda_free(gda_handle_t* handle) {
    if (!handle) {
        return;
    }

    if (handle->op) {
        // DWQOperation destructor handles cleanup
        delete handle->op;
    }

    delete handle;
}

void gda_flush(void) {
    if (initialized && ofi) {
        ofi->flush_work_queue();
    }
}

} // extern "C"
