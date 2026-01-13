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
#include <chrono>
#include <thread>

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
    bool is_ipc;  // true = IPC mode (same-node), false = DWQ mode (RDMA)
    volatile uint64_t* ipc_completion_signal;  // Completion signal for IPC mode
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

#ifdef USE_AMDGPU
    // Check if IPC is available for same-node communication
    if (ofi->is_ipc_available(dest_rank)) {
        void* ipc_ptr = ofi->get_ipc_ptr(dest_rank);
        size_t ipc_size = ofi->get_ipc_size(dest_rank);

        // Validate offset and size
        if (dest_offset + size > ipc_size) {
            fprintf(stderr, "OpenGDA: IPC put exceeds peer buffer bounds\n");
            // Fall through to DWQ path
        } else {
            // Allocate handle for IPC mode
            gda_handle_t* handle = new gda_handle_t;
            handle->op = new gda_op;
            handle->op->dwq_op = nullptr;  // No DWQ operation needed
            handle->op->dest_rank = dest_rank;
            handle->op->is_put = true;
            handle->op->is_ipc = true;

            // Allocate completion signal for IPC mode
            volatile uint64_t* completion_signal = nullptr;
            CompletionSignalPool* pool = ofi->get_signal_pool();
            if (pool && pool->is_initialized()) {
                size_t offset;
                pool->allocate(&completion_signal, &offset);
                if (completion_signal) {
                    hipMemset((void*)completion_signal, 0, sizeof(uint64_t));
                }
            }

            // Fall back to DWQ if no completion signal
            if (!completion_signal) {
                delete handle->op;
                delete handle;
                // Fall through to DWQ path
                goto dwq_path;
            }

            handle->op->ipc_completion_signal = completion_signal;

            // Fill GPU handle for IPC mode
            handle->gpu.is_ipc = 1;
            handle->gpu.ipc_dest_addr = (char*)ipc_ptr + dest_offset;
            handle->gpu.ipc_src_addr = local_buf;
            handle->gpu.ipc_size = size;
            handle->gpu.completion_addr = completion_signal;
            handle->gpu.trigger_addr = nullptr;
            handle->gpu.trigger_threshold = 0;
            handle->gpu.completion_threshold = 1;  // IPC mode: always wait for >= 1

            OPENGDA_Debug("gda", "Created IPC put handle: local=%p -> peer %d offset=%zu size=%zu",
                         local_buf, dest_rank, dest_offset, size);

            return handle;
        }
    }

dwq_path:
#endif

    // Create DWQ operation (standard RDMA path)
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
    handle->op->is_ipc = false;
    handle->op->ipc_completion_signal = nullptr;

    // Fill GPU handle for DWQ mode
    handle->gpu.is_ipc = 0;
    handle->gpu.trigger_addr = handle->op->dwq_op->get_trigger_addr();
    handle->gpu.completion_addr = handle->op->dwq_op->get_completion_signal();
    handle->gpu.trigger_threshold = handle->op->dwq_op->get_trigger_threshold();
    handle->gpu.completion_threshold = handle->op->dwq_op->get_completion_threshold();
    handle->gpu.ipc_dest_addr = nullptr;
    handle->gpu.ipc_src_addr = nullptr;
    handle->gpu.ipc_size = 0;

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

#ifdef USE_AMDGPU
    // Check if IPC is available for same-node communication
    if (ofi->is_ipc_available(src_rank)) {
        void* ipc_ptr = ofi->get_ipc_ptr(src_rank);
        size_t ipc_size = ofi->get_ipc_size(src_rank);

        // Validate offset and size
        if (src_offset + size > ipc_size) {
            fprintf(stderr, "OpenGDA: IPC get exceeds peer buffer bounds\n");
            // Fall through to DWQ path
        } else {
            // Allocate handle for IPC mode
            gda_handle_t* handle = new gda_handle_t;
            handle->op = new gda_op;
            handle->op->dwq_op = nullptr;  // No DWQ operation needed
            handle->op->dest_rank = src_rank;
            handle->op->is_put = false;
            handle->op->is_ipc = true;

            // Allocate completion signal for IPC mode
            volatile uint64_t* completion_signal = nullptr;
            CompletionSignalPool* pool = ofi->get_signal_pool();
            if (pool && pool->is_initialized()) {
                size_t offset;
                pool->allocate(&completion_signal, &offset);
                if (completion_signal) {
                    hipMemset((void*)completion_signal, 0, sizeof(uint64_t));
                }
            }

            // Fall back to DWQ if no completion signal
            if (!completion_signal) {
                delete handle->op;
                delete handle;
                // Fall through to DWQ path
                goto dwq_path_get;
            }

            handle->op->ipc_completion_signal = completion_signal;

            // Fill GPU handle for IPC mode
            // Note: For get, source is the peer (ipc_src_addr), dest is local
            handle->gpu.is_ipc = 1;
            handle->gpu.ipc_src_addr = (char*)ipc_ptr + src_offset;  // Read from peer
            handle->gpu.ipc_dest_addr = local_buf;                    // Write to local
            handle->gpu.ipc_size = size;
            handle->gpu.completion_addr = completion_signal;
            handle->gpu.trigger_addr = nullptr;
            handle->gpu.trigger_threshold = 0;
            handle->gpu.completion_threshold = 1;  // IPC mode: always wait for >= 1

            OPENGDA_Debug("gda", "Created IPC get handle: peer %d offset=%zu -> local=%p size=%zu",
                         src_rank, src_offset, local_buf, size);

            return handle;
        }
    }

dwq_path_get:
#endif

    // Create DWQ operation (standard RDMA path)
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
    handle->op->is_ipc = false;
    handle->op->ipc_completion_signal = nullptr;

    // Fill GPU handle for DWQ mode
    handle->gpu.is_ipc = 0;
    handle->gpu.trigger_addr = handle->op->dwq_op->get_trigger_addr();
    handle->gpu.completion_addr = handle->op->dwq_op->get_completion_signal();
    handle->gpu.trigger_threshold = handle->op->dwq_op->get_trigger_threshold();
    handle->gpu.completion_threshold = handle->op->dwq_op->get_completion_threshold();
    handle->gpu.ipc_dest_addr = nullptr;
    handle->gpu.ipc_src_addr = nullptr;
    handle->gpu.ipc_size = 0;

    return handle;
}

int gda_wait(gda_handle_t* handle) {
    return gda_wait_timeout(handle, -1);
}

int gda_wait_timeout(gda_handle_t* handle, int timeout_ms) {
    if (!handle || !handle->op) {
        return -EINVAL;
    }

    // For IPC mode, completion is signaled immediately by the GPU
    if (handle->op->is_ipc) {
        // Just check if completion signal is set (GPU sets it after memcpy)
        if (handle->op->ipc_completion_signal) {
            // Wait for completion
            auto start = std::chrono::steady_clock::now();
            while (*handle->op->ipc_completion_signal == 0) {
                if (timeout_ms >= 0) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
                    if (elapsed >= timeout_ms) {
                        return -ETIMEDOUT;
                    }
                }
                // Brief pause
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }
        return 0;
    }

    // DWQ mode - use dwq_op
    if (!handle->op->dwq_op) {
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

int gda_test(gda_handle_t* handle) {
    if (!handle || !handle->op) {
        return -EINVAL;
    }

    // IPC mode - check completion signal directly
    if (handle->op->is_ipc) {
        if (handle->op->ipc_completion_signal) {
            return (*handle->op->ipc_completion_signal != 0) ? 1 : 0;
        }
        return 1;  // No signal means already done
    }

    // DWQ mode
    if (!handle->op->dwq_op) {
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

    return cq->poll_one(seq_num) ? 1 : 0;
}

int gda_test_any(gda_handle_t** handles, int count, int* completed_idx) {
    if (!handles || count <= 0) {
        return -EINVAL;
    }

    CompletionQueue* cq = ofi->get_completion_queue();

    for (int i = 0; i < count; i++) {
        gda_handle_t* h = handles[i];
        if (!h || !h->op) {
            continue;  // Skip invalid handles
        }

        // Check IPC mode handles
        if (h->op->is_ipc) {
            if (h->op->ipc_completion_signal && *h->op->ipc_completion_signal != 0) {
                if (completed_idx) {
                    *completed_idx = i;
                }
                return 1;
            }
            continue;
        }

        // Check DWQ mode handles
        if (!h->op->dwq_op || !cq) {
            continue;
        }

        uint64_t seq_num = h->op->dwq_op->get_seq_num();
        if (seq_num == CompletionQueue::INVALID_SEQ) {
            continue;
        }

        if (cq->poll_one(seq_num)) {
            if (completed_idx) {
                *completed_idx = i;
            }
            return 1;
        }
    }

    return 0;  // None completed
}

int gda_test_all(gda_handle_t** handles, int count) {
    if (!handles || count <= 0) {
        return -EINVAL;
    }

    CompletionQueue* cq = ofi->get_completion_queue();

    for (int i = 0; i < count; i++) {
        gda_handle_t* h = handles[i];
        if (!h || !h->op) {
            return -EINVAL;  // Invalid handle in array
        }

        // Check IPC mode handles
        if (h->op->is_ipc) {
            if (h->op->ipc_completion_signal && *h->op->ipc_completion_signal == 0) {
                return 0;  // Not completed
            }
            continue;
        }

        // Check DWQ mode handles
        if (!h->op->dwq_op || !cq) {
            return -EINVAL;
        }

        uint64_t seq_num = h->op->dwq_op->get_seq_num();
        if (seq_num == CompletionQueue::INVALID_SEQ) {
            return -EINVAL;
        }

        if (!cq->poll_one(seq_num)) {
            return 0;  // At least one not completed
        }
    }

    return 1;  // All completed
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
    handle->gpu.completion_threshold = 0;

    return 0;
}

void gda_free(gda_handle_t* handle) {
    if (!handle) {
        return;
    }

    if (handle->op) {
#ifdef USE_AMDGPU
        // Release IPC completion signal back to pool
        if (handle->op->is_ipc && handle->op->ipc_completion_signal) {
            CompletionSignalPool* pool = ofi->get_signal_pool();
            if (pool) {
                pool->release(handle->op->ipc_completion_signal);
            }
            handle->op->ipc_completion_signal = nullptr;
        }
#endif
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

// ============================================================================
// GPU-side Barrier Implementation
// ============================================================================

// Internal structure to hold barrier resources
struct gda_barrier_internal {
    gda_gpu_barrier_t gpu_barrier;           // GPU-accessible part
    std::vector<gda_handle_t*> put_handles;  // DWQ handles for signaling
    uint64_t* host_sync_arr;                 // Host copy for debugging
    size_t sync_arr_offset;                  // Offset in GPU buffer
};

static std::vector<gda_barrier_internal*> barrier_pool;

gda_gpu_barrier_t* gda_gpu_barrier_alloc(int max_iters) {
    if (!initialized || !ofi) {
        fprintf(stderr, "OpenGDA: Not initialized\n");
        return nullptr;
    }

    int mype = ofi->get_rank();
    int npes = ofi->get_size();

    if (npes > GDA_BARRIER_MAX_RANKS) {
        fprintf(stderr, "OpenGDA: Too many ranks for GPU barrier (%d > %d)\n",
                npes, GDA_BARRIER_MAX_RANKS);
        return nullptr;
    }

    // Default max_iters
    if (max_iters <= 0) {
        max_iters = GDA_BARRIER_MAX_ITERS;
    }
    if (max_iters > GDA_BARRIER_MAX_ITERS) {
        fprintf(stderr, "OpenGDA: Too many barrier iterations requested (%d > %d)\n",
                max_iters, GDA_BARRIER_MAX_ITERS);
        return nullptr;
    }

    // Calculate number of phases for dissemination algorithm
    int num_phases = 0;
    int temp = npes - 1;
    while (temp > 0) {
        num_phases++;
        temp /= 2;  // Using radix 2 for simplicity
    }

    if (num_phases > GDA_BARRIER_MAX_PHASES) {
        fprintf(stderr, "OpenGDA: Too many phases for GPU barrier\n");
        return nullptr;
    }

    // Get GPU buffer
    void* gpu_buf = gda_gpu_buf();
    size_t gpu_buf_size = gda_gpu_buf_size();
    if (!gpu_buf) {
        fprintf(stderr, "OpenGDA: No GPU buffer available\n");
        return nullptr;
    }

    // Allocate sync array at the end of GPU buffer
    // Layout: [sync_arr: npes * uint64_t] [sync_counter: uint64_t] [phase_handles array]
    size_t sync_arr_size = npes * sizeof(uint64_t);
    size_t sync_counter_size = sizeof(uint64_t);
    size_t handles_array_size = max_iters * num_phases * sizeof(gda_gpu_handle_t);
    size_t total_sync_size = sync_arr_size + sync_counter_size + handles_array_size;

    // Use a fixed offset near the end of GPU buffer for barrier sync
    size_t sync_offset = gpu_buf_size - total_sync_size - 4096;  // 4KB safety margin
    sync_offset = (sync_offset / 64) * 64;  // Align to 64 bytes

    volatile uint64_t* sync_arr = (volatile uint64_t*)((char*)gpu_buf + sync_offset);
    volatile uint64_t* sync_counter = (volatile uint64_t*)((char*)sync_arr + sync_arr_size);
    gda_gpu_handle_t* phase_handles_arr = (gda_gpu_handle_t*)((char*)sync_counter + sync_counter_size);

    // Allocate internal structure
    gda_barrier_internal* internal = new gda_barrier_internal;
    internal->sync_arr_offset = sync_offset;

    // Initialize GPU barrier structure
    gda_gpu_barrier_t* barrier = &internal->gpu_barrier;
    barrier->sync_arr = sync_arr;
    barrier->sync_counter = sync_counter;
    barrier->mype = mype;
    barrier->npes = npes;
    barrier->num_phases = num_phases;
    barrier->max_iters = max_iters;
    barrier->phase_handles = phase_handles_arr;  // GPU-accessible array

    // Calculate phase targets and sources (same for all iterations)
    for (int phase = 0; phase < num_phases; phase++) {
        int distance = 1 << phase;  // 2^phase
        barrier->phase_targets[phase] = (mype + distance) % npes;
        barrier->phase_sources[phase] = (mype - distance + npes) % npes;
    }

    // Create DWQ handles for each iteration and phase
    // Total handles: max_iters * num_phases
    int total_handles = max_iters * num_phases;
    internal->put_handles.resize(total_handles);

    size_t local_slot_offset = sync_offset + mype * sizeof(uint64_t);
    size_t remote_slot_offset = sync_offset + mype * sizeof(uint64_t);

    for (int iter = 0; iter < max_iters; iter++) {
        for (int phase = 0; phase < num_phases; phase++) {
            int handle_idx = iter * num_phases + phase;
            int target = barrier->phase_targets[phase];

            gda_handle_t* put_handle = gda_put(
                (void*)((char*)gpu_buf + local_slot_offset),
                sizeof(uint64_t),
                target,
                remote_slot_offset
            );

            if (!put_handle) {
                fprintf(stderr, "OpenGDA: Failed to create barrier put handle for iter=%d phase=%d\n",
                        iter, phase);
                // Cleanup already created handles
                for (int j = 0; j < handle_idx; j++) {
                    gda_free(internal->put_handles[j]);
                }
                delete internal;
                return nullptr;
            }

            internal->put_handles[handle_idx] = put_handle;
            phase_handles_arr[handle_idx] = put_handle->gpu;
        }
    }

    // Initialize sync array and counter to 0
#ifdef USE_AMDGPU
    hipMemset((void*)sync_arr, 0, sync_arr_size);
    hipMemset((void*)sync_counter, 0, sync_counter_size);
    hipDeviceSynchronize();
#endif

    // Barrier to ensure all ranks have initialized
    gda_barrier();

    if (mype == 0) {
        fprintf(stderr, "OpenGDA: GPU barrier allocated with %d iterations, %d phases\n",
                max_iters, num_phases);
    }

    barrier_pool.push_back(internal);
    return barrier;
}

void gda_gpu_barrier_free(gda_gpu_barrier_t* barrier) {
    if (!barrier) {
        return;
    }

    // Find internal structure
    gda_barrier_internal* internal = nullptr;
    for (auto it = barrier_pool.begin(); it != barrier_pool.end(); ++it) {
        if (&(*it)->gpu_barrier == barrier) {
            internal = *it;
            barrier_pool.erase(it);
            break;
        }
    }

    if (!internal) {
        fprintf(stderr, "OpenGDA: Barrier not found in pool\n");
        return;
    }

    // Free put handles
    gda_flush();
    for (auto h : internal->put_handles) {
        gda_free(h);
    }

    delete internal;
}

int gda_gpu_barrier_reset(gda_gpu_barrier_t* barrier) {
    if (!barrier) {
        return -EINVAL;
    }

    // Find internal structure
    gda_barrier_internal* internal = nullptr;
    for (auto it = barrier_pool.begin(); it != barrier_pool.end(); ++it) {
        if (&(*it)->gpu_barrier == barrier) {
            internal = *it;
            break;
        }
    }

    if (!internal) {
        return -EINVAL;
    }

    // Reset sync array and counter
#ifdef USE_AMDGPU
    int npes = barrier->npes;
    size_t sync_arr_size = npes * sizeof(uint64_t);
    hipMemset((void*)barrier->sync_arr, 0, sync_arr_size);
    hipMemset((void*)barrier->sync_counter, 0, sizeof(uint64_t));
    hipDeviceSynchronize();
#endif

    // Reset put handles
    for (int i = 0; i < barrier->num_phases; i++) {
        gda_handle_t* h = internal->put_handles[i];
        if (h && h->op && h->op->dwq_op) {
            h->op->dwq_op->reset();
            // Re-prepare the operation
            const PeerInfo* peer = ofi->get_peer_info(barrier->phase_targets[i]);
            if (peer) {
                void* gpu_buf = gda_gpu_buf();
                size_t local_slot_offset = internal->sync_arr_offset + barrier->mype * sizeof(uint64_t);
                size_t remote_slot_offset = internal->sync_arr_offset + barrier->mype * sizeof(uint64_t);

                h->op->dwq_op->prepare_write_explicit(
                    (void*)((char*)gpu_buf + local_slot_offset),
                    sizeof(uint64_t),
                    peer->fi_addr,
                    remote_slot_offset,
                    peer->mr_info.gpu_mr_key
                );

                // Update GPU handle
                h->gpu.trigger_addr = h->op->dwq_op->get_trigger_addr();
                h->gpu.completion_addr = h->op->dwq_op->get_completion_signal();
                h->gpu.trigger_threshold = h->op->dwq_op->get_trigger_threshold();
                h->gpu.completion_threshold = h->op->dwq_op->get_completion_threshold();
                barrier->phase_handles[i] = h->gpu;
            }
        }
    }

    gda_barrier();
    return 0;
}

} // extern "C"
