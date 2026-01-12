/**
 * gda_barrier_proxy.cpp - Implementation of CPU Proxy Barrier
 *
 * Based on P1 prototype: All-to-all barrier with single threshold triggering.
 * Each rank atomically adds +1 to all other ranks' d_arrive counter.
 */

#include "gda_barrier_proxy.h"
#include "gda.h"
#include "network/ofi.hpp"
#include "bootstrap/common.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <chrono>

#ifdef USE_AMDGPU
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#endif

// External OFI instance from gda.cpp
extern std::unique_ptr<OFI> ofi;
extern std::unique_ptr<Bootstrap> bootstrap;

// ============================================================================
// Internal Structures
// ============================================================================

/**
 * Per-slot counter resources for DWQ operations.
 */
struct ProxySlotCounters {
    struct fid_cntr* trigger_cntr;           // Trigger counter
    struct fi_cxi_cntr_ops* trigger_ops;     // CXI counter ops
    void* trigger_mmio_addr;                 // Host MMIO address
    size_t trigger_mmio_len;                 // MMIO length
    volatile uint64_t* dev_trigger_addr;     // GPU-accessible trigger doorbell

    struct fid_cntr* completion_cntr;        // Completion counter

    bool initialized;
};

/**
 * Remote arrive counter info for each peer.
 */
struct RemoteArriveInfo {
    fi_addr_t fi_addr;                       // Peer's fi_addr
    uint64_t remote_addr;                    // Remote d_arrive address
    uint64_t remote_key;                     // Remote MR key
};

/**
 * Full host-side proxy barrier context.
 */
struct gda_proxy_barrier {
    // Device context (GPU-accessible, allocated with hipHostMalloc)
    gda_proxy_barrier_dev_t dev;

    // Basic info
    int mype;
    int npes;
    int num_peers;
    int window_size;

    // Fabric resources
    struct fid_domain* domain;
    struct fid_ep* ep;
    struct fid_cq* cq;
    struct fi_info* info;

    // Slot counters [window_size]
    std::vector<ProxySlotCounters> slots;

    // Remote arrive info [npes] (one per rank, skip self)
    std::vector<RemoteArriveInfo> peer_arrive_info;

    // Local atomic operand (value=1, registered GPU memory)
    uint64_t* d_atomic_operand;
    struct fid_mr* mr_atomic_operand;
    void* desc_atomic_operand;

    // d_arrive registration
    struct fid_mr* mr_arrive;
    void* desc_arrive;
    uint64_t key_arrive;

    // DWQ structures per slot per peer: [window_size][num_peers]
    std::vector<struct fi_deferred_work> atomic_works;
    std::vector<struct fi_op_atomic> atomic_ops;
    std::vector<struct fi_msg_atomic> atomic_msgs;
    std::vector<struct fi_ioc> atomic_iovs;
    std::vector<struct fi_rma_ioc> atomic_rma_iovs;

    // Proxy thread
    std::thread proxy_thread;
    std::atomic<bool> running;
    std::atomic<bool> stop_requested;

    // Statistics
    std::atomic<uint64_t> total_rearms;
    std::atomic<uint64_t> queue_polls;
    std::atomic<uint64_t> cq_events_drained;

    bool initialized;
};

// ============================================================================
// Utility Functions
// ============================================================================

static void bytes_to_hex(const uint8_t* in, size_t len, char* out) {
    static const char* h = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        out[2 * i] = h[(in[i] >> 4) & 0xF];
        out[2 * i + 1] = h[in[i] & 0xF];
    }
    out[2 * len] = '\0';
}

static int hexval(char c) {
    if ('0' <= c && c <= '9') return c - '0';
    if ('a' <= c && c <= 'f') return c - 'a' + 10;
    if ('A' <= c && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int hex_to_bytes(const char* in, uint8_t* out, size_t outlen) {
    size_t n = strlen(in);
    if (n % 2 != 0 || outlen < n / 2) return -1;
    for (size_t i = 0; i < n; i += 2) {
        int hi = hexval(in[i]);
        int lo = hexval(in[i + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[i / 2] = (uint8_t)((hi << 4) | lo);
    }
    return (int)(n / 2);
}

// ============================================================================
// Slot Arming (All-to-all: atomics to all peers with threshold=1)
// ============================================================================

/**
 * Queue DWQ work for a single slot (atomics to all peers).
 * All atomics fire at once when GPU writes 1 to trigger counter.
 */
static int queue_slot_work(gda_proxy_barrier_t* pb, int slot) {
    if (!pb || slot < 0 || slot >= pb->window_size) {
        return -1;
    }

    ProxySlotCounters& sc = pb->slots[slot];

    // Reset counters to 0
    fi_cntr_set(sc.trigger_cntr, 0);
    fi_cntr_set(sc.completion_cntr, 0);

    int num_peers = pb->num_peers;
    int idx_base = slot * num_peers;

    // Queue atomic add to each peer's d_arrive (threshold=1, all fire at once)
    int peer_idx = 0;
    for (int r = 0; r < pb->npes; r++) {
        if (r == pb->mype) continue;  // Skip self

        int idx = idx_base + peer_idx;
        RemoteArriveInfo& ri = pb->peer_arrive_info[r];

        // Setup atomic IOV (source operand = 1)
        pb->atomic_iovs[idx].addr = pb->d_atomic_operand;
        pb->atomic_iovs[idx].count = 1;

        // Setup remote RMA IOV (target: peer's d_arrive)
        pb->atomic_rma_iovs[idx].addr = ri.remote_addr;
        pb->atomic_rma_iovs[idx].count = 1;
        pb->atomic_rma_iovs[idx].key = ri.remote_key;

        // Setup atomic message
        pb->atomic_msgs[idx].msg_iov = &pb->atomic_iovs[idx];
        pb->atomic_msgs[idx].desc = &pb->desc_atomic_operand;
        pb->atomic_msgs[idx].iov_count = 1;
        pb->atomic_msgs[idx].addr = ri.fi_addr;
        pb->atomic_msgs[idx].rma_iov = &pb->atomic_rma_iovs[idx];
        pb->atomic_msgs[idx].rma_iov_count = 1;
        pb->atomic_msgs[idx].datatype = FI_UINT64;
        pb->atomic_msgs[idx].op = FI_SUM;

        // Setup atomic op
        pb->atomic_ops[idx].ep = pb->ep;
        pb->atomic_ops[idx].msg = pb->atomic_msgs[idx];
        pb->atomic_ops[idx].flags = 0;

        // Setup deferred work with threshold = 1 (all fire at once)
        pb->atomic_works[idx].triggering_cntr = sc.trigger_cntr;
        pb->atomic_works[idx].completion_cntr = sc.completion_cntr;
        pb->atomic_works[idx].threshold = 1;
        pb->atomic_works[idx].op_type = FI_OP_ATOMIC;
        pb->atomic_works[idx].op.atomic = &pb->atomic_ops[idx];

        int ret = fi_control(&pb->domain->fid, FI_QUEUE_WORK, &pb->atomic_works[idx]);
        if (ret) {
            OPENGDA_Error("proxy_barrier", "fi_control(atomic slot=%d peer=%d) failed: %s",
                         slot, r, fi_strerror(-ret));
            return ret;
        }

        peer_idx++;
    }

    return 0;
}

// ============================================================================
// Proxy Thread
// ============================================================================

static void proxy_thread_func(gda_proxy_barrier_t* pb) {
    OPENGDA_Debug("proxy_barrier", "Proxy thread started for rank %d", pb->mype);

    while (!pb->stop_requested.load(std::memory_order_relaxed)) {
        bool did_work = false;

        // Scan slots for those needing rearm
        for (int s = 0; s < pb->window_size; s++) {
            int state = __atomic_load_n(&pb->dev.slot_state[s], __ATOMIC_ACQUIRE);

            if (state == GDA_SLOT_NEED_QUEUE) {
                // Rearm this slot
                int ret = queue_slot_work(pb, s);
                if (ret == 0) {
                    // Mark as armed
                    __atomic_store_n(&pb->dev.slot_state[s], GDA_SLOT_ARMED, __ATOMIC_RELEASE);
                    pb->total_rearms.fetch_add(1, std::memory_order_relaxed);
                    did_work = true;
                }
            }
        }

        pb->queue_polls.fetch_add(1, std::memory_order_relaxed);

        // Drain CQ to prevent overflow
        struct fi_cq_entry cq_entries[64];
        int ret;
        while ((ret = fi_cq_read(pb->cq, cq_entries, 64)) > 0) {
            pb->cq_events_drained.fetch_add(ret, std::memory_order_relaxed);
        }

        // If no work done, brief sleep to avoid busy spinning
        if (!did_work) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    OPENGDA_Debug("proxy_barrier", "Proxy thread exiting for rank %d", pb->mype);
}

// ============================================================================
// Public API Implementation
// ============================================================================

gda_proxy_barrier_t* gda_proxy_barrier_alloc(int window_size) {
    if (!ofi) {
        OPENGDA_Error("proxy_barrier", "OpenGDA not initialized (call gda_init first)");
        return nullptr;
    }

    int mype = ofi->get_rank();
    int npes = ofi->get_size();

    if (npes < 2) {
        OPENGDA_Error("proxy_barrier", "Need at least 2 ranks for barrier");
        return nullptr;
    }

    // Validate and adjust window size based on peer count
    // Each slot needs num_peers DWQ ops, so limit window to avoid DWQ overflow
    // DWQ depth is typically 1024, shared by all ranks on a NIC (typically 4)
    // Leave room for put/get handles (~64 per rank)
    // Safe budget per rank: ~200 DWQ ops for barrier
    int num_peers = npes - 1;
    int max_window_for_peers = (num_peers > 0) ? (200 / num_peers) : GDA_PROXY_DEFAULT_WINDOW_SIZE;
    max_window_for_peers = std::max(2, max_window_for_peers);  // At least 2 slots

    if (window_size <= 0) {
        window_size = std::min(GDA_PROXY_DEFAULT_WINDOW_SIZE, max_window_for_peers);
    }
    if (window_size > GDA_PROXY_MAX_WINDOW_SIZE) {
        window_size = GDA_PROXY_MAX_WINDOW_SIZE;
    }
    if (window_size > max_window_for_peers) {
        OPENGDA_Info("proxy_barrier", "Reducing window_size from %d to %d for %d peers (DWQ budget)",
                    window_size, max_window_for_peers, num_peers);
        window_size = max_window_for_peers;
    }
    int initial_arm_count = 0;  // Declared early to avoid goto issues

    OPENGDA_Info("proxy_barrier", "Allocating all-to-all proxy barrier: rank=%d/%d, window=%d, peers=%d",
                mype, npes, window_size, num_peers);

    // Allocate barrier structure with zero-initialization
    gda_proxy_barrier_t* pb = new gda_proxy_barrier();

    // Initialize device context
    pb->dev = gda_proxy_barrier_dev_t{};

    pb->mype = mype;
    pb->npes = npes;
    pb->num_peers = num_peers;
    pb->window_size = window_size;
    pb->initialized = false;
    pb->running.store(false);
    pb->stop_requested.store(false);
    pb->total_rearms.store(0);
    pb->queue_polls.store(0);
    pb->cq_events_drained.store(0);

    // Initialize pointers to nullptr
    pb->domain = nullptr;
    pb->ep = nullptr;
    pb->cq = nullptr;
    pb->info = nullptr;
    pb->d_atomic_operand = nullptr;
    pb->mr_atomic_operand = nullptr;
    pb->mr_arrive = nullptr;

    // Get fabric resources from OFI
    pb->domain = ofi->get_domain();
    pb->ep = ofi->get_endpoint();
    pb->cq = ofi->get_cq();
    pb->info = ofi->get_info();

    // Initialize device context
    gda_proxy_barrier_dev_t& dev = pb->dev;
    dev.mype = mype;
    dev.npes = npes;
    dev.num_peers = num_peers;
    dev.window_size = window_size;

#ifdef USE_AMDGPU
    hipError_t hip_err;

    // Allocate GPU-visible slot_state
    hip_err = hipHostMalloc((void**)&dev.slot_state,
                            sizeof(int) * window_size,
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_barrier", "hipHostMalloc(slot_state) failed: %s",
                     hipGetErrorString(hip_err));
        delete pb;
        return nullptr;
    }

    // Initialize all slots as NEED_QUEUE
    for (int i = 0; i < window_size; i++) {
        ((int*)dev.slot_state)[i] = GDA_SLOT_NEED_QUEUE;
    }

    // Allocate trigger address array (GPU-visible)
    hip_err = hipHostMalloc((void**)&dev.slot_trigger_addrs,
                            sizeof(volatile uint64_t*) * window_size,
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_barrier", "hipHostMalloc(trigger_addrs) failed: %s",
                     hipGetErrorString(hip_err));
        (void)hipHostFree((void*)dev.slot_state);
        delete pb;
        return nullptr;
    }

    // Allocate d_arrive (GPU memory, receives atomic adds from peers)
    hip_err = hipMalloc((void**)&dev.d_arrive, sizeof(uint64_t));
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_barrier", "hipMalloc(d_arrive) failed: %s",
                     hipGetErrorString(hip_err));
        (void)hipHostFree((void*)dev.slot_trigger_addrs);
        (void)hipHostFree((void*)dev.slot_state);
        delete pb;
        return nullptr;
    }
    (void)hipMemset((void*)dev.d_arrive, 0, sizeof(uint64_t));

    // Allocate d_epoch (GPU memory)
    hip_err = hipMalloc((void**)&dev.d_epoch, sizeof(uint64_t));
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_barrier", "hipMalloc(d_epoch) failed: %s",
                     hipGetErrorString(hip_err));
        (void)hipFree((void*)dev.d_arrive);
        (void)hipHostFree((void*)dev.slot_trigger_addrs);
        (void)hipHostFree((void*)dev.slot_state);
        delete pb;
        return nullptr;
    }
    (void)hipMemset((void*)dev.d_epoch, 0, sizeof(uint64_t));

    // Allocate atomic operand (GPU memory, value = 1)
    hip_err = hipMalloc((void**)&pb->d_atomic_operand, sizeof(uint64_t));
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_barrier", "hipMalloc(atomic_operand) failed: %s",
                     hipGetErrorString(hip_err));
        (void)hipFree((void*)dev.d_epoch);
        (void)hipFree((void*)dev.d_arrive);
        (void)hipHostFree((void*)dev.slot_trigger_addrs);
        (void)hipHostFree((void*)dev.slot_state);
        delete pb;
        return nullptr;
    }
    uint64_t one = 1;
    (void)hipMemcpy(pb->d_atomic_operand, &one, sizeof(uint64_t), hipMemcpyHostToDevice);
#else
    OPENGDA_Error("proxy_barrier", "GPU support required but not compiled");
    delete pb;
    return nullptr;
#endif

    // Register d_arrive as MR for remote atomics
    pb->mr_arrive = ofi->register_memory((void*)dev.d_arrive, sizeof(uint64_t), true);
    if (!pb->mr_arrive) {
        OPENGDA_Error("proxy_barrier", "Failed to register d_arrive MR");
        goto cleanup_gpu;
    }
    pb->desc_arrive = fi_mr_desc(pb->mr_arrive);
    pb->key_arrive = fi_mr_key(pb->mr_arrive);

    // Register atomic operand as MR
    pb->mr_atomic_operand = ofi->register_memory(pb->d_atomic_operand, sizeof(uint64_t), true);
    if (!pb->mr_atomic_operand) {
        OPENGDA_Error("proxy_barrier", "Failed to register atomic_operand MR");
        ofi->deregister_memory(pb->mr_arrive);
        goto cleanup_gpu;
    }
    pb->desc_atomic_operand = fi_mr_desc(pb->mr_atomic_operand);

    // Exchange d_arrive addresses via bootstrap (all-to-all)
    {
        struct ExchangeArriveInfo {
            uint64_t addr;
            uint64_t key;
        } my_info;

        if (pb->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
            my_info.addr = (uint64_t)dev.d_arrive;
        } else {
            my_info.addr = 0;
        }
        my_info.key = pb->key_arrive;

        char my_hex[64];
        bytes_to_hex((uint8_t*)&my_info, sizeof(my_info), my_hex);

        char key_str[64];
        snprintf(key_str, sizeof(key_str), "arrive-%d", mype);
        bool rc = bootstrap->bootstrap_kvs_put(key_str, my_hex);
        if (!rc) {
            OPENGDA_Error("proxy_barrier", "KV put failed for arrive");
            ofi->deregister_memory(pb->mr_atomic_operand);
            ofi->deregister_memory(pb->mr_arrive);
            goto cleanup_gpu;
        }

        bootstrap->bootstrap_barrier();

        // Fetch arrive info from all peers
        pb->peer_arrive_info.resize(npes);
        for (int r = 0; r < npes; r++) {
            if (r == mype) continue;

            snprintf(key_str, sizeof(key_str), "arrive-%d", r);
            char peer_hex[128];
            int peer_hex_len = 0;
            rc = bootstrap->bootstrap_kvs_get(key_str, peer_hex, &peer_hex_len);
            if (!rc) {
                OPENGDA_Error("proxy_barrier", "KV get failed for rank %d arrive", r);
                ofi->deregister_memory(pb->mr_atomic_operand);
                ofi->deregister_memory(pb->mr_arrive);
                goto cleanup_gpu;
            }

            ExchangeArriveInfo peer_info;
            hex_to_bytes(peer_hex, (uint8_t*)&peer_info, sizeof(peer_info));

            const PeerInfo* peer = ofi->get_peer_info(r);
            if (!peer || !peer->valid) {
                OPENGDA_Error("proxy_barrier", "Invalid peer info for rank %d", r);
                ofi->deregister_memory(pb->mr_atomic_operand);
                ofi->deregister_memory(pb->mr_arrive);
                goto cleanup_gpu;
            }

            pb->peer_arrive_info[r].fi_addr = peer->fi_addr;
            pb->peer_arrive_info[r].remote_addr = peer_info.addr;
            pb->peer_arrive_info[r].remote_key = peer_info.key;

            OPENGDA_Debug("proxy_barrier", "Peer %d: remote_addr=0x%lx, key=0x%lx",
                         r, pb->peer_arrive_info[r].remote_addr, pb->peer_arrive_info[r].remote_key);
        }
    }

    // Allocate slot counters
    pb->slots.resize(window_size);
    for (int s = 0; s < window_size; s++) {
        ProxySlotCounters& sc = pb->slots[s];
        sc.initialized = false;

        // Create trigger counter
        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;

        int ret = fi_cntr_open(pb->domain, &cntr_attr, &sc.trigger_cntr, NULL);
        if (ret) {
            OPENGDA_Error("proxy_barrier", "fi_cntr_open(trigger %d) failed: %s",
                         s, fi_strerror(-ret));
            goto cleanup_slots;
        }

        // Create completion counter
        ret = fi_cntr_open(pb->domain, &cntr_attr, &sc.completion_cntr, NULL);
        if (ret) {
            OPENGDA_Error("proxy_barrier", "fi_cntr_open(completion %d) failed: %s",
                         s, fi_strerror(-ret));
            fi_close(&sc.trigger_cntr->fid);
            goto cleanup_slots;
        }

        // Get CXI counter ops for MMIO access
        ret = fi_open_ops(&sc.trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                          (void**)&sc.trigger_ops, NULL);
        if (ret) {
            OPENGDA_Error("proxy_barrier", "fi_open_ops(trigger %d) failed: %s",
                         s, fi_strerror(-ret));
            fi_close(&sc.completion_cntr->fid);
            fi_close(&sc.trigger_cntr->fid);
            goto cleanup_slots;
        }

        // Get MMIO address
        ret = sc.trigger_ops->get_mmio_addr(&sc.trigger_cntr->fid,
                                             &sc.trigger_mmio_addr,
                                             &sc.trigger_mmio_len);
        if (ret) {
            OPENGDA_Error("proxy_barrier", "get_mmio_addr(trigger %d) failed: %s",
                         s, fi_strerror(-ret));
            fi_close(&sc.completion_cntr->fid);
            fi_close(&sc.trigger_cntr->fid);
            goto cleanup_slots;
        }

#ifdef USE_AMDGPU
        // Map MMIO to GPU
        hip_err = hipHostRegister(sc.trigger_mmio_addr, sc.trigger_mmio_len,
                                  hipHostRegisterMapped);
        if (hip_err != hipSuccess) {
            OPENGDA_Error("proxy_barrier", "hipHostRegister(trigger %d) failed: %s",
                         s, hipGetErrorString(hip_err));
            fi_close(&sc.completion_cntr->fid);
            fi_close(&sc.trigger_cntr->fid);
            goto cleanup_slots;
        }

        hip_err = hipHostGetDevicePointer((void**)&sc.dev_trigger_addr,
                                          sc.trigger_mmio_addr, 0);
        if (hip_err != hipSuccess) {
            OPENGDA_Error("proxy_barrier", "hipHostGetDevicePointer(trigger %d) failed: %s",
                         s, hipGetErrorString(hip_err));
            (void)hipHostUnregister(sc.trigger_mmio_addr);
            fi_close(&sc.completion_cntr->fid);
            fi_close(&sc.trigger_cntr->fid);
            goto cleanup_slots;
        }

        // Store in device context
        ((volatile uint64_t**)dev.slot_trigger_addrs)[s] = sc.dev_trigger_addr;
#endif

        sc.initialized = true;
    }

    // Allocate DWQ structures: [window_size * num_peers]
    {
        int total_ops = window_size * num_peers;
        pb->atomic_works.resize(total_ops);
        pb->atomic_ops.resize(total_ops);
        pb->atomic_msgs.resize(total_ops);
        pb->atomic_iovs.resize(total_ops);
        pb->atomic_rma_iovs.resize(total_ops);
    }

    pb->initialized = true;

    // Initial arming: only arm first few slots to conserve DWQ resources
    // CPU proxy will arm the rest as needed
    initial_arm_count = std::min(4, window_size);  // Only arm 4 slots initially
    OPENGDA_Debug("proxy_barrier", "Arming initial %d of %d slots", initial_arm_count, window_size);
    for (int s = 0; s < initial_arm_count; s++) {
        int ret = queue_slot_work(pb, s);
        if (ret) {
            OPENGDA_Error("proxy_barrier", "Initial arm of slot %d failed", s);
            goto cleanup_slots;
        }
        __atomic_store_n(&pb->dev.slot_state[s], GDA_SLOT_ARMED, __ATOMIC_RELEASE);
    }
    // Remaining slots stay as NEED_QUEUE - proxy will arm them

    // Barrier to ensure all ranks have initialized
    bootstrap->bootstrap_barrier();

    OPENGDA_Info("proxy_barrier", "All-to-all proxy barrier allocated: rank=%d, window=%d, peers=%d",
                mype, window_size, num_peers);

    return pb;

cleanup_slots:
    for (int s = 0; s < (int)pb->slots.size(); s++) {
        ProxySlotCounters& sc = pb->slots[s];
        if (sc.initialized) {
#ifdef USE_AMDGPU
            (void)hipHostUnregister(sc.trigger_mmio_addr);
#endif
            fi_close(&sc.completion_cntr->fid);
            fi_close(&sc.trigger_cntr->fid);
        }
    }
    ofi->deregister_memory(pb->mr_atomic_operand);
    ofi->deregister_memory(pb->mr_arrive);

cleanup_gpu:
#ifdef USE_AMDGPU
    if (dev.d_epoch) (void)hipFree((void*)dev.d_epoch);
    if (dev.d_arrive) (void)hipFree((void*)dev.d_arrive);
    if (pb->d_atomic_operand) (void)hipFree(pb->d_atomic_operand);
    if (dev.slot_trigger_addrs) (void)hipHostFree((void*)dev.slot_trigger_addrs);
    if (dev.slot_state) (void)hipHostFree((void*)dev.slot_state);
#endif
    delete pb;
    return nullptr;
}

void gda_proxy_barrier_free(gda_proxy_barrier_t* barrier) {
    if (!barrier) {
        return;
    }

    // Ensure proxy thread is stopped
    if (barrier->running.load()) {
        gda_proxy_stop(barrier);
    }

    // Cleanup slots
    for (int s = 0; s < (int)barrier->slots.size(); s++) {
        ProxySlotCounters& sc = barrier->slots[s];
        if (sc.initialized) {
#ifdef USE_AMDGPU
            (void)hipHostUnregister(sc.trigger_mmio_addr);
#endif
            fi_close(&sc.completion_cntr->fid);
            fi_close(&sc.trigger_cntr->fid);
        }
    }

    // Cleanup MRs
    if (barrier->mr_atomic_operand) {
        ofi->deregister_memory(barrier->mr_atomic_operand);
    }
    if (barrier->mr_arrive) {
        ofi->deregister_memory(barrier->mr_arrive);
    }

    // Cleanup GPU memory
#ifdef USE_AMDGPU
    gda_proxy_barrier_dev_t& dev = barrier->dev;
    if (dev.d_epoch) (void)hipFree((void*)dev.d_epoch);
    if (dev.d_arrive) (void)hipFree((void*)dev.d_arrive);
    if (barrier->d_atomic_operand) (void)hipFree(barrier->d_atomic_operand);
    if (dev.slot_trigger_addrs) (void)hipHostFree((void*)dev.slot_trigger_addrs);
    if (dev.slot_state) (void)hipHostFree((void*)dev.slot_state);
#endif

    delete barrier;
}

gda_proxy_barrier_dev_t* gda_proxy_barrier_get_dev(gda_proxy_barrier_t* barrier) {
    if (!barrier || !barrier->initialized) {
        return nullptr;
    }
    return &barrier->dev;
}

int gda_proxy_start(gda_proxy_barrier_t* barrier) {
    if (!barrier || !barrier->initialized) {
        return -1;
    }

    if (barrier->running.load()) {
        OPENGDA_Warn("proxy_barrier", "Proxy already running");
        return 0;
    }

    barrier->stop_requested.store(false);
    barrier->proxy_thread = std::thread(proxy_thread_func, barrier);
    barrier->running.store(true);

    OPENGDA_Info("proxy_barrier", "Proxy thread started for rank %d", barrier->mype);
    return 0;
}

int gda_proxy_stop(gda_proxy_barrier_t* barrier) {
    if (!barrier) {
        return -1;
    }

    if (!barrier->running.load()) {
        return 0;
    }

    barrier->stop_requested.store(true);
    if (barrier->proxy_thread.joinable()) {
        barrier->proxy_thread.join();
    }
    barrier->running.store(false);

    OPENGDA_Info("proxy_barrier", "Proxy thread stopped for rank %d (rearms=%lu, polls=%lu)",
                barrier->mype,
                barrier->total_rearms.load(),
                barrier->queue_polls.load());
    return 0;
}

int gda_proxy_is_running(gda_proxy_barrier_t* barrier) {
    if (!barrier) {
        return -1;
    }
    return barrier->running.load() ? 1 : 0;
}

int gda_proxy_barrier_get_stats(gda_proxy_barrier_t* barrier, gda_proxy_stats_t* stats) {
    if (!barrier || !stats) {
        return -1;
    }

    stats->total_rearms = barrier->total_rearms.load();
    stats->queue_polls = barrier->queue_polls.load();
    stats->cq_events_drained = barrier->cq_events_drained.load();
    stats->gpu_spin_cycles = 0;  // Not tracked in all-to-all variant

    return 0;
}
