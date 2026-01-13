/**
 * gda_barrier.cpp - Implementation of GPU Barrier with CPU Proxy
 *
 * Based on the verified prototype (barrier-verify.cpp) which uses:
 * - Dissemination barrier algorithm: O(log P) phases per barrier
 * - Windowed DWQ slots for unlimited iterations
 * - CPU proxy thread for continuous slot rearming
 */

#include "gda_barrier.h"
#include "gda.h"
#include "network/ofi.hpp"
#include "bootstrap/common.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifdef USE_AMDGPU
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#endif

// External OFI and bootstrap instances from gda.cpp
extern std::unique_ptr<OFI> ofi;
extern std::unique_ptr<Bootstrap> bootstrap;

// ============================================================================
// Internal Structures
// ============================================================================

// Per-slot counter resources
struct BarrierSlot {
    struct fid_cntr* trigger_cntr;
    struct fi_cxi_cntr_ops* trigger_ops;
    void* trigger_mmio_addr;
    size_t trigger_mmio_len;
    volatile uint64_t* dev_trigger_addr;

    struct fid_cntr* completion_cntr;

    bool initialized;
};

// Remote phase counter info (for atomic add targets)
struct RemotePhaseInfo {
    uint64_t addr;
    uint64_t key;
};

// Full barrier structure
struct gda_barrier {
    // Device context (contains GPU-visible pointers)
    gda_barrier_dev_t dev;

    // Basic info
    int rank;
    int size;
    int num_phases;
    int window_size;
    int total_slots;

    // Fabric resources
    struct fid_domain* domain;
    struct fid_ep* ep;
    struct fid_cq* cq;
    struct fi_info* info;

    // Slots array
    std::vector<BarrierSlot> slots;

    // Peer addresses
    std::vector<fi_addr_t> peer_fi_addrs;

    // Per-phase remote info [phase][rank]
    std::vector<std::vector<RemotePhaseInfo>> peer_phase_info;

    // Atomic operand (always 1)
    uint64_t* d_atomic_operand;
    struct fid_mr* mr_atomic_operand;
    void* desc_atomic_operand;

    // Phase counters registration
    struct fid_mr* mr_phase_counters;
    uint64_t key_phase_counters;

    // DWQ structures [total_slots]
    std::vector<struct fi_deferred_work> works;
    std::vector<struct fi_op_atomic> atomics;
    std::vector<struct fi_msg_atomic> msgs;
    std::vector<struct fi_ioc> iovs;
    std::vector<struct fi_rma_ioc> rma_iovs;

    // Proxy thread control
    std::thread proxy_thread;
    std::atomic<bool> running;
    std::atomic<bool> stop_requested;

    // Statistics
    std::atomic<uint64_t> total_rearms;
    std::atomic<uint64_t> polls;

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
// Slot Arming
// ============================================================================

static int queue_slot_work(gda_barrier_t* b, int slot_idx) {
    int phase = slot_idx % b->num_phases;
    int step = 1 << phase;
    int target = (b->rank + step) % b->size;

    BarrierSlot& bs = b->slots[slot_idx];

    auto t0 = std::chrono::high_resolution_clock::now();

    // Reset counters
    fi_cntr_set(bs.trigger_cntr, 0);
    fi_cntr_set(bs.completion_cntr, 0);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Get remote info
    RemotePhaseInfo& rinfo = b->peer_phase_info[phase][target];

    // Setup atomic IOV
    b->iovs[slot_idx].addr = b->d_atomic_operand;
    b->iovs[slot_idx].count = 1;

    // Setup remote RMA IOV
    b->rma_iovs[slot_idx].addr = rinfo.addr;
    b->rma_iovs[slot_idx].count = 1;
    b->rma_iovs[slot_idx].key = rinfo.key;

    // Setup atomic message
    b->msgs[slot_idx].msg_iov = &b->iovs[slot_idx];
    b->msgs[slot_idx].desc = &b->desc_atomic_operand;
    b->msgs[slot_idx].iov_count = 1;
    b->msgs[slot_idx].addr = b->peer_fi_addrs[target];
    b->msgs[slot_idx].rma_iov = &b->rma_iovs[slot_idx];
    b->msgs[slot_idx].rma_iov_count = 1;
    b->msgs[slot_idx].datatype = FI_UINT64;
    b->msgs[slot_idx].op = FI_SUM;

    // Setup atomic op
    b->atomics[slot_idx].ep = b->ep;
    b->atomics[slot_idx].msg = b->msgs[slot_idx];
    b->atomics[slot_idx].flags = 0;

    // Setup deferred work
    b->works[slot_idx].triggering_cntr = bs.trigger_cntr;
    b->works[slot_idx].completion_cntr = bs.completion_cntr;
    b->works[slot_idx].threshold = 1;
    b->works[slot_idx].op_type = FI_OP_ATOMIC;
    b->works[slot_idx].op.atomic = &b->atomics[slot_idx];

    auto t2 = std::chrono::high_resolution_clock::now();

    int ret = fi_control(&b->domain->fid, FI_QUEUE_WORK, &b->works[slot_idx]);

    auto t3 = std::chrono::high_resolution_clock::now();

    if (ret) {
        OPENGDA_Error("barrier", "fi_control(slot %d) failed: %s", slot_idx, fi_strerror(-ret));
        return ret;
    }

    // Log timing if any operation was slow (>100us)
    auto cntr_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto ctrl_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    if (cntr_us > 100 || ctrl_us > 100) {
        OPENGDA_Warn("barrier", "SLOW rearm slot %d: fi_cntr_set=%ldus fi_control=%ldus",
                     slot_idx, cntr_us, ctrl_us);
    }

    return 0;
}

// ============================================================================
// Proxy Thread
// ============================================================================

static void proxy_thread_func(gda_barrier_t* b) {
    OPENGDA_Debug("barrier", "Proxy thread started for rank %d", b->rank);

    while (!b->stop_requested.load(std::memory_order_relaxed)) {
        bool did_work = false;

        for (int s = 0; s < b->total_slots; s++) {
            int state = __atomic_load_n(&b->dev.slot_state[s], __ATOMIC_ACQUIRE);

            if (state == GDA_SLOT_NEED_REARM) {
                int ret = queue_slot_work(b, s);
                if (ret == 0) {
                    __atomic_store_n(&b->dev.slot_state[s], GDA_SLOT_READY, __ATOMIC_RELEASE);
                    b->total_rearms.fetch_add(1, std::memory_order_relaxed);
                    did_work = true;
                }
            }
        }

        b->polls.fetch_add(1, std::memory_order_relaxed);

        // Drain CQ
        struct fi_cq_entry cq_entries[32];
        int ret;
        while ((ret = fi_cq_read(b->cq, cq_entries, 32)) > 0) {
        }

        if (!did_work) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));  // Reduced from 10us
        }
    }

    OPENGDA_Debug("barrier", "Proxy thread exiting for rank %d", b->rank);
}

// ============================================================================
// Public API
// ============================================================================

gda_barrier_t* gda_barrier_alloc(int window_size) {
    if (!ofi) {
        OPENGDA_Error("barrier", "OpenGDA not initialized");
        return nullptr;
    }

    int rank = ofi->get_rank();
    int size = ofi->get_size();

    if (size < 2) {
        OPENGDA_Error("barrier", "Need at least 2 ranks");
        return nullptr;
    }

    if (window_size <= 0) {
        window_size = GDA_BARRIER_DEFAULT_WINDOW;
    }
    if (window_size > GDA_BARRIER_MAX_WINDOW) {
        window_size = GDA_BARRIER_MAX_WINDOW;
    }

    // Calculate phases: ceil(log2(size))
    int num_phases = 0;
    int temp = 1;
    while (temp < size) {
        num_phases++;
        temp *= 2;
    }

    if (num_phases > GDA_BARRIER_MAX_PHASES) {
        OPENGDA_Error("barrier", "Too many ranks (%d > 2^%d)", size, GDA_BARRIER_MAX_PHASES);
        return nullptr;
    }

    int total_slots = window_size * num_phases;

    OPENGDA_Info("barrier", "Allocating barrier: rank=%d/%d, window=%d, phases=%d, slots=%d",
                 rank, size, window_size, num_phases, total_slots);

    gda_barrier_t* b = new struct gda_barrier();
    b->rank = rank;
    b->size = size;
    b->num_phases = num_phases;
    b->window_size = window_size;
    b->total_slots = total_slots;
    b->initialized = false;
    b->running.store(false);
    b->stop_requested.store(false);
    b->total_rearms.store(0);
    b->polls.store(0);

    // Get fabric resources
    b->domain = ofi->get_domain();
    b->ep = ofi->get_endpoint();
    b->cq = ofi->get_cq();
    b->info = ofi->get_info();

    // Initialize device context
    b->dev.rank = rank;
    b->dev.size = size;
    b->dev.num_phases = num_phases;
    b->dev.window_size = window_size;
    b->dev.total_slots = total_slots;

#ifdef USE_AMDGPU
    hipError_t hip_err;

    // Allocate slot_state (pinned, GPU-visible)
    hip_err = hipHostMalloc((void**)&b->dev.slot_state, sizeof(int) * total_slots,
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("barrier", "hipHostMalloc(slot_state) failed: %s", hipGetErrorString(hip_err));
        delete b;
        return nullptr;
    }
    for (int i = 0; i < total_slots; i++) {
        ((int*)b->dev.slot_state)[i] = GDA_SLOT_NEED_REARM;
    }

    // Allocate trigger_addrs (pinned, GPU-visible)
    hip_err = hipHostMalloc((void**)&b->dev.trigger_addrs,
                            sizeof(volatile uint64_t*) * total_slots,
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("barrier", "hipHostMalloc(trigger_addrs) failed");
        hipHostFree((void*)b->dev.slot_state);
        delete b;
        return nullptr;
    }

    // Allocate phase_counters (GPU memory)
    hip_err = hipMalloc((void**)&b->dev.phase_counters, sizeof(uint64_t) * num_phases);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("barrier", "hipMalloc(phase_counters) failed");
        hipHostFree((void*)b->dev.trigger_addrs);
        hipHostFree((void*)b->dev.slot_state);
        delete b;
        return nullptr;
    }
    hipMemset((void*)b->dev.phase_counters, 0, sizeof(uint64_t) * num_phases);

    // Allocate atomic operand (GPU memory, value = 1)
    hip_err = hipMalloc((void**)&b->d_atomic_operand, sizeof(uint64_t));
    if (hip_err != hipSuccess) {
        OPENGDA_Error("barrier", "hipMalloc(atomic_operand) failed");
        hipFree((void*)b->dev.phase_counters);
        hipHostFree((void*)b->dev.trigger_addrs);
        hipHostFree((void*)b->dev.slot_state);
        delete b;
        return nullptr;
    }
    uint64_t one = 1;
    hipMemcpy(b->d_atomic_operand, &one, sizeof(uint64_t), hipMemcpyHostToDevice);
#else
    OPENGDA_Error("barrier", "GPU support required");
    delete b;
    return nullptr;
#endif

    // Register phase_counters MR
    b->mr_phase_counters = ofi->register_memory((void*)b->dev.phase_counters,
                                                 sizeof(uint64_t) * num_phases, true);
    if (!b->mr_phase_counters) {
        OPENGDA_Error("barrier", "Failed to register phase_counters MR");
        goto cleanup_gpu;
    }
    b->key_phase_counters = fi_mr_key(b->mr_phase_counters);

    // Register atomic operand MR
    b->mr_atomic_operand = ofi->register_memory(b->d_atomic_operand, sizeof(uint64_t), true);
    if (!b->mr_atomic_operand) {
        OPENGDA_Error("barrier", "Failed to register atomic_operand MR");
        ofi->deregister_memory(b->mr_phase_counters);
        goto cleanup_gpu;
    }
    b->desc_atomic_operand = fi_mr_desc(b->mr_atomic_operand);

    // Get peer addresses from OFI
    b->peer_fi_addrs.resize(size);
    for (int i = 0; i < size; i++) {
        const PeerInfo* peer = ofi->get_peer_info(i);
        if (!peer || !peer->valid) {
            OPENGDA_Error("barrier", "Invalid peer info for rank %d", i);
            ofi->deregister_memory(b->mr_atomic_operand);
            ofi->deregister_memory(b->mr_phase_counters);
            goto cleanup_gpu;
        }
        b->peer_fi_addrs[i] = peer->fi_addr;
    }

    // Exchange phase counter info
    b->peer_phase_info.resize(num_phases);
    for (int p = 0; p < num_phases; p++) {
        b->peer_phase_info[p].resize(size);
    }

    {
        // My phase info
        for (int p = 0; p < num_phases; p++) {
            RemotePhaseInfo my_info;
            if (b->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
                my_info.addr = (uint64_t)&b->dev.phase_counters[p];
            } else {
                my_info.addr = p * sizeof(uint64_t);
            }
            my_info.key = b->key_phase_counters;

            char info_hex[128];
            bytes_to_hex((uint8_t*)&my_info, sizeof(my_info), info_hex);

            char key_str[64];
            snprintf(key_str, sizeof(key_str), "barrier-phase-%d-rank-%d", p, rank);
            if (!bootstrap->bootstrap_kvs_put(key_str, info_hex)) {
                OPENGDA_Error("barrier", "KVS put failed");
                ofi->deregister_memory(b->mr_atomic_operand);
                ofi->deregister_memory(b->mr_phase_counters);
                goto cleanup_gpu;
            }
        }

        bootstrap->bootstrap_barrier();

        // Get all peers' phase info
        for (int p = 0; p < num_phases; p++) {
            for (int r = 0; r < size; r++) {
                char key_str[64];
                snprintf(key_str, sizeof(key_str), "barrier-phase-%d-rank-%d", p, r);
                char peer_hex[128];
                int peer_hex_len = 0;
                if (!bootstrap->bootstrap_kvs_get(key_str, peer_hex, &peer_hex_len)) {
                    OPENGDA_Error("barrier", "KVS get failed for rank %d phase %d", r, p);
                    ofi->deregister_memory(b->mr_atomic_operand);
                    ofi->deregister_memory(b->mr_phase_counters);
                    goto cleanup_gpu;
                }
                hex_to_bytes(peer_hex, (uint8_t*)&b->peer_phase_info[p][r], sizeof(RemotePhaseInfo));
            }
        }
    }

    // Allocate slots
    b->slots.resize(total_slots);
    for (int i = 0; i < total_slots; i++) {
        BarrierSlot& bs = b->slots[i];
        bs.initialized = false;

        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;

        int ret = fi_cntr_open(b->domain, &cntr_attr, &bs.trigger_cntr, NULL);
        if (ret) {
            OPENGDA_Error("barrier", "fi_cntr_open(trigger %d) failed: %s", i, fi_strerror(-ret));
            goto cleanup_slots;
        }

        ret = fi_cntr_open(b->domain, &cntr_attr, &bs.completion_cntr, NULL);
        if (ret) {
            OPENGDA_Error("barrier", "fi_cntr_open(completion %d) failed", i);
            fi_close(&bs.trigger_cntr->fid);
            goto cleanup_slots;
        }

        ret = fi_open_ops(&bs.trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                          (void**)&bs.trigger_ops, NULL);
        if (ret) {
            OPENGDA_Error("barrier", "fi_open_ops(trigger %d) failed", i);
            fi_close(&bs.completion_cntr->fid);
            fi_close(&bs.trigger_cntr->fid);
            goto cleanup_slots;
        }

        ret = bs.trigger_ops->get_mmio_addr(&bs.trigger_cntr->fid,
                                             &bs.trigger_mmio_addr,
                                             &bs.trigger_mmio_len);
        if (ret) {
            OPENGDA_Error("barrier", "get_mmio_addr(trigger %d) failed", i);
            fi_close(&bs.completion_cntr->fid);
            fi_close(&bs.trigger_cntr->fid);
            goto cleanup_slots;
        }

#ifdef USE_AMDGPU
        hip_err = hipHostRegister(bs.trigger_mmio_addr, bs.trigger_mmio_len,
                                  hipHostRegisterMapped);
        if (hip_err != hipSuccess) {
            OPENGDA_Error("barrier", "hipHostRegister(trigger %d) failed", i);
            fi_close(&bs.completion_cntr->fid);
            fi_close(&bs.trigger_cntr->fid);
            goto cleanup_slots;
        }

        hip_err = hipHostGetDevicePointer((void**)&bs.dev_trigger_addr,
                                          bs.trigger_mmio_addr, 0);
        if (hip_err != hipSuccess) {
            OPENGDA_Error("barrier", "hipHostGetDevicePointer(trigger %d) failed", i);
            hipHostUnregister(bs.trigger_mmio_addr);
            fi_close(&bs.completion_cntr->fid);
            fi_close(&bs.trigger_cntr->fid);
            goto cleanup_slots;
        }

        ((volatile uint64_t**)b->dev.trigger_addrs)[i] = bs.dev_trigger_addr;
#endif

        bs.initialized = true;
    }

    // Allocate DWQ structures
    b->works.resize(total_slots);
    b->atomics.resize(total_slots);
    b->msgs.resize(total_slots);
    b->iovs.resize(total_slots);
    b->rma_iovs.resize(total_slots);

    // Initial arming of all slots
    OPENGDA_Debug("barrier", "Arming initial %d slots", total_slots);
    for (int s = 0; s < total_slots; s++) {
        int ret = queue_slot_work(b, s);
        if (ret) {
            OPENGDA_Error("barrier", "Initial arm of slot %d failed", s);
            goto cleanup_slots;
        }
        __atomic_store_n(&b->dev.slot_state[s], GDA_SLOT_READY, __ATOMIC_RELEASE);
    }

    bootstrap->bootstrap_barrier();

    b->initialized = true;
    OPENGDA_Info("barrier", "Barrier allocated: rank=%d, window=%d, phases=%d",
                 rank, window_size, num_phases);

    return b;

cleanup_slots:
    for (int s = 0; s < (int)b->slots.size(); s++) {
        BarrierSlot& bs = b->slots[s];
        if (bs.initialized) {
#ifdef USE_AMDGPU
            hipHostUnregister(bs.trigger_mmio_addr);
#endif
            fi_close(&bs.completion_cntr->fid);
            fi_close(&bs.trigger_cntr->fid);
        }
    }
    ofi->deregister_memory(b->mr_atomic_operand);
    ofi->deregister_memory(b->mr_phase_counters);

cleanup_gpu:
#ifdef USE_AMDGPU
    if (b->d_atomic_operand) hipFree(b->d_atomic_operand);
    if (b->dev.phase_counters) hipFree((void*)b->dev.phase_counters);
    if (b->dev.trigger_addrs) hipHostFree((void*)b->dev.trigger_addrs);
    if (b->dev.slot_state) hipHostFree((void*)b->dev.slot_state);
#endif
    delete b;
    return nullptr;
}

void gda_barrier_free(gda_barrier_t* barrier) {
    if (!barrier) return;

    if (barrier->running.load()) {
        gda_barrier_stop(barrier);
    }

    for (int s = 0; s < (int)barrier->slots.size(); s++) {
        BarrierSlot& bs = barrier->slots[s];
        if (bs.initialized) {
#ifdef USE_AMDGPU
            hipHostUnregister(bs.trigger_mmio_addr);
#endif
            fi_close(&bs.completion_cntr->fid);
            fi_close(&bs.trigger_cntr->fid);
        }
    }

    if (barrier->mr_atomic_operand) {
        ofi->deregister_memory(barrier->mr_atomic_operand);
    }
    if (barrier->mr_phase_counters) {
        ofi->deregister_memory(barrier->mr_phase_counters);
    }

#ifdef USE_AMDGPU
    if (barrier->d_atomic_operand) hipFree(barrier->d_atomic_operand);
    if (barrier->dev.phase_counters) hipFree((void*)barrier->dev.phase_counters);
    if (barrier->dev.trigger_addrs) hipHostFree((void*)barrier->dev.trigger_addrs);
    if (barrier->dev.slot_state) hipHostFree((void*)barrier->dev.slot_state);
#endif

    delete barrier;
}

gda_barrier_dev_t* gda_barrier_get_dev(gda_barrier_t* barrier) {
    if (!barrier || !barrier->initialized) {
        return nullptr;
    }
    return &barrier->dev;
}

int gda_barrier_start(gda_barrier_t* barrier) {
    if (!barrier || !barrier->initialized) {
        return -1;
    }

    if (barrier->running.load()) {
        OPENGDA_Warn("barrier", "Proxy already running");
        return 0;
    }

    barrier->stop_requested.store(false);
    barrier->proxy_thread = std::thread(proxy_thread_func, barrier);
    barrier->running.store(true);

    OPENGDA_Info("barrier", "Proxy started for rank %d", barrier->rank);
    return 0;
}

int gda_barrier_stop(gda_barrier_t* barrier) {
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

    OPENGDA_Info("barrier", "Proxy stopped for rank %d (rearms=%lu, polls=%lu)",
                 barrier->rank, barrier->total_rearms.load(), barrier->polls.load());
    return 0;
}

int gda_barrier_get_stats(gda_barrier_t* barrier, gda_barrier_stats_t* stats) {
    if (!barrier || !stats) {
        return -1;
    }

    stats->total_rearms = barrier->total_rearms.load();
    stats->polls = barrier->polls.load();
    return 0;
}
