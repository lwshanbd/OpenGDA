/**
 * gda_barrier_proxy.cpp - Implementation of CPU Proxy Barrier
 *
 * Implements the Dissemination barrier with O(log P) rounds using
 * threshold-based DWQ triggering and CPU proxy for unlimited iterations.
 */

#include "gda_barrier_proxy.h"
#include "gda.h"
#include "network/ofi.hpp"
#include "bootstrap/common.hpp"

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
 * Per-round atomic operation info for remote peer.
 */
struct RoundAtomicInfo {
    fi_addr_t peer_fi_addr;                  // Target peer's fi_addr
    uint64_t remote_addr;                    // Remote d_round_recv[k] address
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
    int num_rounds;
    int window_size;

    // Fabric resources
    struct fid_domain* domain;
    struct fid_ep* ep;
    struct fid_cq* cq;
    struct fi_info* info;

    // Slot counters [window_size]
    std::vector<ProxySlotCounters> slots;

    // Per-round atomic info [num_rounds]
    std::vector<RoundAtomicInfo> round_info;

    // Local atomic operand (value=1, registered GPU memory)
    uint64_t* d_atomic_operand;
    struct fid_mr* mr_atomic_operand;
    void* desc_atomic_operand;

    // d_round_recv registration
    struct fid_mr* mr_round_recv;
    void* desc_round_recv;
    uint64_t key_round_recv;

    // d_slot_done registration (for local done atomic)
    struct fid_mr* mr_slot_done;
    void* desc_slot_done;
    uint64_t key_slot_done;

    // Local fi_addr for self-atomics
    fi_addr_t local_fi_addr;

    // DWQ structures per slot per round: [window_size][num_rounds]
    // For outgoing atomics to peers
    std::vector<struct fi_deferred_work> atomic_works;
    std::vector<struct fi_op_atomic> atomic_ops;
    std::vector<struct fi_msg_atomic> atomic_msgs;
    std::vector<struct fi_ioc> atomic_iovs;
    std::vector<struct fi_rma_ioc> atomic_rma_iovs;

    // DWQ structures for local done atomics: [window_size]
    std::vector<struct fi_deferred_work> done_works;
    std::vector<struct fi_op_atomic> done_ops;
    std::vector<struct fi_msg_atomic> done_msgs;
    std::vector<struct fi_ioc> done_iovs;
    std::vector<struct fi_rma_ioc> done_rma_iovs;

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
// Slot Arming
// ============================================================================

/**
 * Queue DWQ work for a single slot (all rounds + done op).
 * This is called by the proxy thread to rearm a slot.
 *
 * DWQ structure:
 * - For each round k: atomic SUM +1 to round_targets[k]'s d_round_recv[k]
 *   with threshold = k+1 (so GPU writes 1,2,3... to trigger rounds sequentially)
 * - Final done op: atomic SUM +1 to local d_slot_done[slot]
 *   with threshold = num_rounds on completion_cntr
 */
static int queue_slot_work(gda_proxy_barrier_t* pb, int slot) {
    if (!pb || slot < 0 || slot >= pb->window_size) {
        return -1;
    }

    ProxySlotCounters& sc = pb->slots[slot];

    // Reset counters to 0
    fi_cntr_set(sc.trigger_cntr, 0);
    fi_cntr_set(sc.completion_cntr, 0);

    // Note: d_slot_done[slot] is NOT cleared here. It's a monotonic counter
    // that increments (+1) each time the slot completes. The GPU computes
    // the expected value based on epoch: expected = (epoch / window_size) + 1
    // This avoids hipMemcpy in the hot path which would add jitter.

    int num_rounds = pb->num_rounds;
    int base_idx = slot * num_rounds;

    // Queue atomic work for each round with increasing threshold
    for (int k = 0; k < num_rounds; k++) {
        int idx = base_idx + k;
        RoundAtomicInfo& ri = pb->round_info[k];

        // Setup atomic IOV (source operand = 1)
        pb->atomic_iovs[idx].addr = pb->d_atomic_operand;
        pb->atomic_iovs[idx].count = 1;

        // Setup remote RMA IOV (target: peer's d_round_recv[k])
        pb->atomic_rma_iovs[idx].addr = ri.remote_addr;
        pb->atomic_rma_iovs[idx].count = 1;
        pb->atomic_rma_iovs[idx].key = ri.remote_key;

        // Setup atomic message
        pb->atomic_msgs[idx].msg_iov = &pb->atomic_iovs[idx];
        pb->atomic_msgs[idx].desc = &pb->desc_atomic_operand;
        pb->atomic_msgs[idx].iov_count = 1;
        pb->atomic_msgs[idx].addr = ri.peer_fi_addr;
        pb->atomic_msgs[idx].rma_iov = &pb->atomic_rma_iovs[idx];
        pb->atomic_msgs[idx].rma_iov_count = 1;
        pb->atomic_msgs[idx].datatype = FI_UINT64;
        pb->atomic_msgs[idx].op = FI_SUM;

        // Setup atomic op
        pb->atomic_ops[idx].ep = pb->ep;
        pb->atomic_ops[idx].msg = pb->atomic_msgs[idx];
        pb->atomic_ops[idx].flags = 0;

        // Setup deferred work with threshold = k+1
        pb->atomic_works[idx].triggering_cntr = sc.trigger_cntr;
        pb->atomic_works[idx].completion_cntr = sc.completion_cntr;
        pb->atomic_works[idx].threshold = k + 1;  // Round k triggered when counter >= k+1
        pb->atomic_works[idx].op_type = FI_OP_ATOMIC;
        pb->atomic_works[idx].op.atomic = &pb->atomic_ops[idx];

        int ret = fi_control(&pb->domain->fid, FI_QUEUE_WORK, &pb->atomic_works[idx]);
        if (ret) {
            OPENGDA_Error("proxy_barrier", "fi_control(atomic slot=%d round=%d) failed: %s",
                         slot, k, fi_strerror(-ret));
            return ret;
        }
    }

    // Queue the "done" atomic to signal when all outbound ops complete
    // This local atomic SUM (+1) to d_slot_done[slot] is triggered when
    // completion_cntr reaches num_rounds (all rounds have completed)
    {
        // Setup atomic IOV (source operand = 1)
        pb->done_iovs[slot].addr = pb->d_atomic_operand;
        pb->done_iovs[slot].count = 1;

        // Setup remote RMA IOV (target: local d_slot_done[slot])
        uint64_t slot_done_addr;
        if (pb->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
            slot_done_addr = (uint64_t)&pb->dev.d_slot_done[slot];
        } else {
            slot_done_addr = slot * sizeof(uint64_t);
        }
        pb->done_rma_iovs[slot].addr = slot_done_addr;
        pb->done_rma_iovs[slot].count = 1;
        pb->done_rma_iovs[slot].key = pb->key_slot_done;

        // Setup atomic message (local atomic to self)
        pb->done_msgs[slot].msg_iov = &pb->done_iovs[slot];
        pb->done_msgs[slot].desc = &pb->desc_atomic_operand;
        pb->done_msgs[slot].iov_count = 1;
        pb->done_msgs[slot].addr = pb->local_fi_addr;  // Target is self
        pb->done_msgs[slot].rma_iov = &pb->done_rma_iovs[slot];
        pb->done_msgs[slot].rma_iov_count = 1;
        pb->done_msgs[slot].datatype = FI_UINT64;
        pb->done_msgs[slot].op = FI_SUM;

        // Setup atomic op
        pb->done_ops[slot].ep = pb->ep;
        pb->done_ops[slot].msg = pb->done_msgs[slot];
        pb->done_ops[slot].flags = 0;

        // Setup deferred work: trigger when completion_cntr >= num_rounds
        // This means all outbound atomic ops have completed
        pb->done_works[slot].triggering_cntr = sc.completion_cntr;
        pb->done_works[slot].completion_cntr = nullptr;  // No further completion needed
        pb->done_works[slot].threshold = num_rounds;
        pb->done_works[slot].op_type = FI_OP_ATOMIC;
        pb->done_works[slot].op.atomic = &pb->done_ops[slot];

        int ret = fi_control(&pb->domain->fid, FI_QUEUE_WORK, &pb->done_works[slot]);
        if (ret) {
            OPENGDA_Error("proxy_barrier", "fi_control(done slot=%d) failed: %s",
                         slot, fi_strerror(-ret));
            return ret;
        }
    }

    return 0;
}

// ============================================================================
// Proxy Thread
// ============================================================================

// Low-latency pause for busy-wait loops
static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    // Fallback: compiler memory barrier
    __asm__ volatile("" ::: "memory");
#endif
}

static void proxy_thread_func(gda_proxy_barrier_t* pb) {
    OPENGDA_Debug("proxy_barrier", "Proxy thread started for rank %d", pb->mype);

    // Consecutive idle iterations before yielding to OS
    constexpr int BUSY_WAIT_ITERS = 1000;
    int idle_count = 0;

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
            did_work = true;  // CQ draining counts as work
        }

        // Low-latency idle strategy:
        // - If work was done, reset idle counter
        // - If no work, do short busy-wait with pause instructions
        // - After BUSY_WAIT_ITERS idle iterations, yield to OS scheduler
        if (did_work) {
            idle_count = 0;
        } else {
            idle_count++;
            if (idle_count < BUSY_WAIT_ITERS) {
                // Short busy-wait with CPU pause instruction
                for (int i = 0; i < 10; i++) {
                    cpu_relax();
                }
            } else {
                // Yield to OS after extended idle period
                std::this_thread::yield();
                idle_count = 0;  // Reset after yield
            }
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

    // Validate window size
    if (window_size <= 0) {
        window_size = GDA_PROXY_DEFAULT_WINDOW_SIZE;
    }
    if (window_size > GDA_PROXY_MAX_WINDOW_SIZE) {
        window_size = GDA_PROXY_MAX_WINDOW_SIZE;
    }

    // Calculate number of rounds for dissemination: ceil(log2(npes))
    int num_rounds = 0;
    int temp = 1;
    while (temp < npes) {
        num_rounds++;
        temp *= 2;
    }

    if (num_rounds > GDA_PROXY_MAX_ROUNDS) {
        OPENGDA_Error("proxy_barrier", "Too many ranks (%d > 2^%d)", npes, GDA_PROXY_MAX_ROUNDS);
        return nullptr;
    }

    OPENGDA_Info("proxy_barrier", "Allocating proxy barrier: rank=%d/%d, window=%d, rounds=%d",
                mype, npes, window_size, num_rounds);

    // Allocate barrier structure with zero-initialization
    gda_proxy_barrier_t* pb = new gda_proxy_barrier();

    // Initialize device context
    pb->dev = gda_proxy_barrier_dev_t{};

    pb->mype = mype;
    pb->npes = npes;
    pb->num_rounds = num_rounds;
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
    pb->mr_round_recv = nullptr;
    pb->mr_slot_done = nullptr;

    // Get fabric resources from OFI
    pb->domain = ofi->get_domain();
    pb->ep = ofi->get_endpoint();
    pb->cq = ofi->get_cq();
    pb->info = ofi->get_info();
    pb->local_fi_addr = ofi->get_local_fi_addr();

    // Initialize device context
    gda_proxy_barrier_dev_t& dev = pb->dev;
    dev.mype = mype;
    dev.npes = npes;
    dev.num_rounds = num_rounds;
    dev.window_size = window_size;

    // Calculate dissemination targets and sources
    for (int k = 0; k < num_rounds; k++) {
        int distance = 1 << k;  // 2^k
        dev.round_targets[k] = (mype + distance) % npes;
        dev.round_sources[k] = (mype - distance + npes) % npes;
    }

    // Allocate GPU-visible slot_state
#ifdef USE_AMDGPU
    hipError_t hip_err;

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

    // Allocate d_round_recv (GPU memory, receives atomic adds from peers)
    hip_err = hipMalloc((void**)&dev.d_round_recv, sizeof(uint64_t) * num_rounds);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_barrier", "hipMalloc(d_round_recv) failed: %s",
                     hipGetErrorString(hip_err));
        (void)hipHostFree((void*)dev.slot_trigger_addrs);
        (void)hipHostFree((void*)dev.slot_state);
        delete pb;
        return nullptr;
    }
    (void)hipMemset((void*)dev.d_round_recv, 0, sizeof(uint64_t) * num_rounds);

    // Allocate d_slot_done (GPU memory, NIC writes completion signals)
    hip_err = hipMalloc((void**)&dev.d_slot_done, sizeof(uint64_t) * window_size);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_barrier", "hipMalloc(d_slot_done) failed: %s",
                     hipGetErrorString(hip_err));
        (void)hipFree((void*)dev.d_round_recv);
        (void)hipHostFree((void*)dev.slot_trigger_addrs);
        (void)hipHostFree((void*)dev.slot_state);
        delete pb;
        return nullptr;
    }
    (void)hipMemset((void*)dev.d_slot_done, 0, sizeof(uint64_t) * window_size);

    // Allocate d_epoch (GPU memory)
    hip_err = hipMalloc((void**)&dev.d_epoch, sizeof(uint64_t));
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_barrier", "hipMalloc(d_epoch) failed: %s",
                     hipGetErrorString(hip_err));
        (void)hipFree((void*)dev.d_slot_done);
        (void)hipFree((void*)dev.d_round_recv);
        (void)hipHostFree((void*)dev.slot_trigger_addrs);
        (void)hipHostFree((void*)dev.slot_state);
        delete pb;
        return nullptr;
    }
    (void)hipMemset((void*)dev.d_epoch, 0, sizeof(uint64_t));

    // Allocate d_spin_cycles (GPU memory, for statistics)
    hip_err = hipMalloc((void**)&dev.d_spin_cycles, sizeof(uint64_t));
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_barrier", "hipMalloc(d_spin_cycles) failed: %s",
                     hipGetErrorString(hip_err));
        (void)hipFree((void*)dev.d_epoch);
        (void)hipFree((void*)dev.d_slot_done);
        (void)hipFree((void*)dev.d_round_recv);
        (void)hipHostFree((void*)dev.slot_trigger_addrs);
        (void)hipHostFree((void*)dev.slot_state);
        delete pb;
        return nullptr;
    }
    (void)hipMemset((void*)dev.d_spin_cycles, 0, sizeof(uint64_t));

    // Allocate atomic operand (GPU memory, value = 1)
    hip_err = hipMalloc((void**)&pb->d_atomic_operand, sizeof(uint64_t));
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_barrier", "hipMalloc(atomic_operand) failed: %s",
                     hipGetErrorString(hip_err));
        (void)hipFree((void*)dev.d_spin_cycles);
        (void)hipFree((void*)dev.d_epoch);
        (void)hipFree((void*)dev.d_slot_done);
        (void)hipFree((void*)dev.d_round_recv);
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

    // Register d_round_recv as MR for remote atomics
    pb->mr_round_recv = ofi->register_memory((void*)dev.d_round_recv,
                                              sizeof(uint64_t) * num_rounds, true);
    if (!pb->mr_round_recv) {
        OPENGDA_Error("proxy_barrier", "Failed to register d_round_recv MR");
        goto cleanup_gpu;
    }
    pb->desc_round_recv = fi_mr_desc(pb->mr_round_recv);
    pb->key_round_recv = fi_mr_key(pb->mr_round_recv);

    // Register d_slot_done as MR for local atomics
    pb->mr_slot_done = ofi->register_memory((void*)dev.d_slot_done,
                                             sizeof(uint64_t) * window_size, true);
    if (!pb->mr_slot_done) {
        OPENGDA_Error("proxy_barrier", "Failed to register d_slot_done MR");
        ofi->deregister_memory(pb->mr_round_recv);
        goto cleanup_gpu;
    }
    pb->desc_slot_done = fi_mr_desc(pb->mr_slot_done);
    pb->key_slot_done = fi_mr_key(pb->mr_slot_done);

    // Register atomic operand as MR
    pb->mr_atomic_operand = ofi->register_memory(pb->d_atomic_operand,
                                                  sizeof(uint64_t), true);
    if (!pb->mr_atomic_operand) {
        OPENGDA_Error("proxy_barrier", "Failed to register atomic_operand MR");
        ofi->deregister_memory(pb->mr_slot_done);
        ofi->deregister_memory(pb->mr_round_recv);
        goto cleanup_gpu;
    }
    pb->desc_atomic_operand = fi_mr_desc(pb->mr_atomic_operand);

    // Exchange d_round_recv addresses via bootstrap
    {
        // Prepare exchange data: [addr, key] for each round
        // All rounds use same base address with different offsets
        struct ExchangeRoundRecv {
            uint64_t base_addr;
            uint64_t key;
        } my_info;

        if (pb->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
            my_info.base_addr = (uint64_t)dev.d_round_recv;
        } else {
            my_info.base_addr = 0;
        }
        my_info.key = pb->key_round_recv;

        char my_hex[64];
        bytes_to_hex((uint8_t*)&my_info, sizeof(my_info), my_hex);

        char key_str[64];
        snprintf(key_str, sizeof(key_str), "round_recv-%d", mype);
        bool rc = bootstrap->bootstrap_kvs_put(key_str, my_hex);
        if (!rc) {
            OPENGDA_Error("proxy_barrier", "KV put failed for round_recv");
            ofi->deregister_memory(pb->mr_atomic_operand);
            ofi->deregister_memory(pb->mr_slot_done);
            ofi->deregister_memory(pb->mr_round_recv);
            goto cleanup_gpu;
        }

        bootstrap->bootstrap_barrier();

        // Build round info by fetching from each target peer
        pb->round_info.resize(num_rounds);
        for (int k = 0; k < num_rounds; k++) {
            int target = dev.round_targets[k];
            snprintf(key_str, sizeof(key_str), "round_recv-%d", target);

            char peer_hex[128];
            int peer_hex_len = 0;
            rc = bootstrap->bootstrap_kvs_get(key_str, peer_hex, &peer_hex_len);
            if (!rc) {
                OPENGDA_Error("proxy_barrier", "KV get failed for rank %d round_recv", target);
                ofi->deregister_memory(pb->mr_atomic_operand);
                ofi->deregister_memory(pb->mr_slot_done);
                ofi->deregister_memory(pb->mr_round_recv);
                goto cleanup_gpu;
            }

            ExchangeRoundRecv peer_info;
            hex_to_bytes(peer_hex, (uint8_t*)&peer_info, sizeof(peer_info));

            const PeerInfo* peer = ofi->get_peer_info(target);
            if (!peer || !peer->valid) {
                OPENGDA_Error("proxy_barrier", "Invalid peer info for rank %d", target);
                ofi->deregister_memory(pb->mr_atomic_operand);
                ofi->deregister_memory(pb->mr_slot_done);
                ofi->deregister_memory(pb->mr_round_recv);
                goto cleanup_gpu;
            }

            pb->round_info[k].peer_fi_addr = peer->fi_addr;
            pb->round_info[k].remote_addr = peer_info.base_addr + k * sizeof(uint64_t);
            pb->round_info[k].remote_key = peer_info.key;

            OPENGDA_Debug("proxy_barrier", "Round %d: target=%d, remote_addr=0x%lx, key=0x%lx",
                         k, target, pb->round_info[k].remote_addr, pb->round_info[k].remote_key);
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

    // Allocate DWQ structures
    {
        int total_atomics = window_size * num_rounds;
        pb->atomic_works.resize(total_atomics);
        pb->atomic_ops.resize(total_atomics);
        pb->atomic_msgs.resize(total_atomics);
        pb->atomic_iovs.resize(total_atomics);
        pb->atomic_rma_iovs.resize(total_atomics);

        pb->done_works.resize(window_size);
        pb->done_ops.resize(window_size);
        pb->done_msgs.resize(window_size);
        pb->done_iovs.resize(window_size);
        pb->done_rma_iovs.resize(window_size);
    }

    pb->initialized = true;

    // Initial arming of all slots
    OPENGDA_Debug("proxy_barrier", "Arming initial %d slots", window_size);
    for (int s = 0; s < window_size; s++) {
        int ret = queue_slot_work(pb, s);
        if (ret) {
            OPENGDA_Error("proxy_barrier", "Initial arm of slot %d failed", s);
            goto cleanup_slots;
        }
        __atomic_store_n(&pb->dev.slot_state[s], GDA_SLOT_ARMED, __ATOMIC_RELEASE);
    }

    // Barrier to ensure all ranks have initialized
    bootstrap->bootstrap_barrier();

    OPENGDA_Info("proxy_barrier", "Proxy barrier allocated and initialized: rank=%d, window=%d, rounds=%d",
                mype, window_size, num_rounds);

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
    ofi->deregister_memory(pb->mr_slot_done);
    ofi->deregister_memory(pb->mr_round_recv);

cleanup_gpu:
#ifdef USE_AMDGPU
    (void)hipFree((void*)dev.d_spin_cycles);
    (void)hipFree((void*)dev.d_epoch);
    (void)hipFree((void*)dev.d_slot_done);
    (void)hipFree((void*)dev.d_round_recv);
    (void)hipFree(pb->d_atomic_operand);
    (void)hipHostFree((void*)dev.slot_trigger_addrs);
    (void)hipHostFree((void*)dev.slot_state);
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
    if (barrier->mr_slot_done) {
        ofi->deregister_memory(barrier->mr_slot_done);
    }
    if (barrier->mr_round_recv) {
        ofi->deregister_memory(barrier->mr_round_recv);
    }

    // Cleanup GPU memory
#ifdef USE_AMDGPU
    gda_proxy_barrier_dev_t& dev = barrier->dev;
    if (dev.d_spin_cycles) (void)hipFree((void*)dev.d_spin_cycles);
    if (dev.d_epoch) (void)hipFree((void*)dev.d_epoch);
    if (dev.d_slot_done) (void)hipFree((void*)dev.d_slot_done);
    if (dev.d_round_recv) (void)hipFree((void*)dev.d_round_recv);
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

#ifdef USE_AMDGPU
    if (barrier->dev.d_spin_cycles) {
        (void)hipMemcpy(&stats->gpu_spin_cycles, (void*)barrier->dev.d_spin_cycles,
                        sizeof(uint64_t), hipMemcpyDeviceToHost);
    } else {
        stats->gpu_spin_cycles = 0;
    }
#else
    stats->gpu_spin_cycles = 0;
#endif

    return 0;
}
