/*
 * counter_pool.hpp - Reusable counter pool for DWQ operations
 *
 * Provides a fixed-size pool of counter pairs that can be acquired/released
 * to support more concurrent streams than the hardware limit.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

#include "fabric_dwq_context.hpp"
#include "memory_region.hpp"
#include "dwq_work_builder.hpp"

// Pool size is the max concurrent DWQ operations supported by hardware
constexpr int POOL_SIZE = 6;

// Slot state visible to both CPU and GPU
struct SlotState {
    volatile uint64_t* dev_trigger_addr;   // GPU writes here to trigger
    volatile uint64_t* atomic_result;      // GPU polls here for completion
    int stream_id;                         // Which stream is using this slot (-1 = free)
    bool ready_for_trigger;                // CPU sets when DWQ is queued
};

class CounterPool {
public:
    FabricDwqContext& fabric;
    int rank;

    // Pool of counter pairs
    FabricDwqContext::CounterPair counters[POOL_SIZE];

    // Pool of DWQ work builders (one per slot)
    DwqWorkBuilder* dwq_builders[POOL_SIZE];

    // Per-slot GPU resources
    uint64_t* atomic_results[POOL_SIZE];      // GPU memory for completion signals
    uint64_t* atomic_operands[POOL_SIZE];     // GPU memory for atomic operands
    MemoryRegion* mr_atomic[POOL_SIZE];
    MemoryRegion* mr_atomic_operand[POOL_SIZE];

    // Slot states (host memory, GPU-accessible)
    SlotState* h_slot_states;
    SlotState* d_slot_states;

    // GPU arrays for kernel
    volatile uint64_t** d_trigger_addrs;
    volatile uint64_t** d_atomic_results;

    CounterPool(FabricDwqContext& fabric_, int rank_)
        : fabric(fabric_), rank(rank_),
          h_slot_states(nullptr), d_slot_states(nullptr),
          d_trigger_addrs(nullptr), d_atomic_results(nullptr)
    {
        init_pool();
    }

    ~CounterPool() {
        // Free GPU arrays
        if (d_trigger_addrs) (void)hipFree(d_trigger_addrs);
        if (d_atomic_results) (void)hipFree(d_atomic_results);
        if (d_slot_states) (void)hipFree(d_slot_states);
        if (h_slot_states) (void)hipHostFree(h_slot_states);

        // Free per-slot resources
        for (int i = 0; i < POOL_SIZE; i++) {
            delete mr_atomic[i];
            delete mr_atomic_operand[i];
            delete dwq_builders[i];
            if (atomic_results[i]) (void)hipFree(atomic_results[i]);
            if (atomic_operands[i]) (void)hipFree(atomic_operands[i]);
            fabric.destroy_counter_pair(counters[i]);
        }
    }

    // No copy/move
    CounterPool(const CounterPool&) = delete;
    CounterPool& operator=(const CounterPool&) = delete;

    // Reset a slot for reuse (call after operation completes)
    void reset_slot(int slot_idx) {
        if (slot_idx < 0 || slot_idx >= POOL_SIZE) return;

        // Reset counters
        fi_cntr_set(counters[slot_idx].trigger_cntr, 0);
        fi_cntr_set(counters[slot_idx].completion_cntr, 0);
        fi_cntr_set(counters[slot_idx].atomic_completion_cntr, 0);

        // Reset atomic_result on GPU
        uint64_t zero = 0;
        (void)hipMemcpy(atomic_results[slot_idx], &zero, sizeof(uint64_t),
                        hipMemcpyHostToDevice);

        // Mark slot as free
        h_slot_states[slot_idx].stream_id = -1;
        h_slot_states[slot_idx].ready_for_trigger = false;
    }

    // Reset all slots
    void reset_all() {
        for (int i = 0; i < POOL_SIZE; i++) {
            reset_slot(i);
        }
        (void)hipDeviceSynchronize();
    }

    // Queue a DWQ RMA write + atomic signal for a slot
    // Returns true on success, false if slot not available
    bool queue_operation(
        int slot_idx,
        int stream_id,
        void* src_buf,
        void* src_desc,
        size_t size,
        fi_addr_t dest_addr,
        uint64_t remote_addr,
        uint64_t remote_key)
    {
        if (slot_idx < 0 || slot_idx >= POOL_SIZE) return false;

        h_slot_states[slot_idx].stream_id = stream_id;

        // Queue RMA write
        dwq_builders[slot_idx]->queue_rma_write(
            fabric.domain, fabric.ep,
            src_buf, src_desc, size,
            dest_addr, remote_addr, remote_key,
            counters[slot_idx].trigger_cntr,
            counters[slot_idx].completion_cntr, 1);

        // Queue atomic signal (self-atomic to local GPU memory)
        uint64_t atomic_result_addr = fabric.is_virt_addr_mode()
            ? (uint64_t)atomic_results[slot_idx] : 0;

        dwq_builders[slot_idx]->queue_atomic_signal(
            fabric.domain, fabric.ep,
            atomic_operands[slot_idx], mr_atomic_operand[slot_idx]->desc,
            atomic_results[slot_idx], mr_atomic[slot_idx]->key, atomic_result_addr,
            fabric.local_addr_in_av,
            counters[slot_idx].completion_cntr,
            counters[slot_idx].atomic_completion_cntr, 1);

        h_slot_states[slot_idx].ready_for_trigger = true;
        return true;
    }

    // Check if a slot's operation has completed
    bool is_slot_complete(int slot_idx) {
        if (slot_idx < 0 || slot_idx >= POOL_SIZE) return false;
        uint64_t val;
        (void)hipMemcpy(&val, atomic_results[slot_idx], sizeof(uint64_t),
                        hipMemcpyDeviceToHost);
        return val >= 1;
    }

    // Get device trigger address array (for kernel)
    volatile uint64_t** get_d_trigger_addrs() { return d_trigger_addrs; }

    // Get device atomic results array (for kernel)
    volatile uint64_t** get_d_atomic_results() { return d_atomic_results; }

private:
    void check_hip(hipError_t err, const char* msg) {
        if (err != hipSuccess) {
            fprintf(stderr, "Rank %d: CounterPool %s failed: %s\n",
                    rank, msg, hipGetErrorString(err));
            exit(1);
        }
    }

    void init_pool() {
        // Allocate host-pinned slot states (GPU accessible)
        check_hip(hipHostMalloc(&h_slot_states, POOL_SIZE * sizeof(SlotState),
                                hipHostMallocMapped), "hipHostMalloc(slot_states)");
        memset(h_slot_states, 0, POOL_SIZE * sizeof(SlotState));

        // Get device pointer for slot states
        check_hip(hipHostGetDevicePointer((void**)&d_slot_states, h_slot_states, 0),
                  "hipHostGetDevicePointer(slot_states)");

        // Initialize per-slot resources
        volatile uint64_t* h_trigger_addrs[POOL_SIZE];
        volatile uint64_t* h_atomic_results[POOL_SIZE];

        for (int i = 0; i < POOL_SIZE; i++) {
            // Create counter pair
            counters[i] = fabric.create_counter_pair();

            // Allocate GPU buffers
            check_hip(hipMalloc(&atomic_results[i], sizeof(uint64_t)),
                      "hipMalloc(atomic_result)");
            check_hip(hipMalloc(&atomic_operands[i], sizeof(uint64_t)),
                      "hipMalloc(atomic_operand)");

            // Initialize
            check_hip(hipMemset(atomic_results[i], 0, sizeof(uint64_t)),
                      "hipMemset(atomic_result)");
            uint64_t one = 1;
            check_hip(hipMemcpy(atomic_operands[i], &one, sizeof(uint64_t),
                                hipMemcpyHostToDevice), "hipMemcpy(atomic_operand)");

            // Register memory
            mr_atomic[i] = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                            atomic_results[i], sizeof(uint64_t), true, 0, rank);
            mr_atomic_operand[i] = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                                    atomic_operands[i], sizeof(uint64_t), true, 0, rank);

            // Create DWQ work builder
            dwq_builders[i] = new DwqWorkBuilder(rank);

            // Setup slot state
            h_slot_states[i].dev_trigger_addr = counters[i].dev_trigger_cntr;
            h_slot_states[i].atomic_result = atomic_results[i];
            h_slot_states[i].stream_id = -1;
            h_slot_states[i].ready_for_trigger = false;

            // Build host arrays
            h_trigger_addrs[i] = counters[i].dev_trigger_cntr;
            h_atomic_results[i] = atomic_results[i];
        }

        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

        // Allocate and copy GPU arrays
        check_hip(hipMalloc(&d_trigger_addrs, POOL_SIZE * sizeof(volatile uint64_t*)),
                  "hipMalloc(d_trigger_addrs)");
        check_hip(hipMalloc(&d_atomic_results, POOL_SIZE * sizeof(volatile uint64_t*)),
                  "hipMalloc(d_atomic_results)");

        check_hip(hipMemcpy(d_trigger_addrs, h_trigger_addrs,
                            POOL_SIZE * sizeof(volatile uint64_t*), hipMemcpyHostToDevice),
                  "hipMemcpy(d_trigger_addrs)");
        check_hip(hipMemcpy(d_atomic_results, h_atomic_results,
                            POOL_SIZE * sizeof(volatile uint64_t*), hipMemcpyHostToDevice),
                  "hipMemcpy(d_atomic_results)");
    }
};
