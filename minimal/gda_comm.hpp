/**
 * gda_comm.hpp - Simple communication wrapper for GPU-Direct Async (DWQ)
 *
 * Provides a simple API for GPU-triggered RDMA operations:
 *   - put(): queue a one-sided write to remote rank
 *   - trigger(): trigger queued DWQ operations from GPU
 *   - wait(): wait for pending operations
 *   - barrier(): global synchronization
 *
 * Simple usage:
 *   GdaComm comm;
 *   auto handle = comm.register_buffer(d_buf, size, true);
 *   comm.set_remote_info_by_index(dest_rank, 0, remote_addr, remote_key);
 *   uint64_t thresh = comm.put(handle, dest_rank, 0, size);
 *   comm.trigger(thresh);  // GPU triggers the DWQ operation
 *   comm.wait(thresh);
 *
 * Advanced usage (custom kernel with trigger):
 *   volatile uint64_t* trigger_addr = comm.get_trigger_addr();
 *   // In your kernel: *trigger_addr = threshold;
 */
#pragma once

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <vector>
#include <unordered_map>

#include "hip_device_context.hpp"
#include "pmi_session.hpp"
#include "device_affinity.hpp"
#include "fabric_dwq_context.hpp"
#include "memory_region.hpp"
#include "dwq_work_builder.hpp"
#include "ofi_barrier.hpp"

// GPU kernel to trigger DWQ operations
__global__ void gda_trigger_kernel(volatile uint64_t* trigger_addr, uint64_t threshold) {
    *trigger_addr = threshold;
}

// Handle to a registered memory region
struct GdaHandle {
    void* buf;              // Local buffer pointer
    size_t size;            // Buffer size
    MemoryRegion* mr;       // Memory region (may be null for raw handles)
    void* local_desc;       // Local descriptor for DWQ operations
    uint64_t rma_addr;      // RMA address
    uint64_t rma_key;       // RMA key
};

// Remote RMA info for a specific buffer
struct GdaRemoteInfo {
    fi_addr_t av_addr;
    uint64_t rma_addr;
    uint64_t rma_key;
};

class GdaComm {
public:
    // Core components (public for advanced usage)
    PmiSession pmi;
    DeviceAffinityDetector* affinity;
    HipDeviceContext* hip;
    FabricDwqContext* fabric;
    OfiBarrier* ofi_barrier;

    // Address book: rank -> fi_addr_t
    std::vector<fi_addr_t> av_addrs;

    // Remote RMA info: (rank, buf_index) -> remote info
    std::unordered_map<uint64_t, GdaRemoteInfo> remote_info;

    // Registered memory regions (for cleanup)
    std::vector<MemoryRegion*> registered_mrs;

    // DWQ work builders (one per pending operation)
    std::vector<DwqWorkBuilder*> pending_ops;

    // Counter writeback ops (for cleanup)
    std::vector<fi_op_cntr*> pending_wb_ops;

    // Current threshold counter
    uint64_t current_threshold;

    /**
     * Initialize GDA communication
     * @param local_rank Local rank for GPU selection (e.g., SLURM_LOCALID)
     */
    explicit GdaComm(int local_rank = -1)
        : affinity(nullptr), hip(nullptr), fabric(nullptr), ofi_barrier(nullptr),
          current_threshold(0)
    {
        // Get local rank from PMI if not provided
        if (local_rank < 0) {
            local_rank = pmi.local_rank;
            if (local_rank < 0) {
                int num_devices;
                (void)hipGetDeviceCount(&num_devices);
                local_rank = pmi.rank % num_devices;
            }
        }

        // Initialize components
        affinity = new DeviceAffinityDetector(local_rank);
        hip = new HipDeviceContext(affinity->selected_gpu_id);
        fabric = new FabricDwqContext(pmi.rank, affinity);

        // Exchange addresses with all ranks
        exchange_addresses();

        // Initialize OFI barrier
        init_barrier();
    }

    ~GdaComm() {
        // Cleanup pending operations
        for (auto* op : pending_ops) delete op;
        pending_ops.clear();

        // Cleanup writeback ops
        for (auto* wb : pending_wb_ops) delete wb;
        pending_wb_ops.clear();

        // Cleanup registered memory regions
        for (auto* mr : registered_mrs) delete mr;
        registered_mrs.clear();

        // Cleanup components (reverse order)
        delete ofi_barrier;
        delete fabric;
        delete hip;
        delete affinity;
    }

    // No copy
    GdaComm(const GdaComm&) = delete;
    GdaComm& operator=(const GdaComm&) = delete;

    /**
     * Register a buffer for RDMA operations
     * @param buf Buffer pointer
     * @param size Buffer size in bytes
     * @param is_device True if buffer is on GPU
     * @return Handle for use in put/get operations
     */
    GdaHandle register_buffer(void* buf, size_t size, bool is_device) {
        auto* mr = new MemoryRegion(
            fabric->domain, fabric->ep, fabric->cxi_info,
            buf, size, is_device, hip->gpu_id, pmi.rank);
        registered_mrs.push_back(mr);

        GdaHandle handle;
        handle.buf = buf;
        handle.size = size;
        handle.mr = mr;
        handle.local_desc = mr->desc;
        handle.rma_addr = (uint64_t)buf;
        handle.rma_key = mr->key;
        return handle;
    }

    /**
     * Set remote RMA info by buffer index (for double-buffering scenarios)
     * @param dest_rank Destination rank
     * @param buf_index Buffer index (used as map key)
     * @param remote_addr Remote buffer address
     * @param remote_key Remote buffer key
     */
    void set_remote_info_by_index(int dest_rank, int buf_index,
                                   uint64_t remote_addr, uint64_t remote_key) {
        uint64_t map_key = make_remote_key(dest_rank, buf_index);
        remote_info[map_key] = {av_addrs[dest_rank], remote_addr, remote_key};
    }

    /**
     * Queue a put operation (GPU-triggered RDMA write with writeback)
     * @param src_handle Source buffer handle
     * @param dest_rank Destination rank
     * @param dest_buf_index Destination buffer index (for double-buffering)
     * @param size Transfer size
     * @return The threshold value to pass to trigger()
     */
    uint64_t put(const GdaHandle& src_handle, int dest_rank,
                 int dest_buf_index, size_t size) {
        current_threshold++;
        uint64_t threshold = current_threshold;

        uint64_t map_key = make_remote_key(dest_rank, dest_buf_index);
        auto it = remote_info.find(map_key);
        if (it == remote_info.end()) {
            fprintf(stderr, "Rank %d: RMA info not set for rank %d buf %d\n",
                    pmi.rank, dest_rank, dest_buf_index);
            exit(1);
        }
        const auto& ri = it->second;

        uint64_t remote_addr = fabric->is_virt_addr_mode() ? ri.rma_addr : 0;

        // Queue RMA write
        auto* dwq = new DwqWorkBuilder(pmi.rank);
        dwq->queue_rma_write(
            fabric->domain, fabric->ep,
            src_handle.buf, src_handle.mr->desc, size,
            ri.av_addr, remote_addr, ri.rma_key,
            fabric->trigger_cntr, fabric->completion_cntr, threshold);
        pending_ops.push_back(dwq);

        // Queue writeback (counter self-increment for stable completion tracking)
        queue_counter_writeback(threshold);

        return threshold;
    }

    /**
     * Queue a raw put operation with explicit remote address and key
     * @param src_handle Source buffer handle (must have local_desc set)
     * @param dest_rank Destination rank
     * @param remote_addr Remote buffer address
     * @param remote_key Remote buffer key
     * @param size Transfer size
     * @return The threshold value to pass to trigger()
     */
    uint64_t put_raw(const GdaHandle& src_handle, int dest_rank,
                     uint64_t remote_addr, uint64_t remote_key, size_t size) {
        current_threshold++;
        uint64_t threshold = current_threshold;

        // Use direct remote address for non-virt_addr mode
        uint64_t rma_addr = fabric->is_virt_addr_mode() ? remote_addr : remote_addr;

        // Queue RMA write
        auto* dwq = new DwqWorkBuilder(pmi.rank);
        dwq->queue_rma_write(
            fabric->domain, fabric->ep,
            src_handle.buf, src_handle.local_desc, size,
            av_addrs[dest_rank], rma_addr, remote_key,
            fabric->trigger_cntr, fabric->completion_cntr, threshold);
        pending_ops.push_back(dwq);

        // Queue writeback
        queue_counter_writeback(threshold);

        return threshold;
    }

    /**
     * Trigger queued DWQ operations from GPU
     * Launches a simple kernel that writes to the trigger counter
     * @param threshold The threshold value returned by put()
     */
    void trigger(uint64_t threshold) {
        hipLaunchKernelGGL(gda_trigger_kernel, dim3(1), dim3(1), 0, 0,
                           fabric->dev_trigger_cntr, threshold);
    }

    /**
     * Wait for completion counter to reach threshold
     * @param threshold Expected completion counter value
     */
    void wait(uint64_t threshold) {
        while (fi_cntr_read(fabric->completion_cntr) < threshold) {
            fi_cq_read(fabric->cq, NULL, 0);
        }
    }

    /**
     * Fast flush: progress CQ without full flush
     * @param expected_completions Expected completion counter value
     */
    void fast_flush(uint64_t expected_completions) {
        fabric->fast_flush(expected_completions);
    }

    /**
     * Get GPU-accessible trigger counter address (for advanced usage)
     * Use this when you want to trigger in your own kernel instead of using trigger()
     * Example: *comm.get_trigger_addr() = threshold;
     */
    volatile uint64_t* get_trigger_addr() const {
        return fabric->dev_trigger_cntr;
    }

    /**
     * Global barrier across all ranks
     */
    void barrier() {
        ofi_barrier->barrier();
    }

    /**
     * Reset counters for next batch of operations
     */
    void reset_counters() {
        fi_cntr_set(fabric->trigger_cntr, 0);
        fi_cntr_set(fabric->completion_cntr, 0);
        current_threshold = 0;

        // Cleanup pending ops
        for (auto* op : pending_ops) delete op;
        pending_ops.clear();

        // Cleanup writeback ops
        for (auto* wb : pending_wb_ops) delete wb;
        pending_wb_ops.clear();
    }

    /**
     * Flush DWQ (slower, use fast_flush when possible)
     */
    void flush() {
        fabric->flush_dwq();
    }

    // Accessors
    int rank() const { return pmi.rank; }
    int size() const { return pmi.size; }
    int gpu_id() const { return hip->gpu_id; }
    bool is_virt_addr_mode() const { return fabric->is_virt_addr_mode(); }

private:
    void queue_counter_writeback(uint64_t threshold) {
        struct fi_deferred_work wb_work = {};
        auto* wb_op = new fi_op_cntr();
        wb_op->cntr = fabric->completion_cntr;
        wb_op->value = 1;
        wb_work.triggering_cntr = fabric->completion_cntr;
        wb_work.completion_cntr = nullptr;
        wb_work.threshold = threshold;
        wb_work.op_type = FI_OP_CNTR_ADD;
        wb_work.op.cntr = wb_op;

        int ret = fi_control(&fabric->domain->fid, FI_QUEUE_WORK, &wb_work);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_control(writeback) failed: %s\n",
                    pmi.rank, fi_strerror(-ret));
            exit(1);
        }
        pending_wb_ops.push_back(wb_op);
    }

    void exchange_addresses() {
        av_addrs.resize(pmi.size);

        char* my_hex = (char*)malloc(2 * fabric->addrlen + 1);
        bytes_to_hex((uint8_t*)fabric->local_addr, fabric->addrlen, my_hex);

        char key[PMI2_MAX_KEYLEN];
        snprintf(key, sizeof(key), "addr-%d", pmi.rank);
        pmi.kvs_put(key, my_hex);
        pmi.barrier();

        uint8_t* peer_bin = (uint8_t*)malloc(fabric->addrlen);
        for (int r = 0; r < pmi.size; r++) {
            char peer_hex[PMI2_MAX_VALLEN];
            snprintf(key, sizeof(key), "addr-%d", r);
            pmi.kvs_get(key, peer_hex, sizeof(peer_hex));
            hex_to_bytes(peer_hex, peer_bin, fabric->addrlen);

            if (fi_av_insert(fabric->av, peer_bin, 1, &av_addrs[r], 0, NULL) != 1) {
                fprintf(stderr, "Rank %d: fi_av_insert(rank %d) failed\n", pmi.rank, r);
                exit(1);
            }
        }

        fabric->local_addr_in_av = av_addrs[pmi.rank];
        free(my_hex);
        free(peer_bin);
    }

    void init_barrier() {
        ofi_barrier = new OfiBarrier(fabric->domain, fabric->av, fabric->cxi_info,
                                      pmi.rank, pmi.size);

        // Exchange barrier EP addresses
        char* my_hex = (char*)malloc(2 * ofi_barrier->addrlen + 1);
        bytes_to_hex((uint8_t*)ofi_barrier->local_addr, ofi_barrier->addrlen, my_hex);

        char key[PMI2_MAX_KEYLEN];
        snprintf(key, sizeof(key), "barrier-ep-%d", pmi.rank);
        pmi.kvs_put(key, my_hex);
        pmi.barrier();

        uint8_t* peer_bin = (uint8_t*)malloc(ofi_barrier->addrlen);
        for (int r = 0; r < pmi.size; r++) {
            char peer_hex[PMI2_MAX_VALLEN];
            snprintf(key, sizeof(key), "barrier-ep-%d", r);
            pmi.kvs_get(key, peer_hex, sizeof(peer_hex));
            hex_to_bytes(peer_hex, peer_bin, ofi_barrier->addrlen);

            fi_addr_t peer_av_addr;
            if (fi_av_insert(fabric->av, peer_bin, 1, &peer_av_addr, 0, NULL) != 1) {
                fprintf(stderr, "Rank %d: fi_av_insert(barrier %d) failed\n", pmi.rank, r);
                exit(1);
            }
            ofi_barrier->set_peer_addr(r, peer_av_addr);
        }

        ofi_barrier->post_initial_recvs();

        free(my_hex);
        free(peer_bin);
    }

    uint64_t make_remote_key(int rank, int buf_index) const {
        return ((uint64_t)rank << 48) | ((uint64_t)buf_index & 0xFFFFFFFFFFFF);
    }

    // Hex conversion utilities
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
};
