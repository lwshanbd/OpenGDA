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
#include <vector>
#include <unordered_map>

#include "hip_device_context.hpp"
#include "gicc/bootstrap/bootstrap.hpp"
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
    uint64_t rma_addr;      // Full address (base + offset) for virt_addr mode
    uint64_t rma_key;
    uint64_t base_addr;     // MR base address (for computing offset in non-virt_addr mode)
};

class GdaComm {
public:
    // Core components (public for advanced usage)
    gicc::Bootstrap& boot;
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

    // GPU-accessible completion signaling (for fused kernel wait)
    uint64_t* atomic_result;        // GPU memory - incremented when operations complete
    uint64_t* atomic_operand;       // GPU memory - value to add (always 1)
    MemoryRegion* mr_atomic_result;
    MemoryRegion* mr_atomic_operand;
    struct fid_cntr* atomic_completion_cntr;  // Counter for atomic ops

    /**
     * Initialize GDA communication
     * @param boot_ Bootstrap instance providing rank, size, and collective ops
     * @param local_rank Local rank for GPU selection (e.g., SLURM_LOCALID)
     */
    explicit GdaComm(gicc::Bootstrap& boot_, int local_rank = -1)
        : boot(boot_),
          affinity(nullptr), hip(nullptr), fabric(nullptr), ofi_barrier(nullptr),
          current_threshold(0),
          atomic_result(nullptr), atomic_operand(nullptr),
          mr_atomic_result(nullptr), mr_atomic_operand(nullptr),
          atomic_completion_cntr(nullptr)
    {
        // Get local rank from Bootstrap if not provided
        if (local_rank < 0) {
            local_rank = boot.local_rank();
            if (local_rank < 0) {
                int num_devices;
                (void)hipGetDeviceCount(&num_devices);
                local_rank = boot.rank() % num_devices;
            }
        }

        // Initialize components
        affinity = new DeviceAffinityDetector(local_rank);
        hip = new HipDeviceContext(affinity->selected_gpu_id);
        fabric = new FabricDwqContext(boot.rank(), affinity);

        // Exchange addresses with all ranks
        exchange_addresses();

        // Initialize OFI barrier
        init_barrier();

        // Initialize GPU-accessible atomic signaling
        init_atomic_signaling();
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

        // Cleanup atomic signaling resources
        delete mr_atomic_result;
        delete mr_atomic_operand;
        if (atomic_completion_cntr) fi_close(&atomic_completion_cntr->fid);
        if (atomic_result) (void)hipFree(atomic_result);
        if (atomic_operand) (void)hipFree(atomic_operand);

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
            buf, size, is_device, hip->gpu_id, boot.rank());
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
     * @param remote_addr Remote buffer address (base + offset)
     * @param remote_key Remote buffer key
     * @param remote_base_addr Remote MR base address (default 0 means remote_addr is the base)
     */
    void set_remote_info_by_index(int dest_rank, int buf_index,
                                   uint64_t remote_addr, uint64_t remote_key,
                                   uint64_t remote_base_addr = 0) {
        uint64_t map_key = make_remote_key(dest_rank, buf_index);
        // If base_addr is not provided, assume remote_addr is the base (offset = 0)
        uint64_t base = (remote_base_addr != 0) ? remote_base_addr : remote_addr;
        remote_info[map_key] = {av_addrs[dest_rank], remote_addr, remote_key, base};
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
                    boot.rank(), dest_rank, dest_buf_index);
            exit(1);
        }
        const auto& ri = it->second;

        // Compute remote address based on MR mode:
        // - VIRT_ADDR mode: use full virtual address (base + offset)
        // - Non-VIRT_ADDR mode: use offset relative to MR base
        uint64_t remote_addr;
        if (fabric->is_virt_addr_mode()) {
            remote_addr = ri.rma_addr;
        } else {
            // Compute offset from base address
            remote_addr = ri.rma_addr - ri.base_addr;
        }

        // Debug: print addresses for first few operations (disabled for performance)
        // static int put_debug_count = 0;
        // if (put_debug_count < 5) {
        //     printf("Rank %d put(): virt_addr=%d, ri.rma_addr=0x%lx, ri.base_addr=0x%lx, computed remote_addr=0x%lx, key=0x%lx\n",
        //            boot.rank(), fabric->is_virt_addr_mode(), ri.rma_addr, ri.base_addr, remote_addr, ri.rma_key);
        //     put_debug_count++;
        // }

        // Queue RMA write
        auto* dwq = new DwqWorkBuilder(boot.rank());
        dwq->queue_rma_write(
            fabric->domain, fabric->ep,
            src_handle.buf, src_handle.mr->desc, size,
            ri.av_addr, remote_addr, ri.rma_key,
            fabric->trigger_cntr, fabric->completion_cntr, threshold);
        pending_ops.push_back(dwq);

        // NOTE: Removed queue_counter_writeback() - it caused a race condition
        // where completion_cntr grew by 2 per operation (1 from RMA + 1 from writeback),
        // causing wait() to return early in subsequent iterations.
        // The RMA completion counter alone is sufficient.

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
        auto* dwq = new DwqWorkBuilder(boot.rank());
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
     * Trigger queued DWQ operations from CPU (alternative to GPU trigger)
     * Uses fi_cntr_set directly - useful for debugging GPU MMIO issues
     * @param threshold The threshold value returned by put()
     */
    void trigger_cpu(uint64_t threshold) {
        fi_cntr_set(fabric->trigger_cntr, threshold);
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
     * Cleanup pending ops without resetting counters
     * Use this to free memory while keeping counter state
     */
    void cleanup_pending_ops() {
        for (auto* op : pending_ops) delete op;
        pending_ops.clear();
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
    int rank() const { return boot.rank(); }
    int size() const { return boot.size(); }
    int gpu_id() const { return hip->gpu_id; }
    bool is_virt_addr_mode() const { return fabric->is_virt_addr_mode(); }
    uint64_t get_current_threshold() const { return current_threshold; }

    /**
     * Get remote RMA info for debugging
     */
    GdaRemoteInfo get_remote_info(int dest_rank, int buf_index) const {
        uint64_t map_key = make_remote_key(dest_rank, buf_index);
        auto it = remote_info.find(map_key);
        if (it != remote_info.end()) {
            return it->second;
        }
        return {0, 0, 0, 0};
    }

    /**
     * Get GPU-accessible atomic result pointer (for fused kernel wait)
     * GPU polls this value to know when operations complete
     */
    volatile uint64_t* get_atomic_result() const {
        return atomic_result;
    }

    /**
     * Reset atomic result counter (call before each batch of operations)
     */
    void reset_atomic_result() {
        uint64_t zero = 0;
        (void)hipMemcpy(atomic_result, &zero, sizeof(uint64_t), hipMemcpyHostToDevice);
    }

    /**
     * Queue a put operation with GPU-signaled completion
     * When RMA completes, atomic_result is incremented (GPU can poll this)
     * @param src_handle Source buffer handle
     * @param dest_rank Destination rank
     * @param dest_buf_index Destination buffer index
     * @param size Transfer size
     * @return The threshold value to pass to trigger()
     */
    uint64_t put_with_signal(const GdaHandle& src_handle, int dest_rank,
                              int dest_buf_index, size_t size) {
        current_threshold++;
        uint64_t threshold = current_threshold;

        uint64_t map_key = make_remote_key(dest_rank, dest_buf_index);
        auto it = remote_info.find(map_key);
        if (it == remote_info.end()) {
            fprintf(stderr, "Rank %d: RMA info not set for rank %d buf %d\n",
                    boot.rank(), dest_rank, dest_buf_index);
            exit(1);
        }
        const auto& ri = it->second;

        uint64_t remote_addr;
        if (fabric->is_virt_addr_mode()) {
            remote_addr = ri.rma_addr;
        } else {
            remote_addr = ri.rma_addr - ri.base_addr;
        }

        // Queue RMA write
        auto* dwq = new DwqWorkBuilder(boot.rank());
        dwq->queue_rma_write(
            fabric->domain, fabric->ep,
            src_handle.buf, src_handle.mr->desc, size,
            ri.av_addr, remote_addr, ri.rma_key,
            fabric->trigger_cntr, fabric->completion_cntr, threshold);

        // Queue atomic signal to increment atomic_result when RMA completes
        uint64_t atomic_result_addr = fabric->is_virt_addr_mode()
            ? (uint64_t)atomic_result : 0;
        dwq->queue_atomic_signal(
            fabric->domain, fabric->ep,
            atomic_operand, mr_atomic_operand->desc,
            atomic_result, mr_atomic_result->key, atomic_result_addr,
            fabric->local_addr_in_av,
            fabric->completion_cntr,
            atomic_completion_cntr, threshold);

        pending_ops.push_back(dwq);
        return threshold;
    }

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
                    boot.rank(), fi_strerror(-ret));
            exit(1);
        }
        pending_wb_ops.push_back(wb_op);
    }

    void exchange_addresses() {
        av_addrs.resize(boot.size());

        // Allgather libfabric addresses (raw bytes, no hex).
        auto all = boot.allgather(fabric->local_addr, (int)fabric->addrlen);

        for (int r = 0; r < boot.size(); r++) {
            if ((int)all[r].size() != (int)fabric->addrlen) {
                fprintf(stderr, "Rank %d: allgather(rank %d) size mismatch: "
                                "expected %zu, got %zu\n",
                        boot.rank(), r, fabric->addrlen, all[r].size());
                exit(1);
            }
            if (fi_av_insert(fabric->av, all[r].data(), 1, &av_addrs[r], 0, NULL) != 1) {
                fprintf(stderr, "Rank %d: fi_av_insert(rank %d) failed\n",
                        boot.rank(), r);
                exit(1);
            }
        }

        fabric->local_addr_in_av = av_addrs[boot.rank()];
    }

    void init_atomic_signaling() {
        // Allocate GPU memory for atomic signaling
        (void)hipMalloc(&atomic_result, sizeof(uint64_t));
        (void)hipMalloc(&atomic_operand, sizeof(uint64_t));

        // Initialize values
        uint64_t zero = 0;
        uint64_t one = 1;
        (void)hipMemcpy(atomic_result, &zero, sizeof(uint64_t), hipMemcpyHostToDevice);
        (void)hipMemcpy(atomic_operand, &one, sizeof(uint64_t), hipMemcpyHostToDevice);

        // Register as memory regions for RDMA
        mr_atomic_result = new MemoryRegion(fabric->domain, fabric->ep, fabric->cxi_info,
                                             atomic_result, sizeof(uint64_t), true, hip->gpu_id, boot.rank());
        mr_atomic_operand = new MemoryRegion(fabric->domain, fabric->ep, fabric->cxi_info,
                                              atomic_operand, sizeof(uint64_t), true, hip->gpu_id, boot.rank());

        // Create completion counter for atomic operations
        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;
        cntr_attr.wait_obj = FI_WAIT_UNSPEC;
        int ret = fi_cntr_open(fabric->domain, &cntr_attr, &atomic_completion_cntr, NULL);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_cntr_open(atomic) failed: %s\n",
                    boot.rank(), fi_strerror(-ret));
            exit(1);
        }
    }

    void init_barrier() {
        ofi_barrier = new OfiBarrier(fabric->domain, fabric->av, fabric->cxi_info,
                                      boot.rank(), boot.size());

        // Allgather barrier-EP addresses (raw bytes, no hex).
        auto all = boot.allgather(ofi_barrier->local_addr,
                                  (int)ofi_barrier->addrlen);

        for (int r = 0; r < boot.size(); r++) {
            if ((int)all[r].size() != (int)ofi_barrier->addrlen) {
                fprintf(stderr, "Rank %d: barrier allgather(rank %d) size mismatch\n",
                        boot.rank(), r);
                exit(1);
            }
            fi_addr_t peer_av_addr;
            if (fi_av_insert(fabric->av, all[r].data(), 1, &peer_av_addr, 0, NULL) != 1) {
                fprintf(stderr, "Rank %d: fi_av_insert(barrier %d) failed\n",
                        boot.rank(), r);
                exit(1);
            }
            ofi_barrier->set_peer_addr(r, peer_av_addr);
        }

        ofi_barrier->post_initial_recvs();
    }

    uint64_t make_remote_key(int rank, int buf_index) const {
        return ((uint64_t)rank << 48) | ((uint64_t)buf_index & 0xFFFFFFFFFFFF);
    }
};
