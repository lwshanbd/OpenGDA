/**
 * fabric.hpp - Low-level fabric primitives for GICC on libfabric/CXI
 *
 * Provides the fabric layer used directly by gicc::Runtime and by
 * subsystems (gicc::Barrier, gicc::am::Am). Offers:
 *   - put_raw(): queue a one-sided write to remote rank
 *   - trigger(): trigger queued DWQ operations from GPU
 *   - wait(): wait for pending operations
 *   - barrier(): global synchronization (host-side, via OfiBarrier)
 *
 * Simple usage:
 *   gicc::Fabric comm(boot);
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

#include <vector>
#include <unordered_map>

#include "gpu_device_context.hpp"
#include "gicc/bootstrap/bootstrap.hpp"
#include "device_affinity.hpp"
#include "fabric_dwq_context.hpp"
#include "memory_region.hpp"
#include "dwq_work_builder.hpp"
#include "ofi_barrier.hpp"

// GPU kernel to trigger DWQ operations.
// `static` (= internal linkage) so multiple TUs that include this header —
// notably the gicc-clang-plugin's generated sidecars and the user's own
// HIP TU — don't fight over the kernel symbol at link time.
static __global__ void gda_trigger_kernel(volatile uint64_t* trigger_addr,
                                          uint64_t threshold) {
    *trigger_addr = threshold;
}

namespace gicc {

// Handle to a registered memory region
struct Handle {
    void* buf;              // Local buffer pointer
    size_t size;            // Buffer size
    MemoryRegion* mr;       // Memory region (may be null for raw handles)
    void* local_desc;       // Local descriptor for DWQ operations
    uint64_t rma_addr;      // RMA address
    uint64_t rma_key;       // RMA key
};

// Remote RMA info for a specific buffer
struct RemoteInfo {
    fi_addr_t av_addr;
    uint64_t rma_addr;      // Full address (base + offset) for virt_addr mode
    uint64_t rma_key;
    uint64_t base_addr;     // MR base address (for computing offset in non-virt_addr mode)
};

class Fabric {
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
    std::unordered_map<uint64_t, RemoteInfo> remote_info;

    // Registered memory regions (for cleanup)
    std::vector<MemoryRegion*> registered_mrs;

    // CPU proxy fleet: per-thread fi_endpoint + fi_cq, sharing the main AV.
    // create_proxy_endpoints(N) populates these; ProxyLibfabric instances
    // pick proxy_eps_[i] / proxy_cqs_[i] by ep_idx. Empty when N=0 (only
    // the DWQ-triggered path is in use).
    //
    // The AV is shared with the main EP: peer addresses are inserted once
    // (in exchange_addresses()) and the resulting fi_addr_t is valid as a
    // destination from any local EP bound to the AV — including these
    // proxy EPs. So the proxy fleet does NOT need its own per-EP AV nor a
    // separate address allgather. Peers only ever target our main EP via
    // the address they learned from exchange(); our proxy EPs are TX-only.
    std::vector<fid_ep*> proxy_eps_;
    std::vector<fid_cq*> proxy_cqs_;

    // Per-(buf_idx, ep_idx) MR pointers for the proxy fleet's local
    // descriptors. Each proxy EP needs its own MR registration of every
    // user buffer because CXI rejects fi_mr_bind to a second EP. The MR
    // objects themselves are owned by registered_mrs; this vector just
    // provides the [buf][ep] indexing that proxy_buf_desc() uses.
    std::vector<std::vector<MemoryRegion*>> proxy_mrs_by_buf_;

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
     * Initialize the OFI fabric layer
     * @param boot_ Bootstrap instance providing rank, size, and collective ops
     * @param local_rank Local rank for GPU selection (e.g., SLURM_LOCALID)
     */
    explicit Fabric(Bootstrap& boot_, int local_rank = -1)
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
                (void)gpuGetDeviceCount(&num_devices);
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

    ~Fabric() {
        // Cleanup pending operations
        for (auto* op : pending_ops) delete op;
        pending_ops.clear();

        // Cleanup writeback ops
        for (auto* wb : pending_wb_ops) delete wb;
        pending_wb_ops.clear();

        // Cleanup registered memory regions before tearing down EPs (MRs
        // hold bind references to those EPs).
        for (auto* mr : registered_mrs) delete mr;
        registered_mrs.clear();

        // Cleanup atomic signaling resources
        delete mr_atomic_result;
        delete mr_atomic_operand;
        if (atomic_completion_cntr) fi_close(&atomic_completion_cntr->fid);
        if (atomic_result) (void)gpuFree(atomic_result);
        if (atomic_operand) (void)gpuFree(atomic_operand);

        // Close proxy EPs + CQs before the FabricDwqContext (which owns the
        // shared AV/domain). Order: EP first, then CQ — fi_close on EP can
        // touch its bound CQ otherwise.
        for (auto* ep : proxy_eps_) {
            if (ep) fi_close(&ep->fid);
        }
        proxy_eps_.clear();
        for (auto* cq : proxy_cqs_) {
            if (cq) fi_close(&cq->fid);
        }
        proxy_cqs_.clear();

        // Cleanup components (reverse order)
        delete ofi_barrier;
        delete fabric;
        delete hip;
        delete affinity;
    }

    // No copy
    Fabric(const Fabric&) = delete;
    Fabric& operator=(const Fabric&) = delete;

    /**
     * Open N additional fi_endpoints (each with its own fi_cq) on the same
     * domain, sharing the main AV. Idempotent: calling with the same N is
     * a no-op; calling with a different N after the first call aborts —
     * proxy_eps_ are bound by subsequent register_buffer() calls and
     * cannot be added/removed after MRs are enabled.
     *
     * Must be called BEFORE the first register_buffer() so that MRs can
     * fi_mr_bind to every proxy EP. Runtime ctor takes care of this when
     * GICC_NUM_PROXY_THREADS > 0.
     */
    void create_proxy_endpoints(int n) {
        if (n <= 0) return;
        if (!proxy_eps_.empty()) {
            if ((int)proxy_eps_.size() != n) {
                fprintf(stderr,
                    "Rank %d: create_proxy_endpoints called twice with "
                    "different N (%zu vs %d) — proxy EPs cannot be resized "
                    "after the first MR is registered against them.\n",
                    boot.rank(), proxy_eps_.size(), n);
                exit(1);
            }
            return;
        }

        proxy_eps_.reserve(n);
        proxy_cqs_.reserve(n);

        // Match the main EP's CQ size policy (GICC_CQ_SIZE), default 128.
        struct fi_cq_attr cq_attr = {};
        cq_attr.size   = 128;
        if (const char* env = std::getenv("GICC_CQ_SIZE")) {
            int sz = std::atoi(env);
            if (sz >= 16 && sz <= 16384) cq_attr.size = sz;
        }
        cq_attr.format = FI_CQ_FORMAT_CONTEXT;

        for (int i = 0; i < n; ++i) {
            fid_cq* cq = nullptr;
            int ret = fi_cq_open(fabric->domain, &cq_attr, &cq, NULL);
            if (ret) {
                fprintf(stderr,
                    "Rank %d: fi_cq_open(proxy_cq[%d]) failed: %s (%d)\n",
                    boot.rank(), i, fi_strerror(-ret), ret);
                exit(1);
            }

            fid_ep* ep = nullptr;
            ret = fi_endpoint(fabric->domain, fabric->cxi_info, &ep, NULL);
            if (ret) {
                fi_close(&cq->fid);
                fprintf(stderr,
                    "Rank %d: fi_endpoint(proxy_ep[%d]) failed: %s (%d)\n",
                    boot.rank(), i, fi_strerror(-ret), ret);
                exit(1);
            }
            // Share the main AV so peer fi_addr_t entries inserted by
            // exchange_addresses() are usable from this EP too.
            ret = fi_ep_bind(ep, &fabric->av->fid, 0);
            if (ret) {
                fi_close(&ep->fid);
                fi_close(&cq->fid);
                fprintf(stderr,
                    "Rank %d: fi_ep_bind(proxy_ep[%d], av) failed: %s (%d)\n",
                    boot.rank(), i, fi_strerror(-ret), ret);
                exit(1);
            }
            // Bind the same CQ for both TX and RX. The proxy never posts
            // recvs (peers only target the main EP — see exchange_addresses
            // — so no traffic arrives at proxy EPs), but the CXI provider
            // refuses fi_enable on an EP that has no RX CQ bound (it
            // returns FI_ENOCQ). Sharing the TX CQ for RX is harmless: an
            // empty RX path produces no completions to drain.
            ret = fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
            if (ret) {
                fi_close(&ep->fid);
                fi_close(&cq->fid);
                fprintf(stderr,
                    "Rank %d: fi_ep_bind(proxy_ep[%d], cq, "
                    "FI_TRANSMIT|FI_RECV) failed: %s (%d)\n",
                    boot.rank(), i, fi_strerror(-ret), ret);
                exit(1);
            }
            ret = fi_enable(ep);
            if (ret) {
                fi_close(&ep->fid);
                fi_close(&cq->fid);
                fprintf(stderr,
                    "Rank %d: fi_enable(proxy_ep[%d]) failed: %s (%d)\n",
                    boot.rank(), i, fi_strerror(-ret), ret);
                exit(1);
            }
            proxy_eps_.push_back(ep);
            proxy_cqs_.push_back(cq);
        }
    }

    int      num_proxy_eps() const { return (int)proxy_eps_.size(); }
    fid_ep*  proxy_ep(int i)       { return proxy_eps_.at(i); }
    fid_cq*  proxy_cq(int i)       { return proxy_cqs_.at(i); }

    /**
     * Register a buffer for RDMA operations
     * @param buf Buffer pointer
     * @param size Buffer size in bytes
     * @param is_device True if buffer is on GPU
     * @return Handle for use in put/get operations
     *
     * Always creates ONE MR bound to the main EP (its rkey is what peers
     * learn via exchange()). Additionally, when the proxy fleet is open
     * (N = num_proxy_eps() > 0), creates N more MRs — one per proxy EP —
     * to provide the per-EP local descriptor each proxy thread needs for
     * fi_write. Those extra MRs are LOCAL-ONLY: their rkeys are never
     * advertised to peers. Peers always target our main EP using the main
     * MR's rkey, regardless of which local proxy EP submits the write.
     *
     * Indexing: per-buffer descs[ep_idx] is exposed via proxy_buf_desc().
     * Buffer index is the order register_buffer() is called.
     */
    Handle register_buffer(void* buf, size_t size, bool is_device) {
        auto* mr = new MemoryRegion(
            fabric->domain, fabric->ep, fabric->cxi_info,
            buf, size, is_device, hip->gpu_id, boot.rank());
        registered_mrs.push_back(mr);

        // Per-proxy-EP MRs. Each is a separate fi_mr_regattr call binding
        // the SAME local buffer to one proxy EP (CXI rejects multi-bind
        // with -EINVAL, so a single MR can't span multiple EPs). Each MR's
        // desc is what proxy_ep[i] needs as the source descriptor for
        // fi_write; the rkey is unused (peers don't address these MRs).
        std::vector<MemoryRegion*> per_ep;
        per_ep.reserve(proxy_eps_.size());
        for (size_t i = 0; i < proxy_eps_.size(); ++i) {
            auto* pmr = new MemoryRegion(
                fabric->domain, proxy_eps_[i], fabric->cxi_info,
                buf, size, is_device, hip->gpu_id, boot.rank());
            registered_mrs.push_back(pmr);   // for cleanup ownership
            per_ep.push_back(pmr);
        }
        proxy_mrs_by_buf_.push_back(std::move(per_ep));

        Handle handle;
        handle.buf = buf;
        handle.size = size;
        handle.mr = mr;
        handle.local_desc = mr->desc;
        handle.rma_addr = (uint64_t)buf;
        handle.rma_key = mr->key;
        return handle;
    }

    /**
     * Per-(buf_idx, ep_idx) local descriptor for fi_write on proxy_ep[ep_idx].
     * buf_idx is the order register_buffer() was called (matches
     * Runtime::local_bufs_ index). Aborts if either index is out of range.
     */
    void* proxy_buf_desc(int buf_idx, int ep_idx) const {
        if (buf_idx < 0 || (size_t)buf_idx >= proxy_mrs_by_buf_.size()) {
            fprintf(stderr,
                "Fabric::proxy_buf_desc: buf_idx %d out of range (n_bufs=%zu)\n",
                buf_idx, proxy_mrs_by_buf_.size());
            std::abort();
        }
        const auto& per_ep = proxy_mrs_by_buf_[(size_t)buf_idx];
        if (ep_idx < 0 || (size_t)ep_idx >= per_ep.size()) {
            fprintf(stderr,
                "Fabric::proxy_buf_desc: ep_idx %d out of range (n_eps=%zu)\n",
                ep_idx, per_ep.size());
            std::abort();
        }
        return per_ep[(size_t)ep_idx]->desc;
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
    uint64_t put(const Handle& src_handle, int dest_rank,
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
    uint64_t put_raw(const Handle& src_handle, int dest_rank,
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
        gpuLaunchKernel(gda_trigger_kernel, dim3(1), dim3(1), 0, 0,
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
    RemoteInfo get_remote_info(int dest_rank, int buf_index) const {
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
        (void)gpuMemcpy(atomic_result, &zero, sizeof(uint64_t), gpuMemcpyHostToDevice);
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
    uint64_t put_with_signal(const Handle& src_handle, int dest_rank,
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
        (void)gpuMalloc(&atomic_result, sizeof(uint64_t));
        (void)gpuMalloc(&atomic_operand, sizeof(uint64_t));

        // Initialize values
        uint64_t zero = 0;
        uint64_t one = 1;
        (void)gpuMemcpy(atomic_result, &zero, sizeof(uint64_t), gpuMemcpyHostToDevice);
        (void)gpuMemcpy(atomic_operand, &one, sizeof(uint64_t), gpuMemcpyHostToDevice);

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

} // namespace gicc
