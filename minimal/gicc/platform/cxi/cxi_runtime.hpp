/**
 * cxi_runtime.hpp - libfabric/CXI implementation of gicc::Runtime
 *
 * Wraps the existing minimal/ gda::Runtime for setup (PMI bootstrap, fabric
 * init, MR registration, address exchange) but bypasses gda::Runtime::put /
 * prepare / reset to implement an OPTIMIZED batched-put path:
 *
 *   - put_no_db queues only the RDMA write (no per-op chained atomic).
 *   - prepare() queues a SINGLE chained atomic at the end of the batch
 *     whose threshold = total ops queued. The atomic fires once after every
 *     RMA in the batch has completed and increments atomic_result by 1.
 *   - The kernel polls atomic_result >= 1 instead of >= N.
 *
 * This eliminates the N-way serialization at the GPU memory atomic_result
 * cacheline that the per-op chained-atomic design suffers from on small
 * messages, while preserving the unified gicc:: API contract.
 *
 * Two patterns are supported, distinguished by which preparer is used:
 *
 *   1. Batched (kernel does flush + quiet):
 *        for (i) rt.put_no_db(...);
 *        auto* ctx = rt.prepare();
 *        kernel<<<>>>(ctx);          // gicc::flush(ctx); gicc::quiet(ctx);
 *        rt.reset();
 *
 *   2. Overlap (kernel only triggers, host waits):
 *        auto tok = rt.put_no_db(...);
 *        auto* tctx = rt.prepare_trigger(tok);
 *        trigger_kernel<<<>>>(tctx); // gicc::flush(tctx)
 *        compute_kernel<<<>>>(...);  // overlap
 *        hipDeviceSynchronize();
 *        rt.wait(tok);
 *
 * In pattern (1), prepare() queues the single batched atomic. In pattern (2),
 * no atomic is queued — the host polls the libfabric RMA completion counter
 * directly via wait(tok).
 */
#pragma once

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gicc/gicc_types.hpp"
#include "gicc/platform/cxi/cxi_device.cuh"

// opengda.hpp brings in gda::Runtime, GdaComm, DwqWorkBuilder, MemoryRegion
// and the libfabric headers for fi_cntr_read / fi_cntr_set.
#include "opengda.hpp"

namespace gicc {

static_assert(sizeof(DeviceCtx) == sizeof(gda::DeviceCtx),
              "gicc::DeviceCtx and gda::DeviceCtx must have identical layout");

/**
 * Token returned by put_no_db. Identifies a specific queued op by its
 * monotonically-increasing trigger threshold. Used by prepare_trigger() and
 * wait() for the overlap pattern. Pattern (1) callers can ignore it.
 */
struct Token {
    uint64_t threshold;
};

class Runtime {
public:
    explicit Runtime(MPI_Comm comm = MPI_COMM_WORLD)
        : impl_(), mpi_comm_(comm),
          h_dev_ctx_(nullptr), d_dev_ctx_(nullptr),
          my_threshold_(0), my_n_ops_(0)
    {
        (void)mpi_comm_;  // currently unused; PMI2 drives bootstrap inside GdaComm

        // Single device context (zero-copy pinned), reused by both prepare()
        // and prepare_trigger(). The trigger MMIO address is fixed at fabric
        // init; only completion_/trigger_val_/n_ops_ change per batch.
        (void)hipHostMalloc(&h_dev_ctx_, sizeof(DeviceCtx), hipHostMallocMapped);
        (void)hipHostGetDevicePointer((void**)&d_dev_ctx_, h_dev_ctx_, 0);
        h_dev_ctx_->trigger_addr_ = impl_.comm().get_trigger_addr();
        h_dev_ctx_->completion_   = nullptr;
        h_dev_ctx_->trigger_val_  = 0;
        h_dev_ctx_->n_ops_        = 0;
    }

    ~Runtime() {
        // Free any leftover queued work (mm-style continuous-threshold pattern
        // accumulates pending DwqWorkBuilders that reset() never collected).
        for (auto* op : my_pending_) delete op;
        my_pending_.clear();
        if (h_dev_ctx_) (void)hipHostFree(h_dev_ctx_);
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    //--------------------------------------------------------------------------
    // Buffer registration (delegates to gda::Runtime — uses GdaComm's MR
    // bookkeeping). The returned gicc::Buffer also stashes the underlying
    // gda::Buffer in gda_bufs_[index] so put_no_db can recover the local desc.
    //--------------------------------------------------------------------------
    Buffer register_buffer(void* buf, size_t size, bool is_device) {
        gda::Buffer gb = impl_.register_buffer(buf, size, is_device);

        Buffer b;
        b.ptr   = gb.ptr;
        b.size  = gb.size;
        b.addr  = (uint64_t)gb.ptr;
        // CXI: lkey/rkey have no direct ibv equivalent. We surface the
        // libfabric remote key as rkey, and use the buffer index as a
        // stand-in for lkey (the actual local descriptor lives in the gda
        // buffer table and is looked up by index in put_no_db).
        b.lkey  = (uint32_t)gb.index;
        b.rkey  = (uint32_t)(gb.key_ & 0xFFFFFFFFu);
        b.index = gb.index;

        if ((int)gda_bufs_.size() <= gb.index) gda_bufs_.resize(gb.index + 1);
        gda_bufs_[gb.index] = gb;
        return b;
    }

    void exchange() { impl_.exchange(); }

    RemoteBufferInfo remote_buffer(int rank, int buf_index) const {
        auto& c = const_cast<gda::Runtime&>(impl_).comm();
        auto ri = c.get_remote_info(rank, buf_index);
        RemoteBufferInfo r;
        r.addr = ri.rma_addr;
        r.rkey = (uint32_t)(ri.rma_key & 0xFFFFFFFFu);
        return r;
    }

    //--------------------------------------------------------------------------
    // put_no_db — queue an RMA WRITE only (no chained atomic). The atomic
    // for the whole batch is queued lazily by prepare(). For the overlap
    // pattern (prepare_trigger + host wait), no atomic is needed at all.
    //--------------------------------------------------------------------------
    Token put_no_db(const Buffer& src, int dest_rank, int dest_buf_index,
                    size_t size, size_t src_offset = 0, size_t dst_offset = 0)
    {
        const gda::Buffer& gb = gda_bufs_.at(src.index);
        auto& c = impl_.comm();

        my_threshold_++;
        my_n_ops_++;
        const uint64_t threshold = my_threshold_;

        GdaRemoteInfo ri = c.get_remote_info(dest_rank, dest_buf_index);
        if (ri.rma_key == 0 && ri.rma_addr == 0) {
            fprintf(stderr, "gicc::Runtime: remote info not set for rank %d "
                    "buf %d (call exchange() first)\n", dest_rank, dest_buf_index);
            exit(1);
        }

        const uint64_t remote_addr = c.is_virt_addr_mode()
            ? (ri.rma_addr + dst_offset)
            : (ri.rma_addr - ri.base_addr) + dst_offset;

        auto* dwq = new DwqWorkBuilder(c.rank());
        dwq->queue_rma_write(
            c.fabric->domain, c.fabric->ep,
            (char*)gb.ptr + src_offset, gb.desc_, size,
            c.av_addrs[dest_rank], remote_addr, ri.rma_key,
            c.fabric->trigger_cntr, c.fabric->completion_cntr,
            threshold);
        my_pending_.push_back(dwq);

        return Token{ threshold };
    }

    //--------------------------------------------------------------------------
    // prepare — finalize a batched put_no_db sequence by queuing ONE chained
    // atomic that fires after all queued RMAs complete, and return a
    // DeviceCtx the kernel will use for {flush; quiet}.
    //
    // n_ops_ is set to 1 (not the RMA count) because the kernel polls the
    // atomic_result counter, not per-op completion targets.
    //
    // The peer_rank / remote_buf_index parameters are accepted for source
    // compatibility with mlx5_runtime.hpp::prepare(int, int) but ignored:
    // the CXI batched DeviceCtx is global to the runtime.
    //--------------------------------------------------------------------------
    DeviceCtx* prepare(int peer_rank = -1, int remote_buf_index = -1) {
        (void)peer_rank;
        (void)remote_buf_index;

        auto& c = impl_.comm();

        // Reset atomic_result to 0 BEFORE queueing the new atomic.
        uint64_t zero = 0;
        (void)hipMemcpy(c.atomic_result, &zero, sizeof(uint64_t),
                        hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();

        // Queue a single batched atomic_signal. Triggers when the libfabric
        // completion_cntr reaches my_threshold_ (i.e. all RMAs have drained).
        if (my_n_ops_ > 0) {
            const uint64_t atomic_result_addr = c.is_virt_addr_mode()
                ? (uint64_t)c.atomic_result : 0;

            auto* dwq = new DwqWorkBuilder(c.rank());
            dwq->queue_atomic_signal(
                c.fabric->domain, c.fabric->ep,
                c.atomic_operand, c.mr_atomic_operand->desc,
                c.atomic_result, c.mr_atomic_result->key,
                atomic_result_addr,
                c.fabric->local_addr_in_av,
                c.fabric->completion_cntr,        // wait for all RMAs
                c.atomic_completion_cntr,         // signal channel
                my_threshold_);                   // fire after the LAST RMA
            my_pending_.push_back(dwq);
        }

        h_dev_ctx_->completion_  = c.atomic_result;
        h_dev_ctx_->trigger_val_ = my_threshold_;
        h_dev_ctx_->n_ops_       = 1;             // one batched atomic
        return d_dev_ctx_;
    }

    //--------------------------------------------------------------------------
    // prepare_trigger — overlap pattern. Returns a DeviceCtx whose flush()
    // writes exactly tok.threshold to the trigger MMIO. completion_ is
    // nulled and n_ops_=0 so a kernel that calls gicc::quiet(ctx) returns
    // immediately. The caller is expected to call rt.wait(tok) on the host
    // after compute completes.
    //--------------------------------------------------------------------------
    DeviceCtx* prepare_trigger(Token tok) {
        h_dev_ctx_->completion_  = nullptr;
        h_dev_ctx_->trigger_val_ = tok.threshold;
        h_dev_ctx_->n_ops_       = 0;
        return d_dev_ctx_;
    }

    //--------------------------------------------------------------------------
    // Host-side wait for a specific token: poll the libfabric RMA completion
    // counter. The background CQ progress thread inside FabricDwqContext
    // drives provider progress.
    //--------------------------------------------------------------------------
    void wait(Token tok) {
        auto& c = impl_.comm();
        while (fi_cntr_read(c.fabric->completion_cntr) < tok.threshold) {}
    }

    //--------------------------------------------------------------------------
    // reset — drain the current batch and zero the hardware counters.
    //--------------------------------------------------------------------------
    void reset() {
        auto& c = impl_.comm();
        // Wait for all RMAs (and the batched atomic, if prepare() queued one)
        // to drain on the libfabric side before freeing the work descriptors.
        while (fi_cntr_read(c.fabric->completion_cntr) < my_threshold_) {}

        for (auto* op : my_pending_) delete op;
        my_pending_.clear();

        fi_cntr_set(c.fabric->trigger_cntr, 0);
        fi_cntr_set(c.fabric->completion_cntr, 0);
        if (c.atomic_completion_cntr)
            fi_cntr_set(c.atomic_completion_cntr, 0);

        my_threshold_ = 0;
        my_n_ops_     = 0;
    }

    void barrier() { impl_.barrier(); }

    int rank()   const { return impl_.rank(); }
    int size()   const { return impl_.size(); }
    int gpu_id() const { return impl_.gpu_id(); }

    // Escape hatch for advanced users that need the underlying gda::Runtime.
    gda::Runtime& gda_runtime() { return impl_; }

private:
    gda::Runtime                  impl_;
    MPI_Comm                      mpi_comm_;
    DeviceCtx*                    h_dev_ctx_;   // pinned host (mapped)
    DeviceCtx*                    d_dev_ctx_;   // device pointer (zero-copy)

    uint64_t                      my_threshold_;  // monotonic trigger threshold
    uint64_t                      my_n_ops_;      // ops queued in current batch
    std::vector<DwqWorkBuilder*>  my_pending_;    // work builders awaiting reset

    std::vector<gda::Buffer>      gda_bufs_;
};

} // namespace gicc
