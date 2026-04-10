/**
 * cxi_runtime.hpp - libfabric/CXI implementation of gicc::Runtime
 *
 * Wraps the existing minimal/ gda::Runtime for setup (PMI bootstrap, fabric
 * init, MR registration, address exchange) but bypasses gda::Runtime::put /
 * prepare / reset to implement a per-stream completion+atomic pool that
 * mirrors the proven-correct benchmark_runner.hpp design while keeping the
 * unified gicc:: API contract.
 *
 *   - put_no_db queues only the RDMA write. Each op consumes one slot from
 *     a pre-allocated pool of N libfabric completion counters. The op's
 *     trigger threshold is its 1-based index within the current batch (so
 *     the kernel only needs to write `n_ops` to the shared trigger MMIO to
 *     fire all queued ops in one shot, exactly as benchmark_runner does).
 *   - prepare() queues a chained atomic_signal per op targeting that op's
 *     own GPU-resident atomic_result slot. The slot pool is reset to 0
 *     before kernel launch. The kernel polls all n_ops slots in parallel.
 *   - prepare_trigger(Token) (overlap pattern) skips queueing atomics and
 *     leaves completion_=nullptr; the host calls wait(Token) which polls
 *     the per-op completion counter directly.
 *   - reset() drains every used slot's RMA and atomic counters, frees the
 *     batch's DwqWorkBuilders, fi_cntr_sets per-slot counters AND the
 *     shared trigger counter to zero, and recycles the slots.
 *
 * The key insight matching baseline is that NO shared completion counter is
 * ever fi_cntr_set across batches with stale per-threshold deferred-work
 * metadata. Each slot's counter is independent, the trigger counter is
 * shared but always reset cleanly, and per-stream resets are confirmed safe
 * by benchmark_runner.hpp on this exact provider.
 */
#pragma once

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gicc/gicc_types.hpp"
#include "gicc/platform/cxi/cxi_device.cuh"

#include "opengda.hpp"

namespace gicc {

static_assert(sizeof(DeviceCtx) == sizeof(gda::DeviceCtx),
              "gicc::DeviceCtx and gda::DeviceCtx must have identical layout");

/**
 * Token returned by put_no_db. Identifies a specific queued op by its slot
 * index in the per-runtime completion-counter pool. wait(Token) polls that
 * slot's completion counter on the host. The simple "queue many → prepare()
 * → kernel does flush+quiet" pattern can ignore the return value.
 */
struct Token {
    int slot_idx;
};

class Runtime {
public:
    static constexpr int POOL_SIZE = 32;   // max ops per batch

    explicit Runtime(MPI_Comm comm = MPI_COMM_WORLD)
        : impl_(), mpi_comm_(comm),
          h_dev_ctx_(nullptr), d_dev_ctx_(nullptr),
          d_slot_pool_(nullptr), mr_slot_pool_(nullptr),
          d_operand_pool_(nullptr), mr_operand_pool_(nullptr),
          my_n_ops_(0), atomic_signals_queued_(false)
    {
        (void)mpi_comm_;
        auto& c = impl_.comm();

        // ONE shared GPU buffer holds POOL_SIZE × uint64_t atomic_result
        // slots, registered with ONE MemoryRegion. The chained atomic for
        // slot i targets offset i*8 within this MR (in non-virt mode) or
        // the absolute address of slot i (in virt mode).
        const size_t POOL_BYTES = POOL_SIZE * sizeof(uint64_t);
        if (hipMalloc(&d_slot_pool_, POOL_BYTES) != hipSuccess) {
            fprintf(stderr, "hipMalloc(slot pool) failed\n"); exit(1);
        }
        (void)hipMemset(d_slot_pool_, 0, POOL_BYTES);
        mr_slot_pool_ = new MemoryRegion(
            c.fabric->domain, c.fabric->ep, c.fabric->cxi_info,
            d_slot_pool_, POOL_BYTES, true, c.gpu_id(), c.rank());

        // Single shared atomic operand (value 1) and its MR. Reused by
        // every chained atomic_signal. The provider only reads from it.
        if (hipMalloc(&d_operand_pool_, sizeof(uint64_t)) != hipSuccess) {
            fprintf(stderr, "hipMalloc(operand) failed\n"); exit(1);
        }
        const uint64_t one = 1;
        (void)hipMemcpy(d_operand_pool_, &one, sizeof(uint64_t),
                        hipMemcpyHostToDevice);
        mr_operand_pool_ = new MemoryRegion(
            c.fabric->domain, c.fabric->ep, c.fabric->cxi_info,
            d_operand_pool_, sizeof(uint64_t), true, c.gpu_id(), c.rank());

        // Per-slot libfabric counters (no GPU buffers — those live in
        // d_slot_pool_ via offset).
        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;
        for (int i = 0; i < POOL_SIZE; i++) {
            int ret = fi_cntr_open(c.fabric->domain, &cntr_attr,
                                    &slots_[i].completion_cntr, NULL);
            if (ret) { fprintf(stderr, "fi_cntr_open(%d c) failed\n", i); exit(1); }
            ret = fi_cntr_open(c.fabric->domain, &cntr_attr,
                                &slots_[i].atomic_completion_cntr, NULL);
            if (ret) { fprintf(stderr, "fi_cntr_open(%d a) failed\n", i); exit(1); }
        }
        (void)hipDeviceSynchronize();

        (void)hipHostMalloc(&h_dev_ctx_, sizeof(DeviceCtx), hipHostMallocMapped);
        (void)hipHostGetDevicePointer((void**)&d_dev_ctx_, h_dev_ctx_, 0);
        h_dev_ctx_->trigger_addr_ = c.get_trigger_addr();
        h_dev_ctx_->completion_   = nullptr;
        h_dev_ctx_->trigger_val_  = 0;
        h_dev_ctx_->n_ops_        = 0;
    }

    ~Runtime() {
        for (auto* op : my_pending_) delete op;
        my_pending_.clear();
        for (int i = 0; i < POOL_SIZE; i++) {
            if (slots_[i].completion_cntr)
                fi_close(&slots_[i].completion_cntr->fid);
            if (slots_[i].atomic_completion_cntr)
                fi_close(&slots_[i].atomic_completion_cntr->fid);
        }
        delete mr_slot_pool_;
        delete mr_operand_pool_;
        if (d_slot_pool_)    (void)hipFree(d_slot_pool_);
        if (d_operand_pool_) (void)hipFree(d_operand_pool_);
        if (h_dev_ctx_)      (void)hipHostFree(h_dev_ctx_);
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    //--------------------------------------------------------------------------
    // Buffer registration (delegates to gda::Runtime)
    //--------------------------------------------------------------------------
    Buffer register_buffer(void* buf, size_t size, bool is_device) {
        gda::Buffer gb = impl_.register_buffer(buf, size, is_device);

        Buffer b;
        b.ptr   = gb.ptr;
        b.size  = gb.size;
        b.addr  = (uint64_t)gb.ptr;
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
    // put_no_db — queue an RMA WRITE only. Consumes one slot from the pool.
    // The slot's per-stream libfabric completion counter is the RMA's target
    // (no shared completion_cntr → no cache staleness across batches).
    //--------------------------------------------------------------------------
    Token put_no_db(const Buffer& src, int dest_rank, int dest_buf_index,
                    size_t size, size_t src_offset = 0, size_t dst_offset = 0)
    {
        const gda::Buffer& gb = gda_bufs_.at(src.index);
        auto& c = impl_.comm();

        if ((int)my_n_ops_ >= POOL_SIZE) {
            fprintf(stderr,
                "gicc::Runtime::put_no_db: batch exceeds POOL_SIZE=%d. "
                "Call rt.reset() between batches or raise POOL_SIZE.\n",
                POOL_SIZE);
            exit(1);
        }

        const int      slot_idx           = (int)my_n_ops_;
        const uint64_t trigger_threshold  = my_n_ops_ + 1;  // 1-based
        my_n_ops_++;

        GdaRemoteInfo ri = c.get_remote_info(dest_rank, dest_buf_index);
        if (ri.rma_key == 0 && ri.rma_addr == 0) {
            fprintf(stderr, "gicc::Runtime: remote info not set for rank %d "
                    "buf %d (call exchange() first)\n", dest_rank, dest_buf_index);
            exit(1);
        }
        const uint64_t remote_addr = c.is_virt_addr_mode()
            ? (ri.rma_addr + dst_offset)
            : (ri.rma_addr - ri.base_addr) + dst_offset;

        // EXACTLY mirror benchmark_runner.hpp's queueing order: each
        // DwqWorkBuilder holds both the RMA and the chained atomic_signal,
        // and the two fi_control(FI_QUEUE_WORK) calls happen back-to-back
        // for stream i before stream i+1. The CXI provider appears to have
        // an ordering constraint that breaks if atomics for streams 0..N-1
        // are queued AFTER all RMAs are queued.
        auto* dwq = new DwqWorkBuilder(c.rank());
        dwq->queue_rma_write(
            c.fabric->domain, c.fabric->ep,
            (char*)gb.ptr + src_offset, gb.desc_, size,
            c.av_addrs[dest_rank], remote_addr, ri.rma_key,
            c.fabric->trigger_cntr,                 // shared trigger
            slots_[slot_idx].completion_cntr,       // per-stream completion
            trigger_threshold);

        // Chained atomic_signal targeting slot_idx's offset in the shared
        // pool MR. The shared operand MR provides the value-to-add (=1).
        uint64_t* slot_addr = (uint64_t*)d_slot_pool_ + slot_idx;
        const uint64_t result_addr = c.is_virt_addr_mode()
            ? (uint64_t)slot_addr : ((uint64_t)slot_idx * sizeof(uint64_t));
        dwq->queue_atomic_signal(
            c.fabric->domain, c.fabric->ep,
            d_operand_pool_, mr_operand_pool_->desc,
            slot_addr, mr_slot_pool_->key,
            result_addr,
            c.fabric->local_addr_in_av,
            slots_[slot_idx].completion_cntr,
            slots_[slot_idx].atomic_completion_cntr,
            1);

        my_pending_.push_back(dwq);
        atomic_signals_queued_ = true;

        return Token{ slot_idx };
    }

    //--------------------------------------------------------------------------
    // prepare — finalize a batched put_no_db sequence. The chained atomics
    // were already queued by put_no_db (one per call). We just reset the
    // slot pool to 0 and configure the DeviceCtx for the kernel to poll.
    //--------------------------------------------------------------------------
    DeviceCtx* prepare(int peer_rank = -1, int remote_buf_index = -1) {
        (void)peer_rank;
        (void)remote_buf_index;
        auto& c = impl_.comm();

        // Reset just the used slots in the shared pool to 0 with one
        // hipMemset of n_ops_ × 8 bytes — far cheaper than n_ops_
        // separate hipMemcpy(0) calls.
        if (my_n_ops_ > 0) {
            (void)hipMemset(d_slot_pool_, 0, my_n_ops_ * sizeof(uint64_t));
            (void)hipDeviceSynchronize();
        }

        h_dev_ctx_->trigger_addr_ = c.get_trigger_addr();
        h_dev_ctx_->trigger_val_  = my_n_ops_;
        // completion_ is the base of the contiguous slot pool. The
        // device-side gicc::quiet polls completion_[i] for i in [0, n_ops_).
        h_dev_ctx_->completion_   = (volatile uint64_t*)d_slot_pool_;
        h_dev_ctx_->n_ops_        = my_n_ops_;
        return d_dev_ctx_;
    }

    //--------------------------------------------------------------------------
    // prepare_trigger — overlap pattern. Returns a DeviceCtx whose flush()
    // fires all currently queued put_no_db ops. completion_ is nulled and
    // n_ops_=0 so a kernel that calls gicc::quiet(ctx) returns immediately.
    // The host calls wait(Token) afterwards to drain the per-op counter.
    //
    // The Token argument is accepted for API symmetry with prepare(Token);
    // the trigger value is computed from the current my_n_ops_ accumulator.
    //--------------------------------------------------------------------------
    DeviceCtx* prepare_trigger(Token /*tok*/) {
        auto& c = impl_.comm();
        h_dev_ctx_->trigger_addr_ = c.get_trigger_addr();
        h_dev_ctx_->trigger_val_  = my_n_ops_;
        h_dev_ctx_->completion_   = nullptr;
        h_dev_ctx_->n_ops_        = 0;
        return d_dev_ctx_;
    }

    //--------------------------------------------------------------------------
    // Host-side wait for a specific token (poll its per-stream completion
    // counter). The background CQ progress thread inside FabricDwqContext
    // drives provider progress.
    //--------------------------------------------------------------------------
    void wait(Token tok) {
        while (fi_cntr_read(slots_[tok.slot_idx].completion_cntr) < 1) {}
    }

    //--------------------------------------------------------------------------
    // reset — drain the current batch and recycle the slots.
    //--------------------------------------------------------------------------
    void reset() {
        auto& c = impl_.comm();

        // Drain per-slot RMA completions. Drive progress synchronously
        // with fi_cq_read in the poll loop, exactly like
        // benchmark_runner.hpp lines 507-518. Relying solely on the
        // background CQ progress thread is NOT sufficient — there is a
        // small window where the host returns from polling before the
        // provider has actually applied incoming completion events.
        for (uint64_t i = 0; i < my_n_ops_; i++) {
            while (fi_cntr_read(slots_[i].completion_cntr) < 1) {
                fi_cq_read(c.fabric->cq, NULL, 0);
            }
        }
        if (atomic_signals_queued_) {
            for (uint64_t i = 0; i < my_n_ops_; i++) {
                while (fi_cntr_read(slots_[i].atomic_completion_cntr) < 1) {
                    fi_cq_read(c.fabric->cq, NULL, 0);
                }
            }
        }

        // Now safe to free the DwqWorkBuilders.
        for (auto* op : my_pending_) delete op;
        my_pending_.clear();

        // Reset the SHARED trigger counter and per-slot counters.
        // Per-slot resets are confirmed safe by benchmark_runner.hpp on
        // this CXI provider — the cache-staleness bug is specific to a
        // shared completion counter being reset across batches.
        fi_cntr_set(c.fabric->trigger_cntr, 0);
        for (uint64_t i = 0; i < my_n_ops_; i++) {
            fi_cntr_set(slots_[i].completion_cntr, 0);
            if (atomic_signals_queued_)
                fi_cntr_set(slots_[i].atomic_completion_cntr, 0);
        }

        my_n_ops_              = 0;
        atomic_signals_queued_ = false;
    }

    void barrier() { impl_.barrier(); }

    int rank()   const { return impl_.rank(); }
    int size()   const { return impl_.size(); }
    int gpu_id() const { return impl_.gpu_id(); }

    gda::Runtime& gda_runtime() { return impl_; }

private:
    struct Slot {
        struct fid_cntr* completion_cntr        = nullptr;
        struct fid_cntr* atomic_completion_cntr = nullptr;
    };

    gda::Runtime                  impl_;
    MPI_Comm                      mpi_comm_;
    DeviceCtx*                    h_dev_ctx_;
    DeviceCtx*                    d_dev_ctx_;

    Slot                          slots_[POOL_SIZE];

    // Single shared GPU pool of POOL_SIZE × uint64_t atomic_result slots.
    void*                         d_slot_pool_;
    MemoryRegion*                 mr_slot_pool_;
    void*                         d_operand_pool_;
    MemoryRegion*                 mr_operand_pool_;

    uint64_t                      my_n_ops_;
    bool                          atomic_signals_queued_;
    std::vector<DwqWorkBuilder*>  my_pending_;

    std::vector<gda::Buffer>      gda_bufs_;
};

} // namespace gicc
