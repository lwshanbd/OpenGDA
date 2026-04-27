/**
 * ofi_runtime.hpp - libfabric (CXI/OFI) implementation of gicc::Runtime
 *
 * Directly owns a Fabric for fabric setup (PMI bootstrap, MR registration,
 * address exchange) and implements a per-stream completion+atomic pool that
 * mirrors the proven-correct benchmark_runner.hpp design.
 *
 * No gda:: namespace types are used — this is a self-contained gicc:: backend.
 *
 *   - put_no_db queues the RDMA write. Each op consumes one slot from a
 *     pre-allocated pool of N libfabric completion counters. The op's
 *     trigger threshold is its 1-based index within the current batch.
 *   - prepare() resets the GPU slot pool and configures DeviceCtx for the
 *     kernel to poll all n_ops slots via gicc::quiet.
 *   - prepare_trigger(Token) (overlap pattern) sets up flush-only DeviceCtx;
 *     the host calls wait(Token) to poll the per-op counter directly.
 *   - reset() drains every used slot's RMA and atomic counters, frees the
 *     batch's DwqWorkBuilders, resets libfabric counters to zero, and
 *     recycles the slots.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gicc/gicc_types.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

// OFI backend internals (Fabric, FabricDwqContext, MemoryRegion, etc.)
#include "internal/hip_device_context.hpp"
#include "gicc/bootstrap/bootstrap.hpp"
#include "internal/device_affinity.hpp"
#include "internal/fabric_dwq_context.hpp"
#include "internal/memory_region.hpp"
#include "internal/dwq_work_builder.hpp"
#include "internal/ofi_barrier.hpp"
#include "internal/fabric.hpp"

namespace gicc {

/**
 * Token returned by put_no_db. Identifies a specific queued op by its slot
 * index in the per-runtime completion-counter pool. wait(Token) polls that
 * slot's completion counter on the host.
 */
struct Token {
    int  slot_idx;
    bool is_local = false;
};

class Runtime {
public:
    static constexpr int POOL_SIZE = 32;   // max ops per batch

    Runtime()
        : comm_(nullptr),
          h_dev_ctx_(nullptr), d_dev_ctx_(nullptr),
          d_slot_pool_(nullptr), mr_slot_pool_(nullptr),
          d_operand_pool_(nullptr), mr_operand_pool_(nullptr),
          my_n_ops_(0), my_n_remote_ops_(0), my_n_local_ops_(0),
          atomic_signals_queued_(false),
          h_local_ops_(nullptr), d_local_ops_(nullptr)
    {
        unset_rocr_visible_devices();
        comm_ = new Fabric(boot_);

        // Locality map from Bootstrap: [rank] = true iff rank shares our node.
        // put_no_db() routes same-node writes through HIP IPC — the copy is
        // done INSIDE the user's kernel by gicc::put_local(), using
        // IPC-mapped peer pointers. Remote peers keep the DWQ/CXI path.
        local_peer_ = boot_.locality_map();
        peer_mapped_ptrs_.assign(boot_.size(), {});

        // Host-mapped array of local-op descriptors the kernel reads.
        (void)hipHostMalloc(&h_local_ops_, POOL_SIZE * sizeof(LocalOp),
                            hipHostMallocMapped);
        (void)hipHostGetDevicePointer((void**)&d_local_ops_, h_local_ops_, 0);

        // ONE shared GPU buffer holds POOL_SIZE × uint64_t atomic_result
        // slots, registered with ONE MemoryRegion.
        const size_t POOL_BYTES = POOL_SIZE * sizeof(uint64_t);
        if (hipMalloc(&d_slot_pool_, POOL_BYTES) != hipSuccess) {
            fprintf(stderr, "hipMalloc(slot pool) failed\n"); exit(1);
        }
        (void)hipMemset(d_slot_pool_, 0, POOL_BYTES);
        mr_slot_pool_ = new MemoryRegion(
            comm_->fabric->domain, comm_->fabric->ep, comm_->fabric->cxi_info,
            d_slot_pool_, POOL_BYTES, true, comm_->gpu_id(), comm_->rank());

        // Single shared atomic operand (value 1) and its MR.
        if (hipMalloc(&d_operand_pool_, sizeof(uint64_t)) != hipSuccess) {
            fprintf(stderr, "hipMalloc(operand) failed\n"); exit(1);
        }
        const uint64_t one = 1;
        (void)hipMemcpy(d_operand_pool_, &one, sizeof(uint64_t),
                        hipMemcpyHostToDevice);
        mr_operand_pool_ = new MemoryRegion(
            comm_->fabric->domain, comm_->fabric->ep, comm_->fabric->cxi_info,
            d_operand_pool_, sizeof(uint64_t), true, comm_->gpu_id(), comm_->rank());

        // Per-slot libfabric counters.
        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;
        for (int i = 0; i < POOL_SIZE; i++) {
            int ret = fi_cntr_open(comm_->fabric->domain, &cntr_attr,
                                    &slots_[i].completion_cntr, NULL);
            if (ret) { fprintf(stderr, "fi_cntr_open(%d c) failed\n", i); exit(1); }
            ret = fi_cntr_open(comm_->fabric->domain, &cntr_attr,
                                &slots_[i].atomic_completion_cntr, NULL);
            if (ret) { fprintf(stderr, "fi_cntr_open(%d a) failed\n", i); exit(1); }
        }
        (void)hipDeviceSynchronize();

        (void)hipHostMalloc(&h_dev_ctx_, sizeof(DeviceCtx), hipHostMallocMapped);
        (void)hipHostGetDevicePointer((void**)&d_dev_ctx_, h_dev_ctx_, 0);
        h_dev_ctx_->trigger_addr_      = comm_->get_trigger_addr();
        h_dev_ctx_->completion_        = nullptr;
        h_dev_ctx_->trigger_val_       = 0;
        h_dev_ctx_->n_ops_             = 0;
        h_dev_ctx_->ipc_map_           = nullptr;
        h_dev_ctx_->max_bufs_per_rank_ = 0;
        h_dev_ctx_->local_ops_         = nullptr;
        h_dev_ctx_->n_local_ops_       = 0;
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

        // Close IPC mapped pointers (one per local peer × buffer).
        for (auto& per_rank : peer_mapped_ptrs_) {
            for (void* p : per_rank) {
                if (p) (void)hipIpcCloseMemHandle(p);
            }
        }
        peer_mapped_ptrs_.clear();

        if (h_local_ops_) (void)hipHostFree(h_local_ops_);

        if (d_ipc_map_) (void)hipFree(d_ipc_map_);

        delete comm_;
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    //--------------------------------------------------------------------------
    // Buffer registration — delegates to Fabric, caches local metadata.
    //--------------------------------------------------------------------------
    Buffer register_buffer(void* buf, size_t size, bool is_device) {
        Handle h = comm_->register_buffer(buf, size, is_device);
        int idx = (int)local_bufs_.size();

        OfiBuffer ob;
        ob.ptr   = buf;
        ob.desc_ = h.local_desc;
        ob.key_  = h.rma_key;
        ob.addr_ = h.rma_addr;

        // Capture an IPC handle for device buffers so same-node peers can
        // open them in exchange(). hipMalloc'd pointers are always valid
        // here; for non-device buffers IPC isn't meaningful.
        if (is_device) {
            if (hipIpcGetMemHandle(&ob.ipc_handle, buf) == hipSuccess) {
                ob.has_ipc_handle = true;
            }
        }
        local_bufs_.push_back(ob);

        Buffer b;
        b.ptr   = buf;
        b.size  = size;
        b.addr  = (uint64_t)buf;
        b.lkey  = (uint32_t)idx;
        b.rkey  = (uint32_t)(h.rma_key & 0xFFFFFFFFu);
        b.index = idx;
        // Index in buffers_ matches b.index (== lkey on this backend).
        buffers_.push_back(b);
        return b;
    }

    //--------------------------------------------------------------------------
    // Collective exchange of registered buffer metadata via Bootstrap allgather.
    //--------------------------------------------------------------------------
    void exchange() {
        int nbuf = (int)local_bufs_.size();
        int nranks = boot_.size();

        struct BufMeta {
            uint64_t           addr;
            uint64_t           key;
            uint8_t            has_ipc;
            hipIpcMemHandle_t  ipc_handle;
        };
        std::vector<BufMeta> my_metas(nbuf);
        for (int i = 0; i < nbuf; i++) {
            my_metas[i].addr       = local_bufs_[i].addr_;
            my_metas[i].key        = local_bufs_[i].key_;
            my_metas[i].has_ipc    = local_bufs_[i].has_ipc_handle ? 1 : 0;
            my_metas[i].ipc_handle = local_bufs_[i].ipc_handle;
        }

        auto raw = boot_.allgather(my_metas.data(),
                                   nbuf * (int)sizeof(BufMeta));

        for (int r = 0; r < nranks; r++) {
            if (raw[r].size() != nbuf * sizeof(BufMeta)) {
                fprintf(stderr,
                    "GICC: exchange() rank %d expected %d buffers (%zu B), "
                    "peer %d sent %zu B\n",
                    boot_.rank(), nbuf, nbuf * sizeof(BufMeta),
                    r, raw[r].size());
                gicc::abort(1, "exchange(): buffer count mismatch");
            }
            const auto* peer_metas =
                reinterpret_cast<const BufMeta*>(raw[r].data());

            peer_mapped_ptrs_[r].assign(nbuf, nullptr);

            for (int i = 0; i < nbuf; i++) {
                comm_->set_remote_info_by_index(r, i,
                    peer_metas[i].addr, peer_metas[i].key);

                // Open IPC mapping for same-node peers (skip self — writing
                // to our own mapped pointer would bypass registered buffer
                // semantics and there is no real peer to target).
                if (r != boot_.rank()
                    && local_peer_[r]
                    && peer_metas[i].has_ipc)
                {
                    void* mapped = nullptr;
                    hipError_t err = hipIpcOpenMemHandle(
                        &mapped, peer_metas[i].ipc_handle,
                        hipIpcMemLazyEnablePeerAccess);
                    if (err == hipSuccess) {
                        peer_mapped_ptrs_[r][i] = mapped;
                    }
                }
            }
        }

        // Build and upload the device-side IPC map. Indexed
        // [peer * n_bufs_ + buf_idx]; mapped_ptr is non-null only when
        // peer's buffer is reachable via direct GPU stores. Used by the
        // device-side put_no_db (peer + dst_buf overload) to dispatch
        // IPC vs DWQ in-kernel without host involvement.
        n_bufs_ = nbuf;
        std::vector<IpcMapEntry> host_map((size_t)nranks * (size_t)nbuf);
        for (int r = 0; r < nranks; r++) {
            for (int b = 0; b < nbuf; b++) {
                IpcMapEntry& e = host_map[(size_t)r * nbuf + b];
                e.mapped_ptr  = peer_mapped_ptrs_[r][b];   // nullptr if off-node / self
                e.remote_base = comm_->get_remote_info(r, b).rma_addr;
            }
        }
        const size_t map_bytes = host_map.size() * sizeof(IpcMapEntry);
        if (hipMalloc(&d_ipc_map_, map_bytes) != hipSuccess) {
            fprintf(stderr, "GICC: hipMalloc(ipc_map) failed\n");
            std::abort();
        }
        if (hipMemcpy(d_ipc_map_, host_map.data(), map_bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            fprintf(stderr, "GICC: hipMemcpy(ipc_map) failed\n");
            std::abort();
        }
    }

    RemoteBufferInfo remote_buffer(int rank, int buf_index) const {
        auto ri = const_cast<Fabric*>(comm_)->get_remote_info(rank, buf_index);
        RemoteBufferInfo r;
        r.addr = ri.rma_addr;
        r.rkey = (uint32_t)(ri.rma_key & 0xFFFFFFFFu);
        return r;
    }

    // O(1) — lkey == buf_index on this backend.
    const Buffer& buffer_by_lkey(uint32_t lkey) const {
        if ((size_t)lkey >= buffers_.size()) {
            fprintf(stderr,
                "gicc::Runtime::buffer_by_lkey(%u): no buffer with that lkey.\n",
                lkey);
            std::abort();
        }
        return buffers_[lkey];
    }

    // Peer's registered base address for buffer index `dst_buf_idx`.
    // Looked up by buffer index (NOT by libfabric MR key) — buf indices are
    // dense and stable across ranks, MR keys are not. The host-side trace
    // gets dst_buf_idx from the gicc::launch call site, not from the
    // user's `rkey` arg in the device-side put_no_db call.
    uint64_t peer_buffer_base(int peer, int dst_buf_idx) const {
        auto ri = const_cast<Fabric*>(comm_)->get_remote_info(peer, dst_buf_idx);
        if (ri.rma_key == 0 && ri.rma_addr == 0) {
            fprintf(stderr,
                "gicc::Runtime::peer_buffer_base(peer=%d, buf_idx=%d): not exchanged.\n",
                peer, dst_buf_idx);
            std::abort();
        }
        return ri.rma_addr;
    }

    //--------------------------------------------------------------------------
    // put_no_db — queue an RMA WRITE only. Consumes one slot from the pool.
    //--------------------------------------------------------------------------
    Token put_no_db(const Buffer& src, int dest_rank, int dest_buf_index,
                    size_t size, size_t src_offset = 0, size_t dst_offset = 0)
    {
        const OfiBuffer& ob = local_bufs_.at(src.index);

        if ((int)my_n_ops_ >= POOL_SIZE) {
            fprintf(stderr,
                "gicc::Runtime::put_no_db: batch exceeds POOL_SIZE=%d. "
                "Call rt.reset() between batches or raise POOL_SIZE.\n",
                POOL_SIZE);
            exit(1);
        }

        // IPC fast path: dest_rank shares our node and its buffer was IPC-
        // mapped during exchange(). Stage a {dst,src,size} descriptor into
        // host-mapped memory; the user's kernel will execute the copy via
        // gicc::put_local(ctx) using direct GPU stores to the peer's
        // IPC-mapped buffer. No DWQ op, no trigger slot consumed.
        if (dest_rank != comm_->rank()
            && local_peer_[dest_rank]
            && (int)peer_mapped_ptrs_[dest_rank].size() > dest_buf_index
            && peer_mapped_ptrs_[dest_rank][dest_buf_index] != nullptr)
        {
            if ((int)my_n_local_ops_ >= POOL_SIZE) {
                fprintf(stderr, "gicc::Runtime::put_no_db: local batch "
                        "exceeds POOL_SIZE=%d\n", POOL_SIZE);
                exit(1);
            }
            const int local_idx = (int)my_n_local_ops_;
            h_local_ops_[local_idx].dst =
                (char*)peer_mapped_ptrs_[dest_rank][dest_buf_index] + dst_offset;
            h_local_ops_[local_idx].src = (char*)ob.ptr + src_offset;
            h_local_ops_[local_idx].size = size;
            my_n_local_ops_++;
            return Token{ local_idx, /*is_local=*/true };
        }

        const int slot_idx = (int)my_n_ops_;
        my_n_ops_++;
        const uint64_t trigger_threshold = my_n_remote_ops_ + 1;  // 1-based
        my_n_remote_ops_++;

        RemoteInfo ri = comm_->get_remote_info(dest_rank, dest_buf_index);
        if (ri.rma_key == 0 && ri.rma_addr == 0) {
            fprintf(stderr, "gicc::Runtime: remote info not set for rank %d "
                    "buf %d (call exchange() first)\n", dest_rank, dest_buf_index);
            exit(1);
        }
        const uint64_t remote_addr = comm_->is_virt_addr_mode()
            ? (ri.rma_addr + dst_offset)
            : (ri.rma_addr - ri.base_addr) + dst_offset;

        auto* dwq = new DwqWorkBuilder(comm_->rank());
        dwq->queue_rma_write(
            comm_->fabric->domain, comm_->fabric->ep,
            (char*)ob.ptr + src_offset, ob.desc_, size,
            comm_->av_addrs[dest_rank], remote_addr, ri.rma_key,
            comm_->fabric->trigger_cntr,
            slots_[slot_idx].completion_cntr,
            trigger_threshold);

        uint64_t* slot_addr = (uint64_t*)d_slot_pool_ + slot_idx;
        const uint64_t result_addr = comm_->is_virt_addr_mode()
            ? (uint64_t)slot_addr : ((uint64_t)slot_idx * sizeof(uint64_t));
        dwq->queue_atomic_signal(
            comm_->fabric->domain, comm_->fabric->ep,
            d_operand_pool_, mr_operand_pool_->desc,
            slot_addr, mr_slot_pool_->key,
            result_addr,
            comm_->fabric->local_addr_in_av,
            slots_[slot_idx].completion_cntr,
            slots_[slot_idx].atomic_completion_cntr,
            1);

        my_pending_.push_back(dwq);
        atomic_signals_queued_ = true;

        return Token{ slot_idx, /*is_local=*/false };
    }

    // get_no_db — queue an RMA READ. Same slot-pool accounting as put_no_db.
    Token get_no_db(const Buffer& local_dst, int src_rank, int src_buf_index,
                    size_t size, size_t local_offset = 0, size_t remote_offset = 0)
    {
        // NOTE: unlike put_no_db, no IPC fast-path here. v1 deliberately
        // routes local get through DWQ — IPC short-circuit for reads is
        // deferred (see unified-compiler-codegen design §12 "Out of Scope").
        const OfiBuffer& ob = local_bufs_.at(local_dst.index);

        if ((int)my_n_ops_ >= POOL_SIZE) {
            fprintf(stderr,
                "gicc::Runtime::get_no_db: batch exceeds POOL_SIZE=%d. "
                "Call rt.reset() between batches or raise POOL_SIZE.\n",
                POOL_SIZE);
            exit(1);
        }

        const int slot_idx = (int)my_n_ops_;
        my_n_ops_++;
        const uint64_t trigger_threshold = my_n_remote_ops_ + 1;  // 1-based
        my_n_remote_ops_++;

        RemoteInfo ri = comm_->get_remote_info(src_rank, src_buf_index);
        if (ri.rma_key == 0 && ri.rma_addr == 0) {
            fprintf(stderr, "gicc::Runtime: remote info not set for rank %d "
                    "buf %d (call exchange() first)\n", src_rank, src_buf_index);
            exit(1);
        }
        const uint64_t remote_addr = comm_->is_virt_addr_mode()
            ? (ri.rma_addr + remote_offset)
            : (ri.rma_addr - ri.base_addr) + remote_offset;

        auto* dwq = new DwqWorkBuilder(comm_->rank());
        dwq->queue_rma_read(
            comm_->fabric->domain, comm_->fabric->ep,
            (char*)ob.ptr + local_offset, ob.desc_, size,
            comm_->av_addrs[src_rank], remote_addr, ri.rma_key,
            comm_->fabric->trigger_cntr,
            slots_[slot_idx].completion_cntr,
            trigger_threshold);

        uint64_t* slot_addr = (uint64_t*)d_slot_pool_ + slot_idx;
        const uint64_t result_addr = comm_->is_virt_addr_mode()
            ? (uint64_t)slot_addr : ((uint64_t)slot_idx * sizeof(uint64_t));
        dwq->queue_atomic_signal(
            comm_->fabric->domain, comm_->fabric->ep,
            d_operand_pool_, mr_operand_pool_->desc,
            slot_addr, mr_slot_pool_->key,
            result_addr,
            comm_->fabric->local_addr_in_av,
            slots_[slot_idx].completion_cntr,
            slots_[slot_idx].atomic_completion_cntr,
            1);

        my_pending_.push_back(dwq);
        atomic_signals_queued_ = true;

        return Token{ slot_idx, /*is_local=*/false };
    }

    //--------------------------------------------------------------------------
    // prepare — finalize a batched put_no_db sequence.
    //--------------------------------------------------------------------------
    DeviceCtx* prepare(int peer_rank = -1, int remote_buf_index = -1) {
        (void)peer_rank;
        (void)remote_buf_index;

        if (my_n_remote_ops_ > 0) {
            (void)hipMemset(d_slot_pool_, 0,
                            my_n_remote_ops_ * sizeof(uint64_t));
            (void)hipDeviceSynchronize();
        }

        h_dev_ctx_->trigger_addr_      = comm_->get_trigger_addr();
        h_dev_ctx_->trigger_val_       = my_n_remote_ops_;
        h_dev_ctx_->completion_        = (volatile uint64_t*)d_slot_pool_;
        h_dev_ctx_->n_ops_             = my_n_remote_ops_;
        h_dev_ctx_->ipc_map_           = d_ipc_map_;
        h_dev_ctx_->max_bufs_per_rank_ = n_bufs_;
        h_dev_ctx_->local_ops_         = d_local_ops_;
        h_dev_ctx_->n_local_ops_       = my_n_local_ops_;
        return d_dev_ctx_;
    }

    //--------------------------------------------------------------------------
    // prepare_trigger — overlap pattern (flush only, host polls later).
    //--------------------------------------------------------------------------
    DeviceCtx* prepare_trigger(Token /*tok*/) {
        h_dev_ctx_->trigger_addr_      = comm_->get_trigger_addr();
        h_dev_ctx_->trigger_val_       = my_n_remote_ops_;
        h_dev_ctx_->completion_        = nullptr;
        h_dev_ctx_->n_ops_             = 0;
        h_dev_ctx_->ipc_map_           = d_ipc_map_;
        h_dev_ctx_->max_bufs_per_rank_ = n_bufs_;
        h_dev_ctx_->local_ops_         = d_local_ops_;
        h_dev_ctx_->n_local_ops_       = my_n_local_ops_;
        return d_dev_ctx_;
    }

    //--------------------------------------------------------------------------
    // Host-side wait for a specific token.
    //--------------------------------------------------------------------------
    void wait(Token tok) {
        // Local ops are executed INSIDE the user's kernel by gicc::put_local().
        // The caller is expected to have already synchronised the kernel (e.g.
        // hipStreamSynchronize / hipDeviceSynchronize) before calling wait(),
        // so the copy has landed and no host-side polling is required.
        if (tok.is_local) return;
        while (fi_cntr_read(slots_[tok.slot_idx].completion_cntr) < 1) {}
    }

    //--------------------------------------------------------------------------
    // reset — drain the current batch and recycle the slots.
    //--------------------------------------------------------------------------
    void reset() {
        // my_n_ops_ only counts remote ops (locals live in a parallel pool),
        // so the existing fi_cntr drain covers the right slots.
        for (uint64_t i = 0; i < my_n_ops_; i++) {
            while (fi_cntr_read(slots_[i].completion_cntr) < 1) {
                fi_cq_read(comm_->fabric->cq, NULL, 0);
            }
        }
        if (atomic_signals_queued_) {
            for (uint64_t i = 0; i < my_n_ops_; i++) {
                while (fi_cntr_read(slots_[i].atomic_completion_cntr) < 1) {
                    fi_cq_read(comm_->fabric->cq, NULL, 0);
                }
            }
        }

        for (auto* op : my_pending_) delete op;
        my_pending_.clear();

        fi_cntr_set(comm_->fabric->trigger_cntr, 0);
        for (uint64_t i = 0; i < my_n_ops_; i++) {
            fi_cntr_set(slots_[i].completion_cntr, 0);
            if (atomic_signals_queued_)
                fi_cntr_set(slots_[i].atomic_completion_cntr, 0);
        }

        // Local ops: the copies fired inside the caller's kernel. Caller is
        // required to have synchronised the kernel before reset(), so the
        // staged descriptors can be discarded outright.
        my_n_ops_              = 0;
        my_n_remote_ops_       = 0;
        my_n_local_ops_        = 0;
        atomic_signals_queued_ = false;
    }

    void barrier() { comm_->barrier(); }

    int rank()   const { return comm_->rank(); }
    int size()   const { return comm_->size(); }
    Bootstrap& boot() noexcept { return boot_; }
    const Bootstrap& boot() const noexcept { return boot_; }
    int gpu_id() const { return comm_->gpu_id(); }

    Fabric& fabric() { return *comm_; }

private:
    // Internal buffer metadata (replaces gda::Buffer).
    struct OfiBuffer {
        void*             ptr;
        void*             desc_;
        uint64_t          key_;
        uint64_t          addr_;
        bool              has_ipc_handle = false;
        hipIpcMemHandle_t ipc_handle{};
    };

    struct Slot {
        struct fid_cntr* completion_cntr        = nullptr;
        struct fid_cntr* atomic_completion_cntr = nullptr;
    };

    gicc::Bootstrap               boot_;
    Fabric*                      comm_;
    DeviceCtx*                    h_dev_ctx_;
    DeviceCtx*                    d_dev_ctx_;

    Slot                          slots_[POOL_SIZE];

    void*                         d_slot_pool_;
    MemoryRegion*                 mr_slot_pool_;
    void*                         d_operand_pool_;
    MemoryRegion*                 mr_operand_pool_;

    uint64_t                      my_n_ops_;         // remote ops queued
    uint64_t                      my_n_remote_ops_;  // mirrors my_n_ops_
    uint64_t                      my_n_local_ops_;   // IPC ops queued
    bool                          atomic_signals_queued_;
    std::vector<DwqWorkBuilder*>  my_pending_;

    std::vector<OfiBuffer>        local_bufs_;

    // Local buffers indexed by their buf_index (== lkey on this backend).
    // Populated in register_buffer; used by buffer_by_lkey().
    std::vector<Buffer>           buffers_;

    // IPC fast-path state. peer_mapped_ptrs_[rank][buf_idx] is the mapped
    // pointer opened via hipIpcOpenMemHandle at exchange() time (or nullptr
    // when the peer is off-node / the buffer had no IPC handle).
    std::vector<bool>                  local_peer_;
    std::vector<std::vector<void*>>    peer_mapped_ptrs_;

    // Host-mapped array of LocalOp descriptors. put_no_db() fills entries
    // here; the kernel reads them via d_local_ops_ during gicc::put_local().
    LocalOp*                           h_local_ops_;
    LocalOp*                           d_local_ops_;

    // Per-runtime IPC map uploaded after exchange(). Indexed
    // [peer * n_bufs_ + buf_idx]. Each entry's mapped_ptr is non-null
    // exactly when peer is on the same node and exposed an IPC handle
    // for that buffer; remote_base is the peer's registered RMA base
    // address (used to translate dst_addr in put_no_db to the local
    // IPC-mapped pointer). nullptr until exchange() runs.
    IpcMapEntry*                       d_ipc_map_      = nullptr;
    int                                n_bufs_         = 0;
};

} // namespace gicc
