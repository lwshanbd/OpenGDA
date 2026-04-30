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
#include <thread>
#include <atomic>

#include "gicc/gicc_types.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/runtime_helpers.h"   // C ABI consumed by LTO IR

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
    // C ABI helpers consumed by LTO-emitted IR (see
    // src/gicc/platform/ofi/runtime_helpers.h). They reach into private
    // state — peer_mapped_ptrs_, local_bufs_, remote_info_cache_, the
    // DWQ pool, and the monotonic counter — so granting friendship is
    // simpler than exposing each accessor.
    friend void *           (::gicc_runtime_peer_ipc_base) (Runtime *, int, int);
    friend void *           (::gicc_runtime_local_buf_base)(Runtime *, int);
    friend ::hipStream_t    (::gicc_runtime_ipc_stream)    (Runtime *);
    friend void             (::gicc_runtime_dwq_enqueue)   (Runtime *, int, int,
                                                            std::size_t, int,
                                                            std::size_t,
                                                            std::size_t);
    friend void             (::gicc_runtime_dwq_enqueue_batched)
                                       (Runtime *, int,
                                        const int *, const int *,
                                        const std::size_t *, const int *,
                                        const std::size_t *, const std::size_t *);
    friend volatile std::uint64_t *(::gicc_runtime_trigger_addr)(Runtime *);
    friend std::uint64_t           (::gicc_runtime_trigger_val) (Runtime *);

public:
    static constexpr int POOL_SIZE = 32;   // max ops per batch

    Runtime()
        : comm_(nullptr),
          h_dev_ctx_(nullptr), d_dev_ctx_(nullptr),
          d_slot_pool_(nullptr), mr_slot_pool_(nullptr),
          d_operand_pool_(nullptr), mr_operand_pool_(nullptr),
          my_n_ops_(0), my_n_remote_ops_(0),
          atomic_signals_queued_(false),
          host_wait_mode_(false),
          shared_completion_cntr_(nullptr),
          mono_total_ops_(0)
    {
        unset_rocr_visible_devices();
        comm_ = new Fabric(boot_);

        // Locality map from Bootstrap: [rank] = true iff rank shares our node.
        // put_no_db() routes same-node writes through HIP IPC entirely
        // device-side: the kernel reads peer_mapped_ptrs_ exposed via
        // DeviceCtx->ipc_map_ and does block-cooperative GPU stores
        // directly. Remote peers keep the DWQ/CXI path.
        local_peer_ = boot_.locality_map();
        peer_mapped_ptrs_.assign(boot_.size(), {});

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
        h_dev_ctx_->local_bufs_        = nullptr;
        h_dev_ctx_->n_local_bufs_      = 0;
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

        if (d_ipc_map_) (void)hipFree(d_ipc_map_);
        if (d_local_bufs_) (void)hipFree(d_local_bufs_);

        if (shared_completion_cntr_)
            fi_close(&shared_completion_cntr_->fid);
        for (auto* op : dwq_pool_) delete op;
        dwq_pool_.clear();
        if (monitor_thread_.joinable()) {
            monitor_stop_.store(true, std::memory_order_release);
            monitor_thread_.join();
        }
        if (h_ipc_head_) (void)hipHostFree(h_ipc_head_);
        if (h_ipc_ring_) (void)hipHostFree(h_ipc_ring_);
        if (ipc_stream_) (void)hipStreamDestroy(ipc_stream_);

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

        // Flatten the per-(peer, buf) RemoteInfo into a contiguous array
        // so put_no_db can do an O(1) array index instead of an
        // unordered_map.find() + RemoteInfo copy on every call.
        remote_info_cache_.assign((size_t)nranks * (size_t)nbuf, RemoteInfo{});
        for (int r = 0; r < nranks; r++) {
            for (int b = 0; b < nbuf; b++) {
                remote_info_cache_[(size_t)r * nbuf + b] =
                    comm_->get_remote_info(r, b);
            }
        }

        std::vector<IpcMapEntry> host_map((size_t)nranks * (size_t)nbuf);
        for (int r = 0; r < nranks; r++) {
            for (int b = 0; b < nbuf; b++) {
                IpcMapEntry& e = host_map[(size_t)r * nbuf + b];
                e.mapped_ptr  = peer_mapped_ptrs_[r][b];   // nullptr if off-node / self
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

        // Build and upload device-side local buffer base table. This is a
        // snapshot at exchange() time — register_buffer calls made AFTER
        // exchange() will not appear here (v1 limitation; matches the IPC
        // map's snapshot semantics).
        {
            std::vector<LocalBufView> host_lb(local_bufs_.size());
            for (size_t i = 0; i < local_bufs_.size(); i++) {
                host_lb[i].base = local_bufs_[i].ptr;
            }
            const size_t lb_bytes = host_lb.size() * sizeof(LocalBufView);
            if (hipMalloc(&d_local_bufs_, lb_bytes) != hipSuccess) {
                fprintf(stderr, "GICC: hipMalloc(local_bufs) failed\n");
                std::abort();
            }
            if (hipMemcpy(d_local_bufs_, host_lb.data(), lb_bytes,
                          hipMemcpyHostToDevice) != hipSuccess) {
                fprintf(stderr, "GICC: hipMemcpy(local_bufs) failed\n");
                std::abort();
            }
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
    // enable_host_wait_mode — switch put_no_db / get_no_db to the GDA-style
    // fast path. After enabling:
    //   - Each put queues ONLY the RMA write (no chained atomic_signal),
    //     halving the NIC ops per put.
    //   - One shared completion counter, monotonic threshold across iters.
    //   - reset() becomes a wait-then-recycle (no per-batch counter reset).
    //   - DwqWorkBuilder objects pulled from / returned to a pool.
    //
    // Caller contract: kernels must NOT use device-side gicc::quiet (no
    // GPU-side completion signalling). Host-side rt.reset() is the wait.
    // Idempotent (safe to call multiple times).
    //--------------------------------------------------------------------------
    void enable_host_wait_mode() {
        if (host_wait_mode_) return;
        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;
        int ret = fi_cntr_open(comm_->fabric->domain, &cntr_attr,
                                &shared_completion_cntr_, NULL);
        if (ret) {
            fprintf(stderr, "GICC: fi_cntr_open(shared_completion) failed: %s\n",
                    fi_strerror(-ret));
            std::abort();
        }
        // Dedicated stream for the IPC drain kernel so it overlaps with
        // user kernel + NIC RDMA. Non-blocking so it really runs
        // concurrent with default stream's flush kernel.
        if (hipStreamCreateWithFlags(&ipc_stream_, hipStreamNonBlocking)
            != hipSuccess) {
            fprintf(stderr, "GICC: hipStreamCreate(ipc_stream) failed\n");
            std::abort();
        }
        // Pinned ring + head, both mapped to device. Device
        // put_no_db's IPC route writes into the ring + atomic-increments
        // the head; CPU monitor thread polls the head and dispatches
        // hipMemcpyAsync onto ipc_stream_.
        if (hipHostMalloc(&h_ipc_ring_,
                          IPC_RING_SIZE_ * sizeof(IpcCmdSlot),
                          hipHostMallocMapped) != hipSuccess) {
            fprintf(stderr, "GICC: hipHostMalloc(ipc_ring) failed\n");
            std::abort();
        }
        if (hipHostGetDevicePointer((void**)&d_ipc_ring_, h_ipc_ring_, 0)
            != hipSuccess) {
            fprintf(stderr, "GICC: hipHostGetDevicePointer(ipc_ring) failed\n");
            std::abort();
        }
        if (hipHostMalloc(&h_ipc_head_, sizeof(uint64_t),
                          hipHostMallocMapped) != hipSuccess) {
            fprintf(stderr, "GICC: hipHostMalloc(ipc_head) failed\n");
            std::abort();
        }
        if (hipHostGetDevicePointer((void**)&d_ipc_head_, h_ipc_head_, 0)
            != hipSuccess) {
            fprintf(stderr, "GICC: hipHostGetDevicePointer(ipc_head) failed\n");
            std::abort();
        }
        *h_ipc_head_ = 0;
        monitor_stop_.store(false, std::memory_order_relaxed);
        monitor_dispatched_.store(0, std::memory_order_relaxed);
        monitor_thread_ = std::thread([this]() { monitor_loop_(); });

        host_wait_mode_ = true;
    }

private:
    // Monitor thread loop. Polls h_ipc_head_ (GPU writes via threadfence
    // _system); for each new entry, issues hipMemcpyAsync on ipc_stream_.
    void monitor_loop_() {
        uint64_t cpu_seen = 0;
        while (!monitor_stop_.load(std::memory_order_acquire)) {
            uint64_t head = __atomic_load_n((volatile uint64_t*)h_ipc_head_,
                                            __ATOMIC_ACQUIRE);
            while (cpu_seen < head) {
                IpcCmdSlot& cmd = h_ipc_ring_[cpu_seen & (IPC_RING_SIZE_ - 1)];
                hipError_t err = hipMemcpyAsync(cmd.dst, cmd.src, cmd.size,
                                                hipMemcpyDeviceToDevice,
                                                ipc_stream_);
                if (err != hipSuccess) {
                    fprintf(stderr,
                        "GICC monitor: hipMemcpyAsync failed: %s\n",
                        hipGetErrorString(err));
                    // Keep going; reset will detect via stream sync.
                }
                cpu_seen++;
            }
            monitor_dispatched_.store(cpu_seen, std::memory_order_release);
            // Pure spin — minimod halo is hot, sched_yield would add ms
            // per iter. CPU usage is fine; this is a dedicated helper.
        }
    }
public:

private:
    // Fetch a recycled DwqWorkBuilder from the pool, or allocate a new one.
    // Used only by the host-wait fast path (the legacy path heap-allocates
    // per put for backwards-compat). queue_rma_write overwrites every field
    // we read, so we skip the defensive memset that the constructor does.
    DwqWorkBuilder* dwq_get_() {
        if (dwq_pool_.empty()) {
            return new DwqWorkBuilder(comm_->rank());
        }
        DwqWorkBuilder* d = dwq_pool_.back();
        dwq_pool_.pop_back();
        return d;
    }

    void dwq_release_all_pending_to_pool_() {
        for (auto* op : my_pending_) dwq_pool_.push_back(op);
        my_pending_.clear();
    }

public:

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

        // IPC fast path. host_wait_mode + Pattern C: don't do anything
        // here — the device-side put_no_db will push a command to the
        // GPU↔CPU ring AT THE put_no_db CALL SITE in the user kernel,
        // and the monitor thread will dispatch hipMemcpyAsync onto
        // ipc_stream_. This preserves "comm fires when you write
        // put_no_db" semantics. Legacy (non-host-wait) mode: device
        // put_no_db does in-kernel block-cooperative memcpy.
        if (dest_rank != comm_->rank()
            && local_peer_[dest_rank]
            && (int)peer_mapped_ptrs_[dest_rank].size() > dest_buf_index
            && peer_mapped_ptrs_[dest_rank][dest_buf_index] != nullptr)
        {
            (void)ob; (void)size; (void)src_offset; (void)dst_offset;
            return Token{ -1, /*is_local=*/true };
        }

        // O(1) flat-array lookup populated at exchange() time.
        const RemoteInfo& ri = remote_info_cache_[
            (size_t)dest_rank * (size_t)n_bufs_ + (size_t)dest_buf_index];
        if (ri.rma_key == 0 && ri.rma_addr == 0) {
            fprintf(stderr, "gicc::Runtime: remote info not set for rank %d "
                    "buf %d (call exchange() first)\n", dest_rank, dest_buf_index);
            exit(1);
        }
        const uint64_t remote_addr = comm_->is_virt_addr_mode()
            ? (ri.rma_addr + dst_offset)
            : (ri.rma_addr - ri.base_addr) + dst_offset;

        // ---------- Host-wait fast path (GDA-compatible) ----------
        if (host_wait_mode_) {
            ++mono_total_ops_;
            ++my_n_remote_ops_;
            DwqWorkBuilder* dwq = dwq_get_();
            dwq->queue_rma_write(
                comm_->fabric->domain, comm_->fabric->ep,
                (char*)ob.ptr + src_offset, ob.desc_, size,
                comm_->av_addrs[dest_rank], remote_addr, ri.rma_key,
                comm_->fabric->trigger_cntr,
                shared_completion_cntr_,
                /*threshold=*/mono_total_ops_);
            // NO atomic_signal queued — saves 1 NIC op per put.
            my_pending_.push_back(dwq);
            return Token{ (int)mono_total_ops_, /*is_local=*/false };
        }

        // ---------- Legacy path (per-slot + atomic_signal for device quiet) ----------
        if ((int)my_n_ops_ >= POOL_SIZE) {
            fprintf(stderr,
                "gicc::Runtime::put_no_db: batch exceeds POOL_SIZE=%d. "
                "Call rt.reset() between batches or raise POOL_SIZE.\n",
                POOL_SIZE);
            exit(1);
        }
        const int slot_idx = (int)my_n_ops_;
        my_n_ops_++;
        const uint64_t trigger_threshold = my_n_remote_ops_ + 1;
        my_n_remote_ops_++;

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

        if (host_wait_mode_) {
            // Pattern C: device put_no_db's IPC route writes (dst,src,
            // size) into ipc_cmd_ring_ + atomic-incs ipc_cmd_head_;
            // monitor thread polls and dispatches hipMemcpyAsync on
            // ipc_stream_. Comm fires AT the put_no_db line in the
            // kernel, not at gicc::launch call site.
            h_dev_ctx_->trigger_addr_      = comm_->get_trigger_addr();
            h_dev_ctx_->trigger_val_       = mono_total_ops_;
            h_dev_ctx_->completion_        = nullptr;
            h_dev_ctx_->n_ops_             = 0;
            h_dev_ctx_->ipc_map_           = d_ipc_map_;
            h_dev_ctx_->max_bufs_per_rank_ = n_bufs_;
            h_dev_ctx_->local_bufs_        = d_local_bufs_;
            h_dev_ctx_->n_local_bufs_      = (int)local_bufs_.size();
            h_dev_ctx_->ipc_cmd_ring_      = d_ipc_ring_;
            h_dev_ctx_->ipc_cmd_head_      = (volatile uint64_t*)d_ipc_head_;
            h_dev_ctx_->ipc_ring_mask_     = IPC_RING_SIZE_ - 1;
            return d_dev_ctx_;
        }

        if (my_n_remote_ops_ > 0) {
            (void)hipMemsetAsync(d_slot_pool_, 0,
                                 my_n_remote_ops_ * sizeof(uint64_t),
                                 /*stream=*/0);
        }

        h_dev_ctx_->trigger_addr_      = comm_->get_trigger_addr();
        h_dev_ctx_->trigger_val_       = my_n_remote_ops_;
        h_dev_ctx_->completion_        = (volatile uint64_t*)d_slot_pool_;
        h_dev_ctx_->n_ops_             = my_n_remote_ops_;
        h_dev_ctx_->ipc_map_           = d_ipc_map_;
        h_dev_ctx_->max_bufs_per_rank_ = n_bufs_;
        h_dev_ctx_->local_bufs_        = d_local_bufs_;
        h_dev_ctx_->n_local_bufs_      = (int)local_bufs_.size();
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
        h_dev_ctx_->local_bufs_        = d_local_bufs_;
        h_dev_ctx_->n_local_bufs_      = (int)local_bufs_.size();
        return d_dev_ctx_;
    }

    //--------------------------------------------------------------------------
    // Host-side wait for a specific token.
    //--------------------------------------------------------------------------
    void wait(Token tok) {
        // IPC-routed ops are executed INSIDE the user's kernel by the
        // device-side gicc::put_no_db (block-cooperative GPU stores
        // through ctx->ipc_map_). The caller is expected to have already
        // synchronised the kernel (e.g. hipStreamSynchronize /
        // hipDeviceSynchronize) before calling wait(), so the copy has
        // landed and no host-side polling is required.
        if (tok.is_local) return;
        while (fi_cntr_read(slots_[tok.slot_idx].completion_cntr) < 1) {}
    }

    //--------------------------------------------------------------------------
    // reset — drain the current batch and recycle the slots.
    //--------------------------------------------------------------------------
    void reset() {
        if (host_wait_mode_) {
            // Fast path: poll the SHARED completion counter against the
            // monotonic threshold. No per-slot loop, no counter reset.
            // DwqWorkBuilders go back to the pool instead of being deleted.
            if (mono_total_ops_ > 0) {
                while (fi_cntr_read(shared_completion_cntr_) < mono_total_ops_) {
                    fi_cq_read(comm_->fabric->cq, NULL, 0);
                }
            }
            dwq_release_all_pending_to_pool_();
            my_n_remote_ops_ = 0;   // per-iter accounting clears (mono is global)

            // IPC monitor handoff: caller has already done
            // hipDeviceSynchronize, so all GPU writes to the ring head
            // are visible. Wait for the monitor thread to dispatch every
            // command the kernel pushed, then drain the IPC SDMA stream
            // so all peer writes are committed before MPI_Barrier.
            uint64_t target = __atomic_load_n((volatile uint64_t*)h_ipc_head_,
                                              __ATOMIC_ACQUIRE);
            while (monitor_dispatched_.load(std::memory_order_acquire) < target) {
                // spin
            }
            if (target > 0) {
                (void)hipStreamSynchronize(ipc_stream_);
            }
            return;
        }

        // ---------- Legacy path ----------
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
        my_n_ops_              = 0;
        my_n_remote_ops_       = 0;
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

    // Per-runtime IPC map uploaded after exchange(). Indexed
    // [peer * n_bufs_ + buf_idx]. Each entry's mapped_ptr is non-null
    // exactly when peer is on the same node and exposed an IPC handle
    // for that buffer. nullptr until exchange() runs.
    // Host-wait mode (GDA-compatible fast path):
    //   - Skip per-slot atomic_signal queueing (saves 1 NIC op per put)
    //   - One shared completion counter, monotonic threshold, no per-batch reset
    //   - DwqWorkBuilder objects pooled instead of new/delete per iter
    // Enabled via enable_host_wait_mode(). Required: user kernel must NOT
    // call gicc::quiet (no GPU-side completion polling); host gates via
    // rt.reset() which becomes a fi_cntr_read busy-poll on the shared cntr.
    bool                               host_wait_mode_;
    struct fid_cntr*                   shared_completion_cntr_;
    uint64_t                           mono_total_ops_;          // monotonic across batches
    std::vector<DwqWorkBuilder*>       dwq_pool_;                // recycled builders

    // GPU→CPU command ring + monitor thread for the IPC fast path.
    // Device-side put_no_db (host_wait_mode + IPC peer) pushes a
    // command slot onto ipc_ring_; monitor_thread_ polls the head and
    // dispatches hipMemcpyAsync on ipc_stream_. This preserves "comm
    // fires AT the put_no_db line" semantics — the SDMA dispatch is
    // triggered by the GPU thread via threadfence_system, not by the
    // host trace function.
    static constexpr int               IPC_RING_SIZE_ = 64;   // power of 2
    hipStream_t                        ipc_stream_     = nullptr;
    IpcCmdSlot*                        h_ipc_ring_     = nullptr;
    IpcCmdSlot*                        d_ipc_ring_     = nullptr;
    uint64_t*                          h_ipc_head_     = nullptr;
    uint64_t*                          d_ipc_head_     = nullptr;
    std::thread                        monitor_thread_;
    std::atomic<bool>                  monitor_stop_   {false};
    std::atomic<uint64_t>              monitor_dispatched_ {0};

    // Flat cache of RemoteInfo (av_addr / rma_addr / rma_key / base_addr)
    // for every (peer, buf) pair, populated during exchange(). Avoids
    // an unordered_map lookup + RemoteInfo copy on every put_no_db.
    // Indexed [peer * n_bufs_ + buf_idx]; same indexing as d_ipc_map_.
    std::vector<RemoteInfo>            remote_info_cache_;

    IpcMapEntry*                       d_ipc_map_      = nullptr;
    int                                n_bufs_         = 0;

    // Per-runtime local-buffer base table uploaded after exchange().
    // Indexed [buf_idx]; base is the GPU pointer registered via
    // register_buffer. Used by the device-side put_no_db / get_no_db to
    // resolve (src_buf, src_offset) → real GPU pointer.
    LocalBufView*                      d_local_bufs_   = nullptr;
};

} // namespace gicc
