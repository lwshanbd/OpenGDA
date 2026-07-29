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

#include "internal/gpu_device_context.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <sched.h>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gicc/gicc_types.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/runtime_helpers.h"   // C ABI consumed by LTO IR

// OFI backend internals (Fabric, FabricDwqContext, MemoryRegion, etc.)
#include "internal/gpu_device_context.hpp"
#include "gicc/bootstrap/bootstrap.hpp"
#include "internal/device_affinity.hpp"
#include "internal/fabric_dwq_context.hpp"
#include "internal/memory_region.hpp"
#include "internal/dwq_work_builder.hpp"
#include "internal/ofi_barrier.hpp"
#include "internal/fabric.hpp"

#ifdef GICC_CPU_PROXY
#include <memory>
#include "proxy/proxy_thread.hpp"
#endif


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
    friend ::GpuStream_t    (::gicc_runtime_ipc_stream)        (Runtime *);
    friend ::GpuStream_t    (::gicc_runtime_ipc_stream_indexed)(Runtime *, int);
    friend void             (::gicc_runtime_dwq_enqueue)   (Runtime *, int, int,
                                                            std::size_t, int,
                                                            std::size_t,
                                                            std::size_t);
    friend void             (::gicc_runtime_arm_dwq_trigger)(Runtime *);
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
          proxy_dispatch_disabled_(false),
          shared_completion_cntr_(nullptr),
          mono_total_ops_(0),
          mono_last_triggered_(0)
    {
        // Read W from env var; hint.json override is applied later when
        // the dispatch-lowering pass produces final hint.
        if (const char* env = std::getenv("GICC_WINDOW")) {
            int w = std::atoi(env);
            if (w >= 1 && w <= 32) {
                window_size_ = w;
            } else {
                fprintf(stderr, "GICC: GICC_WINDOW=%s out of range [1,32], using default %d\n",
                        env, window_size_);
            }
        }
        if (const char* env = std::getenv("GICC_STREAMS_MAX")) {
            int n = std::atoi(env);
            if (n >= 1 && n <= 32) {
                n_streams_max_ = n;
            } else {
                fprintf(stderr, "GICC: GICC_STREAMS_MAX=%s out of range [1,32], using default %d\n",
                        env, n_streams_max_);
            }
        }

        unset_rocr_visible_devices();
        comm_ = new Fabric(boot_);

#ifdef GICC_CPU_PROXY
        // Open the per-thread proxy endpoints + CQs BEFORE any MR is
        // registered, so that subsequent register_buffer() calls (and the
        // slot_pool / operand_pool MRs below — even though those are
        // DWQ-internal we keep registration uniform) can fi_mr_bind to every
        // proxy EP. After fi_mr_enable, more EPs cannot be bound to the MR.
        //
        // Defaulting N to 1 matches the previous CPU-proxy default; users
        // override via GICC_NUM_PROXY_THREADS to fan out the fleet. The
        // env range is [1,32] — prepare() unconditionally calls
        // ensure_proxy_rings() under -DGICC_CPU_PROXY, so a fleet of zero
        // is not currently representable; build without GICC_CPU_PROXY to
        // disable the proxy path entirely.
        {
            int n_proxy = 1;
            if (const char* e = std::getenv("GICC_NUM_PROXY_THREADS")) {
                int parsed = std::atoi(e);
                if (parsed >= 1 && parsed <= 32) {
                    n_proxy = parsed;
                } else if (boot_.rank() == 0) {
                    fprintf(stderr,
                        "[gicc] GICC_NUM_PROXY_THREADS=%s out of [1,32], "
                        "using default %d\n", e, n_proxy);
                }
            }
            n_proxy_threads_ = n_proxy;
            comm_->create_proxy_endpoints(n_proxy);
        }
#endif

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
        if (gpuMalloc(&d_slot_pool_, POOL_BYTES) != GPU_SUCCESS) {
            fprintf(stderr, "gpuMalloc(slot pool) failed\n"); exit(1);
        }
        (void)gpuMemset(d_slot_pool_, 0, POOL_BYTES);
        mr_slot_pool_ = new MemoryRegion(
            comm_->fabric->domain, comm_->fabric->ep, comm_->fabric->cxi_info,
            d_slot_pool_, POOL_BYTES, true, comm_->gpu_id(), comm_->rank());

        // Single shared atomic operand (value 1) and its MR.
        if (gpuMalloc(&d_operand_pool_, sizeof(uint64_t)) != GPU_SUCCESS) {
            fprintf(stderr, "gpuMalloc(operand) failed\n"); exit(1);
        }
        const uint64_t one = 1;
        (void)gpuMemcpy(d_operand_pool_, &one, sizeof(uint64_t),
                        gpuMemcpyHostToDevice);
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
        (void)gpuDeviceSynchronize();

        (void)gpuHostMalloc(&h_dev_ctx_, sizeof(DeviceCtx), gpuHostMallocMapped);
        (void)gpuHostGetDevicePointer((void**)&d_dev_ctx_, h_dev_ctx_, 0);
        h_dev_ctx_->trigger_addr_ = comm_->get_trigger_addr();
        h_dev_ctx_->trigger_val_  = 0;
    }

    ~Runtime() {
        // Stop the async staging worker before touching any libfabric
        // state it may still be submitting to.
        dwq_stage_shutdown_();
#ifdef GICC_CPU_PROXY
        // Stop all proxy workers before tearing down any libfabric state
        // they may still be polling. ProxyThread::stop() joins the worker.
        for (auto& pt : proxy_threads_) {
            if (pt) pt->stop();
        }
        if (proxy_rings_arr_host_) {
            (void)gpuHostFree(proxy_rings_arr_host_);
            proxy_rings_arr_host_ = nullptr;
            proxy_rings_arr_dev_  = nullptr;
        }
#endif
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
        if (d_slot_pool_)    (void)gpuFree(d_slot_pool_);
        if (d_operand_pool_) (void)gpuFree(d_operand_pool_);
        if (h_dev_ctx_)      (void)gpuHostFree(h_dev_ctx_);

        // Close IPC mapped pointers (one per local peer × buffer).
        for (auto& per_rank : peer_mapped_ptrs_) {
            for (void* p : per_rank) {
                if (p) (void)gpuIpcCloseMemHandle(p);
            }
        }
        peer_mapped_ptrs_.clear();

        if (shared_completion_cntr_)
            fi_close(&shared_completion_cntr_->fid);
        for (auto* op : dwq_pool_) delete op;
        dwq_pool_.clear();
        for (auto s : ipc_streams_) {
            if (s) (void)gpuStreamDestroy(s);
        }
        ipc_streams_.clear();

        delete comm_;
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    //--------------------------------------------------------------------------
    // Buffer registration — delegates to Fabric, caches local metadata.
    //--------------------------------------------------------------------------
    Buffer register_buffer(void* buf, size_t size, bool is_device) {
        // Sync any pending GPU work on `buf` before exposing it to peers
        // for RDMA. Without this, an async gpuMemset or gpuMemcpy issued
        // by the user prior to register_buffer can race with subsequent
        // inbound NIC writes: if the user's GPU init completes AFTER an
        // inbound RMA write has landed, the init overwrites that write
        // with stale bytes (typically 0). The bug only manifests on the
        // "silent receiver" side — a rank that registers + waits for
        // inbound but never launches its own GPU kernel — because no
        // later kernel launch implicitly forces the queued init to
        // complete. Cheapest robust fix: pay one device sync per
        // device-buffer registration.
        if (is_device) {
            (void)gpuDeviceSynchronize();
        }
        Handle h = comm_->register_buffer(buf, size, is_device);
        int idx = (int)local_bufs_.size();

        OfiBuffer ob;
        ob.ptr   = buf;
        ob.desc_ = h.local_desc;
        ob.key_  = h.rma_key;
        ob.addr_ = h.rma_addr;

        // Capture an IPC handle for device buffers so same-node peers can
        // open them in exchange(). GPU-allocated pointers are always valid
        // here; for non-device buffers IPC isn't meaningful.
        if (is_device) {
            if (gpuIpcGetMemHandle(&ob.ipc_handle, buf) == GPU_SUCCESS) {
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
            GpuIpcMemHandle_t  ipc_handle;
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
                    GpuError err = gpuIpcOpenMemHandle(
                        &mapped, peer_metas[i].ipc_handle,
                        gpuIpcMemLazyEnablePeerAccess);
                    if (err == GPU_SUCCESS) {
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

        // Pattern C deleted: device-side IPC table + local-buf table no
        // longer needed. The LTO host trace consults peer_mapped_ptrs_
        // and local_bufs_ on the host via the gicc_runtime_*_base
        // helpers; the kernel never reads an IPC map.

        // Locality-aware collectives: re-expose a flat device-readable table
        // of peer IPC-mapped buffer bases [peer * n_bufs_ + buf]. Host-pinned
        // mapped so the GPU can read the pointer values; entries are the
        // peer's device pointers (valid on this device) or null. Lets a
        // collective kernel write a same-node peer's buffer over xGMI.
        {
            const size_t n = (size_t)nranks * (size_t)nbuf;
            if (h_peer_ipc_) { (void)gpuHostFree(h_peer_ipc_); h_peer_ipc_ = nullptr; }
            if (gpuHostMalloc((void**)&h_peer_ipc_, n * sizeof(void*),
                              gpuHostMallocMapped) == GPU_SUCCESS) {
                for (int r = 0; r < nranks; r++)
                    for (int b = 0; b < nbuf; b++)
                        h_peer_ipc_[(size_t)r * nbuf + b] =
                            (local_peer_[r] && b < (int)peer_mapped_ptrs_[r].size())
                                ? peer_mapped_ptrs_[r][b] : nullptr;
                (void)gpuHostGetDevicePointer((void**)&d_peer_ipc_, h_peer_ipc_, 0);
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
    //--------------------------------------------------------------------------
    // register_host_mirror — tell GICC that the device array at `dev_ptr`
    // has an authoritative host-side mirror at `host_ptr`, laid out as
    // `count` elements of `elem_size` bytes each.  This lets the LTO pass
    // synthesize a host-side trace function that reads e.g.
    // `transfers[i].peer` at trace time -- the trace looks the host mirror
    // up via gicc_runtime_host_mirror_of() and does the GEP+load on the
    // host, avoiding a GPU memory dereference and unlocking DWQ_TRIGGER
    // dispatch for kernels that today are stuck on CPU_PROXY_ENQUEUE
    // because their args aren't host-knowable.
    //
    // Caller contract: the host mirror's contents must be valid + match
    // the device array on every kernel launch the pass might fire trace
    // for; if the host mirror drifts the synthesized DWQ pre-stage queues
    // stale args.  ASF updates `h_transfers` in lockstep with `d_transfers`
    // at Freeze time so this invariant holds.
    //
    // (elem_size, count) are recorded for future use (range-bounds checks,
    // batched alloc); the lookup helper currently just returns the host
    // pointer.
    //--------------------------------------------------------------------------
    void register_host_mirror(const void* dev_ptr, const void* host_ptr,
                              size_t /*elem_size*/, size_t /*count*/) {
        host_mirrors_[dev_ptr] = host_ptr;
    }

    const void* host_mirror_of(const void* dev_ptr) const {
        auto it = host_mirrors_.find(dev_ptr);
        return it == host_mirrors_.end() ? nullptr : it->second;
    }

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
        // Pool of IPC streams for host-driven dispatches. Non-blocking so
        // they overlap with user kernel + NIC RDMA. Pool size set by
        // n_streams_max_ (read from GICC_STREAMS_MAX or default 8).
        ipc_streams_.resize(n_streams_max_);
        for (int i = 0; i < n_streams_max_; i++) {
            if (gpuStreamCreateWithFlags(&ipc_streams_[i], gpuStreamNonBlocking)
                != GPU_SUCCESS) {
                fprintf(stderr, "GICC: gpuStreamCreate(ipc_streams_[%d]) failed\n", i);
                std::abort();
            }
        }
        host_wait_mode_ = true;
        // Couple host-wait (DWQ) mode with disabling the CPU-proxy
        // dispatch in DeviceCtx.  Without this, a kernel that calls
        // gicc::put would BOTH push a TransferCmd into the proxy ring
        // (CPU worker -> fi_write) AND let the lead-thread MMIO trigger
        // fire the host-pre-staged DWQ descriptors -- the peer receives
        // the same payload twice.  By writing nullptr to proxy_ring /
        // proxy_rings_arr at the next prepare(), gicc::put / quiet
        // short-circuit to no-op on the device side, letting a single
        // unified source kernel correctly serve both proxy and DWQ
        // paths (the mode is selected here, not inside the kernel).
        proxy_dispatch_disabled_ = true;
    }


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

    // ---- async DWQ staging -------------------------------------------------
    // fi_control(FI_QUEUE_WORK) costs ~10us per descriptor (a per-op CXI
    // driver call with no batched form), so the enqueue helpers push the
    // request here and a worker thread performs the staging off the
    // critical path. This is safe under the monotonic-threshold scheme:
    // staging EARLY cannot fire an op prematurely (its threshold is above
    // the counter), and staging LATE is caught by libfabric's triggered-op
    // rule that a descriptor whose threshold is already satisfied executes
    // immediately. reset() drains the queue before waiting on completions
    // and before recycling builders. Opt-in via GICC_DWQ_ASYNC_STAGE=1:
    // async staging only pays off when a concurrently running kernel or
    // enough per-op wire time hides the worker's serial FI_QUEUE_WORK
    // calls; with nothing to hide behind, the per-op handoff adds cost
    // (measured: batch_e2e n=64 sync 740us vs async 896us, while the
    // jacobi halo improved 50.3->46.9us and its 4-byte launch delta
    // dropped below Proxy's, 26.7 vs 29.0us).
    struct DwqStageReq {
        DwqWorkBuilder    *dwq;
        struct fid_domain *domain;
        struct fid_ep     *ep;
        void              *src;
        void              *desc;
        std::size_t        size;
        fi_addr_t          dest;
        std::uint64_t      raddr;
        std::uint64_t      rkey;
        struct fid_cntr   *trig;
        struct fid_cntr   *comp;
        std::uint64_t      threshold;
    };
    std::thread               dwq_stage_thread_;
    std::mutex                dwq_stage_mu_;
    std::condition_variable   dwq_stage_cv_;
    std::deque<DwqStageReq>   dwq_stage_q_;
    bool                      dwq_stage_stop_      = false;
    int                       dwq_stage_async_     = -1;   // -1 = env unread
    std::uint64_t             dwq_stage_submitted_ = 0;    // producer thread only
    std::atomic<std::uint64_t> dwq_stage_done_{0};

    void dwq_stage_worker_() {
        std::unique_lock<std::mutex> lk(dwq_stage_mu_);
        for (;;) {
            dwq_stage_cv_.wait(lk, [&] {
                return dwq_stage_stop_ || !dwq_stage_q_.empty();
            });
            if (dwq_stage_q_.empty()) {
                if (dwq_stage_stop_) return;
                continue;
            }
            DwqStageReq r = dwq_stage_q_.front();
            dwq_stage_q_.pop_front();
            lk.unlock();
            r.dwq->queue_rma_write(r.domain, r.ep, r.src, r.desc, r.size,
                                   r.dest, r.raddr, r.rkey,
                                   r.trig, r.comp, r.threshold);
            dwq_stage_done_.fetch_add(1, std::memory_order_release);
            lk.lock();
        }
    }

    bool dwq_stage_async_enabled_() {
        if (dwq_stage_async_ < 0) {
            const char *e = std::getenv("GICC_DWQ_ASYNC_STAGE");
            dwq_stage_async_ = (e != nullptr && std::atoi(e) != 0) ? 1 : 0;
            if (dwq_stage_async_)
                dwq_stage_thread_ = std::thread([this] { dwq_stage_worker_(); });
        }
        return dwq_stage_async_ == 1;
    }

    void dwq_stage_push_(DwqStageReq r) {
        {
            std::lock_guard<std::mutex> lk(dwq_stage_mu_);
            dwq_stage_q_.push_back(r);
        }
        ++dwq_stage_submitted_;
        dwq_stage_cv_.notify_one();
    }

    void dwq_stage_drain_() {
        while (dwq_stage_done_.load(std::memory_order_acquire) <
               dwq_stage_submitted_)
            sched_yield();
    }

    void dwq_stage_shutdown_() {
        if (!dwq_stage_thread_.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(dwq_stage_mu_);
            dwq_stage_stop_ = true;
        }
        dwq_stage_cv_.notify_one();
        dwq_stage_thread_.join();
    }

#ifdef GICC_CPU_PROXY
    // Snapshot the proxy-ring head once, then spin until the worker advances
    // tail past it. Pairs with put_no_db's atomic_push on the device side.
    // No-op when the proxy thread was never started (no kernel has called
    // ensure_proxy_ring() yet).
    void drain_proxy_ring_() {
        if (proxy_threads_.empty()) return;
        // Snapshot all rings' heads first, then spin per ring until tail
        // catches up. Snapshot-then-wait gives "all puts that were issued
        // before reset() returned will have been ack'd by libfabric".
        std::vector<uint64_t> snaps;
        snaps.reserve(proxy_threads_.size());
        for (auto& pt : proxy_threads_) {
            snaps.push_back(pt->ring_host()->head_volatile());
        }
        for (size_t i = 0; i < proxy_threads_.size(); ++i) {
            auto* ring = proxy_threads_[i]->ring_host();
            while (ring->tail_volatile() < snaps[i]) {
#if defined(__x86_64__)
                __asm__ __volatile__("pause" ::: "memory");
#else
                __asm__ __volatile__("" ::: "memory");
#endif
            }
        }
    }
#endif

public:

    //--------------------------------------------------------------------------
    // put — queue an RMA WRITE.  Non-blocking; consumes one slot from the
    // pool.  Caller observes completion via rt.reset() (host_wait_mode) or
    // rt.wait(tok).  Renamed from put_no_db: under the unified put/get/quiet
    // API, every device + host put is non-blocking-initiated by construction;
    // the "_no_db" suffix only made sense in the legacy device API where
    // flush was the explicit doorbell.
    //--------------------------------------------------------------------------
    Token put(const Buffer& src, int dest_rank, int dest_buf_index,
              size_t size, size_t src_offset = 0, size_t dst_offset = 0)
    {
        const OfiBuffer& ob = local_bufs_.at(src.index);

        if ((int)my_n_ops_ >= POOL_SIZE) {
            fprintf(stderr,
                "gicc::Runtime::put: batch exceeds POOL_SIZE=%d. "
                "Call rt.reset() between batches or raise POOL_SIZE.\n",
                POOL_SIZE);
            exit(1);
        }

        // IPC fast path. Same-node peer with an IPC-mapped buffer →
        // issue a device-to-device gpuMemcpyAsync directly here (mirror
        // of what GICCDispatchLowering::emitIpcBody emits in the LTO
        // host trace). rt.reset() syncs ipc_streams_ so any outstanding
        // copy completes before the next barrier.
        //
        // Pre-Pattern-C deletion this branch relied on a monitor thread
        // + device-side ring to deliver the copy; that machinery was
        // removed in 25e08e2. Without the inline gpuMemcpyAsync below,
        // a non-LTO caller (bench_pingpong, put_two_rank_intranode) hits
        // the IPC peer fast-path and silently transfers ZERO bytes.
        if (ipc_fastpath_
            && dest_rank != comm_->rank()
            && local_peer_[dest_rank]
            && (int)peer_mapped_ptrs_[dest_rank].size() > dest_buf_index
            && peer_mapped_ptrs_[dest_rank][dest_buf_index] != nullptr)
        {
            void* peer_base = peer_mapped_ptrs_[dest_rank][dest_buf_index];
            void* dst       = static_cast<char*>(peer_base) + dst_offset;
            void* src       = static_cast<char*>(ob.ptr)    + src_offset;
            // ipc_streams_ may be empty if enable_host_wait_mode() was
            // not called (legacy path constructs streams in that setter).
            // Fall back to the default stream in that case.
            GpuStream_t s = ipc_streams_.empty() ? nullptr : ipc_streams_[0];
            (void)gpuMemcpyAsync(dst, src, size,
                                 gpuMemcpyDeviceToDevice, s);
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
                "gicc::Runtime::put: batch exceeds POOL_SIZE=%d. "
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
    //
    // FI_HMEM contract note: the local destination buffer was registered with
    // iface = FI_HMEM_ROCR / FI_HMEM_CUDA, so libfabric's completion event
    // for the queued fi_read implies the response payload has been committed
    // to GPU HBM. Host wait via fi_cntr_read on slots_[].completion_cntr is
    // therefore sufficient — subsequent SM reads see the new data.
    //
    // The chained queue_atomic_signal that publishes a GPU-visible "done"
    // slot is currently disabled by default (GICC_GET_ENABLE_ATOMIC_SIGNAL
    // opt-in). The atomic-signal path through the cxi provider, with both
    // operand and target on FI_HMEM_ROCR memory and a loopback fi_addr_t,
    // segfaults inside fi_control(FI_QUEUE_WORK) on the Tioga/Slingshot
    // stack we ship on (libfabric 2.1 + cxi). Bypassing it costs us a
    // kernel-side mid-flight quiet (the kernel cannot poll completion
    // without leaving via host wait), but every host-driven flow (legacy
    // wait/reset, future LTO-driven post-launch host wait) is unaffected.
    Token get(const Buffer& local_dst, int src_rank, int src_buf_index,
              size_t size, size_t local_offset = 0, size_t remote_offset = 0)
    {
        const OfiBuffer& ob = local_bufs_.at(local_dst.index);

        // IPC fast path. Mirrors put_no_db: same-node peer with IPC-mapped
        // src buffer → pull via gpuMemcpyDeviceToDevice on an IPC stream.
        // The peer's buffer at peer_mapped_ptrs_[src_rank][src_buf_index]
        // is the GPU-address-space mapping of rank src_rank's d_buf.
        if (src_rank != comm_->rank()
            && (size_t)src_rank < local_peer_.size()
            && local_peer_[src_rank]
            && (int)peer_mapped_ptrs_[src_rank].size() > src_buf_index
            && peer_mapped_ptrs_[src_rank][src_buf_index] != nullptr)
        {
            void* peer_base = peer_mapped_ptrs_[src_rank][src_buf_index];
            void* src       = static_cast<char*>(peer_base) + remote_offset;
            void* dst       = static_cast<char*>(ob.ptr)    + local_offset;
            GpuStream_t s   = ipc_streams_.empty() ? nullptr : ipc_streams_[0];
            (void)gpuMemcpyAsync(dst, src, size,
                                 gpuMemcpyDeviceToDevice, s);
            return Token{ -1, /*is_local=*/true };
        }

        // Host-wait-mode fast path: mirror put_no_db's host-wait branch
        // (mono_total_ops_ accounting, shared_completion_cntr_, no
        // atomic_signal, recycled DwqWorkBuilder from the pool). This
        // matches the timing model that DWQ PUT uses in bench_pingpong
        // and the LTO-generated host trace.
        if (host_wait_mode_) {
            RemoteInfo ri = comm_->get_remote_info(src_rank, src_buf_index);
            if (ri.rma_key == 0 && ri.rma_addr == 0) {
                fprintf(stderr, "gicc::Runtime: remote info not set for rank %d "
                        "buf %d (call exchange() first)\n",
                        src_rank, src_buf_index);
                exit(1);
            }
            const uint64_t remote_addr = comm_->is_virt_addr_mode()
                ? (ri.rma_addr + remote_offset)
                : (ri.rma_addr - ri.base_addr) + remote_offset;

            ++mono_total_ops_;
            ++my_n_remote_ops_;
            DwqWorkBuilder* dwq = dwq_get_();
            dwq->queue_rma_read(
                comm_->fabric->domain, comm_->fabric->ep,
                (char*)ob.ptr + local_offset, ob.desc_, size,
                comm_->av_addrs[src_rank], remote_addr, ri.rma_key,
                comm_->fabric->trigger_cntr,
                shared_completion_cntr_,
                /*threshold=*/mono_total_ops_);
            // NO atomic_signal queued — same saving as host-wait PUT.
            my_pending_.push_back(dwq);
            return Token{ (int)mono_total_ops_, /*is_local=*/false };
        }

        if ((int)my_n_ops_ >= POOL_SIZE) {
            fprintf(stderr,
                "gicc::Runtime::get: batch exceeds POOL_SIZE=%d. "
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

        // GPU-visible completion signal via chained atomic. Currently
        // gated behind opt-in env var because cxi 2.1 / Slingshot fails
        // inside fi_control(FI_QUEUE_WORK) when the atomic targets
        // FI_HMEM_ROCR memory with a loopback fi_addr_t. Host wait via
        // slots_[slot_idx].completion_cntr is unaffected (the GET's own
        // completion bumps it).
        if (std::getenv("GICC_GET_ENABLE_ATOMIC_SIGNAL") != nullptr) {
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
            atomic_signals_queued_ = true;
        }

        my_pending_.push_back(dwq);

        return Token{ slot_idx, /*is_local=*/false };
    }

    //--------------------------------------------------------------------------
    // prepare — finalize a batched put_no_db sequence.
    //--------------------------------------------------------------------------
    DeviceCtx* prepare(int peer_rank = -1, int remote_buf_index = -1) {
        (void)peer_rank;
        (void)remote_buf_index;

        // Y' (host-driven) lowering: the LTO host trace function has
        // already pre-staged DWQ writes / IPC memcpys via the runtime
        // helper C ABI. The kernel only needs the trigger MMIO addr
        // and the threshold value so its lead thread can fire all
        // queued DWQ ops with one volatile store at flush() time.
        //
        // CRITICAL: CXI's trigger counter MMIO write is ADD-on-write,
        // not SET-on-write — writing N increments the counter by N.
        // For correct DWQ semantics we must write the DELTA between
        // this iter's pending op count and the previous iter's
        // already-written total, so the counter ends up at exactly
        // mono_total_ops_ (the highest queued threshold).
        h_dev_ctx_->trigger_addr_ = comm_->get_trigger_addr();
        if (host_wait_mode_) {
            const uint64_t delta = mono_total_ops_ - mono_last_triggered_;
            h_dev_ctx_->trigger_val_  = delta;
            mono_last_triggered_ = mono_total_ops_;
        } else {
            h_dev_ctx_->trigger_val_ = my_n_remote_ops_;
        }
        // Locality-aware collectives: hand the device the peer IPC table.
        h_dev_ctx_->peer_ipc_base = d_peer_ipc_;
        h_dev_ctx_->ipc_n_bufs    = n_bufs_;
#ifdef GICC_CPU_PROXY
        // Lazy-start the CPU proxy fleet on first prepare(). Stash both
        // the single ring 0 (DeviceCtx::proxy_ring, for back-compat with
        // single-ring kernels) and the full N-element array of device
        // ring pointers (DeviceCtx::proxy_rings_arr, for kernels that
        // shard across rings — pick by warp_id / blockIdx etc.).
        //
        // Skip when proxy dispatch is disabled (see enable_host_wait_mode):
        // a DWQ-mode kernel must not also push to the proxy ring, or the
        // peer ends up receiving the payload twice.  Nulling the fields
        // makes the inline gicc::put / quiet device bodies short-circuit.
        if (proxy_dispatch_disabled_) {
            h_dev_ctx_->proxy_ring       = nullptr;
            h_dev_ctx_->proxy_rings_arr  = nullptr;
            h_dev_ctx_->num_proxy_rings  = 0;
        } else {
            h_dev_ctx_->proxy_ring       = ensure_proxy_ring();
            h_dev_ctx_->proxy_rings_arr  = ensure_proxy_rings();
            h_dev_ctx_->num_proxy_rings  = num_proxy_rings();
        }
#endif
        return d_dev_ctx_;
    }

    //--------------------------------------------------------------------------
    // prepare_trigger — overlap pattern (flush only, host polls later).
    //--------------------------------------------------------------------------
    DeviceCtx* prepare_trigger(Token /*tok*/) {
        h_dev_ctx_->trigger_addr_ = comm_->get_trigger_addr();
        h_dev_ctx_->trigger_val_  = my_n_remote_ops_;
#ifdef GICC_CPU_PROXY
        h_dev_ctx_->proxy_ring = proxy_dispatch_disabled_
            ? nullptr
            : ensure_proxy_ring();
#endif
        return d_dev_ctx_;
    }

    //--------------------------------------------------------------------------
    // Host-side wait for a specific token.
    //--------------------------------------------------------------------------
    void wait(Token tok) {
        // IPC-routed ops are issued on ipc_streams_[0] inside put_no_db /
        // get_no_db (an async gpuMemcpyAsync). Sync that stream so the
        // local destination buffer is observable when wait() returns.
        // A naked gpuDeviceSynchronize from the caller would also work,
        // but waiting on the specific IPC stream lets wait() preserve
        // its "only this op" semantics. The cross-node path polls the
        // per-slot libfabric counter as before.
        if (tok.is_local) {
            if (!ipc_streams_.empty()) {
                (void)gpuStreamSynchronize(ipc_streams_[0]);
            }
            return;
        }
        while (fi_cntr_read(slots_[tok.slot_idx].completion_cntr) < 1) {}
    }

    //--------------------------------------------------------------------------
    // reset — drain all in-flight communication for this batch.
    //
    // CALLER CONTRACT: any kernel that issued put_no_db / quiet via this
    // Runtime MUST be device-synchronized (cudaDeviceSynchronize() /
    // hipDeviceSynchronize()) BEFORE calling reset(). The proxy-ring drain
    // snapshots the producer head ONCE at entry; if device work is still
    // publishing pushes after that snapshot, those pushes complete
    // asynchronously past reset() and may race with subsequent operations
    // (e.g. MPI_Barrier).
    //
    // Typical call sequence:
    //     gicc::launch<kernel>(rt, ..., args...);
    //     hipDeviceSynchronize();        // or cudaDeviceSynchronize();
    //     rt.reset();
    //     MPI_Barrier(MPI_COMM_WORLD);
    //
    // What reset() actually does:
    //   - Polls the shared completion counter against mono_total_ops_
    //     (host-wait fast path) or every per-slot counter (legacy path)
    //     until libfabric reports every queued op complete.
    //   - Drains the CPU proxy ring (snapshot head, spin until tail
    //     catches up) when GICC_CPU_PROXY is enabled.
    //   - Synchronizes any IPC streams used by the host trace.
    //   - Recycles slots / DwqWorkBuilders for the next batch.
    //--------------------------------------------------------------------------
    void reset() {
        if (host_wait_mode_) {
            // Async staging must be fully submitted before we wait on
            // completions or recycle builders.
            dwq_stage_drain_();
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

#ifdef GICC_CPU_PROXY
            // Drain the CPU proxy ring: snapshot the producer head at this
            // moment, then spin until tail catches up. Guarantees every
            // command the kernel emitted before this reset() has been
            // libfabric-submitted AND CQ-acked by the proxy worker.
            drain_proxy_ring_();
#endif

            // IPC handoff: the LTO host trace dispatched any same-node
            // ops via hipMemcpyAsync on ipc_streams_ BEFORE the kernel
            // launch, so draining all streams guarantees all peer
            // writes are committed before the upcoming MPI_Barrier.
            for (auto s : ipc_streams_) {
                if (s) (void)gpuStreamSynchronize(s);
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
#ifdef GICC_CPU_PROXY
        // Same drain as the host-wait path. Doing it AFTER the slot-counter
        // wait is fine: the proxy worker's progress doesn't depend on those
        // counters; the snapshot+spin only blocks on its own ring.
        drain_proxy_ring_();
#endif
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

    // progress — pump the libfabric CQ once (reads 0 entries: progress only,
    // consumes nothing). DWQ mode has no proxy worker thread, so a rank that
    // only SENT via same-node IPC but is RECEIVING an incoming cross-node RMA
    // write must call this to service the incoming write; otherwise its
    // FI_HMEM target DMA stalls the GPU SDMA engine and a concurrent outgoing
    // IPC gpuMemcpyAsync (>=~32KB, which uses SDMA) deadlocks. Callers that
    // block on the device (hipDeviceSynchronize / hipStreamSynchronize) while
    // such a write is in flight should interleave progress() instead.
    void progress() {
        if (comm_ && comm_->fabric)
            (void)fi_cq_read(comm_->fabric->cq, NULL, 0);
    }

    // First IPC dispatch stream (the one Runtime::put uses for the same-node
    // fast path), or nullptr if host-wait mode was never enabled. Lets a
    // caller poll it with hipStreamQuery while interleaving progress().
    GpuStream_t ipc_stream0() const {
        return ipc_streams_.empty() ? nullptr : ipc_streams_[0];
    }

    // Enable/disable the same-node IPC fast path in put() (default enabled).
    // Disable it for host-orchestrated collectives that would otherwise mix
    // same-node IPC copies with concurrent cross-node DWQ writes (SDMA-engine
    // deadlock on AMD+CXI). See ipc_fastpath_.
    void set_ipc_fastpath(bool on) { ipc_fastpath_ = on; }

    int rank()        const { return comm_->rank(); }
    int size()        const { return comm_->size(); }
    int window_size() const { return window_size_; }
    Bootstrap& boot() noexcept { return boot_; }
    const Bootstrap& boot() const noexcept { return boot_; }
    int gpu_id() const { return comm_->gpu_id(); }

    Fabric& fabric() { return *comm_; }

    //--------------------------------------------------------------------------
    // CPU Proxy accessors (Task 5).
    //
    // ProxyLibfabric needs to dereference local-buffer descriptors and remote
    // (rma_addr, rma_key, base_addr) tuples that already live on Runtime.
    // These thin wrappers expose that state without leaking the OfiBuffer /
    // RemoteInfo / Fabric internals to proxy code.
    //--------------------------------------------------------------------------
    struct LocalBufView { void* ptr; void* desc; };

    LocalBufView local_buf_view(int idx) const {
        if (idx < 0 || (size_t)idx >= local_bufs_.size()) {
            fprintf(stderr,
                "Runtime::local_buf_view: idx %d out of range (%zu)\n",
                idx, local_bufs_.size());
            std::abort();
        }
        const auto& ob = local_bufs_[idx];
        return LocalBufView{ ob.ptr, ob.desc_ };
    }

    const RemoteInfo& remote_info(int rank, int buf_idx) const {
        if (rank < 0 || buf_idx < 0 ||
            (size_t)((size_t)rank * (size_t)n_bufs_ + (size_t)buf_idx)
                >= remote_info_cache_.size()) {
            fprintf(stderr,
                "Runtime::remote_info: oob rank=%d buf=%d (n_bufs=%d)\n",
                rank, buf_idx, n_bufs_);
            std::abort();
        }
        return remote_info_cache_[(size_t)rank * (size_t)n_bufs_
                                  + (size_t)buf_idx];
    }

    bool      is_virt_addr_mode() const { return comm_->is_virt_addr_mode(); }
    fi_addr_t av_addr(int rank)   const { return comm_->av_addrs.at(rank); }

    // Same-node IPC accessors.  Exposes the local_peer / peer_mapped
    // tables built by exchange() so an upper-layer collective can emit
    // in-kernel direct GPU stores to peer-IPC-mapped pointers (NVSHMEM-
    // style) instead of routing same-node traffic through the proxy +
    // libfabric path.  Both must NOT be queried before exchange().
    // Returns nullptr for off-node peers or unmapped slots.
    bool is_local_peer(int rank) const {
        return rank >= 0 && (size_t)rank < local_peer_.size() && local_peer_[rank];
    }
    void* peer_mapped(int rank, int buf_idx) const {
        if (rank < 0 || (size_t)rank >= peer_mapped_ptrs_.size()) return nullptr;
        const auto& pm = peer_mapped_ptrs_[rank];
        if (buf_idx < 0 || (size_t)buf_idx >= pm.size()) return nullptr;
        return pm[buf_idx];
    }

#ifdef GICC_CPU_PROXY
    //--------------------------------------------------------------------------
    // ensure_proxy_rings — lazy-construct + start N CPU proxy workers, where
    // N was decided at Runtime construction time (GICC_NUM_PROXY_THREADS,
    // default 1, range [0,32]). Each worker submits/polls on its own
    // (fi_endpoint, fi_cq) opened by Fabric::create_proxy_endpoints in the
    // Runtime ctor, so completions never cross threads. Returns a
    // device-mapped pointer to an array of N ring pointers. Idempotent.
    //--------------------------------------------------------------------------
    void** ensure_proxy_rings() {
        std::lock_guard<std::mutex> g(proxy_init_mutex_);
        if (!proxy_threads_.empty()) return proxy_rings_arr_dev_;

        int n = n_proxy_threads_;
        if (n != comm_->num_proxy_eps()) {
            // Should be unreachable: ctor sized the EP fleet to match.
            fprintf(stderr,
                "[gicc] proxy fleet/EP count mismatch (threads=%d eps=%d)\n",
                n, comm_->num_proxy_eps());
            std::abort();
        }
        proxy_threads_.reserve(n);
        for (int i = 0; i < n; ++i) {
            proxy_threads_.push_back(
                std::make_unique<gicc::proxy::ProxyThread>(*this, /*ep_idx=*/i));
            proxy_threads_.back()->start();
        }
        // Allocate a host-pinned, GPU-mapped array of N device-ring pointers
        // so the kernel can index proxy_rings_arr[ring_idx] without a host
        // round-trip. Pinned makes the GPU-side load coherent on GH200.
        size_t arr_bytes = sizeof(void*) * n;
        if (gpuHostMalloc(&proxy_rings_arr_host_, arr_bytes,
                          gpuHostMallocMapped) != GPU_SUCCESS) {
            fprintf(stderr,
                "[gicc] gpuHostMalloc(proxy_rings_arr) failed for N=%d\n", n);
            std::abort();
        }
        for (int i = 0; i < n; ++i) {
            proxy_rings_arr_host_[i] = proxy_threads_[i]->ring_device();
        }
        if (gpuHostGetDevicePointer(reinterpret_cast<void**>(&proxy_rings_arr_dev_),
                                    proxy_rings_arr_host_, 0) != GPU_SUCCESS) {
            fprintf(stderr,
                "[gicc] gpuHostGetDevicePointer(proxy_rings_arr) failed\n");
            std::abort();
        }
        if (rank() == 0) {
            fprintf(stderr,
                "[gicc] CPU proxy: %d worker thread(s) per rank\n", n);
        }
        return proxy_rings_arr_dev_;
    }

    int num_proxy_rings() const {
        return static_cast<int>(proxy_threads_.size());
    }

    // Backward-compat single-ring helper. Returns ring 0; existing examples
    // (L1/L2/L4/L5) that use a single ring keep working unchanged.
    gicc::proxy::ProxyRing* ensure_proxy_ring() {
        ensure_proxy_rings();
        return proxy_threads_[0]->ring_device();
    }
#endif

private:
    // Internal buffer metadata (replaces gda::Buffer).
    struct OfiBuffer {
        void*             ptr;
        void*             desc_;
        uint64_t          key_;
        uint64_t          addr_;
        bool              has_ipc_handle = false;
        GpuIpcMemHandle_t ipc_handle{};
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

    // Flat, device-readable (host-pinned mapped) table of peer IPC bases for
    // locality-aware collectives. Indexed [peer * n_bufs_ + buf]; built in
    // exchange(), pointed to by DeviceCtx::peer_ipc_base in prepare().
    void**                             h_peer_ipc_ = nullptr;   // host vaddr
    void**                             d_peer_ipc_ = nullptr;   // device-mapped

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
    // When false, put() skips the same-node IPC gpuMemcpyAsync fast path and
    // routes same-node peers through the DWQ/NIC like any remote peer. Default
    // true. Needed by host-orchestrated collectives that mix same-node IPC
    // copies with concurrent cross-node DWQ writes: on AMD+CXI the IPC copy
    // (GPU SDMA engine, >=~32KB) and an incoming one-sided RMA write (NIC
    // FI_HMEM DMA, same SDMA engine) deadlock when both are in flight on one
    // GPU and there is no proxy worker thread progressing libfabric. Forcing
    // a single uniform transport avoids the contention.
    bool                               ipc_fastpath_ = true;
    // Set by enable_host_wait_mode().  When true, prepare() writes nullptr
    // to DeviceCtx::proxy_ring / proxy_rings_arr so device-side gicc::put
    // and gicc::quiet bodies short-circuit -- prevents double-send when a
    // unified kernel source serves both proxy + DWQ paths.  The CPU proxy
    // fleet is not lazy-started either (ensure_proxy_rings is bypassed).
    bool                               proxy_dispatch_disabled_;
    // Host-side mirrors of device arrays whose contents the LTO pass needs
    // to read at host trace synthesis time.  ASF's transfer descriptors,
    // for example, live as gs->d_transfers (GPU) with a host-side mirror
    // at gs->h_transfers; pass-synthesized DWQ trace calls
    // gicc_runtime_host_mirror_of(rt, dev_ptr) to translate the device
    // formal back to the host array so it can pre-stage one DWQ descriptor
    // per element without depending on a kernel-formal-only HK arg path.
    // See [[asf-rtm-pass-driven-dwq]] for the design rationale.
    std::unordered_map<const void*, const void*> host_mirrors_;
    struct fid_cntr*                   shared_completion_cntr_;
    uint64_t                           mono_total_ops_;          // monotonic across batches
    uint64_t                           mono_last_triggered_;     // last value the kernel's MMIO write added (for delta calc)
    std::vector<DwqWorkBuilder*>       dwq_pool_;                // recycled builders

    // Sliding-window depth passed to Barrier construction. Default 8;
    // overridden by GICC_WINDOW env var (range [1, 32]) or a future
    // hint.json from the dispatch-lowering pass.
    int                                window_size_    = 8;

    // Pool of streams for host-driven IPC dispatches. The LTO host
    // trace queues hipMemcpyAsync's on these streams BEFORE the kernel
    // launch (Y' / GDA-style); rt.reset() drains all of them.
    // Pool size is controlled by GICC_STREAMS_MAX (default 8, range [1,32]).
    std::vector<GpuStream_t>           ipc_streams_;
    int                                n_streams_max_  = 8;

    // Flat cache of RemoteInfo (av_addr / rma_addr / rma_key / base_addr)
    // for every (peer, buf) pair, populated during exchange(). Avoids
    // an unordered_map lookup + RemoteInfo copy on every put_no_db.
    // Indexed [peer * n_bufs_ + buf_idx].
    std::vector<RemoteInfo>            remote_info_cache_;
    int                                n_bufs_         = 0;

#ifdef GICC_CPU_PROXY
    // Number of proxy threads in the fleet. Decided at Runtime construction
    // (GICC_NUM_PROXY_THREADS env var, default 1, clamped to [0,32]); the
    // matching number of fi_endpoints + fi_cqs is opened in Fabric at the
    // same time so MR registration can fi_mr_bind to all of them.
    int                                                    n_proxy_threads_ = 1;
    // Lazy-initialized fleet of CPU proxy workers. ensure_proxy_rings()
    // constructs them on demand. Count is n_proxy_threads_ (default 1 —
    // single-ring callers don't pay for unused workers, advanced callers
    // opt into a wider fleet via GICC_NUM_PROXY_THREADS).
    std::vector<std::unique_ptr<gicc::proxy::ProxyThread>> proxy_threads_;
    // Device-mapped pointer to an array of N ring pointers. Allocated
    // (host-pinned, GPU-mapped) lazily alongside the threads. Stored once
    // in DeviceCtx::proxy_rings_arr by prepare().
    void**                                                 proxy_rings_arr_host_ = nullptr;
    void**                                                 proxy_rings_arr_dev_  = nullptr;
    // Serializes lazy construction. Without it, two host threads racing
    // into prepare() could each see empty proxy_threads_, each construct
    // the fleet, and leak one of them.
    std::mutex                                             proxy_init_mutex_;
#endif
};

} // namespace gicc
