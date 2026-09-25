/**
 * mlx5_runtime.hpp - MLX5 platform implementation of gicc::Runtime
 *
 * Handles all host-side setup for GPU-triggered RDMA over InfiniBand:
 *   - Bootstrap (MPI or PMI2, selected at build time)
 *   - GPU selection
 *   - IB device open, PD allocation
 *   - DevX QP creation per peer (GPU-accessible WQE buf, doorbell, BlueFlame)
 *   - QP connection (RST -> INIT -> RTR -> RTS)
 *   - Memory registration and buffer info exchange
 *   - DeviceCtx preparation for GPU kernels
 */
#pragma once

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <unordered_map>

#include "gicc/gicc_types.hpp"
#include "gicc/platform/mlx5/devx_qp.hpp"
#include "gicc/platform/mlx5/gda_engine.hpp"
#include "gicc/util/memory_region.hpp"
#include "gicc/bootstrap/bootstrap.hpp"

#ifdef GICC_CPU_PROXY
#include "gicc/platform/mlx5/proxy/proxy_thread.hpp"
#include <memory>
#include <mutex>
#endif

// DeviceStateOpt POD layout — the slim type-only header so this runtime
// header can be included from pure host C++ TUs (CPU proxy worker)
// without parsing the heavy __device__ helpers in device_opt.cuh.
// Must come AFTER mlx5dv.h (included by mlx5_devx_qp.hpp) to avoid
// macro conflicts with MLX5 enum constants.
#include "gicc/platform/mlx5/device_state_opt.hpp"

// Simplified GPU context — type-only slice, so this runtime header stays
// includable from pure host TUs (CPU proxy worker). The __device__ helper
// flavour (put / get / flush / quiet on GiccContext) lives in
// gicc_context.cuh and is pulled in by gicc/gicc_device.cuh for user
// kernels compiled by nvcc.
#include "gicc/platform/mlx5/gicc_context_types.hpp"

namespace gicc {

// DeviceCtx alias (same as in mlx5_device.cuh, repeated here so
// mlx5_runtime.hpp can be used without including device headers)
using DeviceCtx = gicc::mlx5::DeviceStateOpt;

inline void gicc_cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "GICC: %s failed: %s\n", what, cudaGetErrorString(err));
        gicc::abort(1, what);
    }
}

class Runtime {
public:
    // `legacy_qps` builds the per-peer DevxQp set behind prepare(peer, buf)
    // and build_context(). The GPU-driven API (enable_gda / prepare()) does
    // not use it, so GiOMP turns it off.
    explicit Runtime(bool legacy_qps = true) {
        // GPU setup
        int num_gpus = 0;
        gicc_cuda_check(cudaGetDeviceCount(&num_gpus), "cudaGetDeviceCount");
        if (num_gpus == 0) {
            fprintf(stderr, "GICC: No CUDA devices found\n");
            gicc::abort(1, "no CUDA devices");
        }
        gpu_id_ = boot_.local_rank() % num_gpus;
        gicc_cuda_check(cudaSetDevice(gpu_id_), "cudaSetDevice");
        gicc_cuda_check(cudaGetDeviceProperties(&gpu_props_, gpu_id_),
                        "cudaGetDeviceProperties");
        clock_rate_khz_ = gpu_props_.clockRate;

        // Open IB device
        open_ib_device();

        // Allocate protection domain
        pd_ = ibv_alloc_pd(ib_ctx_);
        if (!pd_) {
            fprintf(stderr, "GICC Rank %d: ibv_alloc_pd failed\n", boot_.rank());
            exit(1);
        }

        if (legacy_qps) {
            // Create one DevX QP per peer
            for (int i = 0; i < boot_.size(); i++) {
                if (i == boot_.rank()) continue;
                peer_qps_[i] = new gicc::mlx5::DevxQp(
                    ib_ctx_, pd_, boot_.rank(), 1, 256, 512);
            }

            connect_peers();
        }

#ifdef GICC_CPU_PROXY
        // Decide proxy fleet width up front (workers are started lazily
        // by ensure_proxy_rings()). Mirrors the OFI side: default 1,
        // override via GICC_NUM_PROXY_THREADS, clamp to [1,32].
        n_proxy_threads_ = 1;
        if (const char* e = std::getenv("GICC_NUM_PROXY_THREADS")) {
            int parsed = std::atoi(e);
            if (parsed >= 1 && parsed <= 32) {
                n_proxy_threads_ = parsed;
            } else if (boot_.rank() == 0) {
                fprintf(stderr,
                    "[gicc] GICC_NUM_PROXY_THREADS=%s out of [1,32], "
                    "using default 1\n", e);
            }
        }
#endif
    }

    ~Runtime() {
        reset();

#ifdef GICC_CPU_PROXY
        // Stop + join workers (and release their per-thread CQ/QP fleets)
        // BEFORE we tear down MRs and the PD — the proxy QPs were created
        // against the same pd_ and may still hold references to it.
        proxy_threads_.clear();
        for (auto* mr : proxy_aux_mrs_) {
            if (mr) ibv_dereg_mr(mr);
        }
        proxy_aux_mrs_.clear();
        if (proxy_rings_arr_host_) {
            cudaFreeHost(proxy_rings_arr_host_);
            proxy_rings_arr_host_ = nullptr;
            proxy_rings_arr_dev_  = nullptr;
        }
#endif

        gda_.reset();

        for (auto* mr : local_bufs_) delete mr;
        local_bufs_.clear();

        for (auto& [peer, qp] : peer_qps_) delete qp;
        peer_qps_.clear();

        if (pd_) { ibv_dealloc_pd(pd_); pd_ = nullptr; }
        if (ib_ctx_) { ibv_close_device(ib_ctx_); ib_ctx_ = nullptr; }
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    Buffer register_buffer(void* buf, size_t size, bool is_device) {
        auto* mr = new gicc::MemoryRegion(pd_, buf, size, is_device, boot_.rank());
        int idx = (int)local_bufs_.size();
        local_bufs_.push_back(mr);
        return { buf, size, (uint64_t)buf, mr->lkey, mr->rkey, idx };
    }

    void exchange() {
        struct BufEntry { uint64_t addr; uint32_t rkey; };

        int n = (int)local_bufs_.size();
        std::vector<BufEntry> my_entries(n);
        for (int i = 0; i < n; i++) {
            my_entries[i] = { (uint64_t)local_bufs_[i]->buf, local_bufs_[i]->rkey };
        }

        auto raw = boot_.allgather(my_entries.data(), n * (int)sizeof(BufEntry));

        remote_bufs_.resize(boot_.size());
        for (int r = 0; r < boot_.size(); r++) {
            // All ranks must register the same number of buffers in the
            // same order. Catch mismatches here instead of reading OOB.
            if (raw[r].size() != n * sizeof(BufEntry)) {
                fprintf(stderr,
                    "GICC: exchange() rank %d expected %d buffers (%zu B), "
                    "peer %d sent %zu B\n",
                    boot_.rank(), n, n * sizeof(BufEntry), r, raw[r].size());
                gicc::abort(1, "exchange(): buffer count mismatch");
            }
            remote_bufs_[r].resize(n);
            const auto* entries = reinterpret_cast<const BufEntry*>(raw[r].data());
            for (int b = 0; b < n; b++) {
                remote_bufs_[r][b] = { entries[b].addr, entries[b].rkey };
            }
        }
        if (gda_) publish_tables();
    }

    RemoteBufferInfo remote_buffer(int rank, int buf_index) const {
        return remote_bufs_[rank][buf_index];
    }

    DeviceCtx* prepare(int peer_rank, int remote_buf_index) {
        auto it = peer_qps_.find(peer_rank);
        if (it == peer_qps_.end()) {
            fprintf(stderr, "GICC Rank %d: No QP for peer %d\n",
                    boot_.rank(), peer_rank);
            exit(1);
        }
        auto* qp = it->second;
        auto remote = remote_bufs_[peer_rank][remote_buf_index];

        DeviceCtx h_ctx;
        memset(&h_ctx, 0, sizeof(h_ctx));

        h_ctx.qpn = qp->qpn;
        h_ctx.nwqes = 1 << qp->log_wq_size;
        h_ctx.nwqes_mask = h_ctx.nwqes - 1;

        h_ctx.wqe_buf = qp->d_wq_buf;
        h_ctx.wqe_lkey = 0;
        h_ctx.dbrec = qp->d_dbrec;
        h_ctx.bf_reg = (volatile uint64_t*)qp->d_uar_reg;
        h_ctx.prod_idx = qp->d_prod_idx;

        h_ctx.cqe = (volatile gicc::mlx5::Cqe64Opt*)qp->d_cq_buf;
        h_ctx.ncqes = qp->num_cqe;
        h_ctx.ncqes_mask = qp->num_cqe - 1;

        h_ctx.remote_addr = remote.addr;
        h_ctx.remote_rkey = remote.rkey;

        h_ctx.batch_size = 32;
        h_ctx.batch_mask = 31;

        DeviceCtx* d_ctx = nullptr;
        cudaMalloc(&d_ctx, sizeof(DeviceCtx));
        cudaMemcpy(d_ctx, &h_ctx, sizeof(DeviceCtx), cudaMemcpyHostToDevice);
        device_ctxs_.push_back(d_ctx);

        return d_ctx;
    }

    /**
     * Build a simplified GiccContext for NVSHMEM-style device API.
     *
     * Call AFTER exchange().  Creates per-peer DeviceCtxs and populates the
     * buffer registry so GPU kernels can use gicc::put(ctx, dst, src, size, peer).
     *
     * @return GPU-allocated GiccContext pointer (pass to kernels)
     */
    GiccContext* build_context() {
        GiccContext h_ctx;
        memset(&h_ctx, 0, sizeof(h_ctx));
        h_ctx.my_rank = boot_.rank();
        h_ctx.num_peers = boot_.size();

        // Fill local buffer registry
        h_ctx.num_local_bufs = (int)local_bufs_.size();
        for (int i = 0; i < h_ctx.num_local_bufs && i < GICC_MAX_BUFS; i++) {
            h_ctx.local_bufs[i].addr = (uint64_t)local_bufs_[i]->buf;
            h_ctx.local_bufs[i].size = local_bufs_[i]->size;
            h_ctx.local_bufs[i].lkey = local_bufs_[i]->lkey;
            h_ctx.local_bufs[i].rkey = local_bufs_[i]->rkey;
        }

        // Fill remote buffer info and per-peer DeviceCtxs
        for (int peer = 0; peer < boot_.size() && peer < GICC_MAX_PEERS; peer++) {
            // Remote buffer entries
            for (int b = 0; b < h_ctx.num_local_bufs && b < GICC_MAX_BUFS; b++) {
                h_ctx.remote_bufs[peer][b].addr = remote_bufs_[peer][b].addr;
                h_ctx.remote_bufs[peer][b].rkey = remote_bufs_[peer][b].rkey;
            }

            // Per-peer DeviceCtx (reuse prepare() logic, pass buf 0 for default remote)
            if (peer == boot_.rank()) {
                h_ctx.peer_ctxs[peer] = nullptr;
            } else {
                h_ctx.peer_ctxs[peer] = prepare(peer, 0);
            }
        }

        GiccContext* d_ctx = nullptr;
        cudaMalloc(&d_ctx, sizeof(GiccContext));
        cudaMemcpy(d_ctx, &h_ctx, sizeof(GiccContext), cudaMemcpyHostToDevice);
        return d_ctx;
    }

    void reset() {
        if (gda_) drain();
        for (auto* d : device_ctxs_) cudaFree(d);
        device_ctxs_.clear();
    }

    //--------------------------------------------------------------------------
    // GPU-driven transport (see gda_engine.hpp).
    //
    // Collective. Builds one GPU-owned QP per (peer, lane) and one host QP
    // per peer. GICC_IB_LANES (default 1) and GICC_IB_QP_DEPTH (default
    // 1024 WQEs, at most 16384) override the arguments when set.
    //--------------------------------------------------------------------------
    void enable_gda(int lanes = 1, uint32_t depth = 1024) {
        if (gda_) return;
        if (const char* e = std::getenv("GICC_IB_LANES")) lanes = std::atoi(e);
        if (const char* e = std::getenv("GICC_IB_QP_DEPTH")) depth = (uint32_t)std::atoi(e);
        gda_ = std::make_unique<gicc::mlx5::GdaEngine>(boot_, ib_ctx_, pd_, lanes, depth);
        if (!remote_bufs_.empty()) publish_tables();
    }

    bool gda_enabled() const noexcept { return gda_ != nullptr; }

    // The GPU context kernels and target regions communicate through. Same
    // pointer for the life of the runtime; exchange() and set_* refresh the
    // contents.
    gicc::mlx5::GdaCtx* prepare() {
        require_gda("prepare()");
        return gda_->device_ctx();
    }

    void set_symmetric_heap(void* dev_base, int buf_index) {
        require_gda("set_symmetric_heap");
        gda_->set_heap(dev_base, buf_index);
    }

    void set_signal_table(void* dev_base, int buf_index) {
        require_gda("set_signal_table");
        gda_->set_signals(static_cast<uint64_t*>(dev_base), buf_index);
    }

    // Host-issued RDMA WRITE: local (src, src_offset) -> peer's
    // (dest_buf_index, dst_offset). Non-blocking; drain() completes it.
    void put(const Buffer& src, int dest_rank, int dest_buf_index,
             size_t size, size_t src_offset = 0, size_t dst_offset = 0) {
        require_gda("put");
        const auto& mr = *local_bufs_.at(src.index);
        const auto& rb = remote_bufs_.at(dest_rank).at(dest_buf_index);
        gda_->host_write(dest_rank, (uint64_t)mr.buf + src_offset, mr.lkey,
                         rb.addr + dst_offset, rb.rkey, size);
    }

    // Host-issued RDMA READ: peer's (src_buf_index, remote_offset) -> local
    // (local_dst, local_offset).
    void get(const Buffer& local_dst, int src_rank, int src_buf_index,
             size_t size, size_t local_offset = 0, size_t remote_offset = 0) {
        require_gda("get");
        const auto& mr = *local_bufs_.at(local_dst.index);
        const auto& rb = remote_bufs_.at(src_rank).at(src_buf_index);
        gda_->host_read(src_rank, (uint64_t)mr.buf + local_offset, mr.lkey,
                        rb.addr + remote_offset, rb.rkey, size);
    }

    // Write `value` into the 8 bytes at peer's (dest_buf_index, dst_offset),
    // ordered after every host put already issued to that peer.
    void put_u64(int dest_rank, int dest_buf_index, size_t dst_offset, uint64_t value) {
        require_gda("put_u64");
        const auto& rb = remote_bufs_.at(dest_rank).at(dest_buf_index);
        gda_->host_write_u64(dest_rank, rb.addr + dst_offset, rb.rkey, value);
    }

    // Complete every transfer this rank has issued, host- or GPU-posted.
    // GPU-posted work is only seen once the posting kernel has rung it;
    // call after the kernel (or target region) returns.
    void drain() {
        require_gda("drain");
        gda_->host_quiet();
#if defined(__CUDACC__)
        gda_->device_quiet();
#endif
    }

    // Reap host completions without blocking.
    void progress() {
        if (gda_) gda_->host_progress();
    }

    void barrier() { boot_.barrier(); }

    int rank() const { return boot_.rank(); }
    int size() const { return boot_.size(); }
    int gpu_id() const { return gpu_id_; }
    double clock_rate_khz() const { return clock_rate_khz_; }
    const char* gpu_name() const { return gpu_props_.name; }

    Bootstrap& boot() noexcept { return boot_; }
    const Bootstrap& boot() const noexcept { return boot_; }

    //--------------------------------------------------------------------------
    // Cross-backend accessor parity with the OFI runtime.
    //
    // These exist on the OFI side and are sometimes referenced by
    // portable user code. On MLX5 they have trivial answers because the
    // backend always uses RDMA (no same-node IPC fast path) and always
    // operates in virtual-address mode (no FI_MR_BASIC zero-based MR).
    // enable_host_wait_mode is an OFI-specific shared-counter
    // optimization that has no analog on MLX5; we accept the call to
    // keep portable bring-up paths compiling and treat it as a no-op.
    //--------------------------------------------------------------------------
    void  enable_host_wait_mode() noexcept {}
    bool  is_local_peer(int /*rank*/) const noexcept { return false; }
    void* peer_mapped (int /*rank*/, int /*buf_idx*/) const noexcept { return nullptr; }
    bool  is_virt_addr_mode() const noexcept { return true; }

#ifdef GICC_CPU_PROXY
    //--------------------------------------------------------------------------
    // CPU-proxy accessors consumed by mlx5::proxy::{ProxyThread, ProxyVerbs,
    // ProxyQpFleet}. We expose the shared ib_context + PD so the proxy QP
    // fleet can ibv_create_qp / ibv_modify_qp on it without re-opening the
    // device, and thin views over local/remote buffer registries so
    // ProxyVerbs can build SGEs from TransferCmd (rank, buf_idx, offset).
    //--------------------------------------------------------------------------
    struct ProxyLocalBuf  { uint64_t addr; uint32_t lkey; };
    struct ProxyRemoteBuf { uint64_t addr; uint32_t rkey; };

    ibv_context* proxy_ib_context() noexcept { return ib_ctx_; }
    ibv_pd*      proxy_pd()         noexcept { return pd_; }

    ProxyLocalBuf proxy_local_buf(int idx) const {
        const auto& mr = *local_bufs_.at(idx);
        return ProxyLocalBuf{ reinterpret_cast<uint64_t>(mr.buf), mr.lkey };
    }

    ProxyRemoteBuf proxy_remote_buf(int rank, int buf_idx) const {
        const auto& rb = remote_bufs_.at(rank).at(buf_idx);
        return ProxyRemoteBuf{ rb.addr, rb.rkey };
    }

    // Register a host buffer in the shared PD (used by ProxyVerbs for the
    // IBV_WR_ATOMIC_FETCH_AND_ADD scratch landing). Tracked here so it gets
    // ibv_dereg_mr()'d in the dtor before pd_ goes away.
    ibv_mr* proxy_register_host(void* ptr, size_t bytes) {
        ibv_mr* mr = ibv_reg_mr(
            pd_, ptr, bytes,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (mr) proxy_aux_mrs_.push_back(mr);
        return mr;
    }

    //--------------------------------------------------------------------------
    // Lazy-start the proxy fleet. The first call constructs n_proxy_threads_
    // ProxyThreads (each owning its own ring + CQ + per-peer QPs), spawns
    // their worker threads, and publishes a host-pinned device-mapped array
    // of the N ring pointers for kernels to fan out across. Idempotent.
    //--------------------------------------------------------------------------
    void** ensure_proxy_rings() {
        std::lock_guard<std::mutex> g(proxy_init_mutex_);
        if (!proxy_threads_.empty()) return proxy_rings_arr_dev_;

        const int n = n_proxy_threads_;
        proxy_threads_.reserve(n);
        for (int i = 0; i < n; ++i) {
            proxy_threads_.push_back(
                std::make_unique<gicc::mlx5::proxy::ProxyThread>(*this, i));
            proxy_threads_.back()->start();
        }

        const size_t arr_bytes = sizeof(void*) * n;
        if (cudaHostAlloc(reinterpret_cast<void**>(&proxy_rings_arr_host_),
                          arr_bytes, cudaHostAllocMapped) != cudaSuccess) {
            fprintf(stderr,
                "[gicc] cudaHostAlloc(proxy_rings_arr) failed for N=%d\n", n);
            std::abort();
        }
        for (int i = 0; i < n; ++i) {
            proxy_rings_arr_host_[i] = proxy_threads_[i]->ring_device();
        }
        if (cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&proxy_rings_arr_dev_),
                proxy_rings_arr_host_, 0) != cudaSuccess) {
            fprintf(stderr,
                "[gicc] cudaHostGetDevicePointer(proxy_rings_arr) failed\n");
            std::abort();
        }

        if (rank() == 0) {
            fprintf(stderr,
                "[gicc] mlx5 CPU proxy: %d worker thread(s) per rank\n", n);
        }
        return proxy_rings_arr_dev_;
    }

    // Single-ring back-compat helper. Returns ring 0's device pointer.
    void* ensure_proxy_ring() {
        ensure_proxy_rings();
        return proxy_threads_[0]->ring_device();
    }

    int num_proxy_rings() const {
        return static_cast<int>(proxy_threads_.size());
    }
#endif  // GICC_CPU_PROXY

private:
    gicc::Bootstrap boot_;
    struct ibv_context* ib_ctx_ = nullptr;
    struct ibv_pd* pd_ = nullptr;

    std::unordered_map<int, gicc::mlx5::DevxQp*> peer_qps_;

    std::vector<gicc::MemoryRegion*> local_bufs_;
    std::vector<std::vector<RemoteBufferInfo>> remote_bufs_;
    std::vector<DeviceCtx*> device_ctxs_;

    std::unique_ptr<gicc::mlx5::GdaEngine> gda_;

    int gpu_id_ = 0;
    double clock_rate_khz_ = 0;
    cudaDeviceProp gpu_props_;

#ifdef GICC_CPU_PROXY
    int                                                          n_proxy_threads_ = 1;
    std::vector<std::unique_ptr<gicc::mlx5::proxy::ProxyThread>> proxy_threads_;
    void**                                                       proxy_rings_arr_host_ = nullptr;
    void**                                                       proxy_rings_arr_dev_  = nullptr;
    std::mutex                                                   proxy_init_mutex_;
    // Auxiliary MRs registered by proxy_register_host() (e.g. atomic
    // fetch-and-add scratch landings). Owned by us; released before pd_
    // in the dtor.
    std::vector<ibv_mr*>                                         proxy_aux_mrs_;
#endif

    void open_ib_device() {
        ib_ctx_ = gicc::mlx5::gda_open_device(boot_.rank());
    }

    void require_gda(const char* what) const {
        if (!gda_) {
            fprintf(stderr, "GICC: %s needs Runtime::enable_gda() first\n", what);
            gicc::abort(1, what);
        }
    }

    // Hand the engine the address book in the flat form the GPU indexes.
    void publish_tables() {
        const int n = (int)local_bufs_.size();
        const int ranks = boot_.size();
        std::vector<uint64_t> laddr(n), raddr((size_t)ranks * n);
        std::vector<uint32_t> lkey(n), rkey((size_t)ranks * n);
        for (int b = 0; b < n; ++b) {
            laddr[b] = (uint64_t)local_bufs_[b]->buf;
            lkey[b]  = local_bufs_[b]->lkey;
        }
        for (int r = 0; r < ranks; ++r) {
            for (int b = 0; b < n; ++b) {
                raddr[(size_t)r * n + b] = remote_bufs_[r][b].addr;
                rkey[(size_t)r * n + b]  = remote_bufs_[r][b].rkey;
            }
        }
        gda_->set_tables(laddr, lkey, raddr, rkey);
    }

    void connect_peers() {
        struct ConnInfo {
            uint32_t qpn;
            uint16_t lid;
            uint8_t  gid[16];
            uint32_t psn;
        };

        struct ibv_port_attr port_attr;
        ibv_query_port(ib_ctx_, 1, &port_attr);

        union ibv_gid my_gid;
        ibv_query_gid(ib_ctx_, 1, 0, &my_gid);

        for (auto& [peer, qp] : peer_qps_) {
            ConnInfo my_info = {};
            my_info.qpn = qp->qpn;
            my_info.lid = port_attr.lid;
            memcpy(my_info.gid, &my_gid, 16);
            my_info.psn = 0;

            ConnInfo peer_info = {};
            boot_.sendrecv(&my_info, &peer_info, sizeof(ConnInfo), peer);

            qp->rst2init();
            qp->init2rtr(peer_info.qpn, peer_info.lid, peer_info.gid,
                         peer_info.psn);
            qp->rtr2rts(my_info.psn);
        }

        boot_.barrier();
    }
};

} // namespace gicc
