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
#include <unordered_map>

#include "gicc/gicc_types.hpp"
#include "gicc/platform/mlx5/devx_qp.hpp"
#include "gicc/util/memory_region.hpp"
#include "gicc/bootstrap/bootstrap.hpp"

// Include gda_device_opt.cuh for DeviceStateOpt struct definition.
// Must come AFTER mlx5dv.h (included by mlx5_devx_qp.hpp) to avoid
// macro conflicts with MLX5 enum constants.
#include "gicc/platform/mlx5/device_opt.cuh"

// Simplified GPU context (NVSHMEM-style API)
#include "gicc/platform/mlx5/gicc_context.cuh"

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
    Runtime() {
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

        // Create one DevX QP per peer
        for (int i = 0; i < boot_.size(); i++) {
            if (i == boot_.rank()) continue;
            peer_qps_[i] = new gicc::mlx5::DevxQp(
                ib_ctx_, pd_, boot_.rank(), 1, 256, 512);
        }

        connect_peers();
    }

    ~Runtime() {
        reset();

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
        for (auto* d : device_ctxs_) cudaFree(d);
        device_ctxs_.clear();
    }

    void barrier() { boot_.barrier(); }

    int rank() const { return boot_.rank(); }
    int size() const { return boot_.size(); }
    int gpu_id() const { return gpu_id_; }
    double clock_rate_khz() const { return clock_rate_khz_; }
    const char* gpu_name() const { return gpu_props_.name; }

    Bootstrap& boot() noexcept { return boot_; }
    const Bootstrap& boot() const noexcept { return boot_; }

private:
    gicc::Bootstrap boot_;
    struct ibv_context* ib_ctx_ = nullptr;
    struct ibv_pd* pd_ = nullptr;

    std::unordered_map<int, gicc::mlx5::DevxQp*> peer_qps_;

    std::vector<gicc::MemoryRegion*> local_bufs_;
    std::vector<std::vector<RemoteBufferInfo>> remote_bufs_;
    std::vector<DeviceCtx*> device_ctxs_;

    int gpu_id_ = 0;
    double clock_rate_khz_ = 0;
    cudaDeviceProp gpu_props_;

    void open_ib_device() {
        int num_devices = 0;
        struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
        if (!dev_list || num_devices == 0) {
            fprintf(stderr, "GICC Rank %d: No IB devices found\n", boot_.rank());
            exit(1);
        }

        // Prefer mlx5_1 (IB port), fallback to first device
        struct ibv_device* target = nullptr;
        for (int i = 0; i < num_devices; i++) {
            const char* name = ibv_get_device_name(dev_list[i]);
            if (name && strcmp(name, "mlx5_1") == 0) {
                target = dev_list[i];
                break;
            }
        }
        if (!target) target = dev_list[0];

        ib_ctx_ = ibv_open_device(target);
        ibv_free_device_list(dev_list);

        if (!ib_ctx_) {
            fprintf(stderr, "GICC Rank %d: ibv_open_device failed\n", boot_.rank());
            exit(1);
        }
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
