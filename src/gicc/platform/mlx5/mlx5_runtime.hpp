/**
 * mlx5_runtime.hpp - gicc::Runtime on NVIDIA GPUs + InfiniBand (verbs).
 *
 * Host-side setup for GPU-initiated RDMA:
 *   - Bootstrap (MPI or PMI2, selected at build time)
 *   - GPU selection
 *   - IB device open (DevX), PD allocation
 *   - the transport (transport.hpp): GPU-owned QPs that kernels and target
 *     regions post to directly, plus verbs QPs for host-issued transfers
 *   - memory registration and the buffer address book
 *   - CUDA IPC mappings of same-node peers' device buffers (peer_mapped),
 *     which need neither verbs nor the NIC
 *
 * Construction is collective: every rank builds its Runtime at the same point.
 * GICC_IB_LANES (default 1) sets the QPs per peer and GICC_IB_QP_DEPTH
 * (default 1024, at most 16384) their send-queue depth.
 */
#pragma once

#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "gicc/bootstrap/bootstrap.hpp"
#include "gicc/gicc_types.hpp"
#include "gicc/platform/mlx5/device_ctx.hpp"
#include "gicc/platform/mlx5/transport.hpp"
#include "gicc/util/memory_region.hpp"

namespace gicc {

inline void gicc_cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "GICC: %s failed: %s\n", what, cudaGetErrorString(err));
        gicc::abort(1, what);
    }
}

class Runtime {
public:
    Runtime() {
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

        ib_ctx_ = gicc::mlx5::open_device(boot_.rank());
        pd_ = ibv_alloc_pd(ib_ctx_);
        if (!pd_) {
            fprintf(stderr, "GICC Rank %d: ibv_alloc_pd failed\n", boot_.rank());
            gicc::abort(1, "ibv_alloc_pd");
        }

        int lanes = 1;
        uint32_t depth = 1024;
        if (const char* e = std::getenv("GICC_IB_LANES")) lanes = std::atoi(e);
        if (const char* e = std::getenv("GICC_IB_QP_DEPTH")) depth = (uint32_t)std::atoi(e);
        transport_ = std::make_unique<gicc::mlx5::Transport>(boot_, ib_ctx_, pd_,
                                                             lanes, depth);
    }

    ~Runtime() {
        reset();
        close_ipc();
        transport_.reset();
        for (auto* mr : local_bufs_) delete mr;
        local_bufs_.clear();
        if (pd_) { ibv_dealloc_pd(pd_); pd_ = nullptr; }
        if (ib_ctx_) { ibv_close_device(ib_ctx_); ib_ctx_ = nullptr; }
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // A device buffer is also exported for CUDA IPC, so same-node peers can
    // map it (peer_mapped). As on libfabric, the mapping is of the whole
    // allocation `buf` belongs to, so register allocation starts -- as the
    // GiOMP heap and signal inbox are -- when peers are to map them.
    Buffer register_buffer(void* buf, size_t size, bool is_device) {
        auto* mr = new gicc::MemoryRegion(pd_, buf, size, is_device, boot_.rank());
        int idx = (int)local_bufs_.size();
        local_bufs_.push_back(mr);
        IpcExport e{};
        if (is_device && cudaIpcGetMemHandle(&e.handle, buf) == cudaSuccess) e.valid = 1;
        else (void)cudaGetLastError();
        ipc_exports_.push_back(e);
        return { buf, size, (uint64_t)buf, mr->lkey, mr->rkey, idx };
    }

    // Collective: every rank must have registered the same number of buffers
    // in the same order. Refreshes the device address book.
    void exchange() {
        struct BufEntry { uint64_t addr; uint32_t rkey; };

        int n = (int)local_bufs_.size();
        std::vector<BufEntry> my_entries(n);
        for (int i = 0; i < n; i++) {
            my_entries[i] = { (uint64_t)local_bufs_[i]->buf, local_bufs_[i]->rkey };
        }

        auto raw = boot_.allgather(my_entries.data(), n * (int)sizeof(BufEntry));
        open_ipc();

        remote_bufs_.resize(boot_.size());
        for (int r = 0; r < boot_.size(); r++) {
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
        publish_tables();
    }

    RemoteBufferInfo remote_buffer(int rank, int buf_index) const {
        return remote_bufs_[rank][buf_index];
    }

    // The context kernels and target regions communicate through. Same
    // pointer for the life of the runtime; exchange() and set_* refresh the
    // contents.
    DeviceCtx* prepare() { return transport_->device_ctx(); }

    void set_symmetric_heap(void* dev_base, int buf_index) {
        transport_->set_heap(dev_base, buf_index);
    }

    void set_signal_table(void* dev_base, int buf_index) {
        transport_->set_signals(static_cast<uint64_t*>(dev_base), buf_index);
    }

    // Host-issued RDMA WRITE: local (src, src_offset) -> peer's
    // (dest_buf_index, dst_offset). Non-blocking; drain() completes it.
    void put(const Buffer& src, int dest_rank, int dest_buf_index,
             size_t size, size_t src_offset = 0, size_t dst_offset = 0) {
        const auto& mr = *local_bufs_.at(src.index);
        const auto& rb = remote_bufs_.at(dest_rank).at(dest_buf_index);
        transport_->host_write(dest_rank, (uint64_t)mr.buf + src_offset, mr.lkey,
                               rb.addr + dst_offset, rb.rkey, size);
    }

    // Host-issued RDMA READ: peer's (src_buf_index, remote_offset) -> local
    // (local_dst, local_offset).
    void get(const Buffer& local_dst, int src_rank, int src_buf_index,
             size_t size, size_t local_offset = 0, size_t remote_offset = 0) {
        const auto& mr = *local_bufs_.at(local_dst.index);
        const auto& rb = remote_bufs_.at(src_rank).at(src_buf_index);
        transport_->host_read(src_rank, (uint64_t)mr.buf + local_offset, mr.lkey,
                              rb.addr + remote_offset, rb.rkey, size);
    }

    // Write `value` into the 8 bytes at peer's (dest_buf_index, dst_offset),
    // ordered after every host put already issued to that peer.
    void put_u64(int dest_rank, int dest_buf_index, size_t dst_offset, uint64_t value) {
        const auto& rb = remote_bufs_.at(dest_rank).at(dest_buf_index);
        transport_->host_write_u64(dest_rank, rb.addr + dst_offset, rb.rkey, value);
    }

    // Complete every transfer this rank has issued, host- or GPU-posted.
    // GPU-posted work is seen once the posting kernel has returned.
    void drain() {
        transport_->host_quiet();
#if defined(__CUDACC__)
        transport_->device_quiet();
#endif
    }

    void reset() { if (transport_) drain(); }

    // Reap host completions without blocking.
    void progress() { transport_->host_progress(); }

    void barrier() { boot_.barrier(); }

    // Same-node peers and their device buffers mapped through CUDA IPC.
    // peer_mapped(rank, buf) is the local address of `rank`'s buffer `buf`,
    // or nullptr when `rank` is this rank, on another node, or the buffer is
    // not device memory. Valid after exchange().
    bool is_local_peer(int rank) const {
        return rank >= 0 && (size_t)rank < local_peer_.size() && local_peer_[rank];
    }
    void* peer_mapped(int rank, int buf_idx) const {
        if (rank < 0 || (size_t)rank >= peer_mapped_.size()) return nullptr;
        const auto& pm = peer_mapped_[rank];
        if (buf_idx < 0 || (size_t)buf_idx >= pm.size()) return nullptr;
        return pm[buf_idx];
    }

    int rank() const { return boot_.rank(); }
    int size() const { return boot_.size(); }
    int gpu_id() const { return gpu_id_; }
    double clock_rate_khz() const { return clock_rate_khz_; }
    const char* gpu_name() const { return gpu_props_.name; }

    Bootstrap& boot() noexcept { return boot_; }
    const Bootstrap& boot() const noexcept { return boot_; }

    // Parity with the libfabric runtime for portable host code: there is no
    // host-wait mode to enable, and memory is addressed by virtual address.
    void  enable_host_wait_mode() noexcept {}
    bool  is_virt_addr_mode() const noexcept { return true; }

private:
    gicc::Bootstrap boot_;
    ibv_context* ib_ctx_ = nullptr;
    ibv_pd* pd_ = nullptr;
    std::unique_ptr<gicc::mlx5::Transport> transport_;

    std::vector<gicc::MemoryRegion*> local_bufs_;
    std::vector<std::vector<RemoteBufferInfo>> remote_bufs_;

    int gpu_id_ = 0;
    double clock_rate_khz_ = 0;
    cudaDeviceProp gpu_props_;

    struct IpcExport {
        cudaIpcMemHandle_t handle;
        uint8_t            valid;
    };
    std::vector<IpcExport>          ipc_exports_;   // [buf], ours
    std::vector<bool>               local_peer_;    // [rank]
    std::vector<std::vector<void*>> peer_mapped_;   // [rank][buf]

    // Collective: exchange the IPC handles and map every same-node peer's
    // device buffers. Called by exchange(); earlier mappings are replaced.
    void open_ipc() {
        close_ipc();
        const int n = (int)ipc_exports_.size();
        auto all = boot_.allgather(ipc_exports_.data(), n * (int)sizeof(IpcExport));
        local_peer_ = boot_.locality_map();
        peer_mapped_.assign(boot_.size(), std::vector<void*>(n, nullptr));
        for (int r = 0; r < boot_.size(); ++r) {
            if (r == boot_.rank() || !local_peer_[r]) continue;
            const auto* peer = reinterpret_cast<const IpcExport*>(all[r].data());
            for (int b = 0; b < n; ++b) {
                if (!peer[b].valid) continue;
                void* mapped = nullptr;
                if (cudaIpcOpenMemHandle(&mapped, peer[b].handle,
                                         cudaIpcMemLazyEnablePeerAccess) == cudaSuccess) {
                    peer_mapped_[r][b] = mapped;
                } else {
                    (void)cudaGetLastError();   // stays unmapped: use the NIC
                }
            }
        }
    }

    void close_ipc() {
        for (auto& per_rank : peer_mapped_)
            for (void* p : per_rank)
                if (p) (void)cudaIpcCloseMemHandle(p);
        peer_mapped_.clear();
    }

    // Hand the transport the address book in the flat form the GPU indexes.
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
        transport_->set_tables(laddr, lkey, raddr, rkey);
    }
};

} // namespace gicc
