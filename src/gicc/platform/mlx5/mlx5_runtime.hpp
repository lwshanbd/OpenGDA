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
        transport_.reset();
        for (auto* mr : local_bufs_) delete mr;
        local_bufs_.clear();
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

    int rank() const { return boot_.rank(); }
    int size() const { return boot_.size(); }
    int gpu_id() const { return gpu_id_; }
    double clock_rate_khz() const { return clock_rate_khz_; }
    const char* gpu_name() const { return gpu_props_.name; }

    Bootstrap& boot() noexcept { return boot_; }
    const Bootstrap& boot() const noexcept { return boot_; }

    // Parity with the libfabric runtime for portable host code: every peer
    // is reached through the NIC and memory is addressed by virtual address.
    void  enable_host_wait_mode() noexcept {}
    bool  is_local_peer(int /*rank*/) const noexcept { return false; }
    void* peer_mapped (int /*rank*/, int /*buf_idx*/) const noexcept { return nullptr; }
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
