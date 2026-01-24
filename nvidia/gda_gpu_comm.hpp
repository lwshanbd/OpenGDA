/**
 * gda_gpu_comm.hpp - GPU-Triggered Communication API
 *
 * High-level API for GPU-initiated RDMA operations using MLX5 DevX.
 * Unlike the CPU-triggered version (gda_comm.hpp), this allows
 * GPU kernels to directly trigger RDMA without CPU involvement.
 *
 * Usage:
 *   GdaGpuComm comm;
 *   auto handle = comm.register_buffer(gpu_buf, size);
 *   comm.exchange_buffer_info(handle);
 *
 *   // From GPU kernel:
 *   gda_build_rdma_write_wqe(...);
 *   gda_ring_doorbell(...);
 *   gda_poll_cq(...);
 */
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>

#include "mpi_bootstrap.hpp"
#include "mlx5_gda_context.hpp"
#include "memory_region.hpp"
#include "gda_device.cuh"

namespace opengda {

#define CUDA_CHECK(cmd) do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Handle to registered buffer
struct GdaGpuHandle {
    void* buf;
    size_t size;
    MemoryRegion* mr;
    bool is_device;
};

class GdaGpuComm {
public:
    MpiBootstrap mpi;
    Mlx5GdaContext* mlx5;

    // CUDA context
    int gpu_id;
    cudaDeviceProp props;

    // Registered buffers
    std::vector<MemoryRegion*> registered_mrs;

    // Remote buffer info: (rank, buf_index) -> {addr, rkey}
    struct RemoteInfo {
        uint64_t addr;
        uint32_t rkey;
    };
    std::unordered_map<uint64_t, RemoteInfo> remote_info;

    // Device state (GPU-accessible)
    GdaDeviceState* d_state;
    GdaDeviceState h_state;

    // Completion counter (GPU-accessible)
    volatile uint64_t* h_num_completions;
    volatile uint64_t* d_num_completions;

    explicit GdaGpuComm(int local_rank = -1)
        : mlx5(nullptr), gpu_id(0), d_state(nullptr),
          h_num_completions(nullptr), d_num_completions(nullptr)
    {
        // Determine GPU to use
        gpu_id = (local_rank >= 0) ? local_rank : mpi.local_rank;

        int num_gpus = 0;
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        if (num_gpus > 0) {
            gpu_id = gpu_id % num_gpus;
        }

        CUDA_CHECK(cudaSetDevice(gpu_id));
        CUDA_CHECK(cudaGetDeviceProperties(&props, gpu_id));

        // Initialize MLX5 context
        mlx5 = new Mlx5GdaContext(mpi);

        // Exchange connection info and connect
        exchange_and_connect();

        // Setup device state
        setup_device_state();

        if (mpi.rank == 0) {
            printf("GdaGpuComm initialized:\n");
            printf("  GPU: %s\n", props.name);
            printf("  IB device: %s\n", mlx5->dev_name.c_str());
            printf("  Rank %d of %d\n", mpi.rank, mpi.size);
            fflush(stdout);
        }
    }

    ~GdaGpuComm() {
        if (d_state) cudaFree(d_state);
        if (h_num_completions) cudaFreeHost((void*)h_num_completions);
        for (auto* mr : registered_mrs) delete mr;
        delete mlx5;
    }

    // No copy
    GdaGpuComm(const GdaGpuComm&) = delete;
    GdaGpuComm& operator=(const GdaGpuComm&) = delete;

    /**
     * Register a buffer for RDMA
     */
    GdaGpuHandle register_buffer(void* buf, size_t size, bool is_device) {
        auto* mr = new MemoryRegion(mlx5->pd, buf, size, is_device, mpi.rank);
        registered_mrs.push_back(mr);

        GdaGpuHandle handle;
        handle.buf = buf;
        handle.size = size;
        handle.mr = mr;
        handle.is_device = is_device;
        return handle;
    }

    /**
     * Exchange buffer info with all peers
     */
    void exchange_buffer_info(const GdaGpuHandle& handle, int buf_index = 0) {
        struct BufInfo {
            uint64_t addr;
            uint32_t rkey;
        };

        BufInfo my_info;
        my_info.addr = (uint64_t)handle.buf;
        my_info.rkey = handle.mr->rkey;

        auto all_info = mpi.allgather_fixed(my_info);

        for (int r = 0; r < mpi.size; r++) {
            uint64_t key = make_key(r, buf_index);
            remote_info[key] = {all_info[r].addr, all_info[r].rkey};
        }

        // Update device state with first remote info
        if (buf_index == 0) {
            int peer = (mpi.rank == 0) ? 1 : 0;
            if (mpi.size > 1) {
                h_state.remote_addr = all_info[peer].addr;
                h_state.remote_rkey = all_info[peer].rkey;
                update_device_state();
            }
        }
    }

    /**
     * Set remote info for a specific peer/buffer
     */
    void set_remote_target(int dest_rank, int buf_index = 0) {
        uint64_t key = make_key(dest_rank, buf_index);
        auto it = remote_info.find(key);
        if (it != remote_info.end()) {
            h_state.remote_addr = it->second.addr;
            h_state.remote_rkey = it->second.rkey;
            update_device_state();
        }
    }

    /**
     * Get device state pointer for GPU kernels
     */
    GdaDeviceState* get_device_state() {
        return d_state;
    }

    /**
     * Get local buffer info for GPU kernels
     */
    void get_local_buffer_info(const GdaGpuHandle& handle,
                               uint64_t* addr, uint32_t* lkey) {
        *addr = (uint64_t)handle.buf;
        *lkey = handle.mr->lkey;
    }

    /**
     * Get remote buffer info
     */
    bool get_remote_buffer_info(int dest_rank, int buf_index,
                                uint64_t* addr, uint32_t* rkey) {
        uint64_t key = make_key(dest_rank, buf_index);
        auto it = remote_info.find(key);
        if (it == remote_info.end()) return false;

        *addr = it->second.addr;
        *rkey = it->second.rkey;
        return true;
    }

    /**
     * CPU-side put for testing (uses standard ibv)
     */
    void cpu_put(const GdaGpuHandle& handle, int dest_rank, int buf_index,
                 size_t size) {
        uint64_t key = make_key(dest_rank, buf_index);
        auto it = remote_info.find(key);
        if (it == remote_info.end()) {
            fprintf(stderr, "No remote info for rank %d buf %d\n", dest_rank, buf_index);
            return;
        }

        // Use standard ibv_post_send for CPU-triggered operation
        struct ibv_sge sge = {};
        sge.addr = (uint64_t)handle.buf;
        sge.length = size;
        sge.lkey = handle.mr->lkey;

        struct ibv_send_wr wr = {};
        wr.wr_id = 1;
        wr.next = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = it->second.addr;
        wr.wr.rdma.rkey = it->second.rkey;

        struct ibv_send_wr* bad_wr = nullptr;
        int ret = ibv_post_send(mlx5->qp, &wr, &bad_wr);
        if (ret) {
            fprintf(stderr, "ibv_post_send failed: %s\n", strerror(ret));
            return;
        }

        // Wait for completion
        struct ibv_wc wc;
        while (ibv_poll_cq(mlx5->cq, 1, &wc) == 0) {
            // Busy wait
        }
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "WC error: %s\n", ibv_wc_status_str(wc.status));
        }
    }

    /**
     * Barrier
     */
    void barrier() {
        mpi.barrier();
    }

    // Accessors
    int rank() const { return mpi.rank; }
    int size() const { return mpi.size; }

private:
    void exchange_and_connect() {
        // Get local connection info
        GdaConnInfo my_info = mlx5->get_local_info();

        // Exchange with all peers
        for (int peer = 0; peer < mpi.size; peer++) {
            if (peer == mpi.rank) continue;

            GdaConnInfo peer_info;
            mpi.exchange(&my_info, &peer_info, sizeof(GdaConnInfo), peer);

            // Store peer info
            mlx5->remote_peers[peer].qpn = peer_info.qpn;
            mlx5->remote_peers[peer].lid = peer_info.lid;
            memcpy(mlx5->remote_peers[peer].gid, peer_info.gid, 16);
        }

        mpi.barrier();

        // Connect to first peer for initial testing
        // In full implementation, we'd use DC or create multiple QPs
        if (mpi.size == 2) {
            int peer = (mpi.rank == 0) ? 1 : 0;

            GdaConnInfo peer_info;
            peer_info.qpn = mlx5->remote_peers[peer].qpn;
            peer_info.lid = mlx5->remote_peers[peer].lid;
            memcpy(peer_info.gid, mlx5->remote_peers[peer].gid, 16);
            peer_info.psn = 0;  // Use 0 for simplicity

            // Re-exchange PSN
            uint32_t my_psn = mlx5->psn;
            uint32_t peer_psn;
            mpi.exchange(&my_psn, &peer_psn, sizeof(uint32_t), peer);
            peer_info.psn = peer_psn;

            mlx5->connect_to_peer(peer, peer_info);
        }

        mpi.barrier();
    }

    void setup_device_state() {
        memset(&h_state, 0, sizeof(h_state));

        // Allocate completion counter (GPU-accessible)
        cudaError_t err = cudaHostAlloc((void**)&h_num_completions, sizeof(uint64_t),
                                        cudaHostAllocMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostAlloc for num_completions failed\n", mpi.rank);
            exit(1);
        }
        *h_num_completions = 0;

        err = cudaHostGetDevicePointer((void**)&d_num_completions, (void*)h_num_completions, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for num_completions failed\n", mpi.rank);
            exit(1);
        }

        h_state.qpn = mlx5->qp->qp_num;
        h_state.nwqes = mlx5->qp_depth;
        h_state.nwqes_mask = mlx5->qp_depth - 1;
        h_state.wqe_buf = mlx5->d_wqe_buf;
        h_state.wqe_lkey = 0;  // TODO: Register WQE buffer with NIC
        h_state.dbrec = (volatile uint32_t*)mlx5->qp_ex.dbrec;
        h_state.prod_idx = mlx5->d_prod_idx;
        h_state.cqe = (volatile GdaCqe64*)mlx5->d_cqe;
        h_state.ncqes = mlx5->cq_depth;
        h_state.ncqes_mask = mlx5->cq_depth - 1;
        h_state.cq_cons_idx = nullptr;
        h_state.cq_dbrec = (volatile uint32_t*)mlx5->cq_ex.dbrec;
        h_state.remote_addr = 0;
        h_state.remote_rkey = 0;
        h_state.num_completions = d_num_completions;

        // Allocate device state
        CUDA_CHECK(cudaMalloc(&d_state, sizeof(GdaDeviceState)));
        CUDA_CHECK(cudaMemcpy(d_state, &h_state, sizeof(GdaDeviceState),
                              cudaMemcpyHostToDevice));
    }

    void update_device_state() {
        CUDA_CHECK(cudaMemcpy(d_state, &h_state, sizeof(GdaDeviceState),
                              cudaMemcpyHostToDevice));
    }

    uint64_t make_key(int rank, int buf_index) const {
        return ((uint64_t)rank << 32) | (uint64_t)buf_index;
    }
};

}  // namespace opengda
