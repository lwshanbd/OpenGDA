/**
 * gda_gpu_comm.hpp - GPU-Triggered Communication API
 *
 * High-level API for GPU-initiated RDMA operations using MLX5 DevX.
 * Unlike the CPU-triggered version (gda_comm.hpp), this allows
 * GPU kernels to directly trigger RDMA without CPU involvement.
 *
 * MULTI-QP SUPPORT: Creates one QP per neighbor for ring topology.
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

#include "gicc/bootstrap/bootstrap.hpp"
#include "devx_context.hpp"
#include "gicc/util/memory_region.hpp"
#include "device.cuh"

namespace gicc::mlx5 {

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
    gicc::Bootstrap& boot;
    DevxContext* mlx5;

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

    // Device state (GPU-accessible) - per peer
    std::unordered_map<int, GdaDeviceState*> d_state_per_peer;
    std::unordered_map<int, GdaDeviceState> h_state_per_peer;

    // Legacy: single device state (for backward compatibility)
    GdaDeviceState* d_state;
    GdaDeviceState device_state;

    // Completion counter (GPU-accessible)
    volatile uint64_t* h_num_completions;
    volatile uint64_t* d_num_completions;

    // Neighbor ranks (for ring topology)
    int top_neighbor;
    int bottom_neighbor;

    explicit GdaGpuComm(gicc::Bootstrap& boot_, int local_rank = -1)
        : boot(boot_), mlx5(nullptr), gpu_id(0), d_state(nullptr),
          h_num_completions(nullptr), d_num_completions(nullptr),
          top_neighbor(-1), bottom_neighbor(-1)
    {
        // Determine GPU to use
        gpu_id = (local_rank >= 0) ? local_rank : boot.local_rank();

        int num_gpus = 0;
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        if (num_gpus > 0) {
            gpu_id = gpu_id % num_gpus;
        }

        CUDA_CHECK(cudaSetDevice(gpu_id));
        CUDA_CHECK(cudaGetDeviceProperties(&props, gpu_id));

        // Initialize MLX5 context
        mlx5 = new DevxContext(boot);

        // Determine neighbors (ring topology)
        top_neighbor = (boot.rank() > 0) ? boot.rank() - 1 : (boot.size() - 1);
        bottom_neighbor = (boot.rank() + 1) % boot.size();

        // Create QPs for neighbors
        create_neighbor_qps();

        // Exchange connection info and connect
        exchange_and_connect();

        // Setup device state for each neighbor
        setup_device_states();

        // Allocate completion counter
        allocate_completion_counter();

        if (boot.rank() == 0) {
            printf("GdaGpuComm initialized:\n");
            printf("  GPU: %s\n", props.name);
            printf("  IB device: %s\n", mlx5->dev_name.c_str());
            printf("  Rank %d of %d\n", boot.rank(), boot.size());
            printf("  Neighbors: top=%d, bottom=%d\n", top_neighbor, bottom_neighbor);
            fflush(stdout);
        }
    }

    ~GdaGpuComm() {
        for (auto& kv : d_state_per_peer) {
            if (kv.second) cudaFree(kv.second);
        }
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
        auto* mr = new MemoryRegion(mlx5->pd, buf, size, is_device, boot.rank());
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

        auto all_info = boot.allgather_fixed(my_info);

        for (int r = 0; r < boot.size(); r++) {
            uint64_t key = make_key(r, buf_index);
            remote_info[key] = {all_info[r].addr, all_info[r].rkey};
        }

        // Update device state with neighbor's remote info
        if (buf_index == 0) {
            update_device_state_remote_info(top_neighbor, all_info[top_neighbor].addr, all_info[top_neighbor].rkey);
            update_device_state_remote_info(bottom_neighbor, all_info[bottom_neighbor].addr, all_info[bottom_neighbor].rkey);
        }
    }

    /**
     * Set remote info for a specific peer/buffer
     */
    void set_remote_target(int dest_rank, int buf_index = 0) {
        uint64_t key = make_key(dest_rank, buf_index);
        auto it = remote_info.find(key);
        if (it != remote_info.end()) {
            update_device_state_remote_info(dest_rank, it->second.addr, it->second.rkey);
        }
    }

    /**
     * Get device state pointer for a specific peer
     */
    GdaDeviceState* get_device_state_for_peer(int peer) {
        auto it = d_state_per_peer.find(peer);
        if (it != d_state_per_peer.end()) {
            return it->second;
        }
        return nullptr;
    }

    /**
     * Get device state pointer for GPU kernels (legacy - uses first peer)
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

        // Get the QP for this peer
        auto qp_it = mlx5->peer_qps.find(dest_rank);
        if (qp_it == mlx5->peer_qps.end()) {
            fprintf(stderr, "No QP for peer %d\n", dest_rank);
            return;
        }

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
        int ret = ibv_post_send(qp_it->second.qp, &wr, &bad_wr);
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
        boot.barrier();
    }

    // Accessors
    int rank() const { return boot.rank(); }
    int size() const { return boot.size(); }

private:
    void create_neighbor_qps() {
        // Create QPs for top and bottom neighbors
        if (top_neighbor != boot.rank()) {
            mlx5->create_qp_for_peer(top_neighbor);
        }
        if (bottom_neighbor != boot.rank() && bottom_neighbor != top_neighbor) {
            mlx5->create_qp_for_peer(bottom_neighbor);
        }
    }

    void exchange_and_connect() {
        // Exchange connection info with neighbors
        struct ConnExchange {
            uint32_t qpn;
            uint16_t lid;
            uint8_t gid[16];
            uint32_t psn;
        };

        // Exchange with top neighbor
        if (top_neighbor != boot.rank()) {
            GdaConnInfo my_info = mlx5->get_local_info_for_peer(top_neighbor);
            ConnExchange my_ex = {my_info.qpn, my_info.lid, {}, my_info.psn};
            memcpy(my_ex.gid, my_info.gid, 16);

            ConnExchange peer_ex;
            boot.sendrecv(&my_ex, &peer_ex, sizeof(ConnExchange), top_neighbor);

            GdaConnInfo peer_info;
            peer_info.qpn = peer_ex.qpn;
            peer_info.lid = peer_ex.lid;
            memcpy(peer_info.gid, peer_ex.gid, 16);
            peer_info.psn = peer_ex.psn;
            peer_info.buf_addr = 0;
            peer_info.rkey = 0;

            mlx5->connect_to_peer(top_neighbor, peer_info);
        }

        // Exchange with bottom neighbor
        if (bottom_neighbor != boot.rank() && bottom_neighbor != top_neighbor) {
            GdaConnInfo my_info = mlx5->get_local_info_for_peer(bottom_neighbor);
            ConnExchange my_ex = {my_info.qpn, my_info.lid, {}, my_info.psn};
            memcpy(my_ex.gid, my_info.gid, 16);

            ConnExchange peer_ex;
            boot.sendrecv(&my_ex, &peer_ex, sizeof(ConnExchange), bottom_neighbor);

            GdaConnInfo peer_info;
            peer_info.qpn = peer_ex.qpn;
            peer_info.lid = peer_ex.lid;
            memcpy(peer_info.gid, peer_ex.gid, 16);
            peer_info.psn = peer_ex.psn;
            peer_info.buf_addr = 0;
            peer_info.rkey = 0;

            mlx5->connect_to_peer(bottom_neighbor, peer_info);
        }

        boot.barrier();
    }

    void setup_device_states() {
        // Setup device state for each neighbor
        for (auto& kv : mlx5->peer_qps) {
            int peer = kv.first;
            setup_device_state_for_peer(peer);
        }

        // Set legacy d_state to point to first neighbor's state
        if (!d_state_per_peer.empty()) {
            d_state = d_state_per_peer.begin()->second;
            device_state = h_state_per_peer.begin()->second;
        }

        // Also update mlx5 legacy pointers
        if (!mlx5->peer_qps.empty()) {
            mlx5->set_active_peer(mlx5->peer_qps.begin()->first);
        }
    }

    void setup_device_state_for_peer(int peer) {
        auto it = mlx5->peer_qps.find(peer);
        if (it == mlx5->peer_qps.end()) return;

        GdaDeviceState state;
        memset(&state, 0, sizeof(state));

        const PerPeerQp& pqp = it->second;
        state.qpn = pqp.qp->qp_num;
        state.nwqes = mlx5->qp_depth;
        state.nwqes_mask = mlx5->qp_depth - 1;
        state.wqe_buf = pqp.d_wqe_buf;
        state.wqe_lkey = 0;
        state.dbrec = pqp.d_dbrec;
        state.prod_idx = pqp.d_prod_idx;
        state.cqe = (volatile GdaCqe64*)mlx5->d_cqe;
        state.ncqes = mlx5->cq_depth;
        state.ncqes_mask = mlx5->cq_depth - 1;
        state.cq_cons_idx = nullptr;
        state.cq_dbrec = nullptr;
        state.remote_addr = 0;
        state.remote_rkey = 0;
        state.num_completions = d_num_completions;

        h_state_per_peer[peer] = state;

        GdaDeviceState* d_state_ptr;
        CUDA_CHECK(cudaMalloc(&d_state_ptr, sizeof(GdaDeviceState)));
        CUDA_CHECK(cudaMemcpy(d_state_ptr, &state, sizeof(GdaDeviceState),
                              cudaMemcpyHostToDevice));
        d_state_per_peer[peer] = d_state_ptr;
    }

    void update_device_state_remote_info(int peer, uint64_t addr, uint32_t rkey) {
        auto it = h_state_per_peer.find(peer);
        if (it == h_state_per_peer.end()) return;

        it->second.remote_addr = addr;
        it->second.remote_rkey = rkey;

        auto d_it = d_state_per_peer.find(peer);
        if (d_it != d_state_per_peer.end()) {
            CUDA_CHECK(cudaMemcpy(d_it->second, &it->second, sizeof(GdaDeviceState),
                                  cudaMemcpyHostToDevice));
        }
    }

    void allocate_completion_counter() {
        cudaError_t err = cudaHostAlloc((void**)&h_num_completions, sizeof(uint64_t),
                                        cudaHostAllocMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostAlloc for num_completions failed\n", boot.rank());
            exit(1);
        }
        *h_num_completions = 0;

        err = cudaHostGetDevicePointer((void**)&d_num_completions, (void*)h_num_completions, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for num_completions failed\n", boot.rank());
            exit(1);
        }
    }

    uint64_t make_key(int rank, int buf_index) const {
        return ((uint64_t)rank << 32) | (uint64_t)buf_index;
    }
};

}  // namespace gicc::mlx5
