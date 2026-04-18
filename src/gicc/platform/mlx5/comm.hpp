/**
 * gda_comm.hpp - GPU-Direct Async Communication API for NVIDIA + InfiniBand
 *
 * Provides nvshmem-like put/get APIs:
 *   - put(): RDMA write to remote rank
 *   - get(): RDMA read from remote rank
 *   - barrier(): Global synchronization
 *
 * Uses per-peer RC QPs for reliable point-to-point connections.
 */
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <arpa/inet.h>

#include "gicc/util/cuda_device_context.hpp"
#include "gicc/bootstrap/bootstrap.hpp"
#include "gicc/util/ibv_context.hpp"
#include "gicc/util/memory_region.hpp"
#include "types.hpp"

namespace gicc::mlx5 {

#define CUDA_CHECK(cmd) do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// GPU kernel to trigger operations by writing to mapped counter
__global__ void gda_trigger_kernel(volatile uint64_t* trigger_addr, uint64_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *trigger_addr = value;
        __threadfence_system();
    }
}

// Handle to registered memory
struct GdaHandle {
    void* buf;
    size_t size;
    MemoryRegion* mr;
    bool is_device;
};

// Remote buffer info
struct GdaRemoteInfo {
    uint64_t addr;
    uint32_t rkey;
};

class Fabric {
public:
    // Components
    gicc::Bootstrap& boot;
    CudaDeviceContext* cuda;
    IbvContext* ibv;

    // Registered buffers
    std::vector<MemoryRegion*> registered_mrs;

    // Remote info: (rank, buf_index) -> info
    std::unordered_map<uint64_t, GdaRemoteInfo> remote_info;

    // Operation tracking
    std::atomic<uint64_t> op_counter;
    std::atomic<uint64_t> completion_counter;

    // GPU-visible counters (host-mapped memory)
    volatile uint64_t* h_trigger_cntr;
    volatile uint64_t* h_completion_cntr;
    volatile uint64_t* d_trigger_cntr;
    volatile uint64_t* d_completion_cntr;

    explicit Fabric(gicc::Bootstrap& boot_, int local_rank = -1)
        : boot(boot_), cuda(nullptr), ibv(nullptr),
          op_counter(0), completion_counter(0),
          h_trigger_cntr(nullptr), h_completion_cntr(nullptr),
          d_trigger_cntr(nullptr), d_completion_cntr(nullptr)
    {
        // Determine GPU to use
        int gpu_id = (local_rank >= 0) ? local_rank : boot.local_rank();

        int num_gpus = 0;
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        if (num_gpus > 0) {
            gpu_id = gpu_id % num_gpus;
        }

        // Initialize CUDA
        cuda = new CudaDeviceContext(gpu_id);

        // Initialize IB with per-peer QPs
        ibv = new IbvContext(boot.rank(), boot.size());

        // Allocate GPU-visible counters
        allocate_counters();

        // Exchange IB connection info and connect QPs
        exchange_and_connect();
    }

    ~Fabric() {
        for (auto* mr : registered_mrs) delete mr;
        registered_mrs.clear();

        free_counters();

        delete ibv;
        delete cuda;
    }

    // No copy
    Fabric(const Fabric&) = delete;
    Fabric& operator=(const Fabric&) = delete;

    /**
     * Register a buffer for RDMA
     */
    GdaHandle register_buffer(void* buf, size_t size, bool is_device) {
        auto* mr = new MemoryRegion(ibv->pd, buf, size, is_device, boot.rank());
        registered_mrs.push_back(mr);

        GdaHandle handle;
        handle.buf = buf;
        handle.size = size;
        handle.mr = mr;
        handle.is_device = is_device;
        return handle;
    }

    /**
     * Exchange buffer info with all peers
     */
    void exchange_buffer_info(const GdaHandle& handle, int buf_index = 0) {
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
    }

    /**
     * Set remote info directly
     */
    void set_remote_info(int rank, int buf_index, uint64_t addr, uint32_t rkey) {
        uint64_t key = make_key(rank, buf_index);
        remote_info[key] = {addr, rkey};
    }

    /**
     * RDMA put operation
     */
    uint64_t put(const GdaHandle& handle, int dest_rank, int buf_index,
                 size_t size, bool signaled = true) {
        uint64_t key = make_key(dest_rank, buf_index);
        auto it = remote_info.find(key);
        if (it == remote_info.end()) {
            fprintf(stderr, "Rank %d: No remote info for rank %d buf %d\n",
                    boot.rank(), dest_rank, buf_index);
            exit(1);
        }

        uint64_t op_id = ++op_counter;

        if (getenv("GDA_DEBUG")) {
            printf("Rank %d: RDMA PUT to rank %d buf %d: local=%p (lkey=0x%x), "
                   "remote=0x%lx (rkey=0x%x), size=%zu, op_id=%lu\n",
                   boot.rank(), dest_rank, buf_index,
                   handle.buf, handle.mr->lkey,
                   it->second.addr, it->second.rkey, size, op_id);
        }

        int ret = ibv->post_rdma_write(
            dest_rank,
            handle.buf, handle.mr->lkey, size,
            it->second.addr, it->second.rkey,
            op_id, signaled);

        if (ret) {
            fprintf(stderr, "Rank %d: post_rdma_write failed: %s\n",
                    boot.rank(), strerror(ret));
            exit(1);
        }

        return op_id;
    }

    /**
     * RDMA put with explicit remote address
     */
    uint64_t put_raw(const GdaHandle& handle, int dest_rank,
                     uint64_t remote_addr, uint32_t rkey,
                     size_t size, bool signaled = true) {
        uint64_t op_id = ++op_counter;

        int ret = ibv->post_rdma_write(
            dest_rank,
            handle.buf, handle.mr->lkey, size,
            remote_addr, rkey, op_id, signaled);

        if (ret) {
            fprintf(stderr, "Rank %d: post_rdma_write failed: %s\n",
                    boot.rank(), strerror(ret));
            exit(1);
        }

        return op_id;
    }

    /**
     * RDMA get operation
     */
    uint64_t get(const GdaHandle& handle, int src_rank, int buf_index,
                 size_t size, bool signaled = true) {
        uint64_t key = make_key(src_rank, buf_index);
        auto it = remote_info.find(key);
        if (it == remote_info.end()) {
            fprintf(stderr, "Rank %d: No remote info for rank %d buf %d\n",
                    boot.rank(), src_rank, buf_index);
            exit(1);
        }

        uint64_t op_id = ++op_counter;

        int ret = ibv->post_rdma_read(
            src_rank,
            handle.buf, handle.mr->lkey, size,
            it->second.addr, it->second.rkey,
            op_id, signaled);

        if (ret) {
            fprintf(stderr, "Rank %d: post_rdma_read failed: %s\n",
                    boot.rank(), strerror(ret));
            exit(1);
        }

        return op_id;
    }

    /**
     * Wait for completion of signaled operations
     */
    void wait(int count = 1) {
        struct ibv_wc wc;
        int completed = 0;

        while (completed < count) {
            int n = ibv->poll_cq(&wc, 1);
            if (n < 0) {
                fprintf(stderr, "Rank %d: poll_cq failed\n", boot.rank());
                exit(1);
            }
            if (n > 0) {
                if (wc.status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "Rank %d: WC error: %s (op_id=%lu)\n",
                            boot.rank(), ibv_wc_status_str(wc.status), wc.wr_id);
                    exit(1);
                }
                completed++;
                completion_counter++;
            }
        }
    }

    /**
     * Poll for completions (non-blocking)
     */
    int poll(int max_completions = 1) {
        struct ibv_wc wc[16];
        int n = ibv->poll_cq(wc, std::min(max_completions, 16));
        if (n < 0) {
            fprintf(stderr, "Rank %d: poll_cq failed\n", boot.rank());
            return n;
        }
        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                fprintf(stderr, "Rank %d: WC error: %s\n",
                        boot.rank(), ibv_wc_status_str(wc[i].status));
            }
            completion_counter++;
        }
        return n;
    }

    /**
     * Trigger GPU kernel that writes to mapped trigger counter
     */
    void trigger(uint64_t value) {
        gda_trigger_kernel<<<1, 1>>>(d_trigger_cntr, value);
    }

    /**
     * Get GPU-accessible trigger counter address
     */
    volatile uint64_t* get_trigger_addr() const {
        return d_trigger_cntr;
    }

    /**
     * Get GPU-accessible completion counter address
     */
    volatile uint64_t* get_completion_addr() const {
        return d_completion_cntr;
    }

    /**
     * Update GPU-visible completion counter
     */
    void signal_completion(uint64_t value) {
        *h_completion_cntr = value;
    }

    /**
     * Global barrier
     */
    void barrier() {
        boot.barrier();
    }

    /**
     * Reset counters
     */
    void reset_counters() {
        op_counter = 0;
        completion_counter = 0;
        *h_trigger_cntr = 0;
        *h_completion_cntr = 0;
    }

    // Accessors
    int rank() const { return boot.rank(); }
    int size() const { return boot.size(); }
    int gpu_id() const { return cuda->gpu_id; }

private:
    void allocate_counters() {
        CUDA_CHECK(cudaHostAlloc((void**)&h_trigger_cntr, sizeof(uint64_t),
                                 cudaHostAllocMapped | cudaHostAllocWriteCombined));
        CUDA_CHECK(cudaHostAlloc((void**)&h_completion_cntr, sizeof(uint64_t),
                                 cudaHostAllocMapped));

        CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_trigger_cntr,
                                            (void*)h_trigger_cntr, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_completion_cntr,
                                            (void*)h_completion_cntr, 0));

        *h_trigger_cntr = 0;
        *h_completion_cntr = 0;
    }

    void free_counters() {
        if (h_trigger_cntr) cudaFreeHost((void*)h_trigger_cntr);
        if (h_completion_cntr) cudaFreeHost((void*)h_completion_cntr);
    }

    void exchange_and_connect() {
        // For each peer, exchange connection info
        for (int peer = 0; peer < boot.size(); peer++) {
            if (peer == boot.rank()) continue;

            // Get local info for this peer's QP
            IbvConnInfo local_info = ibv->get_local_info(peer);
            IbvConnInfo peer_info;

            // Exchange with peer
            boot.sendrecv(&local_info, &peer_info, sizeof(IbvConnInfo), peer);

            // Store peer's info
            ibv->set_peer_info(peer, peer_info);

            // Debug output
            if (getenv("GDA_DEBUG")) {
                char local_gid[64], peer_gid[64];
                inet_ntop(AF_INET6, local_info.gid, local_gid, sizeof(local_gid));
                inet_ntop(AF_INET6, peer_info.gid, peer_gid, sizeof(peer_gid));
                printf("Rank %d: QP to peer %d - local(qp=%u, lid=%u, psn=%u, gid=%s) -> "
                       "peer(qp=%u, lid=%u, psn=%u, gid=%s)\n",
                       boot.rank(), peer, local_info.qp_num, local_info.lid, local_info.psn, local_gid,
                       peer_info.qp_num, peer_info.lid, peer_info.psn, peer_gid);
            }
        }

        boot.barrier();

        // Connect all QPs
        for (int peer = 0; peer < boot.size(); peer++) {
            if (peer == boot.rank()) continue;
            ibv->connect_to_peer(peer);
        }

        boot.barrier();
    }

    uint64_t make_key(int rank, int buf_index) const {
        return ((uint64_t)rank << 32) | (uint64_t)buf_index;
    }
};

}  // namespace gicc::mlx5
