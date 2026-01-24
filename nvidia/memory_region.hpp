/**
 * memory_region.hpp - IB memory registration with GPU support
 *
 * Supports both host and GPU memory registration.
 * GPU memory requires nvidia_peermem kernel module.
 */
#pragma once

#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

class MemoryRegion {
public:
    struct ibv_mr* mr;
    void* buf;
    size_t size;
    uint32_t lkey;
    uint32_t rkey;
    bool is_device_mem;
    int rank;

    MemoryRegion(struct ibv_pd* pd, void* buf_, size_t size_,
                 bool is_device = false, int rank_ = 0)
        : mr(nullptr), buf(buf_), size(size_), lkey(0), rkey(0),
          is_device_mem(is_device), rank(rank_)
    {
        int access = IBV_ACCESS_LOCAL_WRITE |
                     IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_READ |
                     IBV_ACCESS_REMOTE_ATOMIC;

        // Register memory
        // For GPU memory, nvidia_peermem module handles the registration
        mr = ibv_reg_mr(pd, buf, size, access);
        if (!mr) {
            fprintf(stderr, "Rank %d: ibv_reg_mr failed for %s memory (%p, %zu bytes): %s\n",
                    rank, is_device ? "GPU" : "host", buf, size, strerror(errno));
            fprintf(stderr, "  Note: GPU memory requires nvidia_peermem kernel module\n");
            exit(1);
        }

        lkey = mr->lkey;
        rkey = mr->rkey;
    }

    ~MemoryRegion() {
        if (mr) {
            ibv_dereg_mr(mr);
        }
    }

    // No copy
    MemoryRegion(const MemoryRegion&) = delete;
    MemoryRegion& operator=(const MemoryRegion&) = delete;

    // Get remote key for sharing with peers
    uint32_t get_rkey() const { return rkey; }

    // Get address as uint64_t for RDMA
    uint64_t get_addr() const { return (uint64_t)buf; }
};

// Remote memory info for RDMA operations
struct RemoteMemInfo {
    uint64_t addr;
    uint32_t rkey;
};
