/*
 * memory_region.hpp - Libfabric memory region wrapper with RAII
 */
#pragma once

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <cstdio>
#include <cstdlib>

class MemoryRegion {
public:
    struct fid_mr* mr;
    void* desc;
    uint64_t key;
    void* buf;
    size_t size;
    int rank;  // For error messages

    MemoryRegion(struct fid_domain* domain, struct fid_ep* ep, struct fi_info* info,
                 void* buf_, size_t size_, bool is_device_mem, int device_id, int rank_)
        : mr(nullptr), desc(nullptr), key(0), buf(buf_), size(size_), rank(rank_)
    {
        struct fi_mr_attr mr_attr = {};
        struct iovec iov;
        iov.iov_base = buf;
        iov.iov_len = size;

        mr_attr.mr_iov = &iov;
        mr_attr.iov_count = 1;
        mr_attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE |
                         FI_REMOTE_READ | FI_REMOTE_WRITE;

        if (is_device_mem) {
            mr_attr.iface = FI_HMEM_ROCR;
            mr_attr.device.reserved = device_id;
        } else {
            mr_attr.iface = FI_HMEM_SYSTEM;
        }

        int ret = fi_mr_regattr(domain, &mr_attr, 0, &mr);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_mr_regattr failed: %s (%d)\n",
                    rank, fi_strerror(-ret), ret);
            exit(1);
        }

        // Handle FI_MR_ENDPOINT mode
        if (info->domain_attr->mr_mode & FI_MR_ENDPOINT) {
            ret = fi_mr_bind(mr, &ep->fid, 0);
            if (ret) {
                fi_close(&mr->fid);
                fprintf(stderr, "Rank %d: fi_mr_bind failed: %s (%d)\n",
                        rank, fi_strerror(-ret), ret);
                exit(1);
            }
            ret = fi_mr_enable(mr);
            if (ret) {
                fi_close(&mr->fid);
                fprintf(stderr, "Rank %d: fi_mr_enable failed: %s (%d)\n",
                        rank, fi_strerror(-ret), ret);
                exit(1);
            }
        }

        desc = fi_mr_desc(mr);
        key = fi_mr_key(mr);
    }

    ~MemoryRegion() {
        if (mr) fi_close(&mr->fid);
    }

    // No copy/move
    MemoryRegion(const MemoryRegion&) = delete;
    MemoryRegion& operator=(const MemoryRegion&) = delete;
};
