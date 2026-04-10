/*
 * ofi_barrier.hpp - Optimized dissemination barrier using OFI FI_MSG
 *
 * Algorithm: Dissemination (Bruck) - O(log N) steps
 * Each step k: send to (rank + 2^k) % N, recv from (rank - 2^k + N) % N
 *
 * Uses fi_injectdata for zero-copy sends with immediate data
 */
#pragma once

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

class OfiBarrier {
public:
    static constexpr int MAX_STEPS = 20;
    static constexpr int RECV_POOL_SIZE = 64;

    struct fid_domain* domain;
    struct fid_av* av;
    struct fid_ep* barrier_ep;
    struct fid_cq* barrier_cq;

    void* local_addr;
    size_t addrlen;
    std::vector<fi_addr_t> peer_addrs;

    int rank;
    int size;
    int steps;

    // Barrier state
    uint8_t arrived[2][MAX_STEPS];
    uint8_t phase;

    int peer_fwd[MAX_STEPS];
    int peer_bwd[MAX_STEPS];

    // Small receive buffer
    uint8_t recv_buf[RECV_POOL_SIZE];

    // Deferred repost tracking
    int pending_repost[MAX_STEPS];
    int num_pending;

    OfiBarrier(struct fid_domain* domain_, struct fid_av* av_,
               struct fi_info* info, int rank_, int size_)
        : domain(domain_), av(av_),
          barrier_ep(nullptr), barrier_cq(nullptr),
          local_addr(nullptr), addrlen(0),
          rank(rank_), size(size_), phase(0), num_pending(0)
    {
        steps = 0;
        for (int n = 1; n < size; n <<= 1) steps++;

        for (int k = 0; k < steps; k++) {
            int dist = 1 << k;
            peer_fwd[k] = (rank + dist) % size;
            peer_bwd[k] = (rank - dist + size) % size;
        }

        memset(arrived, 0, sizeof(arrived));
        peer_addrs.resize(size, FI_ADDR_NOTAVAIL);

        // CQ with DATA format to get immediate data
        struct fi_cq_attr cq_attr = {};
        cq_attr.size = RECV_POOL_SIZE * 2;
        cq_attr.format = FI_CQ_FORMAT_DATA;
        int ret = fi_cq_open(domain, &cq_attr, &barrier_cq, NULL);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_cq_open failed: %s\n", rank, fi_strerror(-ret));
            exit(1);
        }

        ret = fi_endpoint(domain, info, &barrier_ep, NULL);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_endpoint failed: %s\n", rank, fi_strerror(-ret));
            exit(1);
        }

        fi_ep_bind(barrier_ep, &av->fid, 0);
        fi_ep_bind(barrier_ep, &barrier_cq->fid, FI_TRANSMIT | FI_RECV);
        fi_enable(barrier_ep);

        addrlen = 0;
        fi_getname(&barrier_ep->fid, NULL, &addrlen);
        local_addr = malloc(addrlen);
        fi_getname(&barrier_ep->fid, local_addr, &addrlen);
    }

    ~OfiBarrier() {
        if (barrier_ep) fi_close(&barrier_ep->fid);
        if (barrier_cq) fi_close(&barrier_cq->fid);
        if (local_addr) free(local_addr);
    }

    OfiBarrier(const OfiBarrier&) = delete;
    OfiBarrier& operator=(const OfiBarrier&) = delete;

    void set_peer_addr(int peer_rank, fi_addr_t addr) {
        if (peer_rank >= 0 && peer_rank < size) {
            peer_addrs[peer_rank] = addr;
        }
    }

    void post_initial_recvs() {
        for (int i = 0; i < RECV_POOL_SIZE; i++) {
            fi_recv(barrier_ep, &recv_buf[i], 1,
                    NULL, FI_ADDR_UNSPEC, (void*)(uintptr_t)i);
        }
    }

    void barrier() {
        if (size == 1) return;

        phase ^= 1;
        for (int k = 0; k < steps; k++) {
            arrived[phase][k] = 0;
        }
        num_pending = 0;

        for (int k = 0; k < steps; k++) {
            // Send using fi_injectdata (zero-size with immediate data)
            // Immediate data: (phase << 8) | step
            uint64_t imm_data = ((uint64_t)phase << 8) | k;

            ssize_t ret = fi_injectdata(barrier_ep, NULL, 0, imm_data,
                                        peer_addrs[peer_fwd[k]]);
            while (ret == -FI_EAGAIN) {
                poll_cq_nodefer();
                ret = fi_injectdata(barrier_ep, NULL, 0, imm_data,
                                    peer_addrs[peer_fwd[k]]);
            }

            // Wait for this step
            while (!arrived[phase][k]) {
                poll_cq_defer();
            }
        }

        // Batch repost
        for (int i = 0; i < num_pending; i++) {
            int buf_idx = pending_repost[i];
            fi_recv(barrier_ep, &recv_buf[buf_idx], 1,
                    NULL, FI_ADDR_UNSPEC, (void*)(uintptr_t)buf_idx);
        }
    }

private:
    inline void poll_cq_defer() {
        struct fi_cq_data_entry entry;
        ssize_t ret = fi_cq_read(barrier_cq, &entry, 1);
        if (ret > 0) {
            // Extract phase and step from immediate data
            uint64_t imm = entry.data;
            int msg_phase = (imm >> 8) & 0xFF;
            int msg_step = imm & 0xFF;

            if (msg_step < MAX_STEPS) {
                arrived[msg_phase][msg_step] = 1;
            }

            int buf_idx = (int)(uintptr_t)entry.op_context;
            pending_repost[num_pending++] = buf_idx;
        }
    }

    inline void poll_cq_nodefer() {
        struct fi_cq_data_entry entry;
        ssize_t ret = fi_cq_read(barrier_cq, &entry, 1);
        if (ret > 0) {
            uint64_t imm = entry.data;
            int msg_phase = (imm >> 8) & 0xFF;
            int msg_step = imm & 0xFF;

            if (msg_step < MAX_STEPS) {
                arrived[msg_phase][msg_step] = 1;
            }

            int buf_idx = (int)(uintptr_t)entry.op_context;
            fi_recv(barrier_ep, &recv_buf[buf_idx], 1,
                    NULL, FI_ADDR_UNSPEC, (void*)(uintptr_t)buf_idx);
        }
    }
};
