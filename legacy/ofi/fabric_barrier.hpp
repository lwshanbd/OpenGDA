/*
 * fabric_barrier.hpp - Fast libfabric-based barrier
 *
 * Uses Bruck (dissemination) algorithm from OpenMPI: O(log N) rounds
 * Each round: sendrecv with partner at distance 2^k
 * Has its own dedicated endpoint/CQ to avoid interference with DWQ
 */
#pragma once

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

class FabricBarrier {
public:
    struct fid_domain* domain;
    struct fid_av* av;
    int rank;
    int size;

    // Our own endpoint and CQ for barrier operations
    struct fid_ep* barrier_ep;
    struct fid_cq* barrier_cq;

    // Addresses of ALL ranks (needed for Bruck algorithm)
    std::vector<fi_addr_t> peer_addrs;

    // Local address for exchange
    void* local_addr;
    size_t addrlen;

    // Number of rounds = ceil(log2(size))
    int num_rounds;

    FabricBarrier(struct fid_domain* domain_, struct fid_av* av_,
                  struct fi_info* info, int rank_, int size_)
        : domain(domain_), av(av_),
          rank(rank_), size(size_),
          barrier_ep(nullptr), barrier_cq(nullptr),
          local_addr(nullptr), addrlen(0), num_rounds(0)
    {
        // Calculate number of rounds
        int n = size;
        while (n > 1) {
            num_rounds++;
            n = (n + 1) / 2;
        }

        peer_addrs.resize(size, FI_ADDR_NOTAVAIL);

        // Create dedicated CQ for barrier
        struct fi_cq_attr cq_attr = {};
        cq_attr.size = 64;
        cq_attr.format = FI_CQ_FORMAT_CONTEXT;
        int ret = fi_cq_open(domain, &cq_attr, &barrier_cq, NULL);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_cq_open(barrier) failed: %s\n",
                    rank, fi_strerror(-ret));
            exit(1);
        }

        // Create dedicated endpoint for barrier
        ret = fi_endpoint(domain, info, &barrier_ep, NULL);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_endpoint(barrier) failed: %s\n",
                    rank, fi_strerror(-ret));
            exit(1);
        }

        ret = fi_ep_bind(barrier_ep, &av->fid, 0);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_ep_bind(av) failed: %s\n",
                    rank, fi_strerror(-ret));
            exit(1);
        }

        ret = fi_ep_bind(barrier_ep, &barrier_cq->fid, FI_TRANSMIT | FI_RECV);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_ep_bind(cq) failed: %s\n",
                    rank, fi_strerror(-ret));
            exit(1);
        }

        ret = fi_enable(barrier_ep);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_enable(barrier_ep) failed: %s\n",
                    rank, fi_strerror(-ret));
            exit(1);
        }

        // Get our endpoint's address
        addrlen = 0;
        fi_getname(&barrier_ep->fid, NULL, &addrlen);
        local_addr = malloc(addrlen);
        fi_getname(&barrier_ep->fid, local_addr, &addrlen);
    }

    ~FabricBarrier() {
        if (barrier_ep) fi_close(&barrier_ep->fid);
        if (barrier_cq) fi_close(&barrier_cq->fid);
        if (local_addr) free(local_addr);
    }

    FabricBarrier(const FabricBarrier&) = delete;
    FabricBarrier& operator=(const FabricBarrier&) = delete;

    // Set peer address (call for each rank after AV insert)
    void set_peer_addr(int peer_rank, fi_addr_t addr) {
        if (peer_rank >= 0 && peer_rank < size) {
            peer_addrs[peer_rank] = addr;
        }
    }

    // Bruck (dissemination) barrier - O(log N) rounds
    // Each round: exchange with partner at distance 2^k
    void barrier() {
        if (size == 1) return;

        for (int k = 0; k < num_rounds; k++) {
            int distance = 1 << k;

            // Send to (rank + distance) % size
            int send_to = (rank + distance) % size;
            // Receive from (rank - distance + size) % size
            int recv_from = (rank - distance + size) % size;

            // Sendrecv: post recv first, then send, then wait for recv
            int err = sendrecv(peer_addrs[send_to], peer_addrs[recv_from]);
            if (err) {
                fprintf(stderr, "Rank %d: barrier round %d failed: %s\n",
                        rank, k, fi_strerror(-err));
                return;
            }
        }
    }

private:
    // Sendrecv: post recv, inject send, wait for recv completion
    int sendrecv(fi_addr_t send_dest, fi_addr_t recv_src) {
        uint64_t recv_buf = 0;
        uint64_t send_buf = rank + 1;

        // Post receive first
        int ret = fi_recv(barrier_ep, &recv_buf, sizeof(recv_buf), NULL, recv_src, NULL);
        if (ret) return ret;

        // Send using inject (no completion needed)
        ret = fi_inject(barrier_ep, &send_buf, sizeof(send_buf), send_dest);
        if (ret == -FI_EAGAIN) {
            progress_cq();
            ret = fi_inject(barrier_ep, &send_buf, sizeof(send_buf), send_dest);
        }
        if (ret) return ret;

        // Wait for receive completion
        return wait_completion();
    }

    int wait_completion() {
        struct fi_cq_entry entry;
        while (true) {
            int ret = fi_cq_read(barrier_cq, &entry, 1);
            if (ret > 0) {
                return 0;
            } else if (ret == -FI_EAGAIN) {
                continue;
            } else if (ret == -FI_EAVAIL) {
                struct fi_cq_err_entry err;
                fi_cq_readerr(barrier_cq, &err, 0);
                fprintf(stderr, "Rank %d: CQ error in barrier: %s (%d)\n",
                        rank, fi_strerror(err.err), err.err);
                return err.err;
            } else {
                return ret;
            }
        }
    }

    void progress_cq() {
        struct fi_cq_entry entry;
        fi_cq_read(barrier_cq, &entry, 1);
    }
};
