// omp_peer_ptr.cpp - same-node direct access through ompx_peer_ptr, checked
// word for word, next to ompx_put / ompx_get to every peer.
//
// Every rank owns four symmetric arrays of W words:
//   mine   filled with a pattern unique to the rank
//   inbox  one slot per rank: peer r stores into slot r of our inbox
//   pulled one slot per rank: filled by ompx_get from peer r's `mine`
//   pushed one slot per rank: peer r's ompx_put of its `mine` lands in slot r
//
// For a same-node peer, ompx_peer_ptr must be non-NULL: a target region reads
// the peer's `mine` through it and stores our pattern into the peer's inbox
// through it. For this rank and for peers on other nodes it must be NULL.
// ompx_put / ompx_get go to every peer (IPC copy on the node, the NIC off it).
//
// Build: bash examples/omp/build_giomp_example.sh examples/omp/omp_peer_ptr.cpp OUT
// Run  : GICC_HALO_IPC=1 srun -N2 -n4 --ntasks-per-node=2 ... ./omp_peer_ptr
//        (without GICC_HALO_IPC every ompx_peer_ptr is NULL; the rest holds)

#include "gicc/omp.h"
#include <omp.h>
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#pragma omp declare target
static inline uint32_t pattern(int rank, size_t i) {
    return (uint32_t)rank * 0x9E3779B1u ^ (uint32_t)i * 0x85EBCA77u ^ 0x5bd1e995u;
}
#pragma omp end declare target

int main() {
    const size_t W = 1 << 16;                        // words per slot
    const size_t B = W * sizeof(uint32_t);

    ompx_init();
    const int me = ompx_get_rank_num();
    const int n  = ompx_get_num_ranks();
    const bool ipc = std::getenv("GICC_HALO_IPC") && std::atoi(std::getenv("GICC_HALO_IPC"));

    // Which ranks share this node, to know what ompx_peer_ptr must return.
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, me, MPI_INFO_NULL, &node);
    int* color = (int*)malloc(sizeof(int) * n);
    int host_id = 0, node_rank = 0;
    MPI_Comm_rank(node, &node_rank);
    host_id = me - node_rank;                         // any rank-unique node id
    MPI_Allreduce(MPI_IN_PLACE, &host_id, 1, MPI_INT, MPI_MIN, node);
    MPI_Allgather(&host_id, 1, MPI_INT, color, 1, MPI_INT, MPI_COMM_WORLD);

    uint32_t* mine   = (uint32_t*)ompx_alloc(B);
    uint32_t* inbox  = (uint32_t*)ompx_alloc(B * n);
    uint32_t* pulled = (uint32_t*)ompx_alloc(B * n);
    uint32_t* pushed = (uint32_t*)ompx_alloc(B * n);

    #pragma omp target teams distribute parallel for is_device_ptr(mine) firstprivate(me)
    for (size_t i = 0; i < W; ++i) mine[i] = pattern(me, i);
    ompx_barrier();

    unsigned long long bad = 0, wrong_ptr = 0;
    int mapped = 0;
    for (int p = 0; p < n; ++p) {
        uint32_t* peer_mine  = (uint32_t*)ompx_peer_ptr(p, mine);
        uint32_t* peer_inbox = (uint32_t*)ompx_peer_ptr(p, inbox + (size_t)me * W);
        const bool expect = ipc && p != me && color[p] == color[me];
        if ((peer_mine != nullptr) != expect || (peer_inbox != nullptr) != expect) {
            ++wrong_ptr;
            std::printf("rank %d: ompx_peer_ptr(%d) is %s, expected %s\n", me, p,
                        peer_mine ? "mapped" : "NULL", expect ? "mapped" : "NULL");
        }
        if (peer_mine == nullptr || peer_inbox == nullptr) continue;
        ++mapped;
        // Read the peer's array and store ours into the peer's inbox, both by
        // plain loads and stores from the target region.
        unsigned long long b = 0;
        #pragma omp target teams distribute parallel for reduction(+ : b) \
                is_device_ptr(peer_mine, peer_inbox) firstprivate(p, me)
        for (size_t i = 0; i < W; ++i) {
            b += peer_mine[i] != pattern(p, i);
            peer_inbox[i] = pattern(me, i);
        }
        bad += b;
    }

    // Host-issued transfers to every peer: IPC copies on the node, the NIC off it.
    for (int p = 0; p < n; ++p) {
        if (p == me) continue;
        ompx_get(p, pulled + (size_t)p * W, mine, B);
        ompx_put(p, pushed + (size_t)me * W, mine, B);
    }
    ompx_fence();

    for (int p = 0; p < n; ++p) {
        if (p == me) continue;
        const bool stored = ipc && color[p] == color[me];
        uint32_t* in = inbox + (size_t)p * W;
        uint32_t* pl = pulled + (size_t)p * W;
        uint32_t* ps = pushed + (size_t)p * W;
        unsigned long long b = 0;
        #pragma omp target teams distribute parallel for reduction(+ : b) \
                is_device_ptr(in, pl, ps) firstprivate(p, stored)
        for (size_t i = 0; i < W; ++i)
            b += (stored && in[i] != pattern(p, i)) + (pl[i] != pattern(p, i)) +
                 (ps[i] != pattern(p, i));
        bad += b;
    }

    std::printf("rank %d: %d same-node peer(s) mapped, bad words %llu, wrong "
                "pointers %llu : %s\n", me, mapped, bad, wrong_ptr,
                bad == 0 && wrong_ptr == 0 ? "PASS" : "FAIL");
    const int rc = bad == 0 && wrong_ptr == 0 ? 0 : 1;
    ompx_barrier();
    free(color);
    MPI_Comm_free(&node);
    ompx_free(pushed); ompx_free(pulled); ompx_free(inbox); ompx_free(mine);
    ompx_finalize();
    return rc;
}
