/**
 * am_context.hpp - Active Message context for NVIDIA IB
 *
 * This file provides host-side AM context management:
 *   - Allocation of per-peer inbox rings in GPU device memory
 *   - Registration of memory regions for RDMA
 *   - Address exchange via Bootstrap collective
 */
#pragma once

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "am_types.hpp"
#include "gicc/bootstrap/bootstrap.hpp"
#include "gicc/util/memory_region.hpp"
#include "devx_qp.hpp"
#include "device_opt.cuh"

namespace gicc::mlx5 {
namespace am {

// =============================================================================
// Connection info for QP exchange
// =============================================================================

struct QpConnInfo {
    uint32_t qpn;
    uint16_t lid;
    uint8_t  gid[16];
    uint32_t psn;
};

// =============================================================================
// Context - Active Message context class
// =============================================================================

class Context {
public:
    // Bootstrap handle (provides rank/size and collectives)
    gicc::Bootstrap& boot;

    // IB context
    struct ibv_context* ib_ctx;
    struct ibv_pd* pd;
    DevxQp* qp;

    // Configuration
    int nslots;

    // Per-peer inbox: device memory for slots
    std::vector<am_slot_t*> d_inbox_slots;
    std::vector<MemoryRegion*> mr_inbox_slots;

    // Device-side context
    am_context_t* d_context;
    am_context_t h_context;

    // Device-side recv state array
    am_recv_state_t* d_recv_states;

    // Host-side copies
    std::vector<am_peer_send_state_t> h_send_states;
    std::vector<am_recv_state_t> h_recv_states;

    // Device state for RDMA operations
    DeviceStateOpt* d_gda_state;
    DeviceStateOpt h_gda_state;

    /**
     * Initialize AM context
     * @param ib_ctx_ IB context
     * @param pd_ Protection domain
     * @param qp_ DevX QP for RDMA
     * @param nslots_ Number of slots per inbox ring (power of 2)
     */
    Context(gicc::Bootstrap& boot_,
                  struct ibv_context* ib_ctx_,
                  struct ibv_pd* pd_,
                  DevxQp* qp_,
                  int nslots_ = AM_DEFAULT_RING_SLOTS)
        : boot(boot_),
          ib_ctx(ib_ctx_),
          pd(pd_),
          qp(qp_),
          nslots(nslots_),
          d_context(nullptr),
          d_recv_states(nullptr),
          d_gda_state(nullptr)
    {

        // Validate parameters
        if ((nslots & (nslots - 1)) != 0) {
            fprintf(stderr, "Rank %d: AM nslots must be power of 2\n", boot.rank());
            exit(1);
        }

        // Allocate per-peer inboxes
        allocate_buffers();

        // Register memory regions
        register_memory();

        // Exchange QP info and connect
        connect_qps();

        // Exchange AM buffer addresses
        exchange_addresses();

        // Setup device state
        setup_device_state();

        if (boot.rank() == 0) {
            printf("NVIB AM Context initialized: %d peers, %d slots/peer\n",
                   boot.size(), nslots);
            printf("  Slot size: %zu bytes, Ring size per peer: %zu bytes\n",
                   sizeof(am_slot_t), nslots * sizeof(am_slot_t));
        }
    }

    ~Context() {
        if (d_context) cudaFree(d_context);
        if (d_recv_states) cudaFree(d_recv_states);
        if (d_gda_state) cudaFree(d_gda_state);

        for (auto* mr : mr_inbox_slots) delete mr;
        for (auto* p : d_inbox_slots) if (p) cudaFree(p);
    }

    // No copy
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    int rank() const { return boot.rank(); }
    int size() const { return boot.size(); }

    am_context_t* get_device_context() const { return d_context; }
    DeviceStateOpt* get_gda_state() const { return d_gda_state; }

    // Get send state for a peer (for building RDMA operations)
    am_peer_send_state_t& get_send_state(int peer) { return h_send_states[peer]; }

    // Update device recv states after modifying host copies
    void sync_recv_states_to_device() {
        cudaMemcpy(d_recv_states, h_recv_states.data(),
                   boot.size() * sizeof(am_recv_state_t), cudaMemcpyHostToDevice);
    }

private:
    void check_cuda(cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: %s failed: %s\n",
                    boot.rank(), msg, cudaGetErrorString(err));
            exit(1);
        }
    }

    void allocate_buffers() {
        size_t slots_size = nslots * sizeof(am_slot_t);

        d_inbox_slots.resize(boot.size(), nullptr);

        for (int p = 0; p < boot.size(); p++) {
            check_cuda(cudaMalloc(&d_inbox_slots[p], slots_size), "cudaMalloc(inbox)");
            check_cuda(cudaMemset(d_inbox_slots[p], 0, slots_size), "cudaMemset(inbox)");
        }

        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    void register_memory() {
        size_t slots_size = nslots * sizeof(am_slot_t);

        mr_inbox_slots.resize(boot.size(), nullptr);

        for (int p = 0; p < boot.size(); p++) {
            mr_inbox_slots[p] = new MemoryRegion(pd, d_inbox_slots[p], slots_size, true, boot.rank());
        }
    }

    void connect_qps() {
        // Query port
        struct ibv_port_attr port_attr;
        ibv_query_port(ib_ctx, 1, &port_attr);

        union ibv_gid my_gid;
        ibv_query_gid(ib_ctx, 1, 0, &my_gid);

        // Prepare my connection info
        QpConnInfo my_conn;
        my_conn.qpn = qp->qpn;
        my_conn.lid = port_attr.lid;
        my_conn.psn = 0;
        memcpy(my_conn.gid, &my_gid, 16);

        // Gather all connection info
        auto all_conns = boot.template allgather_fixed<QpConnInfo>(my_conn);

        // For 2-rank case, connect to peer
        if (boot.size() == 2) {
            int peer = (boot.rank() == 0) ? 1 : 0;
            QpConnInfo& peer_conn = all_conns[peer];

            qp->rst2init();
            qp->init2rtr(peer_conn.qpn, peer_conn.lid, peer_conn.gid, peer_conn.psn, 3);
            qp->rtr2rts(my_conn.psn);
        }

        boot.barrier();
    }

    void exchange_addresses() {
        // Prepare my exchange info for each peer
        std::vector<am_exchange_info_t> my_infos(boot.size());
        for (int p = 0; p < boot.size(); p++) {
            my_infos[p].ring_base = (uint64_t)d_inbox_slots[p];
            my_infos[p].ring_rkey = mr_inbox_slots[p]->rkey;
            my_infos[p].nslots = nslots;
        }

        // All-to-all exchange
        auto raw_all = boot.allgather(my_infos.data(),
                                      boot.size() * (int)sizeof(am_exchange_info_t));
        std::vector<am_exchange_info_t> all_infos(boot.size() * boot.size());
        for (int r = 0; r < boot.size(); ++r) {
            std::memcpy(&all_infos[r * boot.size()], raw_all[r].data(),
                        boot.size() * sizeof(am_exchange_info_t));
        }

        // Setup states
        h_send_states.resize(boot.size());
        h_recv_states.resize(boot.size());

        for (int p = 0; p < boot.size(); p++) {
            // Get peer p's info for ME (peer p published info at index [p * size + my_rank])
            am_exchange_info_t& info_for_me = all_infos[p * boot.size() + boot.rank()];

            // Setup send state for peer p
            h_send_states[p].peer_rank = p;
            h_send_states[p].nslots = info_for_me.nslots;
            h_send_states[p].head_seq = 1;  // Start at 1 (0 = empty slot)
            h_send_states[p].remote_ring_base = info_for_me.ring_base;
            h_send_states[p].remote_ring_rkey = info_for_me.ring_rkey;

            // Setup recv state for peer p
            h_recv_states[p].peer_rank = p;
            h_recv_states[p].nslots = nslots;
            h_recv_states[p].expected_seq = 1;
            h_recv_states[p].tail_seq = 0;
            h_recv_states[p].inbox_slots = d_inbox_slots[p];
        }
    }

    void setup_device_state() {
        // Allocate device recv states
        check_cuda(cudaMalloc(&d_recv_states, boot.size() * sizeof(am_recv_state_t)),
                   "cudaMalloc(d_recv_states)");
        check_cuda(cudaMemcpy(d_recv_states, h_recv_states.data(),
                              boot.size() * sizeof(am_recv_state_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy(d_recv_states)");

        // Setup context structure
        h_context.rank = boot.rank();
        h_context.size = boot.size();
        h_context.nslots = nslots;
        h_context.recv_states = d_recv_states;

        check_cuda(cudaMalloc(&d_context, sizeof(am_context_t)),
                   "cudaMalloc(d_context)");
        check_cuda(cudaMemcpy(d_context, &h_context, sizeof(am_context_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(d_context)");

        // Setup device state for RDMA operations
        memset(&h_gda_state, 0, sizeof(h_gda_state));
        h_gda_state.qpn = qp->qpn;
        h_gda_state.nwqes = 1 << qp->log_wq_size;
        h_gda_state.nwqes_mask = h_gda_state.nwqes - 1;
        h_gda_state.wqe_buf = qp->d_wq_buf;
        h_gda_state.dbrec = qp->d_dbrec;
        h_gda_state.bf_reg = (volatile uint64_t*)qp->d_uar_reg;
        h_gda_state.prod_idx = qp->d_prod_idx;

        // CQ info
        h_gda_state.cqe = (volatile Cqe64Opt*)qp->d_cq_buf;
        h_gda_state.ncqes = qp->num_cqe;
        h_gda_state.ncqes_mask = h_gda_state.ncqes - 1;
        h_gda_state.cq_dbrec = qp->d_cq_dbrec;

        check_cuda(cudaMalloc(&d_gda_state, sizeof(DeviceStateOpt)),
                   "cudaMalloc(d_gda_state)");
        check_cuda(cudaMemcpy(d_gda_state, &h_gda_state, sizeof(DeviceStateOpt),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(d_gda_state)");

        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
};

}  // namespace am
}  // namespace gicc::mlx5
