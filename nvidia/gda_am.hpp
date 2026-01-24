/**
 * gda_am.hpp - Active Message API for NVIDIA + InfiniBand
 *
 * Provides GPU-triggered active messages similar to minimal/ implementation.
 *
 * AM wire format:
 *   - seq (8B) + hdr (8B) + args (48B) = 64 bytes (short AM)
 *   - seq written LAST as release point
 *
 * Usage:
 *   GdaComm comm;
 *   GdaAm am(comm);
 *   AmArgs args;
 *   args[0] = value;
 *   am.send_handle(dest, AM_HANDLER_NOOP, args);
 *   am.trigger_and_wait();
 *   am.poll_once();
 */
#pragma once

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#include "gda_comm.hpp"
#include "gda_types.hpp"

namespace opengda {

// Device kernel to poll for incoming AMs
__global__ void am_poll_kernel(AmDeviceContext* ctx, int max_per_peer, int* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int total = 0;
    for (int peer = 0; peer < ctx->size; peer++) {
        if (peer == ctx->rank) continue;

        AmRecvState* rs = &ctx->recv_states[peer];
        int processed = 0;

        while (processed < max_per_peer) {
            uint64_t expected = rs->expected_seq;
            int slot_idx = expected & (rs->nslots - 1);
            AmSlot* slot = &rs->inbox_slots[slot_idx];

            // Check if new message arrived
            if (slot->seq != expected) break;

            __threadfence_system();

            // Process message based on handler
            // (In real implementation, dispatch to handler functions)

            rs->expected_seq = expected + 1;
            rs->tail_seq = expected + 1;
            processed++;
            total++;
        }
    }

    *result = total;
}

class GdaAm {
public:
    GdaComm& comm;

    // Per-peer inbox rings (device memory)
    std::vector<AmSlot*> d_inbox_rings;
    std::vector<MemoryRegion*> inbox_mrs;
    int nslots;

    // Per-peer send state (host)
    std::vector<AmPeerSendState> h_send_states;

    // Per-peer recv state (device)
    AmRecvState* h_recv_states;
    AmRecvState* d_recv_states;

    // Device context
    AmDeviceContext h_context;
    AmDeviceContext* d_context;

    // Staging buffers for sending
    std::vector<AmSlot*> d_staging;
    std::vector<MemoryRegion*> staging_mrs;
    size_t staging_idx;
    size_t staging_size;

    // Host temp slot
    AmSlot h_slot_temp;

    // Operation tracking
    uint64_t pending_ops;

    explicit GdaAm(GdaComm& comm_, int nslots_ = 128, size_t staging_pool = 16)
        : comm(comm_), nslots(nslots_),
          h_recv_states(nullptr), d_recv_states(nullptr),
          d_context(nullptr),
          staging_idx(0), staging_size(staging_pool),
          pending_ops(0)
    {
        allocate_inboxes();
        exchange_am_info();
        allocate_staging();
        setup_device_context();
    }

    ~GdaAm() {
        // Free staging
        for (auto* mr : staging_mrs) delete mr;
        for (auto* p : d_staging) if (p) cudaFree(p);

        // Free device context
        if (d_context) cudaFree(d_context);
        if (d_recv_states) cudaFree(d_recv_states);
        if (h_recv_states) cudaFreeHost(h_recv_states);

        // Free inbox rings
        for (auto* mr : inbox_mrs) delete mr;
        for (auto* p : d_inbox_rings) if (p) cudaFree(p);
    }

    // No copy
    GdaAm(const GdaAm&) = delete;
    GdaAm& operator=(const GdaAm&) = delete;

    /**
     * Send a handle-only AM (no payload)
     */
    int send_handle(int dest_rank, uint32_t handler_id, const AmArgs& args) {
        if (staging_idx >= staging_size) {
            fprintf(stderr, "AM staging pool exhausted\n");
            return -1;
        }

        AmSlot* d_slot = d_staging[staging_idx];
        MemoryRegion* mr = staging_mrs[staging_idx];
        staging_idx++;

        AmPeerSendState& ss = h_send_states[dest_rank];
        uint64_t seq = ss.head_seq++;

        // Prepare slot on host
        memset(&h_slot_temp, 0, sizeof(AmSlot));
        h_slot_temp.seq = seq;
        h_slot_temp.hdr.handler_id = (uint16_t)handler_id;
        h_slot_temp.hdr.flags = AM_FLAG_HANDLE_ONLY;
        h_slot_temp.hdr.payload_len = 0;
        h_slot_temp.hdr.src_rank = (uint16_t)comm.rank();
        memcpy(&h_slot_temp.args, &args, sizeof(AmArgs));

        // Copy to device staging
        CUDA_CHECK(cudaMemcpy(d_slot, &h_slot_temp, sizeof(AmSlot), cudaMemcpyHostToDevice));

        // Calculate remote slot address
        int slot_idx = seq & (ss.nslots - 1);
        uint64_t dst_slot_addr = ss.remote_ring_base + slot_idx * sizeof(AmSlot);

        // Post body PUT (hdr + args = 56 bytes, starting at offset 8)
        size_t body_offset = offsetof(AmSlot, hdr);
        size_t body_size = sizeof(AmHeader) + sizeof(AmArgs);

        // Connection already established in GdaComm constructor
        comm.put_raw({(void*)((char*)d_slot + body_offset), body_size, mr, true},
                     dest_rank,
                     dst_slot_addr + body_offset,
                     ss.remote_ring_rkey,
                     body_size, false);

        // Post seq PUT (8 bytes at offset 0) - this is the release point
        comm.put_raw({d_slot, 8, mr, true},
                     dest_rank,
                     dst_slot_addr,
                     ss.remote_ring_rkey,
                     8, true);

        pending_ops++;
        return 0;
    }

    /**
     * Trigger and wait for all pending sends
     */
    void trigger_and_wait() {
        if (pending_ops == 0) return;

        // Wait for all completions
        comm.wait(pending_ops);

        // Reset
        staging_idx = 0;
        pending_ops = 0;
    }

    /**
     * Poll for incoming AMs
     */
    int poll_once(int max_per_peer = 16) {
        int* d_result;
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(int)));

        am_poll_kernel<<<1, 1>>>(d_context, max_per_peer, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());

        int result;
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_result));

        return result;
    }

    // Accessors
    int rank() const { return comm.rank(); }
    int size() const { return comm.size(); }
    AmDeviceContext* get_device_context() { return d_context; }

private:
    void allocate_inboxes() {
        d_inbox_rings.resize(comm.size(), nullptr);
        inbox_mrs.resize(comm.size(), nullptr);

        for (int r = 0; r < comm.size(); r++) {
            if (r == comm.rank()) continue;

            // Allocate device memory for inbox ring
            size_t ring_size = nslots * sizeof(AmSlot);
            CUDA_CHECK(cudaMalloc(&d_inbox_rings[r], ring_size));
            CUDA_CHECK(cudaMemset(d_inbox_rings[r], 0, ring_size));

            // Register for RDMA
            inbox_mrs[r] = new MemoryRegion(comm.ibv->pd, d_inbox_rings[r],
                                            ring_size, true, comm.rank());
        }
    }

    void exchange_am_info() {
        h_send_states.resize(comm.size());

        // Prepare local info
        struct AmExchangeInfo {
            uint64_t ring_base;
            uint32_t ring_rkey;
            int nslots;
        };

        std::vector<AmExchangeInfo> my_info(comm.size());
        for (int r = 0; r < comm.size(); r++) {
            if (r == comm.rank()) {
                my_info[r] = {0, 0, 0};
            } else {
                my_info[r].ring_base = (uint64_t)d_inbox_rings[r];
                my_info[r].ring_rkey = inbox_mrs[r]->rkey;
                my_info[r].nslots = nslots;
            }
        }

        // Exchange with all peers
        // Each rank sends its inbox info for peers to write to
        for (int r = 0; r < comm.size(); r++) {
            if (r == comm.rank()) continue;

            AmExchangeInfo peer_info;
            comm.mpi.exchange(&my_info[r], &peer_info, sizeof(AmExchangeInfo), r);

            h_send_states[r].peer_rank = r;
            h_send_states[r].nslots = peer_info.nslots;
            h_send_states[r].head_seq = 1;  // Start at 1
            h_send_states[r].remote_ring_base = peer_info.ring_base;
            h_send_states[r].remote_ring_rkey = peer_info.ring_rkey;
        }
    }

    void allocate_staging() {
        d_staging.resize(staging_size, nullptr);
        staging_mrs.resize(staging_size, nullptr);

        for (size_t i = 0; i < staging_size; i++) {
            CUDA_CHECK(cudaMalloc(&d_staging[i], sizeof(AmSlot)));
            CUDA_CHECK(cudaMemset(d_staging[i], 0, sizeof(AmSlot)));

            staging_mrs[i] = new MemoryRegion(comm.ibv->pd, d_staging[i],
                                              sizeof(AmSlot), true, comm.rank());
        }
    }

    void setup_device_context() {
        // Allocate recv states on host (pinned) and device
        CUDA_CHECK(cudaHostAlloc(&h_recv_states, comm.size() * sizeof(AmRecvState),
                                 cudaHostAllocMapped));
        CUDA_CHECK(cudaMalloc(&d_recv_states, comm.size() * sizeof(AmRecvState)));

        // Initialize recv states
        for (int r = 0; r < comm.size(); r++) {
            h_recv_states[r].peer_rank = r;
            h_recv_states[r].nslots = nslots;
            h_recv_states[r].expected_seq = 1;
            h_recv_states[r].tail_seq = 0;
            h_recv_states[r].inbox_slots = (r == comm.rank()) ? nullptr : d_inbox_rings[r];
        }

        CUDA_CHECK(cudaMemcpy(d_recv_states, h_recv_states,
                              comm.size() * sizeof(AmRecvState),
                              cudaMemcpyHostToDevice));

        // Setup device context
        h_context.rank = comm.rank();
        h_context.size = comm.size();
        h_context.nslots = nslots;
        h_context.recv_states = d_recv_states;

        CUDA_CHECK(cudaMalloc(&d_context, sizeof(AmDeviceContext)));
        CUDA_CHECK(cudaMemcpy(d_context, &h_context, sizeof(AmDeviceContext),
                              cudaMemcpyHostToDevice));
    }
};

}  // namespace opengda
