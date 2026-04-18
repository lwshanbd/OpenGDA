/**
 * gda_am_context.hpp - Simplified Active Message context
 *
 * This file provides host-side AM context management:
 *   - Allocation of per-peer inbox rings in GPU device memory
 *   - Registration of memory regions for RDMA
 *   - Address exchange via Bootstrap allgather
 *
 * The AM subsystem shares DWQ/counter pool with GdaComm.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "am_types.hpp"
#include "gda_comm.hpp"
#include "memory_region.hpp"

namespace gicc {
namespace am {

// =============================================================================
// GdaAmContext - Simplified AM context class
// =============================================================================

class GdaAmContext {
public:
    // Reference to parent GdaComm (for fabric, PMI, etc.)
    GdaComm& comm;

    // Configuration
    int nslots;

    // Per-peer inbox: device memory for slots
    std::vector<am_slot_t*> d_inbox_slots;
    std::vector<MemoryRegion*> mr_inbox_slots;

    // Per-peer tail_seq: device memory for receiver's progress (sender can read this)
    std::vector<uint64_t*> d_tail_seqs;
    std::vector<MemoryRegion*> mr_tail_seqs;

    // Per-peer ack buffer: device memory for lightweight Reply (receiver writes here)
    std::vector<uint64_t*> d_ack_bufs;
    std::vector<MemoryRegion*> mr_ack_bufs;

    // Device-side context
    am_context_t* d_context;
    am_context_t h_context;

    // Device-side state arrays
    am_peer_send_state_t* d_send_states;
    am_recv_state_t* d_recv_states;

    // Host-side copies
    std::vector<am_peer_send_state_t> h_send_states;
    std::vector<am_recv_state_t> h_recv_states;

    /**
     * Initialize AM context
     * @param comm_ Reference to initialized GdaComm
     * @param nslots_ Number of slots per inbox ring (power of 2, default 128)
     */
    GdaAmContext(GdaComm& comm_, int nslots_ = AM_DEFAULT_RING_SLOTS)
        : comm(comm_),
          nslots(nslots_),
          d_context(nullptr),
          d_send_states(nullptr),
          d_recv_states(nullptr)
    {
        // Validate parameters
        if ((nslots & (nslots - 1)) != 0) {
            fprintf(stderr, "Rank %d: AM nslots must be power of 2\n", comm.rank());
            exit(1);
        }

        // Allocate per-peer inboxes and tail_seqs
        allocate_buffers();

        // Register memory regions
        register_memory();

        // Exchange addresses with all peers
        exchange_addresses();

        // Setup device state
        setup_device_state();

        if (comm.rank() == 0) {
            printf("AM Context initialized: %d peers, %d slots/peer\n",
                   comm.size(), nslots);
            printf("  Slot size: %zu bytes, Ring size per peer: %zu bytes\n",
                   sizeof(am_slot_t), nslots * sizeof(am_slot_t));
        }
    }

    ~GdaAmContext() {
        // Cleanup device state
        if (d_context) hipFree(d_context);
        if (d_send_states) hipFree(d_send_states);
        if (d_recv_states) hipFree(d_recv_states);

        // Cleanup per-peer buffers
        for (auto* mr : mr_inbox_slots) delete mr;
        for (auto* mr : mr_tail_seqs) delete mr;
        for (auto* mr : mr_ack_bufs) delete mr;
        for (auto* p : d_inbox_slots) if (p) hipFree(p);
        for (auto* p : d_tail_seqs) if (p) hipFree(p);
        for (auto* p : d_ack_bufs) if (p) hipFree(p);
    }

    // No copy
    GdaAmContext(const GdaAmContext&) = delete;
    GdaAmContext& operator=(const GdaAmContext&) = delete;

    /**
     * Get device-accessible AM context pointer
     */
    am_context_t* get_device_context() const { return d_context; }

private:
    void check_hip(hipError_t err, const char* msg) {
        if (err != hipSuccess) {
            fprintf(stderr, "Rank %d: %s failed: %s\n",
                    comm.rank(), msg, hipGetErrorString(err));
            exit(1);
        }
    }

    void allocate_buffers() {
        int npeers = comm.size();
        size_t slots_size = nslots * sizeof(am_slot_t);

        d_inbox_slots.resize(npeers, nullptr);
        d_tail_seqs.resize(npeers, nullptr);
        d_ack_bufs.resize(npeers, nullptr);

        for (int p = 0; p < npeers; p++) {
            // Allocate inbox slot array on device
            check_hip(hipMalloc(&d_inbox_slots[p], slots_size), "hipMalloc(inbox)");
            check_hip(hipMemset(d_inbox_slots[p], 0, slots_size), "hipMemset(inbox)");

            // Allocate tail_seq on device (for sender to read our progress)
            check_hip(hipMalloc(&d_tail_seqs[p], sizeof(uint64_t)), "hipMalloc(tail_seq)");
            check_hip(hipMemset(d_tail_seqs[p], 0, sizeof(uint64_t)), "hipMemset(tail_seq)");

            // Allocate ack buffer on device (for receiver to write Reply acks)
            check_hip(hipMalloc(&d_ack_bufs[p], sizeof(uint64_t)), "hipMalloc(ack_buf)");
            check_hip(hipMemset(d_ack_bufs[p], 0, sizeof(uint64_t)), "hipMemset(ack_buf)");
        }

        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    }

    void register_memory() {
        int npeers = comm.size();
        size_t slots_size = nslots * sizeof(am_slot_t);

        mr_inbox_slots.resize(npeers, nullptr);
        mr_tail_seqs.resize(npeers, nullptr);
        mr_ack_bufs.resize(npeers, nullptr);

        for (int p = 0; p < npeers; p++) {
            // Register inbox MR
            mr_inbox_slots[p] = new MemoryRegion(
                comm.fabric->domain, comm.fabric->ep, comm.fabric->cxi_info,
                d_inbox_slots[p], slots_size, true, comm.gpu_id(), comm.rank());

            // Register tail_seq MR
            mr_tail_seqs[p] = new MemoryRegion(
                comm.fabric->domain, comm.fabric->ep, comm.fabric->cxi_info,
                d_tail_seqs[p], sizeof(uint64_t), true, comm.gpu_id(), comm.rank());

            // Register ack buffer MR
            mr_ack_bufs[p] = new MemoryRegion(
                comm.fabric->domain, comm.fabric->ep, comm.fabric->cxi_info,
                d_ack_bufs[p], sizeof(uint64_t), true, comm.gpu_id(), comm.rank());
        }
    }

    void exchange_addresses() {
        int npeers = comm.size();

        // Prepare my exchange info for each peer
        std::vector<am_exchange_info_t> my_infos(npeers);
        for (int p = 0; p < npeers; p++) {
            my_infos[p].ring_base = (uint64_t)d_inbox_slots[p];
            my_infos[p].ring_key = mr_inbox_slots[p]->key;
            my_infos[p].tail_seq_addr = (uint64_t)d_tail_seqs[p];
            my_infos[p].tail_seq_key = mr_tail_seqs[p]->key;
            // Ack buffer: peer p will write Reply acks here (to my ack buffer for peer p)
            my_infos[p].ack_addr = (uint64_t)d_ack_bufs[p];
            my_infos[p].ack_key = mr_ack_bufs[p]->key;
            my_infos[p].nslots = nslots;
        }

        // Allgather per-peer info blobs via Bootstrap.
        size_t info_size = sizeof(am_exchange_info_t) * npeers;
        auto all = comm.boot.allgather(my_infos.data(), (int)info_size);

        // Populate send/recv states from each peer's contribution.
        h_send_states.resize(npeers);
        h_recv_states.resize(npeers);

        for (int p = 0; p < npeers; p++) {
            if (all[p].size() != info_size) {
                fprintf(stderr,
                        "Rank %d: AM exchange blob(%d) size mismatch: "
                        "expected %zu, got %zu\n",
                        comm.rank(), p, info_size, all[p].size());
                exit(1);
            }
            const auto* peer_infos =
                reinterpret_cast<const am_exchange_info_t*>(all[p].data());

            // Peer p's info for ME is at index [my_rank]
            const am_exchange_info_t& info_for_me = peer_infos[comm.rank()];

            // Setup send state for peer p (where I send TO peer p)
            h_send_states[p].peer_rank = p;
            h_send_states[p].nslots = info_for_me.nslots;
            h_send_states[p].head_seq = 1;  // Start at 1 (0 = empty slot)
            h_send_states[p].remote_ring_base = info_for_me.ring_base;
            h_send_states[p].remote_ring_key = info_for_me.ring_key;
            h_send_states[p].remote_tail_seq_addr = info_for_me.tail_seq_addr;
            h_send_states[p].remote_tail_seq_key = info_for_me.tail_seq_key;
            // Local ack buffer where peer p will write Reply acks to me
            h_send_states[p].local_ack = d_ack_bufs[p];

            // Setup recv state for peer p (where I receive FROM peer p)
            h_recv_states[p].peer_rank = p;
            h_recv_states[p].nslots = nslots;
            h_recv_states[p].expected_seq = 1;
            h_recv_states[p].tail_seq = 0;
            h_recv_states[p].inbox_slots = d_inbox_slots[p];
            // Remote ack buffer where I write Reply acks to peer p
            h_recv_states[p].remote_ack_addr = info_for_me.ack_addr;
            h_recv_states[p].remote_ack_key = info_for_me.ack_key;
        }
    }

    void setup_device_state() {
        int npeers = comm.size();

        // Allocate device arrays
        check_hip(hipMalloc(&d_send_states, npeers * sizeof(am_peer_send_state_t)),
                  "hipMalloc(d_send_states)");
        check_hip(hipMalloc(&d_recv_states, npeers * sizeof(am_recv_state_t)),
                  "hipMalloc(d_recv_states)");

        // Copy host state to device
        check_hip(hipMemcpy(d_send_states, h_send_states.data(),
                            npeers * sizeof(am_peer_send_state_t),
                            hipMemcpyHostToDevice),
                  "hipMemcpy(d_send_states)");
        check_hip(hipMemcpy(d_recv_states, h_recv_states.data(),
                            npeers * sizeof(am_recv_state_t),
                            hipMemcpyHostToDevice),
                  "hipMemcpy(d_recv_states)");

        // Setup context structure
        h_context.rank = comm.rank();
        h_context.size = npeers;
        h_context.nslots = nslots;
        h_context.send_states = d_send_states;
        h_context.recv_states = d_recv_states;

        // Allocate and copy context to device
        check_hip(hipMalloc(&d_context, sizeof(am_context_t)),
                  "hipMalloc(d_context)");
        check_hip(hipMemcpy(d_context, &h_context, sizeof(am_context_t),
                            hipMemcpyHostToDevice),
                  "hipMemcpy(d_context)");

        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    }

};

}  // namespace am
}  // namespace gicc
