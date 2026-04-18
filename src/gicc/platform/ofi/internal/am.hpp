/**
 * am.hpp - Active Message subsystem for the OFI backend
 *
 * AM subsystem uses Fabric's put mechanism directly.
 * AM = put(slot_body) + put(seq)
 *
 * Usage:
 *   Fabric comm;
 *   Am am(comm);
 *
 *   // Send handle-only AM
 *   am_args64_t args;
 *   args[0] = value;
 *   am.send_handle(dest_rank, AM_HANDLER_NOOP, args);
 *
 *   // Trigger and wait
 *   am.trigger_and_wait();
 *
 *   // Poll for received AMs
 *   am.poll_once();
 */
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "am_types.hpp"
#include "am_context.hpp"
#include "am_device.hpp"
#include "fabric.hpp"

namespace gicc {
namespace am {

// =============================================================================
// Am - Simplified Active Message class
// =============================================================================

class Am {
public:
    // AM context
    Context am_ctx;

    // Staging buffers (device memory, registered for RDMA)
    std::vector<am_slot_t*> d_staging;
    std::vector<MemoryRegion*> staging_mrs;
    size_t staging_idx;
    size_t staging_size;

    // Host temp buffer for preparing slots
    am_slot_t h_slot_temp;

    // Max threshold used in current batch
    uint64_t max_threshold;

    // Reply staging: device memory for 8-byte ack values
    std::vector<uint64_t*> d_reply_staging;
    std::vector<MemoryRegion*> reply_mrs;
    size_t reply_staging_idx;
    size_t reply_staging_size;

    /**
     * Initialize AM subsystem
     * @param comm Reference to initialized Fabric
     * @param nslots Slots per peer inbox ring (default 128)
     * @param staging_pool_size Number of staging slots for sending (default 16)
     */
    Am(Fabric& comm, int nslots = AM_DEFAULT_RING_SLOTS, size_t staging_pool_size = 16)
        : am_ctx(comm, nslots),
          staging_idx(0),
          staging_size(staging_pool_size),
          max_threshold(0),
          reply_staging_idx(0),
          reply_staging_size(staging_pool_size)
    {
        allocate_staging(staging_pool_size);
        allocate_reply_staging(staging_pool_size);
    }

    ~Am() {
        for (auto* mr : staging_mrs) delete mr;
        for (auto* p : d_staging) if (p) hipFree(p);
        for (auto* mr : reply_mrs) delete mr;
        for (auto* p : d_reply_staging) if (p) hipFree(p);
    }

    // No copy
    Am(const Am&) = delete;
    Am& operator=(const Am&) = delete;

    /**
     * Queue a handle-only AM send (no payload).
     * Uses Fabric::put() internally.
     *
     * @param dest_rank Destination rank
     * @param handler_id Handler to invoke
     * @param args 64-byte arguments
     * @return 0 on success, -1 on error
     */
    int send_handle(int dest_rank, uint32_t handler_id, const am_args64_t& args) {
        if (staging_idx >= staging_size) {
            fprintf(stderr, "AM staging pool exhausted (used %zu/%zu)\n",
                    staging_idx, staging_size);
            return -1;
        }

        // Get staging slot
        am_slot_t* d_slot = d_staging[staging_idx];
        MemoryRegion* mr = staging_mrs[staging_idx];
        staging_idx++;

        // Get send state
        am_peer_send_state_t& ss = am_ctx.h_send_states[dest_rank];
        uint64_t seq = ss.head_seq++;

        // Prepare slot on host
        memset(&h_slot_temp, 0, sizeof(am_slot_t));
        h_slot_temp.seq = seq;
        h_slot_temp.hdr.handler_id = (uint16_t)handler_id;
        h_slot_temp.hdr.flags = AM_FLAG_HANDLE_ONLY;
        h_slot_temp.hdr.payload_len = 0;
        h_slot_temp.hdr.src_rank = (uint16_t)am_ctx.comm.rank();
        memcpy(&h_slot_temp.args, &args, sizeof(am_args_t));

        // Copy to device staging
        hipMemcpy(d_slot, &h_slot_temp, sizeof(am_slot_t), hipMemcpyHostToDevice);

        // Calculate remote slot address
        int slot_idx = seq & (ss.nslots - 1);

        // Register staging buffer for this peer
        am_ctx.comm.set_remote_info_by_index(dest_rank, staging_idx - 1,
            ss.remote_ring_base, ss.remote_ring_key);

        // Body size: hdr + args = 8 + 48 = 56 bytes for short AM
        // Total wire: body(56) + seq(8) = 64 bytes (power of 2)
        size_t body_size = sizeof(am_hdr_t) + sizeof(am_args_t);

        // Calculate offsets for put
        // Source offset: hdr starts at offset 8 in am_slot_t
        size_t src_body_offset = offsetof(am_slot_t, hdr);
        // Dest offset: slot_idx * slot_size + field offset
        size_t dst_body_offset = slot_idx * AM_SLOT_SIZE + offsetof(am_slot_t, hdr);
        size_t dst_seq_offset = slot_idx * AM_SLOT_SIZE + offsetof(am_slot_t, seq);

        // Calculate remote address based on addressing mode
        // virt_addr mode: use absolute address
        // non-virt_addr mode: use offset within MR
        uint64_t body_rma_addr = am_ctx.comm.is_virt_addr_mode()
            ? (ss.remote_ring_base + dst_body_offset)
            : dst_body_offset;
        uint64_t seq_rma_addr = am_ctx.comm.is_virt_addr_mode()
            ? (ss.remote_ring_base + dst_seq_offset)
            : dst_seq_offset;

        // Queue body put using Fabric
        Handle body_handle;
        body_handle.buf = (char*)d_slot + src_body_offset;
        body_handle.local_desc = mr->desc;
        body_handle.rma_key = mr->key;

        uint64_t thresh1 = am_ctx.comm.put_raw(
            body_handle, dest_rank,
            body_rma_addr,
            ss.remote_ring_key,
            body_size);
        (void)thresh1;

        // Queue seq put (release point) - must come after body
        Handle seq_handle;
        seq_handle.buf = d_slot;  // seq is at offset 0
        seq_handle.local_desc = mr->desc;
        seq_handle.rma_key = mr->key;

        uint64_t thresh2 = am_ctx.comm.put_raw(
            seq_handle, dest_rank,
            seq_rma_addr,
            ss.remote_ring_key,
            sizeof(uint64_t));

        if (thresh2 > max_threshold) max_threshold = thresh2;

        return 0;
    }

    /**
     * Queue a payload AM send.
     *
     * @param dest_rank Destination rank
     * @param handler_id Handler to invoke
     * @param args 64-byte arguments
     * @param payload Payload data (host memory)
     * @param payload_len Payload length (max 8192)
     * @return 0 on success, -1 on error
     */
    int send_payload(int dest_rank, uint32_t handler_id, const am_args64_t& args,
                     const void* payload, size_t payload_len) {
        if (payload_len > AM_MAX_PAYLOAD_SIZE) {
            fprintf(stderr, "AM payload too large: %zu > %zu\n",
                    payload_len, AM_MAX_PAYLOAD_SIZE);
            return -1;
        }

        if (staging_idx >= staging_size) {
            fprintf(stderr, "AM staging pool exhausted\n");
            return -1;
        }

        am_slot_t* d_slot = d_staging[staging_idx];
        MemoryRegion* mr = staging_mrs[staging_idx];
        staging_idx++;

        am_peer_send_state_t& ss = am_ctx.h_send_states[dest_rank];
        uint64_t seq = ss.head_seq++;

        // Prepare slot
        memset(&h_slot_temp, 0, sizeof(am_slot_t));
        h_slot_temp.seq = seq;
        h_slot_temp.hdr.handler_id = (uint16_t)handler_id;
        h_slot_temp.hdr.flags = AM_FLAG_HAS_PAYLOAD;
        h_slot_temp.hdr.payload_len = (uint16_t)payload_len;
        h_slot_temp.hdr.src_rank = (uint16_t)am_ctx.comm.rank();
        memcpy(&h_slot_temp.args, &args, sizeof(am_args_t));
        memcpy(h_slot_temp.payload, payload, payload_len);

        hipMemcpy(d_slot, &h_slot_temp, sizeof(am_slot_t), hipMemcpyHostToDevice);

        int slot_idx = seq & (ss.nslots - 1);

        // Body size: hdr + args + payload = 8 + 48 + payload_len
        size_t body_size = sizeof(am_hdr_t) + sizeof(am_args_t) + payload_len;
        size_t src_body_offset = offsetof(am_slot_t, hdr);
        size_t dst_body_offset = slot_idx * AM_SLOT_SIZE + offsetof(am_slot_t, hdr);
        size_t dst_seq_offset = slot_idx * AM_SLOT_SIZE + offsetof(am_slot_t, seq);

        // Calculate remote address based on addressing mode
        uint64_t body_rma_addr = am_ctx.comm.is_virt_addr_mode()
            ? (ss.remote_ring_base + dst_body_offset)
            : dst_body_offset;
        uint64_t seq_rma_addr = am_ctx.comm.is_virt_addr_mode()
            ? (ss.remote_ring_base + dst_seq_offset)
            : dst_seq_offset;

        Handle body_handle;
        body_handle.buf = (char*)d_slot + src_body_offset;
        body_handle.local_desc = mr->desc;
        body_handle.rma_key = mr->key;

        am_ctx.comm.put_raw(
            body_handle, dest_rank,
            body_rma_addr,
            ss.remote_ring_key,
            body_size);

        Handle seq_handle;
        seq_handle.buf = d_slot;
        seq_handle.local_desc = mr->desc;
        seq_handle.rma_key = mr->key;

        uint64_t thresh = am_ctx.comm.put_raw(
            seq_handle, dest_rank,
            seq_rma_addr,
            ss.remote_ring_key,
            sizeof(uint64_t));

        if (thresh > max_threshold) max_threshold = thresh;

        return 0;
    }

    /**
     * Trigger all queued sends and wait for completion
     */
    void trigger_and_wait() {
        if (max_threshold == 0) return;

        am_ctx.comm.trigger(max_threshold);
        am_ctx.comm.wait(max_threshold);

        // Reset for next batch
        staging_idx = 0;
        reply_staging_idx = 0;
        max_threshold = 0;
    }

    /**
     * Poll for incoming AMs (one-shot, not persistent)
     * @param max_per_peer Max messages to process per peer
     * @return Total messages processed
     */
    int poll_once(int max_per_peer = 16) {
        int* d_result;
        hipMalloc(&d_result, sizeof(int));
        hipMemset(d_result, 0, sizeof(int));

        hipLaunchKernelGGL(am_poll_once_kernel, dim3(1), dim3(1), 0, 0,
                           am_ctx.d_context, max_per_peer, d_result);
        hipDeviceSynchronize();

        int result;
        hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost);
        hipFree(d_result);

        return result;
    }

    /**
     * Queue a lightweight Reply (8 bytes only).
     * This is more efficient than sending a full AM for simple ack responses.
     *
     * Reply writes a single uint64_t to the sender's ack buffer.
     *
     * @param dest_rank Rank to reply to (original sender)
     * @param ack_value Value to write (typically a sequence/counter)
     * @return 0 on success, -1 on error
     */
    int reply(int dest_rank, uint64_t ack_value) {
        if (reply_staging_idx >= reply_staging_size) {
            fprintf(stderr, "Reply staging pool exhausted (used %zu/%zu)\n",
                    reply_staging_idx, reply_staging_size);
            return -1;
        }

        // Get staging slot for the 8-byte ack value
        uint64_t* d_ack = d_reply_staging[reply_staging_idx];
        MemoryRegion* mr = reply_mrs[reply_staging_idx];
        reply_staging_idx++;

        // Copy ack value to device staging
        hipMemcpy(d_ack, &ack_value, sizeof(uint64_t), hipMemcpyHostToDevice);

        // Get recv state to find where to send the reply
        am_recv_state_t& rs = am_ctx.h_recv_states[dest_rank];

        // Calculate remote address based on addressing mode
        uint64_t rma_addr = am_ctx.comm.is_virt_addr_mode()
            ? rs.remote_ack_addr
            : 0;  // offset 0 within the MR

        // Queue the 8-byte put
        Handle ack_handle;
        ack_handle.buf = d_ack;
        ack_handle.local_desc = mr->desc;
        ack_handle.rma_key = mr->key;

        uint64_t thresh = am_ctx.comm.put_raw(
            ack_handle, dest_rank,
            rma_addr,
            rs.remote_ack_key,
            sizeof(uint64_t));

        if (thresh > max_threshold) max_threshold = thresh;

        return 0;
    }

    /**
     * Optional: Check if we need to wait before sending more AMs.
     * Simple check based on head_seq vs ring size.
     *
     * @param dest_rank Destination to check
     * @param margin How many slots should be free
     * @return true if we can send, false if ring might be full
     */
    bool check_can_send(int dest_rank, int margin = 32) {
        am_peer_send_state_t& ss = am_ctx.h_send_states[dest_rank];
        // Simplified: just check if head_seq is within ring size
        // (Conservative: assume tail=0, so no messages consumed yet)
        uint64_t in_flight = ss.head_seq;
        return in_flight < (uint64_t)(ss.nslots - margin);
    }

    // Accessors
    int rank() const { return am_ctx.comm.rank(); }
    int size() const { return am_ctx.comm.size(); }
    Fabric& comm() { return am_ctx.comm; }

private:
    void allocate_staging(size_t pool_size) {
        d_staging.resize(pool_size, nullptr);
        staging_mrs.resize(pool_size, nullptr);

        for (size_t i = 0; i < pool_size; i++) {
            hipMalloc(&d_staging[i], sizeof(am_slot_t));
            hipMemset(d_staging[i], 0, sizeof(am_slot_t));

            staging_mrs[i] = new MemoryRegion(
                am_ctx.comm.fabric->domain,
                am_ctx.comm.fabric->ep,
                am_ctx.comm.fabric->cxi_info,
                d_staging[i], sizeof(am_slot_t),
                true, am_ctx.comm.gpu_id(), am_ctx.comm.rank());
        }

        hipDeviceSynchronize();
    }

    void allocate_reply_staging(size_t pool_size) {
        d_reply_staging.resize(pool_size, nullptr);
        reply_mrs.resize(pool_size, nullptr);

        for (size_t i = 0; i < pool_size; i++) {
            hipMalloc(&d_reply_staging[i], sizeof(uint64_t));
            hipMemset(d_reply_staging[i], 0, sizeof(uint64_t));

            reply_mrs[i] = new MemoryRegion(
                am_ctx.comm.fabric->domain,
                am_ctx.comm.fabric->ep,
                am_ctx.comm.fabric->cxi_info,
                d_reply_staging[i], sizeof(uint64_t),
                true, am_ctx.comm.gpu_id(), am_ctx.comm.rank());
        }

        hipDeviceSynchronize();
    }
};

}  // namespace am
}  // namespace gicc
