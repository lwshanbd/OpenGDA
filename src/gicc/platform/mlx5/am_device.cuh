/**
 * am_device.cuh - GPU device-side Active Message functions for NVIDIA IB
 *
 * Provides device functions for:
 *   - Building and sending AM via RDMA WRITE
 *   - Polling for incoming AMs
 *   - Handler dispatch
 *
 * AM send is implemented as:
 *   1. RDMA WRITE: body (hdr + args + optional payload)
 *   2. RDMA WRITE: seq (release point, written last)
 *
 * Receiver polls seq field; when seq == expected_seq, message is ready.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "am_types.hpp"
#include "device_opt.cuh"

namespace gicc::mlx5 {
namespace am {

// =============================================================================
// Handler implementations (device functions)
// =============================================================================

/**
 * Handler 0: No-op (for testing/latency measurement)
 */
__device__ __forceinline__
int am_handler_noop(am_context_t* ctx, int src_rank,
                    const am_hdr_t* hdr, const am_args_t* args,
                    const uint8_t* payload, size_t payload_len) {
    (void)ctx; (void)src_rank; (void)hdr; (void)args; (void)payload; (void)payload_len;
    return 0;
}

/**
 * Handler 1: Atomic add to a device counter
 * args[0] = pointer to device counter (uint64_t*)
 * args[1] = value to add
 */
__device__ __forceinline__
int am_handler_counter_add(am_context_t* ctx, int src_rank,
                           const am_hdr_t* hdr, const am_args_t* args,
                           const uint8_t* payload, size_t payload_len) {
    (void)ctx; (void)src_rank; (void)hdr; (void)payload; (void)payload_len;

    uint64_t* counter = (uint64_t*)args->data[0];
    uint64_t value = args->data[1];

    if (counter) {
        atomicAdd((unsigned long long*)counter, (unsigned long long)value);
    }

    return 0;
}

// =============================================================================
// Handler dispatch
// =============================================================================

__device__ __forceinline__
int am_dispatch_handler(am_context_t* ctx, int src_rank,
                        const am_hdr_t* hdr, const am_args_t* args,
                        const uint8_t* payload, size_t payload_len) {
    switch (hdr->handler_id) {
        case AM_HANDLER_NOOP:
            return am_handler_noop(ctx, src_rank, hdr, args, payload, payload_len);
        case AM_HANDLER_COUNTER_ADD:
            return am_handler_counter_add(ctx, src_rank, hdr, args, payload, payload_len);
        default:
            return -1;
    }
}

// =============================================================================
// Polling: check inbox and process incoming AMs
// =============================================================================

/**
 * Poll one peer's inbox for incoming messages
 * @param ctx AM context
 * @param recv_state Receiver state for this peer
 * @param max_poll Max messages to process
 * @return Number of messages processed
 */
__device__ __forceinline__
int am_poll_peer(am_context_t* ctx, am_recv_state_t* recv_state, int max_poll) {
    int processed = 0;

    for (int i = 0; i < max_poll; i++) {
        // Calculate slot index
        int slot_idx = recv_state->expected_seq & (recv_state->nslots - 1);
        am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

        // Check if message arrived (seq matches expected)
        // Use acquire semantics to ensure we see the body after seq
        uint64_t seq;
        asm volatile("ld.acquire.gpu.global.b64 %0, [%1];"
                     : "=l"(seq) : "l"(&slot->seq));

        if (seq != recv_state->expected_seq) {
            break;  // No more messages
        }

        // Memory fence to ensure body is visible
        gda_membar_sys();

        // Extract message components
        const am_hdr_t* hdr = &slot->hdr;
        const am_args_t* args = &slot->args;
        const uint8_t* payload = (hdr->flags & AM_FLAG_HAS_PAYLOAD) ? slot->payload : nullptr;
        size_t payload_len = (hdr->flags & AM_FLAG_HAS_PAYLOAD) ? hdr->payload_len : 0;

        // Dispatch handler
        am_dispatch_handler(ctx, recv_state->peer_rank, hdr, args, payload, payload_len);

        // Update progress
        recv_state->expected_seq++;
        recv_state->tail_seq = recv_state->expected_seq - 1;
        processed++;
    }

    return processed;
}

/**
 * Poll all peers for incoming messages
 * @param ctx AM context
 * @param max_poll_per_peer Max messages per peer
 * @return Total messages processed
 */
__device__ __forceinline__
int am_poll_all(am_context_t* ctx, int max_poll_per_peer) {
    int total = 0;

    for (int p = 0; p < ctx->size; p++) {
        if (p == ctx->rank) continue;  // Skip self
        total += am_poll_peer(ctx, &ctx->recv_states[p], max_poll_per_peer);
    }

    return total;
}

/**
 * Wait for a specific message from a peer
 * @param recv_state Receiver state for the peer
 * @return 0 on success
 */
__device__ __forceinline__
int am_wait_one(am_recv_state_t* recv_state) {
    uint64_t expected = recv_state->expected_seq;
    int slot_idx = expected & (recv_state->nslots - 1);
    am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

    // Spin until message arrives
    // Use relaxed load in tight loop for performance, then acquire at the end
    uint64_t seq;
    do {
        asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                     : "=l"(seq) : "l"(&slot->seq));
    } while (seq != expected);

    // Single acquire fence after detecting message - ensures body is visible
    // This replaces ld.acquire + membar_sys with just one fence
    asm volatile("fence.acq_rel.gpu;" ::: "memory");

    // Update progress
    recv_state->expected_seq++;
    recv_state->tail_seq = expected;

    return 0;
}

// =============================================================================
// AM Send functions (build WQEs for RDMA WRITE)
// =============================================================================

/**
 * Build and send a short AM (handle-only, 64 bytes)
 *
 * OPTIMIZED VERSION: Uses a SINGLE RDMA WRITE for the entire 64-byte slot.
 *
 * The key insight is that RDMA WRITE is byte-ordered within the NIC,
 * meaning lower addresses are written before higher addresses within
 * a single DMA operation. So for ordering, we write seq FIRST in the slot
 * (offset 0), but because the receiver polls seq with relaxed ordering
 * and only does acquire fence after seeing seq, the body at offset 8
 * will be visible by the time the receiver reads it.
 *
 * @param gda_state Device state for RDMA
 * @param local_slot Local staging slot (device memory)
 * @param local_lkey Local MR lkey
 * @param remote_ring_base Remote inbox ring base address
 * @param remote_rkey Remote MR rkey
 * @param slot_idx Destination slot index in remote ring
 * @return New prod_idx after posting
 */
__device__ __forceinline__
uint64_t am_send_short(
    DeviceStateOpt* gda_state,
    am_slot_t* local_slot,
    uint32_t local_lkey,
    uint64_t remote_ring_base,
    uint32_t remote_rkey,
    int slot_idx)
{
    // Calculate destination offset for entire 64-byte short AM
    size_t dst_offset = slot_idx * AM_SLOT_SIZE;
    uint64_t remote_addr = remote_ring_base + dst_offset;

    // Get current prod_idx
    uint64_t prod = gda_load_relaxed_u64(gda_state->prod_idx);

    // Build single WQE for entire 64-byte short AM
    uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);
    gda_build_rdma_write_wqe_opt(
        gda_state,
        (uint64_t)local_slot,  // Write from start of slot
        local_lkey,
        remote_addr,
        remote_rkey,
        AM_SHORT_SIZE,         // 64 bytes
        wqe_slot,
        true                   // signaled
    );

    // Ring doorbell
    uint16_t new_prod = (uint16_t)((prod + 1) & 0xFFFF);
    gda_ring_doorbell_bf(gda_state, new_prod);

    return prod + 1;
}

/**
 * Build and send a payload AM
 *
 * @param gda_state Device state
 * @param local_slot Local staging slot
 * @param local_lkey Local MR lkey
 * @param remote_ring_base Remote inbox ring base
 * @param remote_rkey Remote MR rkey
 * @param slot_idx Destination slot index
 * @param payload_len Payload length
 * @return New prod_idx
 */
__device__ __forceinline__
uint64_t am_send_payload(
    DeviceStateOpt* gda_state,
    am_slot_t* local_slot,
    uint32_t local_lkey,
    uint64_t remote_ring_base,
    uint32_t remote_rkey,
    int slot_idx,
    size_t payload_len)
{
    // Body size includes payload
    size_t body_size = sizeof(am_hdr_t) + sizeof(am_args_t) + payload_len;
    size_t src_body_offset = offsetof(am_slot_t, hdr);
    size_t dst_body_offset = slot_idx * AM_SLOT_SIZE + offsetof(am_slot_t, hdr);
    size_t dst_seq_offset = slot_idx * AM_SLOT_SIZE + offsetof(am_slot_t, seq);

    uint64_t body_remote_addr = remote_ring_base + dst_body_offset;
    uint64_t seq_remote_addr = remote_ring_base + dst_seq_offset;

    uint64_t prod = gda_load_relaxed_u64(gda_state->prod_idx);

    // Build WQE 1: body (hdr + args + payload)
    uint16_t wqe_slot1 = (uint16_t)(prod & 0xFFFF);
    gda_build_rdma_write_wqe_opt(
        gda_state,
        (uint64_t)local_slot + src_body_offset,
        local_lkey,
        body_remote_addr,
        remote_rkey,
        body_size,
        wqe_slot1,
        false
    );

    // Build WQE 2: seq
    uint16_t wqe_slot2 = (uint16_t)((prod + 1) & 0xFFFF);
    gda_build_rdma_write_wqe_opt(
        gda_state,
        (uint64_t)local_slot,
        local_lkey,
        seq_remote_addr,
        remote_rkey,
        sizeof(uint64_t),
        wqe_slot2,
        true
    );

    uint16_t new_prod = (uint16_t)((prod + 2) & 0xFFFF);
    gda_ring_doorbell_bf(gda_state, new_prod);

    return prod + 2;
}

// =============================================================================
// Kernel for one-shot polling
// =============================================================================

__global__
void am_poll_once_kernel(am_context_t* ctx, int max_poll_per_peer, int* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int processed = am_poll_all(ctx, max_poll_per_peer);

    if (result) {
        *result = processed;
    }
}

// =============================================================================
// ReqRep Mode: Request-Reply with lightweight Reply
// =============================================================================

/**
 * Send a lightweight Reply (8 bytes only) - FAST VERSION.
 *
 * Uses pre-prepared reply buffer - NO runtime writes, NO extra fence.
 *
 * @param gda_state Device state for RDMA
 * @param ack_addr Sender's ack buffer address (from reply token)
 * @param ack_rkey Sender's ack buffer rkey (from reply token)
 * @param local_reply_addr Address of pre-prepared reply value (already contains ack_seq)
 * @param local_lkey Local MR lkey
 */
__device__ __forceinline__
void am_send_reply_fast(
    DeviceStateOpt* gda_state,
    uint64_t ack_addr,
    uint32_t ack_rkey,
    uint64_t local_reply_addr,
    uint32_t local_lkey)
{
    // Get current prod_idx
    uint64_t prod = gda_load_relaxed_u64(gda_state->prod_idx);

    // Build single WQE for 8-byte Reply - NO fence needed, data already prepared
    uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);
    gda_build_rdma_write_wqe_opt(
        gda_state,
        local_reply_addr,      // Pre-prepared reply data
        local_lkey,
        ack_addr,
        ack_rkey,
        sizeof(uint64_t),      // Only 8 bytes!
        wqe_slot,
        false                  // unsignaled for speed
    );

    // Ring doorbell
    uint16_t new_prod = (uint16_t)((prod + 1) & 0xFFFF);
    gda_ring_doorbell_bf(gda_state, new_prod);
}

/**
 * Wait for Reply by polling ack buffer.
 *
 * @param ack_entry Pointer to ack entry in device memory
 * @param expected_seq Expected ack sequence
 */
__device__ __forceinline__
void am_wait_reply(volatile am_ack_entry_t* ack_entry, uint64_t expected_seq) {
    // Spin until ack arrives
    // Use relaxed load for tight loop
    uint64_t seq;
    do {
        asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                     : "=l"(seq) : "l"(&ack_entry->seq));
    } while (seq != expected_seq);

    // Acquire fence after detecting reply
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

/**
 * Combined wait for Request and send Reply - FAST VERSION.
 *
 * Uses pre-prepared reply buffer array for zero-copy reply.
 *
 * @param recv_state Receiver state for the peer
 * @param gda_state Device state for RDMA
 * @param reply_buf_base Base address of pre-prepared reply buffer array
 * @param local_lkey Local MR lkey
 * @param iter_idx Current iteration index (to select reply buffer)
 * @return Pointer to received slot
 */
__device__ __forceinline__
am_slot_t* am_recv_and_reply_fast(
    am_recv_state_t* recv_state,
    DeviceStateOpt* gda_state,
    uint64_t reply_buf_base,
    uint32_t local_lkey,
    int iter_idx)
{
    uint64_t expected = recv_state->expected_seq;
    int slot_idx = expected & (recv_state->nslots - 1);
    am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

    // Wait for Request - relaxed polling
    uint64_t seq;
    do {
        asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                     : "=l"(seq) : "l"(&slot->seq));
    } while (seq != expected);

    // NO fence here - we just need to read reply token, ordering not critical

    // Extract reply token from args (minimal reads)
    uint64_t ack_addr = slot->args.data[0];
    uint32_t ack_rkey = (uint32_t)slot->args.data[1];

    // Send Reply using pre-prepared buffer (NO fence, NO write)
    uint64_t local_reply_addr = reply_buf_base + iter_idx * sizeof(uint64_t);
    am_send_reply_fast(gda_state, ack_addr, ack_rkey, local_reply_addr, local_lkey);

    // Update progress
    recv_state->expected_seq++;
    recv_state->tail_seq = expected;

    return slot;
}

/**
 * Send Request with embedded reply token.
 *
 * The reply token tells the receiver where to send the ack.
 * Format in args:
 *   args[0] = ack_addr (sender's ack buffer address)
 *   args[1] = ack_rkey
 *   args[2] = ack_seq (expected reply sequence)
 *
 * @param gda_state Device state
 * @param local_slot Local staging slot (must have reply token in args)
 * @param local_lkey Local MR lkey
 * @param remote_ring_base Remote inbox ring base
 * @param remote_rkey Remote MR rkey
 * @param slot_idx Destination slot index
 */
__device__ __forceinline__
void am_send_request(
    DeviceStateOpt* gda_state,
    am_slot_t* local_slot,
    uint32_t local_lkey,
    uint64_t remote_ring_base,
    uint32_t remote_rkey,
    int slot_idx)
{
    // Same as am_send_short, but caller ensures reply token is in args
    size_t dst_offset = slot_idx * AM_SLOT_SIZE;
    uint64_t remote_addr = remote_ring_base + dst_offset;

    uint64_t prod = gda_load_relaxed_u64(gda_state->prod_idx);

    uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);
    gda_build_rdma_write_wqe_opt(
        gda_state,
        (uint64_t)local_slot,
        local_lkey,
        remote_addr,
        remote_rkey,
        AM_SHORT_SIZE,
        wqe_slot,
        false                  // unsignaled - we wait for Reply instead
    );

    uint16_t new_prod = (uint16_t)((prod + 1) & 0xFFFF);
    gda_ring_doorbell_bf(gda_state, new_prod);
}

}  // namespace am
}  // namespace gicc::mlx5
