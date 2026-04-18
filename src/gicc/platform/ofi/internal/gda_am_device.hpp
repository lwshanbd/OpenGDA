/**
 * gda_am_device.hpp - Simplified GPU device-side Active Message APIs
 *
 * Provides device functions for:
 *   - Polling for incoming AMs
 *   - Handler dispatch
 *
 * All functions are __device__ and called from GPU kernels.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include "am_types.hpp"

namespace gicc {
namespace am {

// =============================================================================
// Handler implementations (device functions)
// =============================================================================

/**
 * Handler 0: No-op (for testing)
 */
__device__
int am_handler_noop(am_context_t* ctx, int src_rank,
                    const am_hdr_t* hdr, const am_args64_t* args,
                    const uint8_t* payload, size_t payload_len) {
    (void)ctx; (void)src_rank; (void)hdr; (void)args; (void)payload; (void)payload_len;
    return 0;
}

/**
 * Handler 1: Atomic add to a device counter
 * args[0] = pointer to device counter (uint64_t*)
 * args[1] = value to add
 */
__device__
int am_handler_counter_add(am_context_t* ctx, int src_rank,
                           const am_hdr_t* hdr, const am_args64_t* args,
                           const uint8_t* payload, size_t payload_len) {
    (void)ctx; (void)src_rank; (void)hdr; (void)payload; (void)payload_len;

    uint64_t* counter = (uint64_t*)args->data[0];
    uint64_t value = args->data[1];

    if (counter) {
        atomicAdd((unsigned long long*)counter, (unsigned long long)value);
    }

    return 0;
}

/**
 * Handler 2: Compute checksum of payload
 * args[0] = pointer to result array (uint64_t*)
 * args[1] = index in result array
 */
__device__
int am_handler_checksum(am_context_t* ctx, int src_rank,
                        const am_hdr_t* hdr, const am_args64_t* args,
                        const uint8_t* payload, size_t payload_len) {
    (void)ctx; (void)src_rank; (void)hdr;

    uint64_t* results = (uint64_t*)args->data[0];
    uint64_t idx = args->data[1];

    if (results && payload && payload_len > 0) {
        uint64_t checksum = 0;
        for (size_t i = 0; i < payload_len; i++) {
            checksum ^= ((uint64_t)payload[i] << ((i % 8) * 8));
        }
        results[idx] = checksum;
    }

    return 0;
}

/**
 * Handler 3: Echo (placeholder)
 */
__device__
int am_handler_echo(am_context_t* ctx, int src_rank,
                    const am_hdr_t* hdr, const am_args64_t* args,
                    const uint8_t* payload, size_t payload_len) {
    (void)ctx; (void)src_rank; (void)hdr; (void)args; (void)payload; (void)payload_len;
    return 0;
}

// =============================================================================
// Handler dispatch
// =============================================================================

/**
 * Dispatch to handler based on handler_id
 */
__device__
int am_dispatch_handler(am_context_t* ctx, int src_rank,
                        const am_hdr_t* hdr, const am_args64_t* args,
                        const uint8_t* payload, size_t payload_len) {
    switch (hdr->handler_id) {
        case AM_HANDLER_NOOP:
            return am_handler_noop(ctx, src_rank, hdr, args, payload, payload_len);
        case AM_HANDLER_COUNTER_ADD:
            return am_handler_counter_add(ctx, src_rank, hdr, args, payload, payload_len);
        case AM_HANDLER_CHECKSUM:
            return am_handler_checksum(ctx, src_rank, hdr, args, payload, payload_len);
        case AM_HANDLER_ECHO:
            return am_handler_echo(ctx, src_rank, hdr, args, payload, payload_len);
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
__device__
int am_poll_peer(am_context_t* ctx, am_recv_state_t* recv_state, int max_poll) {
    int processed = 0;

    for (int i = 0; i < max_poll; i++) {
        // Calculate slot index
        int slot_idx = recv_state->expected_seq & (recv_state->nslots - 1);
        am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

        // Check if message arrived (seq matches expected)
        if (slot->seq != recv_state->expected_seq) {
            break;  // No more messages
        }

        // Memory fence to ensure body is visible
        __threadfence_system();

        // Extract message components
        const am_hdr_t* hdr = &slot->hdr;
        const am_args64_t* args = &slot->args;
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
__device__
int am_poll_all(am_context_t* ctx, int max_poll_per_peer) {
    int total = 0;

    for (int p = 0; p < ctx->size; p++) {
        if (p == ctx->rank) continue;  // Skip self
        total += am_poll_peer(ctx, &ctx->recv_states[p], max_poll_per_peer);
    }

    return total;
}

// =============================================================================
// Kernel for one-shot polling
// =============================================================================

/**
 * One-shot AM poll kernel
 * @param ctx Device AM context
 * @param max_poll_per_peer Max messages per peer
 * @param result Output: total messages processed
 */
__global__
void am_poll_once_kernel(am_context_t* ctx, int max_poll_per_peer, int* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int processed = am_poll_all(ctx, max_poll_per_peer);

    if (result) {
        *result = processed;
    }
}

}  // namespace am
}  // namespace gicc
