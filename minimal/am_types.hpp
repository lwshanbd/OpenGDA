/**
 * am_types.hpp - Active Message type definitions for GPU-Direct Async
 *
 * This file defines all data structures for the GPU-triggered Active Message
 * subsystem. All structures are designed to be device-visible and cache-line
 * aligned where appropriate.
 *
 * Message Types:
 *   1. Handle-only: Fixed 64-byte args, no payload (payload_len = 0)
 *   2. Payload eager: 64-byte args + variable payload (payload_len <= 8192)
 *
 * Wire Format:
 *   - Slot body is written first (hdr + payload for payload msgs)
 *   - 8-byte seq is written LAST as the release point
 *   - Receiver polls seq until it matches expected_seq
 */
#pragma once

#include <cstdint>
#include <cstddef>

namespace opengda {
namespace am {

// =============================================================================
// Configuration Constants
// =============================================================================

// Maximum payload size for eager AM
constexpr size_t AM_MAX_PAYLOAD_SIZE = 8192;

// Ring buffer configuration
constexpr int AM_DEFAULT_RING_SLOTS = 128;     // Slots per peer ring (power of 2)

// Maximum peers supported
constexpr int AM_MAX_PEERS = 64;

// =============================================================================
// Message Flags
// =============================================================================

// AM message type flags (stored in hdr.flags)
constexpr uint8_t AM_FLAG_HANDLE_ONLY = 0x01;  // No payload, args only
constexpr uint8_t AM_FLAG_HAS_PAYLOAD = 0x02;  // Has variable-length payload

// =============================================================================
// am_args64_t - Fixed 64-byte argument structure for handle-only messages
// =============================================================================

/**
 * Exactly 64 bytes of user-defined arguments.
 * Used for handle-only messages and as the args portion of payload messages.
 *
 * The interpretation is up to the handler. Common patterns:
 *   - Small fixed-size data (counters, coordinates, etc.)
 *   - Metadata about a larger payload
 *   - Completion tokens
 */
struct am_args64_t {
    uint64_t data[8];  // 8 * 8 = 64 bytes

    __host__ __device__ void init() {
        for (int i = 0; i < 8; i++) data[i] = 0;
    }

    __host__ __device__ uint64_t& operator[](int idx) { return data[idx]; }
    __host__ __device__ const uint64_t& operator[](int idx) const { return data[idx]; }
};

static_assert(sizeof(am_args64_t) == 64, "am_args64_t must be exactly 64 bytes");

// =============================================================================
// am_hdr_t - AM message header
// =============================================================================

/**
 * AM message header (16 bytes).
 * Stored at the beginning of every AM slot body.
 *
 * For handle-only messages:
 *   - flags = AM_FLAG_HANDLE_ONLY
 *   - payload_len = 0
 *   - Body consists of: hdr (16B) + args64 (64B) = 80B total
 *
 * For payload messages:
 *   - flags = AM_FLAG_HAS_PAYLOAD
 *   - payload_len = actual payload length (1 to 8192)
 *   - Body consists of: hdr (16B) + args64 (64B) + payload (payload_len) = 80B + payload_len
 */
struct am_hdr_t {
    uint32_t handler_id;      // Handler ID for dispatch
    uint8_t  flags;           // Message type flags
    uint8_t  reserved[3];     // Padding for alignment
    uint32_t payload_len;     // Payload length (0 for handle-only)
    uint32_t src_rank;        // Source rank (for debugging/routing)
};

static_assert(sizeof(am_hdr_t) == 16, "am_hdr_t must be 16 bytes");

// =============================================================================
// am_slot_t - Single slot in the inbox ring buffer
// =============================================================================

/**
 * A single slot in the per-peer inbox ring buffer.
 *
 * Memory layout:
 *   Offset 0:      seq (8 bytes) - Written LAST as release point
 *   Offset 8:      hdr (16 bytes)
 *   Offset 24:     args (64 bytes)
 *   Offset 88:     payload (up to 8192 bytes)
 *
 * Total slot size: 8 + 16 + 64 + 8192 = 8280 bytes
 * We use 8280 as the slot size (no extra padding needed)
 *
 * CRITICAL: seq MUST be written last to ensure visibility ordering.
 * Sender writes body first, then puts seq as the final 8-byte write.
 * Receiver polls seq; when seq == expected_seq, body is guaranteed visible.
 */
constexpr size_t AM_SLOT_BODY_SIZE = sizeof(am_hdr_t) + sizeof(am_args64_t) + AM_MAX_PAYLOAD_SIZE;
constexpr size_t AM_SLOT_SIZE = 8 + AM_SLOT_BODY_SIZE;  // seq + body

struct am_slot_t {
    volatile uint64_t seq;           // Sequence number (release point, offset 0)
    am_hdr_t          hdr;           // Message header (offset 8)
    am_args64_t       args;          // 64-byte arguments (offset 24)
    uint8_t           payload[AM_MAX_PAYLOAD_SIZE];  // Payload buffer (offset 88)
};

static_assert(sizeof(am_slot_t) == AM_SLOT_SIZE, "am_slot_t must match AM_SLOT_SIZE");
static_assert(offsetof(am_slot_t, seq) == 0, "seq must be at offset 0");
static_assert(offsetof(am_slot_t, hdr) == 8, "hdr must be at offset 8");

// =============================================================================
// am_inbox_ring_t - Per-peer inbox ring buffer (GPU device memory)
// =============================================================================

/**
 * Inbox ring buffer for receiving AMs from a single peer.
 * Allocated in GPU device memory and registered for RDMA.
 *
 * The receiver maintains expected_seq (starts at 1, since seq=0 is empty).
 * Each slot is initially zeroed (seq=0 means empty).
 *
 * Slot indexing: slot_idx = seq & (nslots - 1)
 * Requires nslots to be power of 2.
 */
struct am_inbox_ring_t {
    int nslots;              // Number of slots (power of 2)
    int slot_size;           // Size of each slot (AM_SLOT_SIZE)
    uint64_t expected_seq;   // Next expected sequence number (starts at 1)
    uint64_t tail_seq;       // Consumed up to this seq (for flow control)
    uint64_t msgs_processed; // Total messages processed (stats)
    am_slot_t* slots;        // Pointer to slot array (device memory)
};

// =============================================================================
// am_peer_send_state_t - Per-peer sender state (device-visible)
// =============================================================================

/**
 * Sender-side state for one peer.
 *
 * Remote addressing:
 *   slot_idx = head_seq & (nslots - 1)
 *   remote_slot_addr = remote_ring_base + slot_idx * slot_size
 */
struct am_peer_send_state_t {
    int peer_rank;                    // Target peer rank
    int nslots;                       // Number of slots in remote ring

    uint64_t head_seq;                // Next sequence to send (starts at 1)

    // Remote ring buffer info (for RDMA put)
    uint64_t remote_ring_base;        // Remote slot array base address
    uint64_t remote_ring_key;         // Remote MR key for ring

    // Remote tail_seq address (for DWQ read to check progress)
    uint64_t remote_tail_seq_addr;    // Receiver's tail_seq address
    uint64_t remote_tail_seq_key;     // MR key for tail_seq
};

// =============================================================================
// am_recv_state_t - Per-peer receiver state (device-visible)
// =============================================================================

/**
 * Receiver-side state for one peer. Device-visible for GPU polling.
 *
 * The receiver polls slots[expected_seq & (nslots-1)].seq until it
 * equals expected_seq, then processes the message.
 */
struct am_recv_state_t {
    int peer_rank;                    // Source peer rank
    int nslots;                       // Number of slots in local ring

    uint64_t expected_seq;            // Next expected sequence (starts at 1)
    volatile uint64_t tail_seq;       // Consumed up to this seq (sender can read this)

    // Local inbox ring (device memory)
    am_slot_t* inbox_slots;           // Pointer to local slot array
};

// =============================================================================
// am_context_t - Complete AM context (host-managed, device-accessible)
// =============================================================================

/**
 * Main AM context structure. Created during init and copied to device.
 * Contains arrays of per-peer send and receive states.
 */
struct am_context_t {
    int rank;                         // My rank
    int size;                         // Total number of ranks/peers
    int nslots;                       // Slots per ring

    // Device-accessible state arrays (allocated on device)
    am_peer_send_state_t* send_states;// Array of size 'size' send states
    am_recv_state_t* recv_states;     // Array of size 'size' recv states
};

// =============================================================================
// Handler ID enumeration (extend as needed)
// =============================================================================

/**
 * Handler IDs for dispatch. Define your handlers here.
 * Dispatch uses switch(handler_id) to avoid device function pointer issues.
 */
enum AmHandlerId : uint32_t {
    AM_HANDLER_NOOP = 0,              // No operation (for testing)
    AM_HANDLER_COUNTER_ADD = 1,       // Atomic add to a device counter
    AM_HANDLER_CHECKSUM = 2,          // Compute checksum of payload
    AM_HANDLER_ECHO = 3,              // Echo back a response
    AM_HANDLER_USER_BASE = 100,       // Start of user-defined handlers
};

// =============================================================================
// Exchange info structure (for init-time address exchange)
// =============================================================================

/**
 * Per-peer info exchanged during initialization via PMI KVS.
 * Each rank publishes its inbox info for other ranks to send to it.
 */
struct am_exchange_info_t {
    uint64_t ring_base;               // Base address of inbox ring slots
    uint64_t ring_key;                // MR key for ring buffer
    uint64_t tail_seq_addr;           // Address of receiver's tail_seq (for sender to read)
    uint64_t tail_seq_key;            // MR key for tail_seq
    int nslots;                       // Number of slots
};

}  // namespace am
}  // namespace opengda
