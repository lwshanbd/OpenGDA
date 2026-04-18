/**
 * am_types.hpp - Active Message type definitions for GPU-Direct Async
 *
 * This file defines all data structures for the GPU-triggered Active Message
 * subsystem. All structures are designed to be device-visible and cache-line
 * aligned where appropriate.
 *
 * Message Types:
 *   1. Short AM (handle-only): 64 bytes total (power of 2 for efficiency)
 *   2. Long AM (with payload): variable size
 *
 * Short AM Wire Format (64 bytes):
 *   - seq (8B) + hdr (8B) + args (48B) = 64 bytes
 *   - seq is written LAST as the release point
 */
#pragma once

#include <cstdint>
#include <cstddef>

namespace gicc {
namespace am {

// =============================================================================
// Configuration Constants
// =============================================================================

// Maximum payload size for long AM
constexpr size_t AM_MAX_PAYLOAD_SIZE = 8192;

// Ring buffer configuration
constexpr int AM_DEFAULT_RING_SLOTS = 128;     // Slots per peer ring (power of 2)

// Maximum peers supported
constexpr int AM_MAX_PEERS = 64;

// Short AM total size (power of 2)
constexpr size_t AM_SHORT_SIZE = 64;

// =============================================================================
// Message Flags
// =============================================================================

// AM message type flags (stored in hdr.flags)
constexpr uint8_t AM_FLAG_HANDLE_ONLY = 0x01;  // Short AM, no payload
constexpr uint8_t AM_FLAG_HAS_PAYLOAD = 0x02;  // Long AM with payload

// =============================================================================
// am_args_t - Arguments for short AM (48 bytes = 6 x uint64_t)
// =============================================================================

/**
 * 48 bytes of user-defined arguments for short (handle-only) AM.
 * Designed so that seq(8) + hdr(8) + args(48) = 64 bytes (power of 2).
 */
struct am_args_t {
    uint64_t data[6];  // 6 * 8 = 48 bytes

    __host__ __device__ void init() {
        for (int i = 0; i < 6; i++) data[i] = 0;
    }

    __host__ __device__ uint64_t& operator[](int idx) { return data[idx]; }
    __host__ __device__ const uint64_t& operator[](int idx) const { return data[idx]; }
};

static_assert(sizeof(am_args_t) == 48, "am_args_t must be exactly 48 bytes");

// =============================================================================
// am_hdr_t - Compact AM header (8 bytes)
// =============================================================================

/**
 * Compact AM header (8 bytes).
 * Designed for efficiency: seq(8) + hdr(8) + args(48) = 64 bytes.
 */
struct am_hdr_t {
    uint16_t handler_id;      // Handler ID for dispatch (0-65535)
    uint16_t src_rank;        // Source rank (0-65535)
    uint16_t payload_len;     // Payload length (0 for short AM, >0 for long AM)
    uint8_t  flags;           // Message type flags
    uint8_t  reserved;        // Padding/future use
};

static_assert(sizeof(am_hdr_t) == 8, "am_hdr_t must be 8 bytes");

// =============================================================================
// am_short_t - Short AM slot (64 bytes, power of 2)
// =============================================================================

/**
 * Short AM slot for handle-only messages.
 * Total size: 64 bytes (optimal for cache lines and DMA).
 *
 * Memory layout:
 *   Offset 0:  seq (8 bytes) - Written LAST as release point
 *   Offset 8:  hdr (8 bytes)
 *   Offset 16: args (48 bytes)
 */
struct am_short_t {
    volatile uint64_t seq;    // Sequence number (release point)
    am_hdr_t          hdr;    // Compact header
    am_args_t         args;   // Arguments
};

static_assert(sizeof(am_short_t) == 64, "am_short_t must be 64 bytes");
static_assert(sizeof(am_short_t) == AM_SHORT_SIZE, "am_short_t must match AM_SHORT_SIZE");

// =============================================================================
// am_slot_t - Full AM slot (for long messages with payload)
// =============================================================================

/**
 * Full AM slot for messages with payload.
 * Uses the same header format but includes payload buffer.
 *
 * Memory layout:
 *   Offset 0:  seq (8 bytes)
 *   Offset 8:  hdr (8 bytes)
 *   Offset 16: args (48 bytes)
 *   Offset 64: payload (up to 8192 bytes)
 *
 * Total: 64 + 8192 = 8256 bytes
 */
constexpr size_t AM_SLOT_SIZE = AM_SHORT_SIZE + AM_MAX_PAYLOAD_SIZE;

struct am_slot_t {
    volatile uint64_t seq;                        // Sequence number (release point)
    am_hdr_t          hdr;                        // Compact header
    am_args_t         args;                       // Arguments
    uint8_t           payload[AM_MAX_PAYLOAD_SIZE];  // Payload buffer
};

static_assert(sizeof(am_slot_t) == AM_SLOT_SIZE, "am_slot_t must match AM_SLOT_SIZE");
static_assert(offsetof(am_slot_t, seq) == 0, "seq must be at offset 0");
static_assert(offsetof(am_slot_t, hdr) == 8, "hdr must be at offset 8");
static_assert(offsetof(am_slot_t, args) == 16, "args must be at offset 16");
static_assert(offsetof(am_slot_t, payload) == 64, "payload must be at offset 64");

// =============================================================================
// Legacy compatibility: am_args64_t (alias to am_args_t for code that uses it)
// =============================================================================

// Note: am_args_t is now 48 bytes, not 64. If you need 64-byte args,
// use the payload field for additional data.
using am_args64_t = am_args_t;

// =============================================================================
// Per-peer state structures
// =============================================================================

/**
 * Sender-side state for one peer.
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

    // Local ack buffer for lightweight Reply (receiver writes here)
    volatile uint64_t* local_ack;     // Pointer to my ack buffer for this peer
};

/**
 * Receiver-side state for one peer. Device-visible for GPU polling.
 */
struct am_recv_state_t {
    int peer_rank;                    // Source peer rank
    int nslots;                       // Number of slots in local ring

    uint64_t expected_seq;            // Next expected sequence (starts at 1)
    volatile uint64_t tail_seq;       // Consumed up to this seq (sender can read this)

    // Local inbox ring (device memory)
    am_slot_t* inbox_slots;           // Pointer to local slot array

    // Remote ack buffer address for lightweight Reply
    // (when I receive a Request from this peer, I reply to this address)
    uint64_t remote_ack_addr;         // Sender's ack buffer address
    uint64_t remote_ack_key;          // MR key for ack buffer
};

/**
 * Main AM context structure.
 */
struct am_context_t {
    int rank;                         // My rank
    int size;                         // Total number of ranks/peers
    int nslots;                       // Slots per ring

    // Device-accessible state arrays (allocated on device)
    am_peer_send_state_t* send_states;
    am_recv_state_t* recv_states;
};

// =============================================================================
// Handler ID enumeration
// =============================================================================

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

struct am_exchange_info_t {
    uint64_t ring_base;               // Base address of inbox ring slots
    uint64_t ring_key;                // MR key for ring buffer
    uint64_t tail_seq_addr;           // Address of receiver's tail_seq
    uint64_t tail_seq_key;            // MR key for tail_seq
    uint64_t ack_addr;                // Address of sender's ack buffer
    uint64_t ack_key;                 // MR key for ack buffer
    int nslots;                       // Number of slots
};

}  // namespace am
}  // namespace gicc
