/**
 * nvib_am_types.hpp - Active Message type definitions for NVIDIA IB
 *
 * Message Types:
 *   1. Short AM (handle-only): 64 bytes total (power of 2 for efficiency)
 *   2. Long AM (with payload): variable size up to 8KB
 *
 * Wire Format (64 bytes for short AM):
 *   - seq (8B) + hdr (8B) + args (48B) = 64 bytes
 *   - seq is written LAST as the release point
 */
#pragma once

#include <cstdint>
#include <cstddef>

namespace opengda {
namespace nvib_am {

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

constexpr uint8_t AM_FLAG_HANDLE_ONLY = 0x01;  // Short AM, no payload
constexpr uint8_t AM_FLAG_HAS_PAYLOAD = 0x02;  // Long AM with payload

// =============================================================================
// am_args_t - Arguments for short AM (48 bytes = 6 x uint64_t)
// =============================================================================

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

struct am_hdr_t {
    uint16_t handler_id;      // Handler ID for dispatch (0-65535)
    uint16_t src_rank;        // Source rank (0-65535)
    uint16_t payload_len;     // Payload length (0 for short AM)
    uint8_t  flags;           // Message type flags
    uint8_t  reserved;        // Padding/future use
};

static_assert(sizeof(am_hdr_t) == 8, "am_hdr_t must be 8 bytes");

// =============================================================================
// am_slot_t - Full AM slot (for messages with optional payload)
// =============================================================================

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
// Per-peer state structures
// =============================================================================

/**
 * Sender-side state for one peer (host-side tracking).
 */
struct am_peer_send_state_t {
    int peer_rank;                    // Target peer rank
    int nslots;                       // Number of slots in remote ring

    uint64_t head_seq;                // Next sequence to send (starts at 1)

    // Remote ring buffer info (for RDMA put)
    uint64_t remote_ring_base;        // Remote slot array base address
    uint32_t remote_ring_rkey;        // Remote MR rkey for ring
};

/**
 * Receiver-side state for one peer. Device-visible for GPU polling.
 */
struct am_recv_state_t {
    int peer_rank;                    // Source peer rank
    int nslots;                       // Number of slots in local ring

    uint64_t expected_seq;            // Next expected sequence (starts at 1)
    volatile uint64_t tail_seq;       // Consumed up to this seq

    // Local inbox ring (device memory)
    am_slot_t* inbox_slots;           // Pointer to local slot array
};

/**
 * Main AM context structure (device-accessible).
 */
struct am_context_t {
    int rank;                         // My rank
    int size;                         // Total number of ranks/peers
    int nslots;                       // Slots per ring

    // Device-accessible state arrays
    am_recv_state_t* recv_states;
};

// =============================================================================
// Handler ID enumeration
// =============================================================================

enum AmHandlerId : uint32_t {
    AM_HANDLER_NOOP = 0,              // No operation (for testing)
    AM_HANDLER_COUNTER_ADD = 1,       // Atomic add to a device counter
    AM_HANDLER_ECHO = 2,              // Echo back a response
    AM_HANDLER_USER_BASE = 100,       // Start of user-defined handlers
};

// =============================================================================
// Exchange info structure (for init-time address exchange)
// =============================================================================

struct am_exchange_info_t {
    uint64_t ring_base;               // Base address of inbox ring slots
    uint32_t ring_rkey;               // MR rkey for ring buffer
    int nslots;                       // Number of slots
};

// =============================================================================
// ReqRep mode: Lightweight Reply support
// =============================================================================

/**
 * Reply token - passed in Request, used by Receiver to send Reply.
 * Receiver writes ack_seq to sender's ack_addr via RDMA WRITE.
 */
struct am_reply_token_t {
    uint64_t ack_addr;                // Sender's ack buffer address (GPU memory)
    uint32_t ack_rkey;                // Sender's ack buffer rkey
    uint32_t ack_seq;                 // Sequence number to write for ack
};

/**
 * Ack entry - what sender polls for Reply completion.
 * Just 8 bytes, written atomically by receiver.
 */
struct am_ack_entry_t {
    volatile uint64_t seq;            // Ack sequence (0 = not received)
};

static_assert(sizeof(am_ack_entry_t) == 8, "am_ack_entry_t must be 8 bytes");

}  // namespace nvib_am
}  // namespace opengda
