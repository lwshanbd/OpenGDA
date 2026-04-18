/**
 * types.hpp - Type definitions for the MLX5 backend
 *
 * Common types used across the MLX5 (NVIDIA + InfiniBand) implementation.
 */
#pragma once

#include <cstdint>
#include <cstddef>

namespace gicc::mlx5 {

// Configuration constants
constexpr int MAX_PEERS = 64;
constexpr int MAX_PENDING_OPS = 256;

// Operation types
enum class OpType : uint8_t {
    PUT = 0,
    GET = 1,
    ATOMIC_ADD = 2,
    ATOMIC_CAS = 3
};

// Operation status
enum class OpStatus : uint8_t {
    PENDING = 0,
    TRIGGERED = 1,
    COMPLETED = 2,
    ERROR = 3
};

// Work request for GPU-triggered operations
// This structure is written by GPU and read by NIC/driver
struct __attribute__((aligned(64))) WorkRequest {
    volatile uint64_t trigger;         // GPU writes to trigger operation
    uint64_t local_addr;               // Local buffer address
    uint64_t remote_addr;              // Remote buffer address
    uint32_t length;                   // Transfer size
    uint32_t lkey;                     // Local memory key
    uint32_t rkey;                     // Remote memory key
    uint16_t dest_rank;                // Destination rank
    uint8_t op_type;                   // OpType
    uint8_t flags;                     // Reserved flags
    volatile uint64_t completion;      // Set when operation completes
};

static_assert(sizeof(WorkRequest) == 64, "WorkRequest must be 64 bytes");

// Completion queue entry visible to GPU
struct __attribute__((aligned(8))) Completion {
    volatile uint64_t seq;             // Sequence number (incremented on completion)
};

// Device-side context passed to GPU kernels
struct DeviceContext {
    // Trigger counter - GPU writes here to trigger operations
    volatile uint64_t* trigger_cntr;

    // Completion counter - GPU polls here for completion
    volatile uint64_t* completion_cntr;

    // Work request array (GPU can see pending operations)
    WorkRequest* work_requests;
    int num_work_requests;

    // Rank info
    int my_rank;
    int num_ranks;
};

// Active Message types (similar to minimal/)

// AM header - 8 bytes
struct AmHeader {
    uint16_t handler_id;
    uint16_t src_rank;
    uint16_t payload_len;
    uint8_t flags;
    uint8_t reserved;
};

static_assert(sizeof(AmHeader) == 8, "AmHeader must be 8 bytes");

// AM arguments - 48 bytes (6 x uint64_t)
struct AmArgs {
    uint64_t data[6];

    __host__ __device__ void init() {
        for (int i = 0; i < 6; i++) data[i] = 0;
    }

    __host__ __device__ uint64_t& operator[](int idx) { return data[idx]; }
    __host__ __device__ const uint64_t& operator[](int idx) const { return data[idx]; }
};

static_assert(sizeof(AmArgs) == 48, "AmArgs must be 48 bytes");

// AM slot - 64 bytes total (power of 2)
struct __attribute__((aligned(64))) AmSlot {
    volatile uint64_t seq;             // Sequence number (written last as release)
    AmHeader hdr;                      // Header
    AmArgs args;                       // Arguments
};

static_assert(sizeof(AmSlot) == 64, "AmSlot must be 64 bytes");

// AM slot with payload - 8KB total
constexpr size_t AM_MAX_PAYLOAD = 8192;
constexpr size_t AM_SLOT_SIZE = 64 + AM_MAX_PAYLOAD;

struct AmSlotFull {
    volatile uint64_t seq;
    AmHeader hdr;
    AmArgs args;
    uint8_t payload[AM_MAX_PAYLOAD];
};

// AM handler IDs
enum AmHandlerId : uint32_t {
    AM_HANDLER_NOOP = 0,
    AM_HANDLER_COUNTER_ADD = 1,
    AM_HANDLER_ECHO = 2,
    AM_HANDLER_USER_BASE = 100
};

// AM flags
constexpr uint8_t AM_FLAG_HANDLE_ONLY = 0x01;
constexpr uint8_t AM_FLAG_HAS_PAYLOAD = 0x02;

// Per-peer send state
struct AmPeerSendState {
    int peer_rank;
    int nslots;
    uint64_t head_seq;
    uint64_t remote_ring_base;
    uint32_t remote_ring_rkey;
};

// Per-peer receive state (device-visible)
struct AmRecvState {
    int peer_rank;
    int nslots;
    uint64_t expected_seq;
    volatile uint64_t tail_seq;
    AmSlot* inbox_slots;
};

// AM context for device
struct AmDeviceContext {
    int rank;
    int size;
    int nslots;
    AmRecvState* recv_states;
};

}  // namespace gicc::mlx5
