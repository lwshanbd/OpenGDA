/*
 * transfer_cmd.hpp - 24-byte command from GPU kernel to CPU proxy.
 *
 * Supports WRITE, QUIET, and ATOMIC (FI_SUM, FI_UINT32 only). ATOMIC
 * fields reuse the WRITE layout: src points at the value to add (one
 * 4-byte uint32), dst points at the remote counter slot. `bytes` is
 * implicitly 4 for ATOMIC and the proxy ignores any other value.
 */
#pragma once

#include <cstdint>

namespace gicc {
namespace proxy {

enum class CmdType : uint8_t {
    EMPTY  = 0,
    WRITE  = 1,
    QUIET  = 2,
    ATOMIC = 4,   // FI_SUM, FI_UINT32; non-fetching remote add. See
                  // ProxyLibfabric::submit_atomic_add for semantics.
    // Reserved: GET = 3, BARRIER = 5
};

#pragma pack(push, 1)
struct TransferCmd {
    CmdType  cmd_type;
    uint8_t  dst_rank;
    uint8_t  src_buf;
    uint8_t  dst_buf;
    uint32_t bytes;
    uint64_t src_offset;
    uint64_t dst_offset;
};
#pragma pack(pop)
static_assert(sizeof(TransferCmd) == 24, "TransferCmd must be 24 bytes");

} // namespace proxy
} // namespace gicc
