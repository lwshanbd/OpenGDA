/*
 * transfer_cmd.hpp - 24-byte command from GPU kernel to CPU proxy.
 * MVP supports WRITE and QUIET; other types reserved for later phases.
 */
#pragma once

#include <cstdint>

namespace gicc {
namespace proxy {

enum class CmdType : uint8_t {
    EMPTY = 0,
    WRITE = 1,
    QUIET = 2,
    // Reserved: GET = 3, ATOMIC = 4, BARRIER = 5
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
