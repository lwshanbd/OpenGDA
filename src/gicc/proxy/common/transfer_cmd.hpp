/*
 * transfer_cmd.hpp - 24-byte command from GPU kernel to CPU proxy.
 *
 * Supports WRITE, READ, QUIET, and ATOMIC (FI_SUM, FI_UINT32 only).
 * ATOMIC fields reuse the WRITE layout: src points at the value to add
 * (one 4-byte uint32), dst points at the remote counter slot. `bytes` is
 * implicitly 4 for ATOMIC and the proxy ignores any other value.
 *
 * Field-direction convention (stable across cmd types):
 *   - `src_buf` / `src_offset` always name the LOCAL buffer slice.
 *   - `dst_buf` / `dst_offset` always name the REMOTE buffer slice on
 *     `dst_rank`.
 * For WRITE this matches the natural "src -> dst" reading. For READ the
 * data flow is reversed (remote -> local), but the FIELD assignment stays
 * the same — the device-side get_no_db remaps API args so src_* is still
 * local and dst_* is still remote, letting the host code use the same
 * proxy_local_buf / proxy_remote_buf accessors without branching on type.
 */
#pragma once

#include <cstdint>

namespace gicc {
namespace proxy {

enum class CmdType : uint8_t {
    EMPTY  = 0,
    WRITE  = 1,
    QUIET  = 2,
    READ   = 3,   // RDMA READ. NIC pulls (dst_rank, dst_buf, dst_offset, bytes)
                  // into local (src_buf, src_offset). CQ completion fires once
                  // the data has landed in local memory, so the same QUIET
                  // mechanism used for WRITE doubles as the read-ordering fence
                  // (proxy advances tail after CQE, kernel quiet() does
                  // __threadfence_system, then subsequent loads see the data).
    ATOMIC = 4,   // FI_SUM, FI_UINT32; non-fetching remote add. See
                  // ProxyLibfabric::submit_atomic_add for semantics.
    // Reserved: BARRIER = 5
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
