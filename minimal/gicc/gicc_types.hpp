/**
 * gicc_types.hpp - GICC common type definitions
 *
 * Platform-neutral types shared by both the NVIB/mlx5 and libfabric/CXI
 * backends.  Mirrors src/gicc/gicc_types.hpp on the nvib branch verbatim
 * so that user code is portable between backends.
 */
#pragma once

#include <cstdint>
#include <cstddef>

namespace gicc {

struct Buffer {
    void*    ptr;     // user pointer (host or device)
    size_t   size;    // bytes
    uint64_t addr;    // device-visible base address
    uint32_t lkey;    // local key (mlx5: ibv lkey; cxi: index into local desc table)
    uint32_t rkey;    // remote key (mlx5: ibv rkey; cxi: fi_mr key truncated)
    int      index;   // registration index — pass to put_no_db as dest_buf_index
};

struct RemoteBufferInfo {
    uint64_t addr;
    uint32_t rkey;
};

} // namespace gicc
