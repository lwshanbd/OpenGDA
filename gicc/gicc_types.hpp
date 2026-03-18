/**
 * gicc_types.hpp - GICC common type definitions
 *
 * Platform-neutral types used by both gicc.hpp and platform implementations.
 */
#pragma once

#include <cstdint>
#include <cstddef>

namespace gicc {

struct Buffer {
    void*    ptr;
    size_t   size;
    uint64_t addr;
    uint32_t lkey;
    uint32_t rkey;
    int      index;
};

struct RemoteBufferInfo {
    uint64_t addr;
    uint32_t rkey;
};

} // namespace gicc
