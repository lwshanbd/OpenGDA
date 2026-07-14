// Small OpenMP-target helpers shared by the GiOMP examples.  Runtime state and
// communication live in libgicc_omp; these helpers only initialize/check an
// example's already allocated device buffer.
#pragma once

#include <cstddef>

#include "gicc/omp.h"

namespace giomp_example {

inline void fill_region(ompx_buffer buffer, size_t offset,
                        unsigned char value, size_t bytes) {
    auto* ptr = static_cast<unsigned char*>(buffer.ptr) + offset;
    #pragma omp target teams distribute parallel for is_device_ptr(ptr) \
            firstprivate(value, bytes)
    for (size_t i = 0; i < bytes; ++i) ptr[i] = value;
}

inline void fill_buffer(ompx_buffer buffer, unsigned char value) {
    fill_region(buffer, 0, value, buffer.bytes);
}

inline size_t count_region_mismatches(ompx_buffer buffer, size_t offset,
                                      unsigned char expected, size_t bytes) {
    auto* ptr = static_cast<unsigned char*>(buffer.ptr) + offset;
    size_t bad = 0;
    #pragma omp target teams distribute parallel for reduction(+:bad) \
            is_device_ptr(ptr) firstprivate(expected, bytes)
    for (size_t i = 0; i < bytes; ++i)
        if (ptr[i] != expected) ++bad;
    return bad;
}

inline size_t count_mismatches(ompx_buffer buffer, unsigned char expected) {
    return count_region_mismatches(buffer, 0, expected, buffer.bytes);
}

}  // namespace giomp_example
