// gicc/omp_compiler.h - the compiler interface, not the user API.
//
// The LTO pass recognizes the markers below and synthesizes the host trace
// that pre-stages their descriptors, so an application compiled with
// -fpass-plugin -foffload-lto writes the transfer inside its target region and
// the staging is generated for it. Their arguments are in the buffer-index
// form the pass analyses, which is why the two heap queries live here too.
//
// Nothing in gicc/omp.h needs this header; include it only when building
// through the pass.
#pragma once

#include "gicc/omp.h"

#ifdef __cplusplus
extern "C" {
#endif

int    ompx_heap_index(void);                 // address-book index of the heap
size_t ompx_heap_offset_of(const void* addr); // heap offset of an address

#ifdef __cplusplus
}

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#include "examples/omp/gicc_omp_dwq.hpp"
#pragma omp declare target
inline void ompx_dwq_put_dev(ompx_ctx* ctx, int node,
                             int dst_buf, size_t dst_offset,
                             int src_buf, size_t src_offset, size_t bytes) {
    gicc::omp_dwq::put(ctx, node, dst_buf, dst_offset, src_buf, src_offset, bytes);
}
inline void ompx_dwq_flush_dev(ompx_ctx* ctx) { gicc::omp_dwq::flush(ctx); }
#pragma omp end declare target
#endif
#endif
