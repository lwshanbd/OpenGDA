// gicc/omp_pipeline.h - compiler-pipelined sends.
//
// Write a kernel the ordinary way and state what it sends at the end of the
// teams region:
//
//     ompx_prepare();
//     #pragma omp target teams is_device_ptr(src)
//     {
//         #pragma omp distribute parallel for [dist_schedule(static, C)]
//         for (i = 0; i < n; ++i) src[i] = ...;
//         ompx_pipelined_put(peer, dst, src, n * sizeof(*src));
//     }
//     ompx_fence();
//
// ompx_pipelined_put means: send these bytes once the kernel's writes to them
// are complete, as a host ompx_put after the kernel would.
//
// The put may instead be stated in the body of a combined construct, every
// iteration with the same arguments:
//
//     #pragma omp target teams distribute parallel for is_device_ptr(src)
//     for (i = 0; i < n; ++i) {
//         src[i] = ...;
//         ompx_pipelined_put(peer, dst, src, n * sizeof(*src));
//     }
//
// It means the same as the put after the loop, for a loop that runs at least
// once, and keeps the kernel in SPMD mode: any statement after the loop in
// a teams region makes clang emit the kernel in generic mode, which cost a
// stencil kernel a fifth of its speed on MI250X. This form always takes the
// box lowering described below. The gicc-passes
// plugin in GICC_MODE=chunk-lower proves the send can be split along the
// distribute blocks (tools/gicc-passes/src/GICCChunkAnalysis.cpp) and
// rewrites it at the grain GICC_CHUNK_GRAIN selects:
//
//   element (default)  each store to the source is repeated into a same-node
//                      peer's copy, so every word leaves as it is produced;
//                      a peer that is not IPC-mapped gets one
//                      ompx__block_put_one per block instead.
//   block              one ompx__block_put per block, issued as soon as that
//                      block's parallel region joins.
//
// The loop may also be a collapse(D) nest writing a D-dimensional box, and
// the put any contiguous range of the written object: part of what the loop
// writes (a halo face of a slab), bytes the kernel does not write at all
// (ghost cells between the rows), or both. Such a put is split at element
// grain whatever GICC_CHUNK_GRAIN says: to a same-node peer each store that
// lands in the range is repeated into the peer's copy, and the bytes of the
// range the kernel never writes are sent once, at the start of the kernel.
// To any other peer the range goes whole, once every team has stored its
// part of it: the kernel runs the loop's chunks that store into the range
// first and sends it through the CPU proxy while the rest of the loop runs.
// That needs an SPMD kernel (the put in the loop's body) and the proxy;
// otherwise the next ompx_quiet or ompx_fence sends it, as it would a put
// after the kernel.
//
// A negative peer sends nothing, so a kernel can state a put that applies
// to only some ranks (no neighbour at the edge of a decomposition).
//
// When the pass cannot prove the split, compilation fails with the reason.
// ompx_pipelined_put has no device definition: a build without the pass
// does not link, so the marker can never run untransformed. Build with
// -foffload-lto: otherwise the device runtime is inlined before the pass
// runs and the pass finds no worksharing loop.
//
// Call ompx_prepare() in every translation unit that launches such a kernel:
// the helpers read that unit's device context.
//
// C++ only.
#pragma once

#include "gicc/omp.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

// Host fallback of a target region. libomp runs the teams concurrently and
// the marker runs in each, after only its own share of the loop: there is
// no point at which one host put would carry the kernel's final data.
#if !defined(__NVPTX__) && !defined(__AMDGCN__)
extern "C" __attribute__((weak)) void
ompx_pipelined_put(int, void*, const void*, size_t) {
    std::fprintf(stderr, "ompx_pipelined_put: the target region fell back to the host, "
                         "which it cannot run correctly\n");
    std::abort();
}
#endif

// What the pass's rewrite calls; not for application code.
#include "gicc/omp_pipeline_impl.h"
