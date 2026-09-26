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
// are complete, as a host ompx_put after the kernel would. The gicc-passes
// plugin in GICC_MODE=chunk-lower proves the send can be split along the
// distribute blocks (tools/gicc-passes/src/GICCChunkAnalysis.cpp) and
// replaces it with one ompx__block_put per block, issued as soon as that
// block's parallel region joins. When it cannot prove that, compilation
// fails with the reason. ompx_pipelined_put has no definition: a build
// without the pass does not link, so the marker can never run untransformed.
//
// C++ only; the block send uses the device ompx_put fallback.
#pragma once

#include "gicc/omp.h"

#include <cstddef>
#include <cstdint>

#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

// Host fallback of a target region: the region runs on the CPU, the loop
// finishes before this call, so it is exactly the host put it stands for.
#if !defined(__NVPTX__) && !defined(__AMDGCN__)
extern "C" __attribute__((weak)) void
ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes) {
    ompx_put_host(peer, dst, src, bytes);
}
#endif

#pragma omp declare target

// Lowering target, called by a team's main thread right after one block's
// parallel region joins, so every word of [src, src + bytes) is written. The
// team's threads store the block into a same-node peer's copy of dst over
// NVLink / xGMI; for any other peer the main thread issues a device ompx_put.
// Weak so every TU including this header can emit it; `used` keeps it in the
// device module until the pass has inserted its calls.
extern "C" __attribute__((weak, used)) void
ompx__block_put(int peer, void* dst, const void* src, size_t bytes) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    ompx_ctx* c = ompx__ctx;
    char* base = c->peer_ipc_base
                     ? static_cast<char*>(c->peer_ipc_base[peer * c->ipc_n_bufs + c->heap_buf])
                     : nullptr;
    if (base == nullptr) {
        ompx_put(peer, dst, src, bytes);
        return;
    }
    char* d = base + (static_cast<char*>(dst) - static_cast<char*>(c->heap_base));
    const char* s = static_cast<const char*>(src);
    // Everything the copy loops use is firstprivate: a variable they shared
    // by reference would be globalized, a device-heap allocation per call.
    // schedule(static, 1) gives neighbouring threads neighbouring words.
    if (((reinterpret_cast<uintptr_t>(d) | reinterpret_cast<uintptr_t>(s)) & 3) == 0) {
        uint32_t* dw = reinterpret_cast<uint32_t*>(d);
        const uint32_t* sw = reinterpret_cast<const uint32_t*>(s);
        const size_t words = bytes / 4;
        #pragma omp parallel for schedule(static, 1) firstprivate(dw, sw, words)
        for (size_t i = 0; i < words; ++i) dw[i] = sw[i];
        for (size_t i = words * 4; i < bytes; ++i) d[i] = s[i];
    } else {
        #pragma omp parallel for schedule(static, 1) firstprivate(d, s, bytes)
        for (size_t i = 0; i < bytes; ++i) d[i] = s[i];
    }
#else
    (void)peer; (void)dst; (void)src; (void)bytes;
#endif
}
#pragma omp end declare target
