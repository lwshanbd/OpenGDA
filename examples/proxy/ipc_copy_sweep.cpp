/*
 * ipc_copy_sweep.cpp - Tunable intra-node IPC copy microbenchmark.
 *
 * Purpose (ML-for-comm research): the same-node IPC fast path currently uses
 * a single hardcoded copy strategy (byte-wise block-cooperative loop in
 * gicc::ipc_copy_block). That one strategy is rarely optimal across the
 * full transfer-size range. This benchmark exposes a *family* of copy
 * strategies as runtime knobs and measures achieved bandwidth per size,
 * so a compiler-pass-fed ML decider has a real decision space to optimize.
 *
 * Knobs (env vars, one binary sweeps all):
 *   GICC_CP_VEC    element width in bytes: 1 | 4 | 8 | 16   (default 16)
 *   GICC_CP_BLOCK  threads per block: 64..1024              (default 256)
 *   GICC_CP_GRID   number of blocks (0 = size-derived)      (default 0)
 *   GICC_CP_MECH   copy mechanism: kernel | memcpy          (default kernel)
 *   GICC_CP_UNROLL inner unroll factor: 1 | 2 | 4 | 8       (default 1)
 *
 * Run (Tioga, single node, 2 ranks share a node):
 *   GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
 *     srun -p pci -t 2 -N 1 -n 2 --gpu-bind=none ./ipc_copy_sweep
 */

#include <mpi.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"

// ----------------------------------------------------------------------------
// Tunable copy kernels. Each is the SAME algorithm (grid-stride cooperative
// copy) parameterized by the vector element type and an unroll factor. The
// vector width is the dominant knob: wider elements = fewer loop iterations
// and wider per-thread memory transactions, but require alignment and waste
// bandwidth on tails.
// ----------------------------------------------------------------------------
// Element-wise copy helper. NT=1 emits non-temporal (streaming) loads/stores
// that bypass the L2 write-allocate. __builtin_nontemporal_* only accepts
// scalar/integer/pointer operands (NOT HIP_vector_type), so for the wide
// vector types we decompose into the underlying 32-bit lanes and NT each.
template <typename VecT, int NT>
__device__ __forceinline__ void copy_elem(VecT* d, const VecT* s) {
    if (!NT) { *d = *s; return; }
    // Reinterpret the vector element as a pack of uint32 lanes and NT each.
    constexpr int LANES = sizeof(VecT) / sizeof(uint32_t);
    auto* dl = reinterpret_cast<uint32_t*>(d);
    auto* sl = reinterpret_cast<const uint32_t*>(s);
#pragma unroll
    for (int l = 0; l < LANES; ++l)
        __builtin_nontemporal_store(__builtin_nontemporal_load(&sl[l]), &dl[l]);
}
// uint8 specialization: a single byte lane, NT on the byte directly.
template <>
__device__ __forceinline__ void copy_elem<uint8_t, 1>(uint8_t* d,
                                                       const uint8_t* s) {
    __builtin_nontemporal_store(__builtin_nontemporal_load(s), d);
}

// NT=1 uses non-temporal (streaming) stores: bypass L2 write-allocate.
// Classic size-inverting knob — helps large no-reuse streaming, hurts small.
template <typename VecT, int UNROLL, int NT>
__global__ void copy_vec_kernel(void* __restrict__ dst,
                                const void* __restrict__ src,
                                size_t bytes) {
    auto* d = reinterpret_cast<VecT*>(dst);
    auto* s = reinterpret_cast<const VecT*>(src);
    size_t n = bytes / sizeof(VecT);
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = tid;
    // Unrolled grid-stride main body.
    for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            copy_elem<VecT, NT>(&d[i + u * stride], &s[i + u * stride]);
    }
    // Remainder of the unrolled stride.
    for (; i < n; i += stride) copy_elem<VecT, NT>(&d[i], &s[i]);
    // Byte tail (when bytes not divisible by sizeof(VecT)).
    size_t done = n * sizeof(VecT);
    auto* db = reinterpret_cast<char*>(dst);
    auto* sb = reinterpret_cast<const char*>(src);
    for (size_t b = done + tid; b < bytes; b += stride) db[b] = sb[b];
}

// Dispatch by runtime knobs.
struct CopyCfg {
    int vec = 16;
    int block = 256;
    int grid = 0;      // 0 = size-derived
    int unroll = 1;
    int nt = 0;        // non-temporal stores
    int nstream = 1;   // split copy across K streams (overlap)
    std::string mech = "kernel";
};

static int env_int(const char* k, int dflt) {
    if (const char* v = std::getenv(k)) { int x = atoi(v); if (x > 0) return x; }
    return dflt;
}

// Launch one kernel covering [off, off+len) of the buffers.
static void launch_one(const CopyCfg& c, char* dst, const char* src,
                       size_t len, GpuStream_t stream) {
    int block = c.block;
    int grid = c.grid;
    if (grid <= 0) {
        size_t elems = len / (size_t)c.vec;
        grid = (int)((elems + block - 1) / block);
        if (grid < 1) grid = 1;
        if (grid > 65535) grid = 65535;
    }
    dim3 g(grid), b(block);
#define DISPATCH(VEC, T) \
    if (c.vec == VEC) { \
        if (c.nt) switch (c.unroll) { \
            case 8: copy_vec_kernel<T,8,1><<<g,b,0,stream>>>(dst,src,len); break; \
            case 4: copy_vec_kernel<T,4,1><<<g,b,0,stream>>>(dst,src,len); break; \
            case 2: copy_vec_kernel<T,2,1><<<g,b,0,stream>>>(dst,src,len); break; \
            default: copy_vec_kernel<T,1,1><<<g,b,0,stream>>>(dst,src,len); break; \
        } else switch (c.unroll) { \
            case 8: copy_vec_kernel<T,8,0><<<g,b,0,stream>>>(dst,src,len); break; \
            case 4: copy_vec_kernel<T,4,0><<<g,b,0,stream>>>(dst,src,len); break; \
            case 2: copy_vec_kernel<T,2,0><<<g,b,0,stream>>>(dst,src,len); break; \
            default: copy_vec_kernel<T,1,0><<<g,b,0,stream>>>(dst,src,len); break; \
        } return; \
    }
    DISPATCH(16, uint4)
    DISPATCH(8,  uint2)
    DISPATCH(4,  uint32_t)
    DISPATCH(1,  uint8_t)
#undef DISPATCH
    copy_vec_kernel<uint4,1,0><<<g,b,0,stream>>>(dst, src, len);
}

static void launch_copy(const CopyCfg& c, void* dst, const void* src,
                        size_t bytes, GpuStream_t stream,
                        GpuStream_t* streams) {
    if (c.mech == "memcpy") {
        (void)hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream);
        return;
    }
    if (c.nstream <= 1) {
        launch_one(c, (char*)dst, (const char*)src, bytes, stream);
        return;
    }
    // Split the transfer into nstream contiguous chunks on separate streams
    // to overlap. Chunk boundary aligned to vec width.
    size_t chunk = bytes / c.nstream;
    size_t align = (size_t)c.vec;
    chunk = (chunk / align) * align;
    if (chunk == 0) { launch_one(c, (char*)dst, (const char*)src, bytes, stream); return; }
    size_t off = 0;
    for (int k = 0; k < c.nstream; ++k) {
        size_t len = (k == c.nstream - 1) ? (bytes - off) : chunk;
        launch_one(c, (char*)dst + off, (const char*)src + off, len, streams[k]);
        off += len;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CopyCfg cfg;
    cfg.vec   = env_int("GICC_CP_VEC", 16);
    cfg.block = env_int("GICC_CP_BLOCK", 256);
    cfg.grid  = env_int("GICC_CP_GRID", 0);
    cfg.unroll= env_int("GICC_CP_UNROLL", 1);
    cfg.nt    = env_int("GICC_CP_NT", 0);
    cfg.nstream = env_int("GICC_CP_NSTREAM", 1);
    if (const char* m = std::getenv("GICC_CP_MECH")) cfg.mech = m;

    const size_t kSizes[] = {
        256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216
    };
    const int kNSizes = sizeof(kSizes) / sizeof(kSizes[0]);
    const size_t kMax = kSizes[kNSizes - 1];
    const int kIters = env_int("GICC_CP_ITERS", 200);
    const int kWarmup = 20;

    gicc::Runtime rt;
    void *d_src = nullptr, *d_dst = nullptr;
    (void)gpuMalloc(&d_src, kMax);
    (void)gpuMalloc(&d_dst, kMax);
    (void)gpuMemset(d_src, 0xAB, kMax);
    (void)gpuMemset(d_dst, 0x00, kMax);

    auto bufDst = rt.register_buffer(d_dst, kMax, true);
    auto bufSrc = rt.register_buffer(d_src, kMax, true);
    rt.exchange();
    rt.prepare();

    int peer = rank ^ 1;  // 0<->1
    // Resolve peer's IPC-mapped dst base (same-node).
    void* peer_dst = gicc_runtime_peer_ipc_base(&rt, peer, bufDst.index);
    GpuStream_t stream = gicc_runtime_ipc_stream(&rt);

    // Multi-stream pool for GICC_CP_NSTREAM > 1. Allocate up to 8 own
    // streams (don't reuse the runtime's single ipc_stream for splits).
    const int kMaxStreams = 8;
    if (cfg.nstream > kMaxStreams) cfg.nstream = kMaxStreams;
    if (cfg.nstream < 1) cfg.nstream = 1;
    GpuStream_t streams[kMaxStreams];
    for (int k = 0; k < cfg.nstream; ++k)
        (void)hipStreamCreateWithFlags(&streams[k], hipStreamNonBlocking);

    if (rank == 0 && peer_dst == nullptr) {
        printf("[ERROR] peer IPC base null — not same-node? Aborting.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("# cfg vec=%d block=%d grid=%d unroll=%d nt=%d nstream=%d mech=%s iters=%d\n",
               cfg.vec, cfg.block, cfg.grid, cfg.unroll, cfg.nt, cfg.nstream,
               cfg.mech.c_str(), kIters);
        printf("# size_bytes,us_per_op,GBps\n");
    }

    for (int si = 0; si < kNSizes; ++si) {
        size_t bytes = kSizes[si];
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            // Warmup.
            for (int it = 0; it < kWarmup; ++it)
                launch_copy(cfg, peer_dst, d_src, bytes, stream, streams);
            (void)gpuStreamSynchronize(stream);
            for (int k = 0; k < cfg.nstream; ++k)
                (void)gpuStreamSynchronize(streams[k]);

            double t0 = MPI_Wtime();
            for (int it = 0; it < kIters; ++it)
                launch_copy(cfg, peer_dst, d_src, bytes, stream, streams);
            (void)gpuStreamSynchronize(stream);
            for (int k = 0; k < cfg.nstream; ++k)
                (void)gpuStreamSynchronize(streams[k]);
            double t1 = MPI_Wtime();

            double us = (t1 - t0) * 1e6 / kIters;
            double gbps = bytes / (us * 1e3);  // bytes/us = GB/s (1e-9*1e6)
            printf("%zu,%.3f,%.2f\n", bytes, us, gbps);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    rt.reset();
    MPI_Barrier(MPI_COMM_WORLD);
    (void)gpuFree(d_src);
    (void)gpuFree(d_dst);
    MPI_Finalize();
    return 0;
}
