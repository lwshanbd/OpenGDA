/*
 * halo_face_sweep.cpp - Tunable STRIDED 3D-halo-face IPC copy microbench.
 *
 * GICC's real intra-node traffic is not a contiguous blob — it's a halo
 * FACE of a 3D sub-volume: a strided gather (src) written into the peer's
 * ghost region (dst) via the IPC fast path. The contiguous ipc_copy_sweep
 * found vec=16 best at 1 MB; strided access usually BREAKS that (the wide
 * vector load straddles a stride gap and mis-aligns), so the optimal knob
 * should INVERT vs the contiguous case. That inversion is the ML signal.
 *
 * Domain model: a cubic sub-volume DIM^3 of float (4B). A halo face of
 * width HALO is exchanged. Three orientations exercise three stride classes:
 *   - Z face (orient=2): innermost contiguous run = DIM*DIM (fully contiguous)
 *   - Y face (orient=1): contiguous run = DIM, gap each DIM (medium stride)
 *   - X face (orient=0): contiguous run = HALO, gap each row (worst stride)
 *
 * Knobs (env):
 *   GICC_HF_DIM     sub-volume edge (default 256) -> face bytes scale DIM^2
 *   GICC_HF_HALO    halo width in cells (default 4)
 *   GICC_HF_VEC     element bytes 4|8|16 (default 4 = one float, safe stride)
 *   GICC_HF_LAYOUT  thread->work mapping: face | row   (default face)
 *   GICC_HF_GATHER  0 = direct strided store to peer; 1 = gather to local
 *                   contiguous staging then one contiguous put (default 0)
 *   GICC_HF_BLOCK   threads/block (default 256)
 *
 * Run (Tioga single node, 2 ranks, NO MPICH_GPU_SUPPORT_ENABLED):
 *   GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
 *     srun -p pci -t 2 -N 1 -n 2 --gpu-bind=none ./halo_face_sweep
 */

#include <mpi.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"

static int env_int(const char* k, int d) {
    if (const char* v = std::getenv(k)) { int x = atoi(v); if (x > 0) return x; }
    return d;
}

// --- Direct strided store: each thread copies one face cell from the strided
// src position to the matching strided dst position in the peer ghost region.
// orient selects which axis is the face normal. The face is HALO cells thick.
template <typename VecT>
__global__ void face_copy_direct(VecT* __restrict__ dst,
                                  const VecT* __restrict__ src,
                                  int dim, int halo, int orient) {
    // Face cell count = halo * dim * dim (a slab of thickness `halo`).
    size_t face_cells = (size_t)halo * dim * dim;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t f = tid; f < face_cells; f += stride) {
        // Decompose f into (h, a, b) within the slab.
        int h = (int)(f / ((size_t)dim * dim));
        size_t rem = f - (size_t)h * dim * dim;
        int a = (int)(rem / dim);
        int b = (int)(rem % dim);
        // Map (h,a,b) to a linear index in the DIM^3 volume per orientation.
        size_t idx;
        if (orient == 2)      idx = ((size_t)h * dim + a) * dim + b;          // Z: contiguous
        else if (orient == 1) idx = ((size_t)a * dim + h) * dim + b;          // Y: row stride
        else                  idx = ((size_t)a * dim + b) * dim + h;          // X: worst stride
        dst[idx] = src[idx];
    }
}

// --- Gather to contiguous staging: pack face cells densely into a staging
// buffer (contiguous), which a real impl would then put() in one shot. Here
// we measure just the gather kernel cost (the contiguous put is already
// characterized by ipc_copy_sweep).
template <typename VecT>
__global__ void face_gather(VecT* __restrict__ stage,
                            const VecT* __restrict__ src,
                            int dim, int halo, int orient) {
    size_t face_cells = (size_t)halo * dim * dim;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t f = tid; f < face_cells; f += stride) {
        int h = (int)(f / ((size_t)dim * dim));
        size_t rem = f - (size_t)h * dim * dim;
        int a = (int)(rem / dim);
        int b = (int)(rem % dim);
        size_t idx;
        if (orient == 2)      idx = ((size_t)h * dim + a) * dim + b;
        else if (orient == 1) idx = ((size_t)a * dim + h) * dim + b;
        else                  idx = ((size_t)a * dim + b) * dim + h;
        stage[f] = src[idx];   // dense write, strided read
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dim    = env_int("GICC_HF_DIM", 256);
    int halo   = env_int("GICC_HF_HALO", 4);
    int vec    = env_int("GICC_HF_VEC", 4);
    int block  = env_int("GICC_HF_BLOCK", 256);
    int gather = env_int("GICC_HF_GATHER", 0);
    std::string layout = "face";
    if (const char* l = std::getenv("GICC_HF_LAYOUT")) layout = l;
    (void)layout;  // reserved; current kernels use grid-stride face layout

    const int kIters = env_int("GICC_HF_ITERS", 200), kWarmup = 20;

    // Volume of DIM^3 floats; pad to vec alignment.
    size_t vol_cells = (size_t)dim * dim * dim;
    size_t vol_bytes = vol_cells * 4;            // float
    size_t face_cells = (size_t)halo * dim * dim;
    size_t face_bytes = face_cells * 4;

    gicc::Runtime rt;
    void *d_vol = nullptr, *d_dst = nullptr, *d_stage = nullptr;
    (void)gpuMalloc(&d_vol, vol_bytes);
    (void)gpuMalloc(&d_dst, vol_bytes);
    (void)gpuMalloc(&d_stage, face_bytes);
    (void)gpuMemset(d_vol, 0xAB, vol_bytes);
    (void)gpuMemset(d_dst, 0, vol_bytes);

    auto bufDst = rt.register_buffer(d_dst, vol_bytes, true);
    auto bufVol = rt.register_buffer(d_vol, vol_bytes, true);
    rt.exchange();
    rt.prepare();
    int peer = rank ^ 1;
    void* peer_dst = gicc_runtime_peer_ipc_base(&rt, peer, bufDst.index);
    GpuStream_t stream = gicc_runtime_ipc_stream(&rt);
    if (rank == 0 && peer_dst == nullptr) {
        printf("[ERROR] peer IPC base null — not same-node?\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int grid = (int)((face_cells + block - 1) / block);
    if (grid > 65535) grid = 65535;
    if (grid < 1) grid = 1;

    if (rank == 0) {
        printf("# dim=%d halo=%d vec=%d block=%d gather=%d face_bytes=%zu iters=%d\n",
               dim, halo, vec, block, gather, face_bytes, kIters);
        printf("# orient,face_bytes,us_per_op,GBps\n");
    }

    for (int orient = 0; orient < 3; ++orient) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            auto launch = [&](){
                void* tgt = gather ? d_stage : peer_dst;
                if (gather) {
                    if (vec==16) face_gather<uint4><<<grid,block,0,stream>>>((uint4*)tgt,(const uint4*)d_vol,dim,halo,orient);
                    else if (vec==8) face_gather<uint2><<<grid,block,0,stream>>>((uint2*)tgt,(const uint2*)d_vol,dim,halo,orient);
                    else face_gather<float><<<grid,block,0,stream>>>((float*)tgt,(const float*)d_vol,dim,halo,orient);
                } else {
                    if (vec==16) face_copy_direct<uint4><<<grid,block,0,stream>>>((uint4*)tgt,(const uint4*)d_vol,dim,halo,orient);
                    else if (vec==8) face_copy_direct<uint2><<<grid,block,0,stream>>>((uint2*)tgt,(const uint2*)d_vol,dim,halo,orient);
                    else face_copy_direct<float><<<grid,block,0,stream>>>((float*)tgt,(const float*)d_vol,dim,halo,orient);
                }
            };
            for (int it=0; it<kWarmup; ++it) launch();
            (void)gpuStreamSynchronize(stream);
            double t0 = MPI_Wtime();
            for (int it=0; it<kIters; ++it) launch();
            (void)gpuStreamSynchronize(stream);
            double t1 = MPI_Wtime();
            double us = (t1-t0)*1e6/kIters;
            double gbps = face_bytes / (us*1e3);
            printf("%d,%zu,%.3f,%.2f\n", orient, face_bytes, us, gbps);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    rt.reset();
    MPI_Barrier(MPI_COMM_WORLD);
    (void)gpuFree(d_vol); (void)gpuFree(d_dst); (void)gpuFree(d_stage);
    MPI_Finalize();
    return 0;
}
