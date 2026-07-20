/*
 * fence_scope_sweep.cpp - Tunable completion-FENCE-scope microbench.
 *
 * After an intra-node IPC copy, the GPU must publish the written bytes so the
 * consumer (this rank's later read, or — for the real proxy path — a flag the
 * peer polls) observes them. GICC currently always uses
 * __threadfence_system() (line ofi_device.cuh:257), the *heaviest* fence:
 * it orders against the whole system including the host and other GPUs over
 * the fabric.
 *
 * But the scope you actually need depends on WHO consumes the data:
 *   - consumer is an SM on the SAME GPU      -> __threadfence_block / none
 *   - consumer is another SM, same GPU       -> __threadfence() (device scope)
 *   - consumer is the peer GPU / host / NIC  -> __threadfence_system()
 *
 * The fence is a per-copy fixed cost, so its relative weight EXPLODES at small
 * sizes (where the copy itself is cheap) and vanishes at large. So fence scope
 * should be a clean size-inverting knob: cheap fence wins small, irrelevant
 * large. The GICC pass knows the consumer locality (same-GPU pack vs peer
 * delivery), so it can pick the scope — an ML-decidable correctness×perf lever.
 *
 * Knobs:
 *   GICC_CP_FENCE  system | device | block | none   (default system = today)
 *   GICC_CP_VEC / _BLOCK   reuse copy config (default 16 / 256)
 *
 * Run (Tioga single node, 2 ranks, NO MPICH_GPU_SUPPORT_ENABLED):
 *   GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
 *     srun -p pci -t 2 -N 1 -n 2 --gpu-bind=none ./fence_scope_sweep
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

// FENCE selects the post-copy memory fence the lead thread issues. The copy
// itself is identical; only the fence differs. This isolates fence cost.
//   0 = none, 1 = block, 2 = device (__threadfence), 3 = system (default today)
template <typename VecT, int FENCE>
__global__ void copy_fence_kernel(void* __restrict__ dst,
                                  const void* __restrict__ src,
                                  size_t bytes) {
    auto* d = reinterpret_cast<VecT*>(dst);
    auto* s = reinterpret_cast<const VecT*>(src);
    size_t n = bytes / sizeof(VecT);
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < n; i += stride) d[i] = s[i];
    // byte tail
    size_t doneb = n * sizeof(VecT);
    auto* db = reinterpret_cast<char*>(dst);
    auto* sb = reinterpret_cast<const char*>(src);
    for (size_t b = doneb + tid; b < bytes; b += stride) db[b] = sb[b];
    // Lead thread issues the publish fence (mirrors gicc::flush / quiet).
    if (tid == 0) {
        if (FENCE == 3)      __threadfence_system();
        else if (FENCE == 2) __threadfence();
        else if (FENCE == 1) __threadfence_block();
        // FENCE == 0: no fence
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0,size=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int vec   = env_int("GICC_CP_VEC", 16);
    int block = env_int("GICC_CP_BLOCK", 256);
    std::string fence = "system";
    if (const char* f = std::getenv("GICC_CP_FENCE")) fence = f;
    int fcode = (fence=="none")?0:(fence=="block")?1:(fence=="device")?2:3;

    const size_t kSizes[] = {256,1024,4096,16384,65536,262144,1048576,4194304};
    const int kN = sizeof(kSizes)/sizeof(kSizes[0]);
    const size_t kMax = kSizes[kN-1];
    const int kIters = env_int("GICC_CP_ITERS", 300), kWarmup=30;

    gicc::Runtime rt;
    void *d_src=nullptr,*d_dst=nullptr;
    (void)gpuMalloc(&d_src,kMax); (void)gpuMalloc(&d_dst,kMax);
    (void)gpuMemset(d_src,0xAB,kMax); (void)gpuMemset(d_dst,0,kMax);
    auto bufDst = rt.register_buffer(d_dst,kMax,true);
    auto bufSrc = rt.register_buffer(d_src,kMax,true);
    rt.exchange(); rt.prepare();
    int peer = rank^1;
    void* peer_dst = gicc_runtime_peer_ipc_base(&rt,peer,bufDst.index);
    GpuStream_t stream = gicc_runtime_ipc_stream(&rt);
    if (rank==0 && !peer_dst){ printf("[ERR] no IPC base\n"); MPI_Abort(MPI_COMM_WORLD,1); }

    if (rank==0){
        printf("# fence=%s vec=%d block=%d iters=%d\n",fence.c_str(),vec,block,kIters);
        printf("# size_bytes,us_per_op,GBps\n");
    }
    for (int si=0; si<kN; ++si){
        size_t bytes=kSizes[si];
        int grid=(int)((bytes/(size_t)vec + block-1)/block); if(grid<1)grid=1; if(grid>65535)grid=65535;
        dim3 g(grid),b(block);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank==0){
            auto launch=[&](){
#define L(V,F) copy_fence_kernel<V,F><<<g,b,0,stream>>>(peer_dst,d_src,bytes)
#define LV(F) do{ if(vec==16)L(uint4,F); else if(vec==8)L(uint2,F); else if(vec==4)L(uint32_t,F); else L(uint8_t,F);}while(0)
                if(fcode==3)LV(3); else if(fcode==2)LV(2); else if(fcode==1)LV(1); else LV(0);
#undef L
#undef LV
            };
            for(int it=0;it<kWarmup;++it)launch();
            (void)gpuStreamSynchronize(stream);
            double t0=MPI_Wtime();
            for(int it=0;it<kIters;++it)launch();
            (void)gpuStreamSynchronize(stream);
            double t1=MPI_Wtime();
            double us=(t1-t0)*1e6/kIters, gbps=bytes/(us*1e3);
            printf("%zu,%.4f,%.2f\n",bytes,us,gbps); fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    rt.reset(); MPI_Barrier(MPI_COMM_WORLD);
    (void)gpuFree(d_src); (void)gpuFree(d_dst);
    MPI_Finalize(); return 0;
}
