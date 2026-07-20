/*
 * direction_sweep.cpp - PUT (push) vs GET (pull) direction microbench.
 *
 * For an intra-node IPC transfer the same bytes can move two ways over xGMI:
 *   PUT/push: local rank STORES into the peer's GPU buffer (remote write)
 *   GET/pull: local rank LOADS from the peer's GPU buffer (remote read)
 * xGMI read vs write bandwidth is asymmetric, so the cheaper direction is
 * size-dependent. A halo exchange can be expressed either way, so direction
 * is a pass-choosable ML action.
 *
 * Both variants run the identical grid-stride copy kernel; only which operand
 * is the peer-mapped pointer differs.
 *
 * Knob: GICC_DIR = push | pull   (default push)
 * Run (Tioga single node, 2 ranks, do NOT set MPICH_GPU_SUPPORT_ENABLED):
 *   GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
 *     srun -p pci -t 2 -N 1 -n 2 --gpu-bind=none ./direction_sweep
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

static int env_int(const char* k,int d){ if(const char*v=std::getenv(k)){int x=atoi(v);if(x>0)return x;} return d; }

template <typename VecT>
__global__ void copy_kernel(void* __restrict__ dst, const void* __restrict__ src, size_t bytes){
    auto* d=reinterpret_cast<VecT*>(dst); auto* s=reinterpret_cast<const VecT*>(src);
    size_t n=bytes/sizeof(VecT), tid=blockIdx.x*blockDim.x+threadIdx.x, stride=(size_t)gridDim.x*blockDim.x;
    for(size_t i=tid;i<n;i+=stride) d[i]=s[i];
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank=0,size=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    int vec=env_int("GICC_CP_VEC",16), block=env_int("GICC_CP_BLOCK",256);
    std::string dir="push"; if(const char*d=std::getenv("GICC_DIR")) dir=d;
    bool pull=(dir=="pull");
    const size_t kSizes[]={256,1024,4096,16384,65536,262144,1048576,4194304,16777216};
    const int kN=sizeof(kSizes)/sizeof(kSizes[0]); const size_t kMax=kSizes[kN-1];
    const int kIters=env_int("GICC_CP_ITERS",300),kWarmup=30;

    gicc::Runtime rt;
    void *d_local=nullptr,*d_peerbuf=nullptr;
    (void)gpuMalloc(&d_local,kMax); (void)gpuMalloc(&d_peerbuf,kMax);
    (void)gpuMemset(d_local,0xAB,kMax); (void)gpuMemset(d_peerbuf,0xCD,kMax);
    auto bufPeer=rt.register_buffer(d_peerbuf,kMax,true);
    auto bufLocal=rt.register_buffer(d_local,kMax,true);
    rt.exchange(); rt.prepare();
    int peer=rank^1;
    void* peer_ptr=gicc_runtime_peer_ipc_base(&rt,peer,bufPeer.index);
    GpuStream_t stream=gicc_runtime_ipc_stream(&rt);
    if(rank==0 && !peer_ptr){ printf("[ERR] no IPC base\n"); MPI_Abort(MPI_COMM_WORLD,1);}

    if(rank==0){
        printf("# dir=%s vec=%d block=%d iters=%d\n",dir.c_str(),vec,block,kIters);
        printf("# size_bytes,us_per_op,GBps\n");
    }
    for(int si=0;si<kN;++si){
        size_t bytes=kSizes[si];
        int grid=(int)((bytes/(size_t)vec+block-1)/block); if(grid<1)grid=1; if(grid>65535)grid=65535;
        dim3 g(grid),b(block);
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==0){
            void* dst = pull ? d_local  : peer_ptr;
            void* src = pull ? peer_ptr : d_local;
            auto launch=[&](){
                if(vec==16) copy_kernel<uint4><<<g,b,0,stream>>>(dst,src,bytes);
                else if(vec==8) copy_kernel<uint2><<<g,b,0,stream>>>(dst,src,bytes);
                else if(vec==4) copy_kernel<uint32_t><<<g,b,0,stream>>>(dst,src,bytes);
                else copy_kernel<uint8_t><<<g,b,0,stream>>>(dst,src,bytes);
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
    (void)gpuFree(d_local); (void)gpuFree(d_peerbuf);
    MPI_Finalize(); return 0;
}
