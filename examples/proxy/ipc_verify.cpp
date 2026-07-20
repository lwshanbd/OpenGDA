/*
 * ipc_verify.cpp - Minimal "did the data actually arrive?" check on the
 * PROVEN one-sided IPC pattern (same shape as ipc_copy_sweep, which runs clean).
 *
 * One-sided, no concurrent kernels, no device-side polled flag:
 *   rank 0: fill src=pat(i); copy src -> peer dst over IPC on ipc_stream;
 *           <FENCE>; stream-sync; barrier.
 *   rank 1: barrier; verify kernel reads its OWN dst, counts bytes != pat(i);
 *           copy count back; print PASS/FAIL.
 * dst is pre-zeroed, so if data never arrives the verify reports FAIL (teeth).
 * GICC_VERIFY_SKIPCOPY=1 makes rank0 send nothing -> control that MUST FAIL.
 *
 * NOTE: a host MPI_Barrier separates write from read, so this validates DATA
 * DELIVERY (weak/no fence doesn't corrupt or drop data), NOT no-host-sync fence
 * isolation. Honest baseline.
 *
 * Knob: GICC_CP_FENCE = system|device|block|none (default system)
 * Run: GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
 *        srun -p pci -t 2 -N 1 -n 2 --gpu-bind=none ./ipc_verify
 */

#include <mpi.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"

__device__ __forceinline__ uint32_t pat(uint32_t i){ return i*2654435761u + 12345u; }

__global__ void fill_kernel(uint32_t* __restrict__ buf, int n){
    int t=blockIdx.x*blockDim.x+threadIdx.x, st=gridDim.x*blockDim.x;
    for(int i=t;i<n;i+=st) buf[i]=pat((uint32_t)i);
}

template <int FENCE>
__global__ void copy_then_fence(uint32_t* __restrict__ dst,
                                const uint32_t* __restrict__ src, int n){
    int t=blockIdx.x*blockDim.x+threadIdx.x, st=gridDim.x*blockDim.x;
    for(int i=t;i<n;i+=st) dst[i]=src[i];
    if(t==0){
        if(FENCE==3)      __threadfence_system();
        else if(FENCE==2) __threadfence();
        else if(FENCE==1) __threadfence_block();
    }
}

__global__ void verify_kernel(const uint32_t* __restrict__ dst, int n,
                              unsigned long long* __restrict__ mism){
    int t=blockIdx.x*blockDim.x+threadIdx.x, st=gridDim.x*blockDim.x;
    unsigned long long local=0;
    for(int i=t;i<n;i+=st) if(dst[i]!=pat((uint32_t)i)) local++;
    if(local) atomicAdd(mism,local);
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank=0,size=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size!=2){ if(rank==0)fprintf(stderr,"need 2 ranks\n"); MPI_Finalize(); return 1; }
    std::string fence="system"; if(const char*f=std::getenv("GICC_CP_FENCE")) fence=f;
    int fcode=(fence=="none")?0:(fence=="block")?1:(fence=="device")?2:3;
    int skip_copy = std::getenv("GICC_VERIFY_SKIPCOPY") ? 1 : 0;

    const int kSizes[]={1024,16384,262144,1048576,4194304};   // elems (×4B)
    const int kN=sizeof(kSizes)/sizeof(kSizes[0]);
    size_t maxbytes=(size_t)kSizes[kN-1]*4;

    uint32_t *d_src=nullptr,*d_dst=nullptr; unsigned long long* d_mism=nullptr;
    (void)gpuMalloc((void**)&d_src,maxbytes);
    (void)gpuMalloc((void**)&d_dst,maxbytes);
    (void)gpuMalloc((void**)&d_mism,sizeof(unsigned long long));

    gicc::Runtime rt;
    auto bufDst=rt.register_buffer(d_dst,maxbytes,true);
    auto bufSrc=rt.register_buffer(d_src,maxbytes,true);
    rt.exchange(); rt.prepare();
    int peer=rank^1;
    uint32_t* peer_dst=(uint32_t*)gicc_runtime_peer_ipc_base(&rt,peer,bufDst.index);
    GpuStream_t s=gicc_runtime_ipc_stream(&rt);
    if(rank==0 && !peer_dst){ fprintf(stderr,"[ERR] no IPC base\n"); MPI_Abort(MPI_COMM_WORLD,1);}

    if(rank==0) printf("# fence=%s skip_copy=%d\n# size_bytes,mismatches,result\n",fence.c_str(),skip_copy);

    for(int si=0; si<kN; ++si){
        int n=kSizes[si];
        int blk=256, grd=(n+blk-1)/blk; if(grd>65535)grd=65535;
        (void)gpuMemset(d_dst,0,(size_t)n*4);     // clear target: no-show => FAIL
        (void)gpuMemset(d_mism,0,sizeof(unsigned long long));
        (void)gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        if(rank==0 && !skip_copy){
            fill_kernel<<<grd,blk,0,s>>>(d_src,n);
            switch(fcode){
                case 3: copy_then_fence<3><<<grd,blk,0,s>>>(peer_dst,d_src,n); break;
                case 2: copy_then_fence<2><<<grd,blk,0,s>>>(peer_dst,d_src,n); break;
                case 1: copy_then_fence<1><<<grd,blk,0,s>>>(peer_dst,d_src,n); break;
                default:copy_then_fence<0><<<grd,blk,0,s>>>(peer_dst,d_src,n); break;
            }
            (void)gpuStreamSynchronize(s);
        }
        MPI_Barrier(MPI_COMM_WORLD);              // write done before read

        if(rank==1){
            verify_kernel<<<grd,blk,0,s>>>(d_dst,n,d_mism);
            (void)gpuStreamSynchronize(s);
            unsigned long long mism=0;
            (void)gpuMemcpy(&mism,d_mism,sizeof(mism),gpuMemcpyDeviceToHost);
            printf("%zu,%llu,%s\n",(size_t)n*4,mism,mism?"FAIL":"PASS");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    rt.reset(); MPI_Barrier(MPI_COMM_WORLD);
    (void)gpuFree(d_src);(void)gpuFree(d_dst);(void)gpuFree(d_mism);
    MPI_Finalize(); return 0;
}
