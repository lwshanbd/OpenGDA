/*
 * fence_correctness.cpp - Does a WEAKER producer fence still deliver correct
 * data across xGMI to a PEER GPU that consumes via a polled flag?
 *
 * Lockstep producer (GPU0) / consumer (GPU1), NO host sync in the path.
 * One data region, reused each round. Round r:
 *   Producer: write data[i]=pat(r,i) into PEER (consumer) data; <FENCE>;
 *             set PEER flag=r+1; then WAIT on local backflag>=r+1.
 *   Consumer: spin local flag>=r+1; __threadfence_system (acquire);
 *             byte-check local data==pat(r,i); set PEER backflag=r+1.
 * Back-pressure keeps the producer from outrunning the consumer, so the
 * data-vs-flag race is live every round. If the producer fence is too weak,
 * flag overtakes data and the consumer sees stale bytes -> mismatch>0.
 *
 * GICC_FC_INVERT=1 is a CONTROL: producer writes flag BEFORE data, no fence
 * (deliberately wrong). A sensitive test MUST report mismatches for it.
 *
 * Lessons baked in (each was a real crash while building this):
 *  - peer-IPC-accessing kernels MUST launch on the runtime ipc_stream, not
 *    the default stream (default-stream peer-IPC store faults).
 *  - polled flag words MUST be fine-grained (hipExtMallocWithFlags +
 *    hipDeviceMallocFinegrained); coarse-grained faults under concurrent
 *    cross-GPU write+poll.
 *
 * Knob: GICC_CP_FENCE = system|device|block|none (default system)
 * Run: GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
 *        srun -p pci -t 2 -N 1 -n 2 --gpu-bind=none ./fence_correctness
 */

#include <mpi.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"

static int env_int(const char* k,int d){ if(const char*v=std::getenv(k)){int x=atoi(v);if(x>0)return x;} return d; }
__device__ __forceinline__ uint32_t pat(int r,int i){ return (uint32_t)(r*2654435761u + i*40503u + 1u); }

template <int FENCE>
__global__ void producer(uint32_t* __restrict__ data_peer,
                         volatile uint32_t* __restrict__ flag_peer,
                         volatile uint32_t* __restrict__ back_local,
                         int niter, int elems, int invert) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int nth = gridDim.x*blockDim.x;
    for (int r=0; r<niter; ++r) {
        if (invert) {                         // CONTROL: wrong order, no fence
            __syncthreads();
            if (t==0) *flag_peer = (uint32_t)(r+1);
            // Deliberately DELAY the data write so the consumer (which reads as
            // soon as it sees the flag) is guaranteed to read STALE data. This
            // gives the test detection power: a correct test MUST report
            // mismatches here. ~1e6-iter spin on every thread before any store.
            for (volatile int d=0; d<1000000; ++d) { }
            __syncthreads();
            for (int i=t;i<elems;i+=nth) data_peer[i] = pat(r,i);
        } else {
            for (int i=t;i<elems;i+=nth) data_peer[i] = pat(r,i);
            if (FENCE==3)      __threadfence_system();
            else if (FENCE==2) __threadfence();
            else if (FENCE==1) __threadfence_block();
            __syncthreads();
            if (t==0) *flag_peer = (uint32_t)(r+1);
        }
        if (t==0) {                           // back-pressure: wait for ack
            unsigned long long sp=0; const unsigned long long kMax=4000000000ULL;
            while (*back_local < (uint32_t)(r+1)) if(++sp>kMax) break;
        }
        __syncthreads();
    }
}

__global__ void consumer(const uint32_t* __restrict__ data_local,
                         volatile uint32_t* __restrict__ flag_local,
                         volatile uint32_t* __restrict__ back_peer,
                         int niter, int elems,
                         unsigned long long* __restrict__ mism,
                         int* __restrict__ timed_out) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int nth = gridDim.x*blockDim.x;
    __shared__ int ready;
    for (int r=0; r<niter; ++r) {
        if (t==0) {
            ready=1;
            unsigned long long sp=0; const unsigned long long kMax=4000000000ULL;
            while (*flag_local < (uint32_t)(r+1)) if(++sp>kMax){ready=0;*timed_out=1;break;}
            __threadfence_system();           // acquire
        }
        __syncthreads();
        if(!ready) return;
        unsigned long long local=0;
        for (int i=t;i<elems;i+=nth) if (data_local[i] != pat(r,i)) local++;
        if(local) atomicAdd(mism,local);
        __syncthreads();
        if (t==0) *back_peer = (uint32_t)(r+1);   // release producer
        __syncthreads();
    }
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank=0,size=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size!=2){ if(rank==0)fprintf(stderr,"need 2 ranks\n"); MPI_Finalize(); return 1; }
    int niter=env_int("GICC_FC_NITER",128), elems=env_int("GICC_FC_ELEMS",16384);
    std::string fence="system"; if(const char*f=std::getenv("GICC_CP_FENCE")) fence=f;
    int fcode=(fence=="none")?0:(fence=="block")?1:(fence=="device")?2:3;
    int invert = std::getenv("GICC_FC_INVERT")?1:0;

    size_t data_bytes=(size_t)elems*sizeof(uint32_t);
    void *d_data=nullptr,*d_flag=nullptr,*d_back=nullptr;
    (void)gpuMalloc(&d_data,data_bytes);
    if(hipExtMallocWithFlags(&d_flag,4096,hipDeviceMallocFinegrained)!=hipSuccess ||
       hipExtMallocWithFlags(&d_back,4096,hipDeviceMallocFinegrained)!=hipSuccess){
        fprintf(stderr,"[ERR] finegrained alloc failed\n"); MPI_Abort(MPI_COMM_WORLD,2);}
    (void)gpuMemset(d_data,0,data_bytes); (void)gpuMemset(d_flag,0,4096); (void)gpuMemset(d_back,0,4096);
    unsigned long long* d_mism=nullptr; int* d_to=nullptr;
    (void)gpuMalloc(&d_mism,sizeof(unsigned long long)); (void)gpuMemset(d_mism,0,sizeof(unsigned long long));
    (void)gpuMalloc(&d_to,sizeof(int)); (void)gpuMemset(d_to,0,sizeof(int));

    gicc::Runtime rt;
    auto bD=rt.register_buffer(d_data,data_bytes,true);
    auto bF=rt.register_buffer(d_flag,4096,true);
    auto bB=rt.register_buffer(d_back,4096,true);
    rt.exchange(); rt.prepare();
    int peer=rank^1;
    uint32_t* peer_data=(uint32_t*)gicc_runtime_peer_ipc_base(&rt,peer,bD.index);
    uint32_t* peer_flag=(uint32_t*)gicc_runtime_peer_ipc_base(&rt,peer,bF.index);
    uint32_t* peer_back=(uint32_t*)gicc_runtime_peer_ipc_base(&rt,peer,bB.index);
    if(!peer_data||!peer_flag||!peer_back){ if(rank==0)fprintf(stderr,"[ERR] no IPC base\n"); MPI_Abort(MPI_COMM_WORLD,1);}
    GpuStream_t s = gicc_runtime_ipc_stream(&rt);

    dim3 g(1),b(256);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0){
        switch(fcode){
            case 3: producer<3><<<g,b,0,s>>>(peer_data,peer_flag,(volatile uint32_t*)d_back,niter,elems,invert); break;
            case 2: producer<2><<<g,b,0,s>>>(peer_data,peer_flag,(volatile uint32_t*)d_back,niter,elems,invert); break;
            case 1: producer<1><<<g,b,0,s>>>(peer_data,peer_flag,(volatile uint32_t*)d_back,niter,elems,invert); break;
            default:producer<0><<<g,b,0,s>>>(peer_data,peer_flag,(volatile uint32_t*)d_back,niter,elems,invert); break;
        }
    } else {
        consumer<<<g,b,0,s>>>((const uint32_t*)d_data,(volatile uint32_t*)d_flag,
                              peer_back,niter,elems,d_mism,d_to);
    }
    (void)gpuStreamSynchronize(s);
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank==1){
        unsigned long long mism=0; int to=0;
        (void)gpuMemcpy(&mism,d_mism,sizeof(mism),gpuMemcpyDeviceToHost);
        (void)gpuMemcpy(&to,d_to,sizeof(to),gpuMemcpyDeviceToHost);
        printf("fence=%s%s niter=%d elems=%d (%zuKB/round): mismatches=%llu timed_out=%d -> %s\n",
               fence.c_str(), invert?"(INVERT-CONTROL)":"", niter,elems,data_bytes/1024,mism,to,
               to?"TIMEOUT":(mism?"FAIL(stale)":"PASS"));
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    (void)gpuFree(d_data);(void)gpuFree(d_flag);(void)gpuFree(d_back);(void)gpuFree(d_mism);(void)gpuFree(d_to);
    MPI_Finalize(); return 0;
}
