/*
 * xnode_proxy_bench.cpp - CROSS-NODE CPU-proxy put bandwidth/latency sweep.
 *
 * Built on the PROVEN put_two_rank_intranode pattern (gicc::put on device ->
 * proxy ring -> CPU proxy thread fi_write -> CQ -> rt.reset() drains). Run with
 * -N 2 -n 2 --ntasks-per-node=1 so rank0 and rank1 are on DIFFERENT nodes and
 * the put goes over the NIC (Slingshot/CXI), not xGMI.
 *
 * rank 0 issues BATCH puts of `bytes` to rank 1 in one kernel, device-syncs,
 * rt.reset() (drains proxy ring + CQ = completion). We time the full batch over
 * the wire and divide. rank 1 just prepare()/reset() (passive target).
 *
 * Knobs exercised (env, read by the GICC proxy runtime):
 *   GICC_PROXY_SUBMIT_BATCH  (default 32)  cmds submitted per proxy loop pass
 *   GICC_PROXY_CQ_BATCH      (default 256) completions reaped per poll
 *   GICC_NUM_PROXY_THREADS   (default 1)   proxy fleet width
 *   GICC_PROXY_KIDLE_YIELD / _SLEEP        idle backoff (if present in build)
 *
 * Run:
 *   GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 FI_MR_CACHE_MAX_COUNT=0 \
 *     srun -p pci -N 2 -n 2 --ntasks-per-node=1 --gpu-bind=none -t 2 \
 *     ./xnode_proxy_bench
 */

#include <mpi.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

static int env_int(const char* k,int d){ if(const char*v=std::getenv(k)){int x=atoi(v);if(x>0)return x;} return d; }

// rank 0 kernel: issue `batch` puts of `bytes` to dst_rank, all on lane 0.
__global__ void put_batch_kernel(gicc::DeviceCtx* ctx, int dst_rank,
                                 int buf_idx, size_t bytes, int batch){
    if(threadIdx.x==0 && blockIdx.x==0){
        for(int i=0;i<batch;++i)
            gicc::put(ctx, dst_rank, buf_idx, /*dst_off=*/0, buf_idx, /*src_off=*/0, bytes);
    }
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank=0,nranks=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nranks);
    if(nranks!=2){ if(rank==0)fprintf(stderr,"need 2 ranks\n"); MPI_Finalize(); return 1; }

    const size_t kSizes[]={256,1024,4096,16384,65536,262144,1048576,4194304};
    const int kN=sizeof(kSizes)/sizeof(kSizes[0]);
    size_t maxbytes=kSizes[kN-1];
    const int kIters=env_int("GICC_XN_ITERS",100), kWarm=10;
    const int batch=env_int("GICC_XN_BATCH",16);  // puts per kernel (amortize launch)

    gicc::Runtime rt;
    int my=rt.rank(), peer=my^1;
    void* d_buf=nullptr; (void)gpuMalloc(&d_buf,maxbytes); (void)gpuMemset(d_buf,0,maxbytes);
    auto bh=rt.register_buffer(d_buf,maxbytes,true);
    rt.exchange();
    gicc::DeviceCtx* d_ctx=rt.prepare();

    if(my==0){
        printf("# xnode proxy put: batch=%d iters=%d submit_batch=%s cq_batch=%s nthreads=%s\n",
               batch,kIters,
               getenv("GICC_PROXY_SUBMIT_BATCH")?:"32",
               getenv("GICC_PROXY_CQ_BATCH")?:"256",
               getenv("GICC_NUM_PROXY_THREADS")?:"1");
        printf("# size_bytes,us_per_put,GBps\n");
    }

    for(int s=0;s<kN;++s){
        size_t bytes=kSizes[s];
        // warmup
        for(int it=0; it<kWarm; ++it){
            if(my==0){ put_batch_kernel<<<1,1>>>(d_ctx,peer,bh.index,bytes,batch);
                       (void)gpuDeviceSynchronize(); rt.reset(); }
            else { rt.reset(); }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t0=MPI_Wtime();
        for(int it=0; it<kIters; ++it){
            if(my==0){ put_batch_kernel<<<1,1>>>(d_ctx,peer,bh.index,bytes,batch);
                       (void)gpuDeviceSynchronize(); rt.reset(); }
            else { rt.reset(); }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double t1=MPI_Wtime();
        if(my==0){
            double total_puts = (double)kIters*batch;
            double us = (t1-t0)*1e6/total_puts;
            double gbps = bytes/(us*1e3);
            printf("%zu,%.3f,%.2f\n",bytes,us,gbps); fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    (void)gpuFree(d_buf);
    MPI_Finalize(); return 0;
}
