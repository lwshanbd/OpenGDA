/**
 * test_gicc.cpp - Smoke test for the unified gicc::launch API on the
 * libfabric/CXI backend.
 *
 * The kernel below is structurally identical to what one would write for
 * the NVIB/mlx5 backend: an HK for-loop calling gicc::put_no_db inline,
 * followed by gicc::flush(ctx) + gicc::quiet(ctx). On OFI the
 * gicc-clang-plugin lifts the put_no_db loop into a host-side
 * kernel_trace<&put_kernel> specialization that pre-stages each RDMA
 * write via the DWQ; gicc::launch invokes that trace, then prepare(),
 * then the kernel itself (where flush + quiet take over device-side).
 *
 * Run: FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 ./test_gicc
 */
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"

__global__ void put_kernel(gicc::DeviceCtx* ctx,
                           int peer, int dst_buf,
                           int n, uint64_t la, uint32_t lk,
                           uint64_t ra, uint32_t rk, uint32_t s)
{
    for (int i = 0; i < n; i++) {
        gicc::put_no_db(ctx, peer, dst_buf,
                        la + (uint64_t)i * s, lk,
                        ra + (uint64_t)i * s, rk, s);
    }
    if (threadIdx.x == 0) {
        gicc::flush(ctx);
        gicc::quiet(ctx);
    }
}

__global__ void verify_copy_kernel(const uint8_t* src, uint8_t* dst, size_t n) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

static void verify_dst(int rank, void* d_dst, size_t size, const char* name) {
    if (rank != 1) return;
    uint8_t* h_pinned = nullptr;
    (void)hipHostMalloc(&h_pinned, size, hipHostMallocMapped);
    uint8_t* d_pinned = nullptr;
    (void)hipHostGetDevicePointer((void**)&d_pinned, h_pinned, 0);
    hipLaunchKernelGGL(verify_copy_kernel,
                       dim3((size + 255) / 256), dim3(256), 0, 0,
                       (const uint8_t*)d_dst, d_pinned, size);
    (void)hipDeviceSynchronize();
    int errors = 0;
    for (size_t i = 0; i < size; i++) if (h_pinned[i] != 0xAB) errors++;
    printf("%s: %s (%zu bytes)\n", name, errors == 0 ? "PASSED" : "FAILED", size);
    (void)hipHostFree(h_pinned);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    gicc::Runtime rt;
    int rank = rt.rank();
    int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "Need exactly 2 ranks\n");
        return 1;
    }

    int peer = 1 - rank;
    constexpr size_t SIZE = 4096;
    constexpr int    N_PUTS = 4;

    void* d_src = nullptr;
    void* d_dst = nullptr;
    (void)hipMalloc(&d_src, SIZE * N_PUTS);
    (void)hipMalloc(&d_dst, SIZE * N_PUTS);
    if (rank == 0) (void)hipMemset(d_src, 0xAB, SIZE * N_PUTS);
    (void)hipMemset(d_dst, 0, SIZE * N_PUTS);
    (void)hipDeviceSynchronize();

    auto src_buf = rt.register_buffer(d_src, SIZE * N_PUTS, true);
    auto dst_buf = rt.register_buffer(d_dst, SIZE * N_PUTS, true);
    rt.exchange();
    rt.boot().barrier();

    if (rank == 0) {
        printf("=== gicc::launch unified API: %d puts of %zu bytes ===\n",
               N_PUTS, SIZE);
        auto ri = rt.remote_buffer(peer, dst_buf.index);
        gicc::launch<put_kernel>(rt, dim3(1), dim3(1),
                                 peer, (int)dst_buf.index,
                                 N_PUTS,
                                 (uint64_t)src_buf.addr, src_buf.lkey,
                                 (uint64_t)ri.addr,     ri.rkey,
                                 (uint32_t)SIZE);
        (void)hipDeviceSynchronize();
        rt.reset();
    }
    rt.boot().barrier();
    verify_dst(rank, d_dst, SIZE * N_PUTS, "gicc::launch + put_no_db loop");
    rt.boot().barrier();

    (void)hipFree(d_src);
    (void)hipFree(d_dst);
    return 0;
}
