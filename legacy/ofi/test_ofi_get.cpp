#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void get_kernel(gicc::DeviceCtx* ctx) {
    if (threadIdx.x == 0) { gicc::flush(ctx); gicc::quiet(ctx); }
}

__global__ void verify_copy_kernel(const unsigned char* src, unsigned char* dst, int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

int main() {
    gicc::Runtime rt;
    if (rt.size() != 2) return 1;
    int peer = 1 - rt.rank();

    void *d_local, *d_remote;
    hipMalloc(&d_local, 4096); hipMemset(d_local, 0, 4096);
    hipMalloc(&d_remote, 4096);
    if (rt.rank() == 1) hipMemset(d_remote, 0xAB, 4096);
    hipDeviceSynchronize();

    auto lb = rt.register_buffer(d_local, 4096, true);
    auto rb = rt.register_buffer(d_remote, 4096, true);
    rt.exchange();
    rt.boot().barrier();

    if (rt.rank() == 0) {
        rt.get(lb, peer, rb.index, 4096, 0, 0);
        auto* ctx = rt.prepare(peer, rb.index);
        hipLaunchKernelGGL(get_kernel, dim3(1), dim3(1), 0, 0, ctx);
        hipDeviceSynchronize();
        rt.reset();

        // Verify via kernel-driven copy (kernel reads bypass stale L2 — see
        // project notes on hipMemcpy(D2H) reading stale cache for <16KB).
        unsigned char* h_pinned = nullptr;
        hipHostMalloc(&h_pinned, 4096, hipHostMallocMapped);
        unsigned char* d_pinned = nullptr;
        hipHostGetDevicePointer((void**)&d_pinned, h_pinned, 0);
        hipLaunchKernelGGL(verify_copy_kernel, dim3(16), dim3(256), 0, 0,
                           (const unsigned char*)d_local, d_pinned, 4096);
        hipDeviceSynchronize();
        int errs = 0;
        for (int i = 0; i < 4096; i++) if (h_pinned[i] != 0xAB) errs++;
        printf("ofi get_no_db: %s (errs=%d, h[0]=0x%02x h[1]=0x%02x h[4095]=0x%02x)\n",
               errs == 0 ? "PASS" : "FAIL", errs, h_pinned[0], h_pinned[1], h_pinned[4095]);
        hipHostFree(h_pinned);
    }
    rt.boot().barrier();
    return 0;
}
