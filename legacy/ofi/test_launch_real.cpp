// Smoke test for Phase G (CMake plugin integration). Mirrors test_ofi_get
// but uses gicc::launch instead of an explicit rt.put_no_db + prepare +
// hipLaunchKernelGGL sequence. The plugin must lift the put_no_db inside
// put_kernel into a host-side kernel_trace specialization; without that
// specialization, the no-op default in launch.hpp would issue zero RDMA
// ops and rank 1's verify would see all zeros.
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

#define CHECK(x) do {                                                       \
    if (!(x)) {                                                             \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);        \
        return 1;                                                           \
    }                                                                       \
} while (0)

__global__ void put_kernel(gicc::DeviceCtx* ctx,
                           int peer, int dst_buf,
                           uint64_t la, uint32_t lk,
                           uint64_t ra, uint32_t rk, uint32_t s) {
    gicc::put_no_db(ctx, peer, dst_buf, la, lk, ra, rk, s);
    if (threadIdx.x == 0) {
        gicc::flush(ctx);
        gicc::quiet(ctx);
    }
}

__global__ void verify_kernel(const uint8_t* src, uint8_t* dst, size_t n) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

int main() {
    gicc::Runtime rt;
    if (rt.size() != 2) return 1;
    int peer = 1 - rt.rank();
    constexpr size_t SIZE = 4096;

    void *ds = nullptr, *dd = nullptr;
    CHECK(hipMalloc(&ds, SIZE) == hipSuccess);
    CHECK(hipMalloc(&dd, SIZE) == hipSuccess);
    CHECK(hipMemset(dd, 0, SIZE) == hipSuccess);
    if (rt.rank() == 0) CHECK(hipMemset(ds, 0xAB, SIZE) == hipSuccess);
    CHECK(hipDeviceSynchronize() == hipSuccess);

    auto sb = rt.register_buffer(ds, SIZE, true);
    auto db = rt.register_buffer(dd, SIZE, true);
    rt.exchange();
    rt.boot().barrier();

    if (rt.rank() == 0) {
        auto ri = rt.remote_buffer(peer, db.index);
        // peer + dst_buf are now per-CALL on put_no_db (passed as kernel
        // args), so launch itself doesn't need them. The kernel can issue
        // puts to multiple peers in one launch.
        gicc::launch<put_kernel>(rt, dim3(1), dim3(1),
                                 peer, (int)db.index,
                                 (uint64_t)sb.addr, sb.lkey,
                                 (uint64_t)ri.addr, ri.rkey,
                                 (uint32_t)SIZE);
        CHECK(hipDeviceSynchronize() == hipSuccess);
        rt.reset();
    }
    rt.boot().barrier();

    if (rt.rank() == 1) {
        // Verify via kernel-driven copy (project gotcha: hipMemcpy(D2H)
        // reads stale L2 cache for sizes <16KB; a kernel read invalidates
        // L2 implicitly).
        uint8_t* h_pinned = nullptr;
        CHECK(hipHostMalloc(&h_pinned, SIZE, hipHostMallocMapped) == hipSuccess);
        uint8_t* d_pinned = nullptr;
        CHECK(hipHostGetDevicePointer((void**)&d_pinned, h_pinned, 0) == hipSuccess);
        hipLaunchKernelGGL(verify_kernel,
                           dim3((SIZE + 255) / 256), dim3(256), 0, 0,
                           (const uint8_t*)dd, d_pinned, SIZE);
        CHECK(hipDeviceSynchronize() == hipSuccess);
        int errs = 0;
        for (size_t i = 0; i < SIZE; i++) {
            if (h_pinned[i] != 0xAB) errs++;
        }
        printf("launch_real: %s (errs=%d, h[0]=0x%02x h[%zu]=0x%02x)\n",
               errs == 0 ? "PASS" : "FAIL", errs,
               h_pinned[0], SIZE - 1, h_pinned[SIZE - 1]);
        CHECK(hipHostFree(h_pinned) == hipSuccess);
    }
    rt.boot().barrier();
    return 0;
}
