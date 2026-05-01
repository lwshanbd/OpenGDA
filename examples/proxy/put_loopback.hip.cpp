/*
 * put_loopback.hip.cpp - HIP twin of put_loopback.cu. Single-rank L1
 * loopback for the CPU proxy path: register one GPU buffer split into
 * src/dst halves, kernel issues one gicc::put_no_db that copies src ->
 * dst inside the same rank via the proxy worker, host verifies dst
 * matches the src pattern after rt.reset().
 *
 * Run (single rank, no MPI required):
 *   GICC_PROXY_ENABLED=1 ./examples/proxy/put_loopback
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

__global__ void put_kernel(gicc::DeviceCtx* ctx,
                           int self_rank, int buf_idx,
                           size_t half_bytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::put_no_db(ctx,
                        self_rank,
                        buf_idx, /*dst_offset=*/half_bytes,
                        buf_idx, /*src_offset=*/0,
                        half_bytes);
    }
}

int main() {
    gicc::Runtime rt;

    constexpr size_t HALF  = 4096;
    constexpr size_t TOTAL = HALF * 2;

    void* d_buf = nullptr;
    if (hipMalloc(&d_buf, TOTAL) != hipSuccess) {
        fprintf(stderr, "put_loopback: hipMalloc failed\n");
        return 1;
    }

    auto* h_init = static_cast<uint8_t*>(std::malloc(TOTAL));
    std::memset(h_init,        0xAB, HALF);
    std::memset(h_init + HALF, 0x00, HALF);
    (void)hipMemcpy(d_buf, h_init, TOTAL, hipMemcpyHostToDevice);

    auto bh = rt.register_buffer(d_buf, TOTAL, /*is_device=*/true);
    rt.exchange();

    gicc::DeviceCtx* d_ctx = rt.prepare();
    gpuLaunchKernel(put_kernel, dim3(1), dim3(1), 0, 0,
                    d_ctx, rt.rank(), bh.index, HALF);
    if (gpuDeviceSynchronize() != GPU_SUCCESS) {
        fprintf(stderr, "put_loopback: kernel sync failed\n");
        return 2;
    }
    rt.reset();

    auto* h_check = static_cast<uint8_t*>(std::malloc(HALF));
    (void)hipMemcpy(h_check, static_cast<uint8_t*>(d_buf) + HALF, HALF,
                    hipMemcpyDeviceToHost);
    int errors = 0;
    for (size_t i = 0; i < HALF; ++i) {
        if (h_check[i] != 0xAB) ++errors;
    }
    printf("put_loopback: %s (%d errors over %zu bytes)\n",
           errors == 0 ? "PASS" : "FAIL", errors, HALF);

    std::free(h_init);
    std::free(h_check);
    (void)hipFree(d_buf);
    return errors == 0 ? 0 : 3;
}
