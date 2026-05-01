/*
 * put_loopback.cu - Single-rank L1 loopback for the CPU proxy path.
 *
 * Registers ONE GPU buffer split into a "src" half (filled with 0xAB on
 * the host) and a "dst" half (zeroed). A 1-thread kernel issues a single
 * gicc::put_no_db that copies src -> dst INTO THE SAME RANK via the CPU
 * proxy: device-side put_no_db atomic_pushes a TransferCmd into the
 * DeviceCtx::proxy_ring; the proxy worker pops it, submits an fi_write to
 * the local av_addr, and CQ-acks. After cudaDeviceSynchronize() we call
 * rt.reset() which drains the proxy ring (snapshot head, spin until tail
 * catches up). Finally the host memcpy's the dst half back and verifies
 * every byte equals 0xAB.
 *
 * Run (single rank, no MPI required):
 *   GICC_PROXY_ENABLED=1 ./examples/proxy/put_loopback
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

__global__ void put_kernel(gicc::DeviceCtx* ctx,
                           int self_rank, int buf_idx,
                           size_t half_bytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // src lives at offset 0; dst at offset half_bytes within the same
        // registered buffer. Self-loopback: dst_rank == self_rank.
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
    if (cudaMalloc(&d_buf, TOTAL) != cudaSuccess) {
        fprintf(stderr, "put_loopback: cudaMalloc failed\n");
        return 1;
    }

    // Initialize: src half = 0xAB, dst half = 0x00.
    auto* h_init = static_cast<uint8_t*>(std::malloc(TOTAL));
    std::memset(h_init,            0xAB, HALF);
    std::memset(h_init + HALF,     0x00, HALF);
    (void)cudaMemcpy(d_buf, h_init, TOTAL, cudaMemcpyHostToDevice);

    auto bh = rt.register_buffer(d_buf, TOTAL, /*is_device=*/true);
    rt.exchange();   // single rank: no peers to discover.

    // First prepare() lazy-starts the proxy worker and writes the device-
    // mapped ring pointer into DeviceCtx. Capture the device pointer so we
    // can hand it to the kernel directly.
    gicc::DeviceCtx* d_ctx = rt.prepare();

    gpuLaunchKernel(put_kernel, dim3(1), dim3(1), 0, 0,
                    d_ctx, rt.rank(), bh.index, HALF);
    if (gpuDeviceSynchronize() != GPU_SUCCESS) {
        fprintf(stderr, "put_loopback: kernel sync failed\n");
        return 2;
    }
    rt.reset();   // drains the proxy ring + CQ.

    // Verify the dst half now matches src.
    auto* h_check = static_cast<uint8_t*>(std::malloc(HALF));
    (void)cudaMemcpy(h_check, static_cast<uint8_t*>(d_buf) + HALF, HALF,
                     cudaMemcpyDeviceToHost);
    int errors = 0;
    for (size_t i = 0; i < HALF; ++i) {
        if (h_check[i] != 0xAB) ++errors;
    }
    printf("put_loopback: %s (%d errors over %zu bytes)\n",
           errors == 0 ? "PASS" : "FAIL", errors, HALF);

    std::free(h_init);
    std::free(h_check);
    (void)cudaFree(d_buf);
    return errors == 0 ? 0 : 3;
}
