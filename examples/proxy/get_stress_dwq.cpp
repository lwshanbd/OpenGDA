/*
 * get_stress_dwq.cpp - DWQ GET stress test.
 *
 * Sibling of get_stress.cpp (CPU proxy path). Hammers the DWQ READ
 * + libfabric/FI_HMEM completion path with N_ITERS back-to-back GETs
 * of WIN_BYTES each, checking the full landing region byte-by-byte
 * for any mismatch.
 *
 * Each iteration:
 *   - rt.get_no_db (host enqueues queue_rma_read into DWQ)
 *   - launch single-thread flush kernel (writes trigger MMIO)
 *   - rt.wait(tok) host-polls slots_[].completion_cntr
 *   - rt.reset() (recycle batch state)
 *   - SM-driven verify kernel reads landing into pinned mem; host compares
 *
 * Build / run:
 *   make -j get_stress_dwq
 *   srun -p pci -t 5 -N 1 -n 2 ./examples/proxy/get_stress_dwq
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

#ifndef N_ITERS
#define N_ITERS 200
#endif
#ifndef WIN_BYTES
#define WIN_BYTES (256u * 1024u)
#endif

__global__ void dwq_flush_kernel(gicc::DeviceCtx* ctx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::flush(ctx);
    }
}

// Kernel-driven verification copy. Bypasses the hipMemcpy(D2H) stale-L2
// pathology for small reads on Tioga.
__global__ void verify_copy_kernel(const uint8_t* __restrict__ src,
                                   uint8_t* __restrict__ dst, int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

int main(int /*argc*/, char** /*argv*/) {
    gicc::Runtime rt;
    const int rank   = rt.rank();
    const int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) {
            fprintf(stderr, "get_stress_dwq: need exactly 2 ranks (got %d)\n", nranks);
        }
        return 1;
    }

    const size_t SOURCE_BYTES = (size_t)N_ITERS * WIN_BYTES;
    const size_t LAND_OFF     = SOURCE_BYTES;
    const size_t BUF_BYTES    = LAND_OFF + WIN_BYTES;

    void* d_buf = nullptr;
    if (gpuMalloc(&d_buf, BUF_BYTES) != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: gpuMalloc(%zu) failed\n", rank, BUF_BYTES);
        return 2;
    }
    (void)gpuMemset(d_buf, 0, BUF_BYTES);

    // Init N distinct source slots; each byte k of slot s is (rank*31+s+k) & 0xFF.
    {
        auto* h_src = static_cast<uint8_t*>(std::malloc(SOURCE_BYTES));
        for (int s = 0; s < N_ITERS; ++s) {
            uint8_t base = (uint8_t)((rank * 31 + s) & 0xFF);
            uint8_t* dst = h_src + (size_t)s * WIN_BYTES;
            for (uint32_t i = 0; i < WIN_BYTES; ++i) {
                dst[i] = (uint8_t)((base + (uint8_t)(i & 0xFF)) & 0xFF);
            }
        }
        (void)gpuMemcpy(d_buf, h_src, SOURCE_BYTES, gpuMemcpyHostToDevice);
        std::free(h_src);
    }

    auto bh = rt.register_buffer(d_buf, BUF_BYTES, /*is_device=*/true);
    rt.exchange();
    rt.barrier();

    const int peer = 1 - rank;
    uint8_t* d_landing = static_cast<uint8_t*>(d_buf) + LAND_OFF;

    // Reusable pinned check buffer.
    uint8_t* h_check = nullptr;
    (void)gpuHostMalloc(&h_check, WIN_BYTES, gpuHostMallocMapped);
    uint8_t* d_check = nullptr;
    (void)gpuHostGetDevicePointer((void**)&d_check, h_check, 0);

    uint64_t total_errors = 0;
    int bad_iters = 0;
    int first_bad = -1, last_bad = -1;

    for (int k = 0; k < N_ITERS; ++k) {
        auto tok = rt.get_no_db(bh,
                                peer, bh.index,
                                WIN_BYTES,
                                /*local_offset=*/LAND_OFF,
                                /*remote_offset=*/(size_t)k * WIN_BYTES);

        gicc::DeviceCtx* d_ctx = rt.prepare();

        gpuLaunchKernel(dwq_flush_kernel, dim3(1), dim3(1), 0, 0, d_ctx);
        if (gpuDeviceSynchronize() != GPU_SUCCESS) {
            fprintf(stderr, "rank %d iter %d: flush sync failed\n", rank, k);
            return 3;
        }
        rt.wait(tok);
        rt.reset();

        // SM-driven copy + verify
        gpuLaunchKernel(verify_copy_kernel,
                        dim3(16), dim3(256), 0, 0,
                        (const uint8_t*)d_landing, d_check, (int)WIN_BYTES);
        (void)gpuDeviceSynchronize();

        uint8_t base = (uint8_t)((peer * 31 + k) & 0xFF);
        uint32_t iter_errors = 0;
        for (uint32_t i = 0; i < WIN_BYTES; ++i) {
            uint8_t want = (uint8_t)((base + (uint8_t)(i & 0xFF)) & 0xFF);
            if (h_check[i] != want) ++iter_errors;
        }
        if (iter_errors) {
            ++bad_iters;
            if (first_bad < 0) first_bad = k;
            last_bad = k;
            total_errors += iter_errors;
            if (bad_iters <= 4) {
                fprintf(stderr,
                    "  rank %d iter %4d: %u/%u byte mismatches\n",
                    rank, k, iter_errors, (unsigned)WIN_BYTES);
            }
        }
    }
    rt.barrier();

    printf("rank %d get_stress_dwq: %s "
           "(bad_iters=%d/%d, total_mismatches=%lu, range=[%d,%d], "
           "win=%uB, iters=%d)\n",
           rank, total_errors == 0 ? "PASS" : "FAIL",
           bad_iters, N_ITERS, (unsigned long)total_errors,
           first_bad, last_bad,
           WIN_BYTES, N_ITERS);

    (void)gpuHostFree(h_check);
    (void)gpuFree(d_buf);
    return total_errors == 0 ? 0 : 4;
}
