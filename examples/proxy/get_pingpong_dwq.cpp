/*
 * get_pingpong_dwq.cpp - DWQ (libfabric deferred work + CXI trigger) GET smoke test.
 *
 * Sibling of get_pingpong.cpp (which exercises the CPU proxy path).
 * Here we exercise the *DWQ* path:
 *   - rt.get_no_db() queues queue_rma_read + chained queue_atomic_signal
 *     into cxi's deferred work queue, gated on the trigger counter.
 *   - The kernel only does flush() — one MMIO write to the trigger counter.
 *   - The NIC sees the trigger cross the queued threshold and fires the
 *     READ; on completion the atomic_signal pushes 1 into d_slot_pool[i].
 *   - Host-side rt.reset() / rt.wait() polls libfabric counters until done.
 *
 * Read-ordering / HBM visibility:
 *   libfabric's FI_HMEM contract (the dst buffer was registered with iface
 *   = FI_HMEM_ROCR / FI_HMEM_CUDA) says the provider must ensure the data
 *   has been *committed to GPU HBM* before signalling completion. So once
 *   rt.reset() returns, GPU SMs can read dst freely. No CST READ needed.
 *
 * Verification quirk on Tioga: hipMemcpy(D2H) on small buffers can read
 * from a stale GPU L2 cache and miss the just-arrived NIC data. So we
 * verify via a kernel-driven copy into pinned memory, which goes through
 * SMs and bypasses that pathology. This mirrors legacy/ofi/test_ofi_get.cpp.
 *
 * Build / run on Tioga:
 *   make -j get_pingpong_dwq
 *   srun -p pci -t 2 -N 1 -n 2 ./examples/proxy/get_pingpong_dwq
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

// Single-thread flush kernel: writes the cxi trigger MMIO counter once,
// which fires every DWQ op queued before launch (the GET + its chained
// atomic_signal).
__global__ void dwq_flush_kernel(gicc::DeviceCtx* ctx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::flush(ctx);
    }
}

// Kernel-driven verification copy. hipMemcpy(D2H) on small buffers can
// satisfy reads from a stale GPU L2, missing NIC-deposited data; reads
// from an SM kernel go through the GPU's memory subsystem in the usual
// way and see HBM truth. We copy d_src -> mapped h_pinned and verify on
// the host.
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
            fprintf(stderr,
                "get_pingpong_dwq: need exactly 2 ranks (got %d)\n",
                nranks);
        }
        return 1;
    }

    constexpr size_t WIN       = 1024;
    constexpr size_t SRC_OFF   = 0;
    constexpr size_t LAND_OFF  = 4096;
    constexpr size_t BUF_BYTES = LAND_OFF + WIN;   // 5120

    void* d_buf = nullptr;
    if (gpuMalloc(&d_buf, BUF_BYTES) != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: gpuMalloc failed\n", rank); return 2;
    }
    (void)gpuMemset(d_buf, 0, BUF_BYTES);

    // SOURCE [0,WIN): our per-rank pattern. Peer will GET it.
    auto* h_src = static_cast<uint8_t*>(std::malloc(WIN));
    for (size_t i = 0; i < WIN; ++i) {
        h_src[i] = (uint8_t)((rank * 31 + i) & 0xFF);
    }
    (void)gpuMemcpy(d_buf, h_src, WIN, gpuMemcpyHostToDevice);
    std::free(h_src);

    auto bh = rt.register_buffer(d_buf, BUF_BYTES, /*is_device=*/true);
    rt.exchange();
    rt.barrier();

    const int peer = 1 - rank;

    auto tok = rt.get_no_db(bh,
                            peer, bh.index,
                            WIN,
                            /*local_offset=*/LAND_OFF,
                            /*remote_offset=*/SRC_OFF);

    gicc::DeviceCtx* d_ctx = rt.prepare();

    gpuLaunchKernel(dwq_flush_kernel, dim3(1), dim3(1), 0, 0, d_ctx);
    if (gpuDeviceSynchronize() != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: flush kernel sync failed\n", rank);
        return 3;
    }
    // FI_HMEM contract: rt.wait returns only after libfabric signals the
    // GET as locally complete, which for an FI_HMEM_ROCR destination
    // implies the data is committed to HBM.
    rt.wait(tok);
    rt.reset();
    rt.barrier();

    // Verify via SM-driven copy into pinned memory (avoids hipMemcpy(D2H)
    // L2-cache pathology for small reads).
    uint8_t* h_pinned = nullptr;
    (void)gpuHostMalloc(&h_pinned, WIN, gpuHostMallocMapped);
    uint8_t* d_pinned = nullptr;
    (void)gpuHostGetDevicePointer((void**)&d_pinned, h_pinned, 0);

    auto* d_landing = static_cast<uint8_t*>(d_buf) + LAND_OFF;
    gpuLaunchKernel(verify_copy_kernel,
                    dim3(16), dim3(256), 0, 0,
                    (const uint8_t*)d_landing, d_pinned, (int)WIN);
    (void)gpuDeviceSynchronize();

    int errors = 0;
    for (size_t i = 0; i < WIN; ++i) {
        uint8_t want = (uint8_t)((peer * 31 + i) & 0xFF);
        if (h_pinned[i] != want) {
            if (errors < 4) {
                fprintf(stderr,
                    "  rank %d: mismatch @%zu got=0x%02x want=0x%02x\n",
                    rank, i, h_pinned[i], (unsigned)want);
            }
            ++errors;
        }
    }
    printf("rank %d get_pingpong_dwq: %s (%d errors over %zu bytes)\n",
           rank, errors == 0 ? "PASS" : "FAIL", errors, WIN);

    (void)gpuHostFree(h_pinned);
    rt.barrier();
    (void)gpuFree(d_buf);
    return errors == 0 ? 0 : 4;
}
