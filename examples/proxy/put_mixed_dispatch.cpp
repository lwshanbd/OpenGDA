/*
 * put_mixed_dispatch.cpp - L4 mixed-dispatch CPU-proxy example.
 *
 * Two MPI ranks. Rank 0's kernel issues TWO gicc::put_no_db calls into
 * non-overlapping regions of rank 1's buffer:
 *
 *   site 0  → DWQ_TRIGGER       (host trace pre-stages; device call
 *                                erased by GICCDeviceLowering)
 *   site 1  → CPU_PROXY_ENQUEUE (device pushes a TransferCmd into the
 *                                proxy ring; host trace emits nothing
 *                                for this site)
 *
 * The whole point of this example is to exercise Task 8's hint-aware
 * device-lowering: it must PRESERVE the device-side put_no_db body for
 * the kernel (so the proxy site runs) while still erasing the DWQ site
 * via the host trace. Because both sites live in the same kernel
 * function, the preservation decision is per-kernel (proxy_aware bit
 * read from per-kernel JSON), not per-site — the DWQ-routed call is
 * simply a no-op in the device body once GICC_CPU_PROXY is defined,
 * since DeviceCtx::proxy_ring belongs to a single ring shared by both
 * sites and the DWQ leg is fully owned by the host trace.
 *
 * After rt.reset() and a barrier, rank 1 verifies:
 *   bytes [0,    OFF_DWQ + LEN) match rank 0's source pattern (DWQ leg)
 *   bytes [OFF_PROXY, OFF_PROXY + LEN) match rank 0's source pattern
 *                                       (PROXY leg)
 * The gap [LEN, OFF_PROXY) stays zero — confirms the two transfers
 * landed at distinct destinations.
 *
 * Build / run flow (Delta or Tioga):
 *   1. Configure the runtime build with -DGICC_ENABLE_CPU_PROXY=ON.
 *   2. Make sure the LTO pass plugin (libgicc-passes.so) is built.
 *   3. To exercise the hint-aware path end-to-end, the kernel must be
 *      compiled with -fpass-plugin=libgicc-passes.so plus
 *      GICC_MODE=lower / GICC_HINT_IN=hint_mixed.json /
 *      GICC_META_DIR=<dir>. The hint file is provided alongside this
 *      source as `hint_mixed.json.template`; replace the site_id
 *      placeholders with the actual ids that feature-extraction emits
 *      (run once with GICC_MODE=feature-extract to see them).
 *   4. At runtime, set GICC_PROXY_ENABLED=1 so the proxy worker starts
 *      and the dispatch-lowering cross-check passes.
 *
 *   GICC_PROXY_ENABLED=1 GICC_HINT_IN=/path/to/hint_mixed.json \
 *       GICC_META_DIR=/tmp/gicc-meta \
 *       srun -p pci -t 2 -N 1 -n 2 ./examples/proxy/put_mixed_dispatch
 *
 * Without the LTO pass plugin both put_no_db calls fall through the
 * non-LTO path. Under GICC_CPU_PROXY that path pushes both into the
 * proxy ring, so the example still PASSes (no DWQ leg actually fires,
 * but bytes still arrive on rank 1 via the proxy). Treat that as a
 * partial smoke run; only the LTO build actually validates the
 * hint-aware preservation.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

__global__ void mixed_kernel(gicc::DeviceCtx* ctx,
                             int dst_rank, int dst_buf_idx, int src_buf_idx,
                             size_t off_dwq, size_t off_proxy, size_t bytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // site 0: routed to DWQ_TRIGGER by hint_mixed.json. With the
        // LTO pipeline the device-side call here is the body in
        // ofi_device.cuh; for proxy_aware kernels it's preserved but
        // the DWQ work is performed by the host trace ahead of the
        // kernel launch. Without LTO, see file header.
        gicc::put_no_db(ctx,
                        dst_rank,
                        dst_buf_idx, /*dst_offset=*/off_dwq,
                        src_buf_idx, /*src_offset=*/off_dwq,
                        bytes);
        // site 1: routed to CPU_PROXY_ENQUEUE. Device push to the
        // proxy ring; CPU proxy worker submits via libfabric.
        gicc::put_no_db(ctx,
                        dst_rank,
                        dst_buf_idx, /*dst_offset=*/off_proxy,
                        src_buf_idx, /*src_offset=*/off_proxy,
                        bytes);
        // flush: lead-thread MMIO write triggers all queued DWQ ops
        // (a no-op for proxy-only sites).
        gicc::flush(ctx);
    }
}

int main(int /*argc*/, char** /*argv*/) {
    gicc::Runtime rt;
    const int rank   = rt.rank();
    const int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) {
            fprintf(stderr,
                "put_mixed_dispatch: need exactly 2 ranks (got %d)\n",
                nranks);
        }
        return 1;
    }

    constexpr size_t LEN       = 4096;
    constexpr size_t OFF_DWQ   = 0;
    constexpr size_t OFF_PROXY = 8192;
    constexpr size_t BUF_BYTES = OFF_PROXY + LEN;   // 12288

    void* d_buf = nullptr;
    if (gpuMalloc(&d_buf, BUF_BYTES) != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: gpuMalloc failed\n", rank);
        return 2;
    }
    (void)gpuMemset(d_buf, 0, BUF_BYTES);

    if (rank == 0) {
        // Pre-fill the entire local buffer with a deterministic byte
        // pattern. Both transfers copy slices of this pattern into the
        // destination at matching offsets so rank 1 can verify by
        // comparing against the same formula.
        auto* h_pattern = static_cast<uint8_t*>(std::malloc(BUF_BYTES));
        for (size_t i = 0; i < BUF_BYTES; ++i)
            h_pattern[i] = (uint8_t)((i * 37 + 11) & 0xFF);
        (void)gpuMemcpy(d_buf, h_pattern, BUF_BYTES, gpuMemcpyHostToDevice);
        std::free(h_pattern);
    }

    auto bh = rt.register_buffer(d_buf, BUF_BYTES, /*is_device=*/true);
    rt.exchange();

    rt.barrier();

    if (rank == 0) {
        gicc::DeviceCtx* d_ctx = rt.prepare();
        gpuLaunchKernel(mixed_kernel, dim3(1), dim3(1), 0, 0,
                        d_ctx, /*dst_rank=*/1,
                        /*dst_buf=*/bh.index, /*src_buf=*/bh.index,
                        OFF_DWQ, OFF_PROXY, LEN);
        if (gpuDeviceSynchronize() != GPU_SUCCESS) {
            fprintf(stderr, "rank 0: kernel sync failed\n");
            return 3;
        }
        rt.reset();   // drains the proxy ring + CQ.
    } else {
        (void)rt.prepare();
        rt.reset();
    }

    rt.barrier();

    int rc = 0;
    if (rank == 1) {
        auto* h_check = static_cast<uint8_t*>(std::malloc(BUF_BYTES));
        (void)gpuMemcpy(h_check, d_buf, BUF_BYTES, gpuMemcpyDeviceToHost);
        int errors_dwq   = 0;
        int errors_proxy = 0;
        int errors_gap   = 0;
        for (size_t i = 0; i < LEN; ++i) {
            uint8_t want = (uint8_t)(((OFF_DWQ + i) * 37 + 11) & 0xFF);
            if (h_check[OFF_DWQ + i] != want) ++errors_dwq;
        }
        for (size_t i = 0; i < LEN; ++i) {
            uint8_t want = (uint8_t)(((OFF_PROXY + i) * 37 + 11) & 0xFF);
            if (h_check[OFF_PROXY + i] != want) ++errors_proxy;
        }
        for (size_t i = LEN; i < OFF_PROXY; ++i) {
            if (h_check[i] != 0) ++errors_gap;
        }
        const int total = errors_dwq + errors_proxy + errors_gap;
        printf("put_mixed_dispatch: %s "
               "(dwq=%d proxy=%d gap=%d over %zu bytes)\n",
               total == 0 ? "PASS" : "FAIL",
               errors_dwq, errors_proxy, errors_gap, BUF_BYTES);
        std::free(h_check);
        rc = (total == 0) ? 0 : 4;
    }

    rt.barrier();
    (void)gpuFree(d_buf);
    return rc;
}
