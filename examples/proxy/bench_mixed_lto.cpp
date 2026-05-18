/*
 * bench_mixed_lto.cpp — LTO mixed-dispatch experiment.
 *
 * Question: can ONE program use BOTH DWQ-triggered RDMA and CPU-proxy
 * RDMA simultaneously? NCCL on InfiniBand requires two separate
 * communicators for this kind of split; we want to know if our
 * unified Runtime supports it via per-site dispatch routing.
 *
 * Two put_no_db sites in one kernel, both writing to non-overlapping
 * regions on the same peer:
 *   site 0 (OFF_DWQ region)   → hint.json routes to DWQ_TRIGGER
 *                                (host trace pre-stages via fi_control,
 *                                 kernel fires via MMIO trigger)
 *   site 1 (OFF_PROXY region) → hint.json routes to CPU_PROXY_ENQUEUE
 *                                (kernel pushes TransferCmd to ring,
 *                                 proxy worker submits via fi_write)
 *
 * Both legs target the SAME peer buffer at DIFFERENT offsets so rank 1
 * can byte-verify each region independently.
 *
 * Resource check:
 *   - DWQ uses fabric->ep + main MR
 *   - Proxy uses proxy_eps_[0..N] + per-EP MRs
 *   - AV is shared (peer addrs valid across both)
 *   - mono_total_ops_ counter advances ONLY for DWQ leg
 *   - Proxy ring counter advances for proxy leg
 *
 * If everything works, this proves the answer to the user's question
 * is YES — mixed dispatch in one program is architecturally supported
 * AND has zero resource conflict.
 */

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/launch.hpp"
#include "gicc/platform/ofi/runtime_helpers.h"

constexpr size_t LEN       = 4096;
constexpr size_t OFF_DWQ   = 0;
constexpr size_t OFF_PROXY = 16384;       // well-separated so any sloppy
                                          //   stride bug surfaces immediately
constexpr size_t BUF_BYTES = OFF_PROXY + LEN;

// Both sites must be HK-capable for the LTO pass to accept routing.
// All args here are kernel formals or constants — HK by construction.
__global__ void mixed_kernel(gicc::DeviceCtx* ctx,
                             int dst_rank, int dst_buf, int src_buf,
                             size_t bytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[kernel] entering. ctx=%p proxy_ring=%p trigger_addr=%p\n",
               ctx, ctx ? ctx->proxy_ring : nullptr,
               ctx ? (void*)ctx->trigger_addr_ : nullptr);
        // site 0 — intended DWQ_TRIGGER (constant offset 0)
        gicc::put_no_db(ctx, dst_rank,
                        dst_buf, /*dst_off=*/0,
                        src_buf, /*src_off=*/0,
                        bytes);
        printf("[kernel] after site 0 (DWQ-routed put)\n");
        // site 1 — intended CPU_PROXY_ENQUEUE (constant offset 16384)
        gicc::put_no_db(ctx, dst_rank,
                        dst_buf, /*dst_off=*/16384,
                        src_buf, /*src_off=*/16384,
                        bytes);
        printf("[kernel] after site 1 (PROXY-routed put)\n");
        // flush: device-pass replaces with lead-thread MMIO write,
        //   which fires the DWQ-routed site. Proxy site already
        //   completed asynchronously via its own ring path.
        gicc::flush(ctx);
        printf("[kernel] after flush\n");
        gicc::quiet(ctx);
        printf("[kernel] after quiet (exiting)\n");
    }
}

int main(int /*argc*/, char** /*argv*/) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    gicc::Runtime rt;
    rt.enable_host_wait_mode();   // needed for DWQ-trigger path
    int rank   = rt.rank();
    int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "need 2 ranks\n");
        return 1;
    }

    void* d_buf = nullptr;
    if (gpuMalloc(&d_buf, BUF_BYTES) != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: gpuMalloc failed\n", rank);
        return 2;
    }

    // Rank 0: stamp two DIFFERENT byte patterns into the two regions
    //         so we can tell DWQ leg vs proxy leg apart at the receiver.
    //         Region [0..LEN) gets 0xAA (DWQ).
    //         Region [OFF_PROXY..OFF_PROXY+LEN) gets 0xBB (proxy).
    //         Gap stays 0x00 (catches stride bugs).
    if (rank == 0) {
        (void)gpuMemset(d_buf, 0x00, BUF_BYTES);
        (void)gpuMemset(d_buf, 0xAA, LEN);
        (void)gpuMemset((char*)d_buf + OFF_PROXY, 0xBB, LEN);
    } else {
        // Rank 1: sentinel-fill so we can detect missing writes.
        (void)gpuMemset(d_buf, 0x55, BUF_BYTES);
    }
    (void)gpuDeviceSynchronize();

    auto bh = rt.register_buffer(d_buf, BUF_BYTES, /*is_device=*/true);
    rt.exchange();
    rt.barrier();

    int peer = 1 - rank;
    if (rank == 0) {
        // Sample call. LTO inserts trace function here that:
        //   - For site 0 (DWQ): emits gicc_runtime_dwq_enqueue once
        //   - For site 1 (proxy): emits nothing (device pushes via ring)
        gicc::launch<mixed_kernel>(rt, dim3(1), dim3(1),
                                   peer, bh.index, bh.index, LEN);
        if (gpuDeviceSynchronize() != GPU_SUCCESS) {
            fprintf(stderr, "rank 0: sync failed\n"); return 3;
        }
        rt.reset();  // drains DWQ completion + proxy ring
    }
    rt.barrier();

    // Verify on rank 1.
    int result = 0;
    if (rank == 1) {
        uint8_t* h_check = (uint8_t*)malloc(BUF_BYTES);
        (void)gpuMemcpy(h_check, d_buf, BUF_BYTES, gpuMemcpyDeviceToHost);

        // Region 0 [0..LEN)  must be all 0xAA  — DWQ leg landed
        // Gap   [LEN..OFF_PROXY) must remain 0x55 — sentinel intact
        // Region 1 [OFF_PROXY..OFF_PROXY+LEN) must be all 0xBB — proxy leg landed
        int dwq_ok = 1, proxy_ok = 1, gap_ok = 1;
        for (size_t i = 0; i < LEN; ++i) {
            if (h_check[i] != 0xAA) { dwq_ok = 0; break; }
        }
        for (size_t i = LEN; i < OFF_PROXY; ++i) {
            if (h_check[i] != 0x55) { gap_ok = 0; break; }
        }
        for (size_t i = OFF_PROXY; i < OFF_PROXY + LEN; ++i) {
            if (h_check[i] != 0xBB) { proxy_ok = 0; break; }
        }
        printf("[verify rank=1] DWQ_leg=%s  GAP=%s  PROXY_leg=%s\n",
               dwq_ok ? "OK (0xAA)" : "FAIL",
               gap_ok ? "OK (0x55)" : "FAIL",
               proxy_ok ? "OK (0xBB)" : "FAIL");
        // Diagnostic dump if proxy leg failed — show byte values across
        // all 3 regions so we can tell if proxy fired to wrong offset,
        // partial, or not at all.
        if (!proxy_ok) {
            printf("[diag rank=1] byte snapshot:\n");
            printf("  region[0]      first 8 bytes: ");
            for (int i = 0; i < 8; i++) printf("%02x ", h_check[i]); printf("\n");
            printf("  region[OFF/2]  first 8 bytes: ");
            for (int i = 0; i < 8; i++) printf("%02x ", h_check[OFF_PROXY/2 + i]); printf("\n");
            printf("  region[OFF]    first 8 bytes: ");
            for (int i = 0; i < 8; i++) printf("%02x ", h_check[OFF_PROXY + i]); printf("\n");
            printf("  region[OFF+1K] first 8 bytes: ");
            for (int i = 0; i < 8; i++) printf("%02x ", h_check[OFF_PROXY + 1024 + i]); printf("\n");
            printf("  region[OFF+LEN-8] last 8: ");
            for (int i = 0; i < 8; i++) printf("%02x ", h_check[OFF_PROXY + LEN - 8 + i]); printf("\n");
        }
        if (!dwq_ok || !proxy_ok || !gap_ok) result = 99;
        free(h_check);
    }

    // Audit DWQ enqueue counter — should equal 1 (one site routed to DWQ,
    // one launch). Proves the LTO pass actually routed site 0 to DWQ.
    if (rank == 0) {
        uint64_t mono = gicc_runtime_trigger_val(&rt);
        printf("[audit rank=0] mono_total_ops=%lu  expected=1  match=%s\n",
               (unsigned long)mono, (mono == 1 ? "YES" : "NO"));
        if (mono != 1) result = 98;
    }

    rt.barrier();
    (void)gpuFree(d_buf);
    return result;
}
