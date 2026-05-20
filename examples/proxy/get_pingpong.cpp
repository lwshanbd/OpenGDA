/*
 * get_pingpong.cpp - CPU Proxy GET smoke test.
 *
 * Sibling of quiet_pingpong.cpp, but exercises gicc::get_no_db instead of
 * put_no_db. Both ranks publish a per-rank pattern in the LOW half of
 * their buffer, then each kernel does:
 *
 *   get_no_db(peer, bh.index, src_off=0, bh.index, dst_off=LAND, WIN);
 *   quiet();
 *
 * The kernel-side quiet() is the read-ordering fence the user asked
 * about: under CPU Proxy mode the NIC DMAs the GET reply into local GPU
 * HBM; the proxy advances the ring tail only after the libfabric CQE
 * fires (which is provider-guaranteed to mean the bytes have landed in
 * local memory). The kernel's quiet() spins on the tail and ends with
 * __threadfence_system(), which invalidates the GPU's L2 for the
 * destination range. So a kernel-internal read of the landed bytes
 * after quiet() is correct without any host-side ordering.
 *
 * Cross-rank ordering still needs an MPI barrier (the kernel's quiet()
 * fences only its own proxy), which the test enforces between the
 * source-publish step and the GETs.
 *
 * Scope honesty: this matches quiet_pingpong's pattern — raw
 * gpuLaunchKernel, no gicc::launch<>, no LTO trace. It validates the
 * device-side ring push + host-side fi_read + ack path under
 * GICC_CPU_PROXY only.
 *
 * Build / run:
 *   GICC_PROXY_ENABLED=1 srun -p pci -t 2 -N 1 -n 2 \
 *       ./examples/proxy/get_pingpong
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

__global__ void get_then_quiet(gicc::DeviceCtx* ctx,
                               int peer, int peer_buf_idx, int my_buf_idx,
                               size_t landing_offset, size_t bytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Pull bytes from peer's [0, bytes) into our [landing_offset, ...).
        gicc::get_no_db(ctx,
                        peer,
                        peer_buf_idx, /*src_offset=*/0,
                        my_buf_idx,   landing_offset,
                        bytes);
        // Kernel-side completion + read-ordering fence: spins until the
        // proxy acks every in-flight slot (including this GET), then
        // __threadfence_system() invalidates the GPU's L2 for the
        // landing range.
        gicc::quiet(ctx);
    }
}

int main(int /*argc*/, char** /*argv*/) {
    gicc::Runtime rt;
    const int rank   = rt.rank();
    const int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) {
            fprintf(stderr,
                "get_pingpong: need exactly 2 ranks (got %d)\n",
                nranks);
        }
        return 1;
    }

    constexpr size_t WIN       = 1024;
    constexpr size_t LAND_OFF  = 4096;
    constexpr size_t BUF_BYTES = LAND_OFF + WIN;   // 5120

    void* d_buf = nullptr;
    if (gpuMalloc(&d_buf, BUF_BYTES) != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: gpuMalloc failed\n", rank);
        return 2;
    }
    (void)gpuMemset(d_buf, 0, BUF_BYTES);

    // SOURCE region [0, WIN): our per-rank pattern. The peer will GET
    // these bytes from us. The landing region [LAND_OFF, LAND_OFF+WIN)
    // starts zeroed; after the GET it must hold the peer's pattern.
    auto* h_pattern = static_cast<uint8_t*>(std::malloc(WIN));
    for (size_t i = 0; i < WIN; ++i) {
        h_pattern[i] = (uint8_t)((rank * 31 + i) & 0xFF);
    }
    (void)gpuMemcpy(d_buf, h_pattern, WIN, gpuMemcpyHostToDevice);
    std::free(h_pattern);

    auto bh = rt.register_buffer(d_buf, BUF_BYTES, /*is_device=*/true);
    rt.exchange();

    // Make sure both ranks have their source patterns on the device and
    // address books exchanged before the peer issues a GET. Without this
    // barrier the GET could pull pre-init zeros.
    rt.barrier();

    const int peer = 1 - rank;
    gicc::DeviceCtx* d_ctx = rt.prepare();
    gpuLaunchKernel(get_then_quiet, dim3(1), dim3(1), 0, 0,
                    d_ctx, peer,
                    /*peer_buf=*/bh.index, /*my_buf=*/bh.index,
                    LAND_OFF, WIN);
    if (gpuDeviceSynchronize() != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: kernel sync failed\n", rank);
        return 3;
    }
    // Deliberately do NOT call rt.reset() before the readback — the
    // device-side quiet() is the *whole point* of this test. The MPI
    // barrier below just keeps both ranks from teardown-racing each
    // other; cross-rank ordering of the readback isn't required since
    // each rank reads its OWN landing region.
    rt.barrier();

    auto* h_check = static_cast<uint8_t*>(std::malloc(WIN));
    (void)gpuMemcpy(h_check,
                    static_cast<uint8_t*>(d_buf) + LAND_OFF,
                    WIN, gpuMemcpyDeviceToHost);

    int errors = 0;
    for (size_t i = 0; i < WIN; ++i) {
        const uint8_t want = (uint8_t)((peer * 31 + i) & 0xFF);
        if (h_check[i] != want) {
            if (errors < 4) {
                fprintf(stderr,
                    "  rank %d: mismatch @%zu got=0x%02x want=0x%02x\n",
                    rank, i, h_check[i], (unsigned)want);
            }
            ++errors;
        }
    }
    printf("rank %d get_pingpong: %s (%d errors over %zu bytes)\n",
           rank, errors == 0 ? "PASS" : "FAIL", errors, WIN);
    std::free(h_check);

    rt.reset();   // drain anything still queued (defensive).
    rt.barrier();
    (void)gpuFree(d_buf);
    return errors == 0 ? 0 : 4;
}
