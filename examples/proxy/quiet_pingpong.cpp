/*
 * quiet_pingpong.cpp - L5 device-side quiet() SMOKE TEST.
 *
 * SCOPE LIMITATION (read this before claiming this validates anything):
 * Like the L2 / L4 proxy examples, this uses raw gpuLaunchKernel rather
 * than gicc::launch<>. That deliberately bypasses the host-side LTO
 * pipeline (host-discovery, feature-extraction, trace-synthesis,
 * dispatch-lowering). What we actually exercise is the device-side body
 * of gicc::quiet in src/gicc/platform/ofi/ofi_device.cuh under the
 * GICC_CPU_PROXY define: the kernel pushes a QUIET TransferCmd, then
 * spins on D2HRing::device_tail_volatile() until the proxy thread acks
 * the slot via ProxyThread::handle_quiet (Task 6).
 *
 * What this example proves
 * ------------------------
 * The kernel layout is:
 *
 *   put_no_db(peer's slot) → quiet() → return
 *
 * If quiet() correctly waits for the proxy to drain the in-flight
 * WRITEs, then by the time gpuDeviceSynchronize() returns on the host:
 *   - rank 0's WRITE to rank 1's buffer has been submitted *and* has
 *     produced a libfabric CQ completion (proxy ProxyThread::handle_quiet
 *     waits on in_flight_), and
 *   - rank 1 symmetrically.
 *
 * After an MPI barrier (the global ordering point — each rank's quiet
 * only fences ITS OWN proxy), each rank reads back its OWN buffer at
 * the offset the peer wrote into and checks for the peer's pattern.
 * If quiet() returned early (i.e. the spin condition was wrong), the
 * read after the barrier would race with the still-in-flight proxy
 * submission and observe stale zeros.
 *
 * What this example does NOT prove
 * --------------------------------
 *   - End-to-end LTO routing (no gicc::launch<>, no host trace, no
 *     dispatch-lowering). The proxy_aware-hint-aware preserve from
 *     Task 8 is not exercised here.
 *   - That __builtin_amdgcn_s_sleep / __nanosleep produce a useful
 *     backoff cadence under contention; this is a 2-rank test, no
 *     contention.
 *
 * Promoting this to a real end-to-end LTO validator requires switching
 * to gicc::launch<>(rt, ...) plus a proxy_aware hint.json so device
 * lowering preserves both put_no_db and quiet bodies.
 *
 * Build / run flow (Delta or Tioga):
 *   1. Configure with -DGICC_ENABLE_CPU_PROXY=ON.
 *   2. At runtime: GICC_PROXY_ENABLED=1 srun -n 2 ./quiet_pingpong
 *
 *   GICC_PROXY_ENABLED=1 srun -p pci -t 2 -N 1 -n 2 \
 *       ./examples/proxy/quiet_pingpong
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

__global__ void put_then_quiet(gicc::DeviceCtx* ctx,
                               int peer, int dst_buf_idx, int src_buf_idx,
                               size_t dst_offset, size_t bytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Write our local source half [0, bytes) into the peer's
        // [dst_offset, dst_offset + bytes) slot.
        gicc::put_no_db(ctx,
                        peer,
                        dst_buf_idx, dst_offset,
                        src_buf_idx, /*src_offset=*/0,
                        bytes);
        // In-kernel completion fence: pushes QUIET cmd; spins until the
        // proxy thread's handle_quiet acks past our slot.
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
                "quiet_pingpong: need exactly 2 ranks (got %d)\n",
                nranks);
        }
        return 1;
    }

    constexpr size_t WIN       = 1024;
    constexpr size_t DST_OFF   = 4096;
    constexpr size_t BUF_BYTES = DST_OFF + WIN;   // 5120

    void* d_buf = nullptr;
    if (gpuMalloc(&d_buf, BUF_BYTES) != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: gpuMalloc failed\n", rank);
        return 2;
    }
    (void)gpuMemset(d_buf, 0, BUF_BYTES);

    // Source half [0, WIN): per-rank pattern. The peer will write its
    // pattern into our [DST_OFF, DST_OFF + WIN) slot.
    auto* h_pattern = static_cast<uint8_t*>(std::malloc(WIN));
    for (size_t i = 0; i < WIN; ++i) {
        h_pattern[i] = (uint8_t)((rank * 31 + i) & 0xFF);
    }
    (void)gpuMemcpy(d_buf, h_pattern, WIN, gpuMemcpyHostToDevice);
    std::free(h_pattern);

    auto bh = rt.register_buffer(d_buf, BUF_BYTES, /*is_device=*/true);
    rt.exchange();

    // Make sure both ranks have published their address books and the
    // initial gpuMemset is visible before either side fires its put.
    rt.barrier();

    const int peer = 1 - rank;
    gicc::DeviceCtx* d_ctx = rt.prepare();
    gpuLaunchKernel(put_then_quiet, dim3(1), dim3(1), 0, 0,
                    d_ctx, peer,
                    /*dst_buf=*/bh.index, /*src_buf=*/bh.index,
                    DST_OFF, WIN);
    if (gpuDeviceSynchronize() != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: kernel sync failed\n", rank);
        return 3;
    }
    // NOTE: deliberately DO NOT call rt.reset() before the readback —
    // we want to prove the device-side quiet did the draining, not the
    // host-side reset. The barrier below provides the cross-rank
    // ordering each side's quiet() cannot guarantee on its own.

    rt.barrier();

    auto* h_check = static_cast<uint8_t*>(std::malloc(WIN));
    (void)gpuMemcpy(h_check,
                    static_cast<uint8_t*>(d_buf) + DST_OFF,
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
    printf("rank %d quiet_pingpong: %s (%d errors over %zu bytes)\n",
           rank, errors == 0 ? "PASS" : "FAIL", errors, WIN);
    std::free(h_check);

    rt.reset();   // drain anything still queued (defensive).
    rt.barrier();
    (void)gpuFree(d_buf);
    return errors == 0 ? 0 : 4;
}
