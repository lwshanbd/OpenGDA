/*
 * get_stress.cpp - CPU Proxy GET PCIe-ordering stress test.
 *
 * Hypothesis we are trying to falsify (or confirm):
 *
 *   After fi_read's CQE fires on the proxy side, the GET response's
 *   posted-write TLPs to local GPU HBM may still be queued in PCIe
 *   buffers (root complex / switch / GPU BAR). The proxy thread advances
 *   the ring tail; the kernel's quiet() spin exits and __threadfence_system()
 *   runs. But threadfence_system is a GPU-internal ordering primitive — it
 *   does not push pending PCIe TLPs into HBM. So the very next read of
 *   the landing region from the SM may see stale bytes.
 *
 *   On Tioga (MI250, gfx90a) `gpuDirectRDMAWritesOrdering = 0` (= NONE),
 *   which per the NVIDIA terminology means "external GDR writes are not
 *   automatically ordered with respect to GPU reads — explicit flush
 *   required". ROCm 6.4 does not expose `hipFlushGPUDirectRDMAWrites`,
 *   so we have no driver-side fence available. NVSHMEM's IBGDA-style fix
 *   would be an extra same-QP RDMA READ to drain the PCIe pipe via the
 *   ordering rule. We haven't added that yet.
 *
 * Stress design:
 *   - Peer publishes N_ITERS distinct source slots, each WIN bytes,
 *     pattern = (peer_rank*31 + k + offset) byte-wise.
 *   - Kernel back-to-back GETs slot k into the SAME landing region,
 *     then immediately reads the *tail* TAIL_CHECK bytes (the most
 *     likely to still be in-flight when CQE fires) and counts byte
 *     mismatches against the expected pattern.
 *   - If we're racing, a stale byte would look like slot k-1's value
 *     at that offset, which (with this pattern) differs from slot k's
 *     value by exactly 31 — so it WILL register as a mismatch.
 *   - err_counts[k] is published per-iter so we can see if errors
 *     cluster early/late/randomly.
 *
 * Tunables (compile-time):
 *   N_ITERS      number of back-to-back GET iterations (default 200)
 *   WIN_BYTES    bytes per GET (default 256*1024 — multi-TLP)
 *   TAIL_CHECK   how many tail bytes to verify per iter (default 256)
 *
 * Build / run (Tioga):
 *   GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
 *       srun -p pci -t 2 -N 1 -n 2 ./examples/proxy/get_stress
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

#ifndef N_ITERS
#define N_ITERS 1000
#endif
#ifndef WIN_BYTES
#define WIN_BYTES (1u * 1024u * 1024u)
#endif
#ifndef TAIL_CHECK
#define TAIL_CHECK WIN_BYTES   /* check every byte for max coverage */
#endif

__global__ void get_stress_kernel(gicc::DeviceCtx* ctx,
                                  int peer, int peer_buf, int my_buf,
                                  size_t land_off,
                                  uint8_t* landing,
                                  uint32_t* err_counts /* [N_ITERS] */)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int k = 0; k < N_ITERS; ++k) {
        gicc::get(ctx, peer, peer_buf, (size_t)k * WIN_BYTES,
                        my_buf,  land_off,  (size_t)WIN_BYTES);
        gicc::quiet(ctx);

        // IMMEDIATELY check the TAIL bytes (last to be PCIe-written,
        // most likely to still be in-flight after CQE). Pattern is
        // byte-wise: landing[i] = (peer*31 + k + i) & 0xFF.
        const uint8_t base = (uint8_t)((peer * 31 + k) & 0xFF);
        uint32_t local_err = 0;
        for (uint32_t off = (uint32_t)WIN_BYTES - TAIL_CHECK;
             off < (uint32_t)WIN_BYTES; ++off) {
            uint8_t got  = landing[off];
            uint8_t want = (uint8_t)((base + (uint8_t)(off & 0xFF)) & 0xFF);
            if (got != want) ++local_err;
        }
        err_counts[k] = local_err;
    }
}

int main(int /*argc*/, char** /*argv*/) {
    gicc::Runtime rt;
    const int rank   = rt.rank();
    const int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) {
            fprintf(stderr, "get_stress: need exactly 2 ranks (got %d)\n", nranks);
        }
        return 1;
    }

    const size_t SOURCE_BYTES = (size_t)N_ITERS * WIN_BYTES;
    const size_t LAND_OFF     = SOURCE_BYTES;
    const size_t ERR_OFF      = LAND_OFF + WIN_BYTES;
    const size_t ERR_BYTES    = (size_t)N_ITERS * sizeof(uint32_t);
    const size_t BUF_BYTES    = ERR_OFF + ERR_BYTES;

    void* d_buf = nullptr;
    if (gpuMalloc(&d_buf, BUF_BYTES) != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: gpuMalloc(%zu) failed\n", rank, BUF_BYTES);
        return 2;
    }
    (void)gpuMemset(d_buf, 0, BUF_BYTES);

    // Init source: N slots, each WIN_BYTES, byte k of slot s has value
    // (rank * 31 + s + k) & 0xFF. Distinct slots have patterns that
    // differ by exactly 31 at every byte position — so a stale read of
    // slot k-1 instead of k will produce WIN_BYTES mismatches per iter.
    {
        auto* h_src = static_cast<uint8_t*>(std::malloc(SOURCE_BYTES));
        if (!h_src) { fprintf(stderr, "malloc fail\n"); return 2; }
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
    gicc::DeviceCtx* d_ctx = rt.prepare();

    uint8_t*  landing    = static_cast<uint8_t*>(d_buf) + LAND_OFF;
    uint32_t* err_counts = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(d_buf) + ERR_OFF);

    gpuLaunchKernel(get_stress_kernel, dim3(1), dim3(1), 0, 0,
                    d_ctx, peer,
                    /*peer_buf=*/bh.index, /*my_buf=*/bh.index,
                    LAND_OFF, landing, err_counts);
    if (gpuDeviceSynchronize() != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: kernel sync failed\n", rank);
        return 3;
    }
    rt.barrier();

    auto* h_err = static_cast<uint32_t*>(std::malloc(ERR_BYTES));
    (void)gpuMemcpy(h_err, err_counts, ERR_BYTES, gpuMemcpyDeviceToHost);

    uint64_t total = 0;
    int bad_iters = 0;
    int first_bad = -1, last_bad = -1;
    int printed = 0;
    for (int k = 0; k < N_ITERS; ++k) {
        if (h_err[k]) {
            ++bad_iters;
            if (first_bad < 0) first_bad = k;
            last_bad = k;
            total += h_err[k];
            if (printed < 8) {
                fprintf(stderr,
                    "  rank %d iter %4d: %u/%u tail-byte mismatches\n",
                    rank, k, h_err[k], TAIL_CHECK);
                ++printed;
            }
        }
    }
    printf("rank %d get_stress: %s "
           "(bad_iters=%d/%d, total_mismatches=%lu, range=[%d,%d], "
           "win=%uB, tail_check=%uB, iters=%d)\n",
           rank, total == 0 ? "PASS" : "FAIL",
           bad_iters, N_ITERS, (unsigned long)total,
           first_bad, last_bad,
           WIN_BYTES, TAIL_CHECK, N_ITERS);
    std::free(h_err);

    rt.reset();
    rt.barrier();
    (void)gpuFree(d_buf);
    return total == 0 ? 0 : 4;
}
