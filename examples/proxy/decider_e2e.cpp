/*
 * decider_e2e.cpp - end-to-end check that the COMPILER decides the
 * dispatch, using facts it derived itself.
 *
 * One kernel, two call sites that differ only in program context a
 * compiler can see and a communication runtime cannot:
 *
 *   site FAR   a loop with a compile-time trip count, whose result is
 *              used only after a long stretch of arithmetic. High trip
 *              count + large issue-to-first-use distance is the regime
 *              where the trigger path wins, because once the compute
 *              hides the wire time only the per-op issue cost is left
 *              and host staging is ~2.8x cheaper than a ring push.
 *
 *   site DYN   one put whose offset is LOADED FROM DEVICE MEMORY. The
 *              host cannot reconstruct that descriptor before the
 *              launch, so HK analysis marks it hk_capable=false and the
 *              trigger path is not merely slower, it is illegal. The
 *              only legal dispatch is the CPU proxy.
 *
 * Both sites move the same number of bytes to the same peer in the same
 * kernel. Nothing a runtime sees at the moment of the call distinguishes
 * them; trip count, distance-to-first-use, and host-knowability are all
 * static properties.
 *
 * The routing is then PROVEN rather than assumed: staged_ops() counts
 * descriptors that went through host staging (trigger) and
 * proxy_pushes() counts commands that went through the ring (proxy).
 * Byte-level verification cannot tell the paths apart -- both deliver
 * the same bytes -- so the counters are the actual test.
 *
 * Built by examples/proxy/build_decider_e2e.sh, which runs the real
 * three-pass flow: discover -> feature-extract -> gicc_decider.py ->
 * lower. No hand-written hint.json anywhere in the loop.
 */

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/launch.hpp"

// Ops issued by the FAR site. A literal so ScalarEvolution can prove the
// trip count; that is the whole point of the site.
#ifndef FAR_OPS
#define FAR_OPS 64
#endif
static constexpr size_t kMsgBytes = 4096;
static constexpr size_t kFarBase  = 0;
static constexpr size_t kDynBase  = 8u << 20;
static constexpr size_t kBufBytes = 32u << 20;

__global__ void two_site_kernel(gicc::DeviceCtx* ctx,
                                int peer, int buf,
                                const size_t* __restrict__ dyn_off,
                                float* __restrict__ scratch,
                                int work)
{
    // ---- site FAR: constant trip count, result needed much later ----
    for (int i = 0; i < FAR_OPS; ++i) {
        size_t off = kFarBase + (size_t)i * kMsgBytes;
        gicc::put(ctx, peer, buf, off, buf, off, kMsgBytes);
    }

    // ---- the distance: arithmetic between the issue and the wait ----
    // Deliberately branch-free and floating point so it lands in the
    // pass's arithmetic count and shows up as flops_to_first_use.
    float acc = 1.0f;
    for (int i = 0; i < work; ++i) {
        acc = acc * 1.000001f + 0.5f;
        acc = acc * 0.999999f - 0.25f;
        acc = acc + acc * 0.5f;
        acc = acc * 1.5f - acc * 0.25f;
    }
    if (scratch && threadIdx.x == 0) scratch[0] = acc;

    // ---- site DYN: descriptor comes out of device memory ----
    // The host cannot know this offset before the launch, so the site is
    // not host-knowable and only the proxy path is legal for it.
    if (dyn_off) {
        size_t off = dyn_off[0];
        gicc::put(ctx, peer, buf, off, buf, off, kMsgBytes);
    }

    if (threadIdx.x == 0) {
        gicc::flush(ctx);
        gicc::quiet(ctx);
    }
}

__global__ void k_make_offset(size_t* out, const float* seed) {
    // Value-dependent so it cannot be constant-folded back to the host.
    if (threadIdx.x == 0)
        out[0] = kDynBase + (size_t)((seed[0] > 0.0f) ? 0 : 0) * kMsgBytes;
}

__global__ void k_fill(uint8_t* p, size_t n, uint8_t v) {
    for (size_t i = threadIdx.x + (size_t)blockIdx.x * blockDim.x; i < n;
         i += (size_t)blockDim.x * gridDim.x)
        p[i] = v;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int work = 2000;
    for (int i = 1; i < argc; ++i)
        if (!strncmp(argv[i], "--work=", 7)) work = atoi(argv[i] + 7);

    gicc::Runtime rt;
    rt.enable_host_wait_mode();
    rt.enable_mixed_dispatch();
    const int rank = rt.rank(), nranks = rt.size();
    if (nranks != 2) {
        if (!rank) fprintf(stderr, "decider_e2e: need exactly 2 ranks\n");
        return 1;
    }
    const int peer = 1 - rank;

    void* d_buf = nullptr;
    if (hipMalloc(&d_buf, kBufBytes) != hipSuccess) return 3;
    hipLaunchKernelGGL(k_fill, dim3(64), dim3(256), 0, 0,
                       (uint8_t*)d_buf, kBufBytes, (uint8_t)(rank ? 0x00 : 0xC7));
    (void)hipDeviceSynchronize();
    auto bh = rt.register_buffer(d_buf, kBufBytes, /*is_device=*/true);
    rt.exchange();

    size_t* d_off = nullptr; float* d_seed = nullptr; float* d_scratch = nullptr;
    (void)hipMalloc((void**)&d_off, sizeof(size_t));
    (void)hipMalloc((void**)&d_seed, sizeof(float));
    (void)hipMalloc((void**)&d_scratch, sizeof(float));
    (void)hipMemset(d_seed, 1, sizeof(float));
    hipLaunchKernelGGL(k_make_offset, dim3(1), dim3(1), 0, 0, d_off, d_seed);
    (void)hipDeviceSynchronize();
    rt.barrier();

    const uint64_t staged0 = rt.staged_ops();
    const uint64_t pushed0 = rt.proxy_pushes();

    if (rank == 0) {
        gicc::launch<two_site_kernel>(rt, dim3(1), dim3(1),
                                      peer, bh.index, (const size_t*)d_off,
                                      d_scratch, work);
    }
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();

    const uint64_t staged = rt.staged_ops()   - staged0;
    const uint64_t pushed = rt.proxy_pushes() - pushed0;

    int ok = 1;
    if (rank == 1) {
        // Both regions must have arrived, whichever path carried them.
        std::vector<uint8_t> h(kMsgBytes);
        for (auto probe : {kFarBase, kDynBase}) {
            (void)hipMemcpy(h.data(), (uint8_t*)d_buf + probe, kMsgBytes,
                            hipMemcpyDeviceToHost);
            for (size_t i = 0; i < kMsgBytes; i += 256)
                if (h[i] != 0xC7) {
                    printf("[data] FAIL at offset %zu (+%zu): 0x%02x\n",
                           probe, i, h[i]);
                    ok = 0; break;
                }
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 1, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== decider_e2e: who carried which call site ===\n");
        printf("  host-staged descriptors (trigger path): %llu\n",
               (unsigned long long)staged);
        printf("  proxy ring pushes       (proxy path)  : %llu\n",
               (unsigned long long)pushed);
        // The proxy count is the DYN put plus one command for the
        // kernel's gicc::quiet, which is also a ring command.
        printf("  expected if the compiler routed FAR->trigger, DYN->proxy:"
               " %d staged, 2 pushed (1 put + 1 quiet)\n", FAR_OPS);
        const bool routed = (staged == (uint64_t)FAR_OPS) && (pushed == 2);
        printf("  routing: %s\n", routed ? "AS DECIDED" : "NOT AS DECIDED");
        printf("  data   : %s\n", ok ? "all regions delivered" : "MISSING DATA");
        printf("  RESULT : %s\n", (routed && ok) ? "PASS" : "FAIL");
    }

    rt.barrier();
    (void)hipFree(d_buf); (void)hipFree(d_off);
    (void)hipFree(d_seed); (void)hipFree(d_scratch);
    return 0;
}
