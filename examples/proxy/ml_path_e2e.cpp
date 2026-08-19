/*
 * ml_path_e2e.cpp -- end-to-end performance test for a learned per-site
 * lowering, not just an offline regret table.
 *
 * The source contains 64 distinct, host-knowable 4 KiB puts released by one
 * completion point.  The no-hint compiler default lowers cross-node peers to
 * DWQ trigger staging.  ml_path_decider.py is trained on ctx_bench data with
 * the entire 4 KiB message size held out; given features.json from this file,
 * it may instead choose the CPU-proxy lowering.  GICCDeviceLowering then
 * spreads the 64 preserved puts over the eight blocks/rings in this launch.
 *
 * Both binaries execute this exact source.  The benchmark times the full
 * launch + completion path and verifies every received byte.
 */

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/launch.hpp"

static constexpr size_t kMsgBytes = 4096;
static constexpr int    kOps      = 64;
static constexpr size_t kRegion   = kMsgBytes * kOps;
static constexpr size_t kBufBytes = 1u << 20;

#define ONE_PUT(I)                                                           \
    gicc::put(ctx, peer, buf, (size_t)(I) * kMsgBytes,                       \
              buf, (size_t)(I) * kMsgBytes, kMsgBytes)
#define PUT8(B)                                                              \
    ONE_PUT((B) + 0); ONE_PUT((B) + 1); ONE_PUT((B) + 2); ONE_PUT((B) + 3); \
    ONE_PUT((B) + 4); ONE_PUT((B) + 5); ONE_PUT((B) + 6); ONE_PUT((B) + 7)

__global__ void k_put64(gicc::DeviceCtx* ctx, int peer, int buf) {
    PUT8(0);  PUT8(8);  PUT8(16); PUT8(24);
    PUT8(32); PUT8(40); PUT8(48); PUT8(56);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

#undef PUT8
#undef ONE_PUT

__global__ void k_fill(uint8_t* p, size_t n, bool source_pattern) {
    for (size_t i = threadIdx.x + (size_t)blockIdx.x * blockDim.x; i < n;
         i += (size_t)blockDim.x * gridDim.x)
        p[i] = source_pattern
                   ? static_cast<uint8_t>(1 + (i / kMsgBytes) % kOps)
                   : uint8_t{0};
}

static double now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::micro>(
               clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int warmup = 10, runs = 100;
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "--warmup=", 9)) warmup = atoi(argv[i] + 9);
        if (!strncmp(argv[i], "--runs=", 7))   runs   = atoi(argv[i] + 7);
    }

    gicc::Runtime rt;
    rt.enable_host_wait_mode();
    rt.enable_mixed_dispatch();
    const int rank = rt.rank();
    if (rt.size() != 2) {
        if (rank == 0) fprintf(stderr, "ml_path_e2e requires two ranks\n");
        return 2;
    }
    const int peer = 1 - rank;

    void* d_buf = nullptr;
    if (hipMalloc(&d_buf, kBufBytes) != hipSuccess) return 3;
    hipLaunchKernelGGL(k_fill, dim3(64), dim3(256), 0, 0,
                       static_cast<uint8_t*>(d_buf), kBufBytes,
                       rank == 0);
    (void)hipDeviceSynchronize();
    auto bh = rt.register_buffer(d_buf, kBufBytes, true);
    rt.exchange();
    rt.barrier();

    auto phase = [&]() {
        if (rank == 0)
            gicc::launch<k_put64>(rt, dim3(8), dim3(1), peer, bh.index);
        (void)hipDeviceSynchronize();
        rt.reset();
    };

    for (int i = 0; i < warmup; ++i) phase();
    // Warmup necessarily writes the receive buffer.  Clear it before the
    // measured phases so the final check proves that the timed path delivered
    // every static site; stale warmup bytes cannot make a dropped site pass.
    if (rank == 1) {
        hipLaunchKernelGGL(k_fill, dim3(64), dim3(256), 0, 0,
                           static_cast<uint8_t*>(d_buf), kRegion, false);
        (void)hipDeviceSynchronize();
    }
    rt.barrier();
    const uint64_t staged0 = rt.staged_ops();
    const uint64_t pushed0 = rt.proxy_pushes();
    std::vector<double> samples;
    samples.reserve(runs);
    const double t0 = now_us();
    for (int i = 0; i < runs; ++i) {
        const double p0 = now_us();
        phase();
        samples.push_back(now_us() - p0);
    }
    const double elapsed = now_us() - t0;
    rt.barrier();

    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double q) {
        if (samples.empty()) return 0.0;
        const double pos = q * static_cast<double>(samples.size() - 1);
        const size_t lo = static_cast<size_t>(pos);
        const size_t hi = std::min(lo + 1, samples.size() - 1);
        return samples[lo] + (samples[hi] - samples[lo]) * (pos - lo);
    };

    int ok = 1;
    uint64_t hash = 1469598103934665603ull;
    if (rank == 1) {
        std::vector<uint8_t> got(kRegion);
        (void)hipMemcpy(got.data(), d_buf, got.size(), hipMemcpyDeviceToHost);
        for (size_t i = 0; i < got.size(); ++i) {
            const uint8_t expected =
                static_cast<uint8_t>(1 + (i / kMsgBytes) % kOps);
            hash ^= got[i];
            hash *= 1099511628211ull;
            if (got[i] != expected) ok = 0;
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 1, MPI_COMM_WORLD);

    if (rank == 0) {
        const uint64_t staged = rt.staged_ops() - staged0;
        const uint64_t pushed = rt.proxy_pushes() - pushed0;
        printf("ML_PATH_RESULT runs=%d total_us=%.3f mean_us=%.3f "
               "median_us=%.3f p25_us=%.3f p75_us=%.3f "
               "staged=%llu pushed=%llu data=%s\n",
               runs, elapsed, elapsed / runs, percentile(0.50),
               percentile(0.25), percentile(0.75),
               (unsigned long long)staged, (unsigned long long)pushed,
               ok ? "OK" : "WRONG");
    }
    if (rank == 1)
        printf("ML_PATH_CHECKSUM rank=1 fnv64=%016llx data=%s\n",
               (unsigned long long)hash, ok ? "OK" : "WRONG");

    rt.barrier();
    (void)hipFree(d_buf);
    return ok ? 0 : 4;
}
