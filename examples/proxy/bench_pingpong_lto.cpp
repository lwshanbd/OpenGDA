/*
 * bench_pingpong_lto.cpp — LTO-pipeline-compiled ping-pong bench.
 *
 * Differs from bench_pingpong --mode=dwq in ONE meaningful way:
 *   - This file contains ZERO explicit rt.put_no_db() calls.
 *   - The kernel has the put_no_db loop inline; the LTO pass
 *     (GICCHostDiscovery + GICCTraceSynthesis + GICCDispatchLowering)
 *     synthesizes the equivalent host trace function and inserts a
 *     call to it before each gicc::launch<> site.
 *
 * If the compiler-generated trace path costs the same as the
 * hand-written rt.put_no_db loop, the median wall time per msg should
 * match bench_pingpong --mode=dwq exactly. Any delta isolates the
 * LTO-trace overhead from the underlying DWQ enqueue cost.
 *
 * Build:  build_bench_lto.sh  (manual hipcc + -fpass-plugin)
 * Run:    FI_MR_CACHE_MAX_COUNT=0 srun -p pci -N 2 -n 2 \
 *           --ntasks-per-node=1 --cpus-per-task=64 -t 2 \
 *           ./bench_pingpong_lto --batch=32
 */

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/launch.hpp"
#include "gicc/platform/ofi/runtime_helpers.h"  // gicc_runtime_trigger_val

static const size_t kSizes[] = {
    1, 2, 4, 8, 64, 256, 1024, 4*1024, 16*1024, 64*1024, 256*1024,
    512*1024, 1024*1024, 2*1024*1024, 4*1024*1024, 16*1024*1024,
};
static constexpr int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);

static constexpr int NUM_OUTER  = 21;
static constexpr int NUM_WARMUP = 10;
static constexpr size_t kBufBytes = 32 * 1024 * 1024;

// The ENTIRE communication pattern is encoded here. The LTO pass walks
// this function's IR, finds the put_no_db inside the HK-bounded loop
// (bound = kernel formal `n`), and synthesizes a host-side trace
// function with the same loop structure that calls
// gicc_runtime_dwq_enqueue once per iteration. The trace is then
// inserted before each gicc::launch<dwq_loop_kernel>(...) site.
__global__ void dwq_loop_kernel(gicc::DeviceCtx* ctx, int peer,
                                int buf_idx, size_t bytes, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n; ++i) {
            gicc::put(ctx, peer, buf_idx, /*dst_off=*/0,
                                         buf_idx, /*src_off=*/0, bytes);
        }
        gicc::flush(ctx);
        gicc::quiet(ctx);
    }
}

static const char* fmt_size(size_t s, char* buf) {
    if (s < 1024)            snprintf(buf, 32, "%zuB",  s);
    else if (s < 1024*1024)  snprintf(buf, 32, "%zuKB", s/1024);
    else                     snprintf(buf, 32, "%zuMB", s/(1024*1024));
    return buf;
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    int batch = 32;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--batch=", 0) == 0) {
            batch = std::atoi(a.substr(8).c_str());
            if (batch < 1) batch = 1;
        }
    }

    gicc::Runtime rt;
    rt.enable_host_wait_mode();         // canonical DWQ fast path
    int rank   = rt.rank();
    int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "need 2 ranks (got %d)\n", nranks);
        return 1;
    }
    int peer = 1 - rank;

    void* d_buf = nullptr;
    if (gpuMalloc(&d_buf, kBufBytes) != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: gpuMalloc failed\n", rank);
        return 3;
    }
    (void)gpuMemset(d_buf, 0xCC, kBufBytes);
    (void)gpuDeviceSynchronize();
    auto bh = rt.register_buffer(d_buf, kBufBytes, /*is_device=*/true);
    rt.exchange();
    rt.barrier();

    if (rank == 0) {
        printf("\n=== bench_pingpong_lto (LTO-generated host trace) ===\n");
        printf("ranks=%d  outer_iters=%d  batch=%d  warmup=%d\n",
               nranks, NUM_OUTER, batch, NUM_WARMUP);
        printf("\n%-10s %12s %14s %14s\n",
               "size", "iters_total", "mean_us/msg", "median_us/msg");
        printf("---------------------------------------------------\n");
    }

    for (int s = 0; s < kNumSizes; ++s) {
        size_t bytes = kSizes[s];
        if (rank == 0) {
            char sb[32]; fmt_size(bytes, sb);
            printf("  starting size=%s ...\n", sb);
        }

        // --- correctness setup: rank 0 stamps a size-keyed pattern into
        //     its src; rank 1 wipes its dst to a sentinel. After all
        //     timing for this size completes, rank 1 reads back the
        //     first/last bytes of dst and verifies they match the
        //     pattern (proving the LTO-generated trace actually issued
        //     the fi_writes). Done outside the timing window.
        const uint8_t pat_byte = (uint8_t)((s * 17 + 0xA1) & 0xFF);
        if (rank == 0) {
            (void)hipMemset(d_buf, pat_byte, std::max<size_t>(bytes, 64));
        } else {
            (void)hipMemset(d_buf, 0x00, std::max<size_t>(bytes, 64));
        }
        (void)hipDeviceSynchronize();
        rt.barrier();

        // --- warmup ---
        if (rank == 0) {
            for (int w = 0; w < NUM_WARMUP; ++w) {
                gicc::launch<dwq_loop_kernel>(rt, dim3(1), dim3(1),
                                              peer, bh.index, bytes, batch);
                (void)hipDeviceSynchronize();
                rt.reset();
            }
        }
        rt.barrier();

        std::vector<double> samples;
        samples.reserve(NUM_OUTER);
        for (int o = 0; o < NUM_OUTER; ++o) {
            rt.barrier();
            double t0 = MPI_Wtime();
            if (rank == 0) {
                // ONE call. The LTO pass injects a call to the
                // synthesized trace function right before this body,
                // which pre-stages `batch` RDMA writes via the
                // gicc_runtime_dwq_enqueue C ABI. The kernel then
                // fires them all with one MMIO trigger.
                gicc::launch<dwq_loop_kernel>(rt, dim3(1), dim3(1),
                                              peer, bh.index, bytes, batch);
                (void)hipDeviceSynchronize();
                rt.reset();
            }
            double t1 = MPI_Wtime();
            if (rank == 0) {
                samples.push_back((t1 - t0) * 1e6 / batch);
            }
        }
        if (rank == 0) {
            double sum = 0.0;
            for (double v : samples) sum += v;
            double mean = sum / samples.size();
            double med  = median(samples);
            char sb[32]; fmt_size(bytes, sb);
            printf("%-10s %12d %14.3f %14.3f\n",
                   sb, NUM_OUTER * batch, mean, med);
            fflush(stdout);
        }
        rt.barrier();

        // --- correctness check: rank 1 reads the first byte AND the
        //     last byte of its dst region, must equal pat_byte. If LTO
        //     erased the trace (no fi_writes ever issued), dst stays at
        //     the sentinel 0x00 and we detect the regression here.
        if (rank == 1) {
            uint8_t first_dst = 0xFF, last_dst = 0xFF;
            size_t check_len = std::max<size_t>(bytes, 1);
            (void)hipMemcpy(&first_dst, d_buf, 1, hipMemcpyDeviceToHost);
            (void)hipMemcpy(&last_dst,
                            (char*)d_buf + (check_len - 1), 1,
                            hipMemcpyDeviceToHost);
            if (first_dst != pat_byte || last_dst != pat_byte) {
                char sb[32]; fmt_size(bytes, sb);
                fprintf(stderr,
                    "[VERIFY-FAIL] size=%s first=0x%02x last=0x%02x "
                    "expected=0x%02x (LTO trace may not have fired!)\n",
                    sb, first_dst, last_dst, pat_byte);
                fflush(stderr);
            }
        }
        rt.barrier();
    }

    if (rank == 0) printf("---------------------------------------------------\n");

    // Final accounting: print actual cumulative enqueue count vs the
    // expected count. If the LTO pass dropped the trace function, or
    // if some inlining removed enqueue calls, this will be far below
    // the expected value and the speedup was bogus.
    if (rank == 0) {
        uint64_t actual = gicc_runtime_trigger_val(&rt);
        uint64_t expected_per_size = (uint64_t)(NUM_WARMUP + NUM_OUTER) * batch;
        uint64_t expected_total = expected_per_size * kNumSizes;
        printf("[enqueue-audit] mono_total_ops=%lu  expected=%lu  "
               "match=%s\n",
               (unsigned long)actual, (unsigned long)expected_total,
               (actual == expected_total) ? "YES" :
                   (actual > expected_total ? "MORE" : "LESS"));
    }

    rt.barrier();
    (void)gpuFree(d_buf);
    return 0;
}
