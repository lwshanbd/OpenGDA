/**
 * gda_benchmark_gicc.cpp - 32-stream concurrent DWQ micro-benchmark, ported
 * to the unified gicc:: high-level API. Equivalent to the original
 * main.cpp + benchmark_runner.hpp but written entirely against
 * gicc::Runtime / gicc::flush / gicc::quiet, with no direct use of the
 * underlying libfabric / DWQ primitives.
 *
 * Pattern (one per size, NUM_ITERATIONS times):
 *   - queue N_STREAMS rt.put_no_db()  (host-side DWQ enqueue)
 *   - rt.prepare()                    (sets trigger_val, n_ops)
 *   - launch kernel { gicc::flush; gicc::quiet; }
 *   - measure hipDeviceSynchronize() time
 *   - rt.reset()
 *
 * Run: FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 ./gda_benchmark_gicc
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <hip/hip_runtime.h>

#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"

constexpr int    N_STREAMS      = 32;
constexpr int    NUM_ITERATIONS = 20;
constexpr size_t MAX_SIZE       = 16 * 1024 * 1024;

constexpr size_t TEST_SIZES[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024,
    2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024
};
constexpr int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);

__global__ void flush_quiet_kernel(gicc::DeviceCtx* ctx) {
    // Single-thread flush+quiet: matches the baseline benchmark_runner's
    // pattern where the trigger and the wait happen on the same thread.
    // Polling completion_ from many threads only causes cache-line bouncing.
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

// GPU-side verify copy. hipMemcpy(D2H) for sizes <16KB can read stale L2
// cache after a NIC RDMA write; a GPU kernel copy bypasses that.
__global__ void verify_copy_kernel(const uint8_t* src, uint8_t* dst, size_t n) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

// Kernel-based memset with explicit system fence. Bypasses the
// hipMemset(<16KB) → kernel-memset path whose L2 writeback can race with
// subsequent NIC PCIe writes to the same address. The threadfence_system
// at the end forces all preceding writes to be visible in HBM before
// the kernel exits, so MPI_Barrier afterwards is a true memory ordering
// boundary.
__global__ void memset_fenced_kernel(uint8_t* dst, uint8_t value, size_t n) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x) {
        dst[i] = value;
    }
    __threadfence_system();
}

static const char* fmt_size(size_t s, char* buf) {
    if      (s < 1024)        snprintf(buf, 32, "%zuB",  s);
    else if (s < 1024 * 1024) snprintf(buf, 32, "%zuKB", s / 1024);
    else                      snprintf(buf, 32, "%zuMB", s / (1024 * 1024));
    return buf;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    gicc::Runtime rt;
    int rank   = rt.rank();
    int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "Need exactly 2 ranks\n");
        MPI_Finalize();
        return 1;
    }
    int peer = 1 - rank;

    // -------------------------------------------------------------------------
    // Allocate ONE source region and ONE destination region, each with
    // N_STREAMS sub-regions of MAX_SIZE. Stream i lives at byte offset
    // i * MAX_SIZE within both. This keeps the registered-buffer count to
    // 2 per rank so the PMI KVS value used by Runtime::exchange() fits
    // comfortably regardless of N_STREAMS.
    // -------------------------------------------------------------------------
    const size_t TOTAL = MAX_SIZE * (size_t)N_STREAMS;

    void* d_src = nullptr;
    void* d_dst = nullptr;
    (void)hipMalloc(&d_src, TOTAL);
    (void)hipMalloc(&d_dst, TOTAL);
    auto src_buf = rt.register_buffer(d_src, TOTAL, true);
    auto dst_buf = rt.register_buffer(d_dst, TOTAL, true);

    rt.exchange();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%-8s  %12s  %12s  %s\n", "Size", "Total(us)", "Per-xfer(us)", "Statistics");
        printf("========  ============  ============  =====================================\n");
        printf("Note: %d concurrent streams via gicc::Runtime, per-xfer = total/%d\n",
               N_STREAMS, N_STREAMS);
        printf("      Unified gicc:: API on libfabric/CXI backend\n\n");
    }

    // Pinned/mapped buffer for GPU-kernel verify path (avoids L2 staleness).
    uint8_t* h_verify = nullptr;
    uint8_t* d_verify = nullptr;
    (void)hipHostMalloc(&h_verify, MAX_SIZE, hipHostMallocMapped);
    (void)hipHostGetDevicePointer((void**)&d_verify, h_verify, 0);

    for (int sidx = 0; sidx < NUM_TEST_SIZES; sidx++) {
        size_t cur = TEST_SIZES[sidx];
        double iter_us[NUM_ITERATIONS];
        int    n_done = 0;
        int    verify_failures = 0;

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Init buffers
            if (rank == 0) {
                for (int i = 0; i < N_STREAMS; i++) {
                    uint8_t pat = (uint8_t)((iter + 0xA0 + i) & 0xFF);
                    (void)hipMemset((uint8_t*)d_src + (size_t)i * MAX_SIZE, pat, cur);
                }
                (void)hipDeviceSynchronize();
            } else {
                for (int i = 0; i < N_STREAMS; i++) {
                    hipLaunchKernelGGL(memset_fenced_kernel,
                        dim3((cur + 255) / 256), dim3(256), 0, 0,
                        (uint8_t*)d_dst + (size_t)i * MAX_SIZE,
                        (uint8_t)0xFF, cur);
                }
                (void)hipDeviceSynchronize();
            }
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                // Queue all 32 puts (host-side DWQ enqueue)
                for (int i = 0; i < N_STREAMS; i++) {
                    rt.put_no_db(src_buf, peer, dst_buf.index, cur,
                                 /*src_off=*/(size_t)i * MAX_SIZE,
                                 /*dst_off=*/(size_t)i * MAX_SIZE);
                }
                auto* ctx = rt.prepare();

                // Time the kernel: flush + quiet from a single block.
                auto t0 = std::chrono::high_resolution_clock::now();
                hipLaunchKernelGGL(flush_quiet_kernel, dim3(1), dim3(1), 0, 0, ctx);
                (void)hipDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();
                iter_us[n_done++] =
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

                rt.reset();
            }
            MPI_Barrier(MPI_COMM_WORLD);

            // Verification (rank 1) — GPU-kernel copy to pinned host memory
            // bypasses the hipMemcpy L2 cache staleness for sizes <16KB.
            if (rank == 1) {
                int total_errors = 0;
                for (int i = 0; i < N_STREAMS; i++) {
                    hipLaunchKernelGGL(verify_copy_kernel,
                        dim3((cur + 255) / 256), dim3(256), 0, 0,
                        (uint8_t*)d_dst + (size_t)i * MAX_SIZE, d_verify, cur);
                    (void)hipDeviceSynchronize();
                    uint8_t pat = (uint8_t)((iter + 0xA0 + i) & 0xFF);
                    int errs = 0;
                    for (size_t j = 0; j < cur; j++) {
                        if (h_verify[j] != pat) {
                            if (errs < 5) {
                                fprintf(stderr,
                                    "FAIL size=%zu iter=%d stream=%d off=%zu "
                                    "got=0x%02x expected=0x%02x\n",
                                    cur, iter, i, j, h_verify[j], pat);
                            }
                            errs++;
                        }
                    }
                    total_errors += errs;
                }
                if (total_errors > 0) verify_failures++;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0 && n_done > 0) {
            std::sort(iter_us, iter_us + n_done);
            int samples = (n_done < 10) ? n_done : 10;
            double sum = 0;
            for (int i = 0; i < samples; i++) sum += iter_us[i];
            double avg = sum / samples;
            char buf[32];
            printf("%-8s  %12.2f  %12.2f  (min=%.2f max=%.2f)\n",
                   fmt_size(cur, buf), avg, avg / N_STREAMS,
                   iter_us[0], iter_us[samples - 1]);
            fflush(stdout);
        }
        if (rank == 1 && verify_failures > 0) {
            fprintf(stderr, "Size %zu: %d/%d iters failed verification\n",
                    cur, verify_failures, NUM_ITERATIONS);
        }
    }

    (void)hipFree(d_src);
    (void)hipFree(d_dst);
    (void)hipHostFree(h_verify);
    MPI_Finalize();
    return 0;
}
