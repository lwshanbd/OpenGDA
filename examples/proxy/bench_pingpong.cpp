/*
 * bench_pingpong.cpp - Latency comparison: CPU Proxy vs Cray MPI.
 *
 * Two ranks. Rank 0 sends a message to rank 1; one direction only (this is
 * not a true ping-pong — both ranks measure unidirectional cost). Loop
 * NUM_ITERATIONS times per size; report mean per-message wall time.
 *
 * Two modes (selected by --mode):
 *
 *   mpi     rank 0 calls MPI_Send(d_buf, size, MPI_BYTE, peer, ...)
 *           rank 1 calls MPI_Recv(d_buf, size, MPI_BYTE, peer, ...)
 *           Requires Cray MPICH with MPICH_GPU_SUPPORT_ENABLED=1.
 *
 *   proxy   rank 0 launches one kernel that does N iterations of
 *           gicc::put_no_db(peer=1, ...) + gicc::quiet(ctx).
 *           Each quiet() spins until the proxy thread has the slot acked
 *           by the libfabric CQ; that ack means the fi_write has reached
 *           rank 1's HBM at the network level.
 *           Rank 1 only constructs Runtime + prepare()/reset() (no
 *           device work — one-sided RDMA semantics).
 *
 * Both modes share the same MPI_Barrier-based outer timing harness. The
 * proxy mode pays the same kernel-launch overhead per outer iteration as
 * the MPI mode pays once per MPI_Send. To amortize launch cost, each
 * outer timing iteration submits BATCH_PER_OUTER puts inside one kernel.
 *
 * Build with -DGICC_ENABLE_CPU_PROXY=ON. Run inside a 2-GPU allocation
 * with --gpu-bind=none so each task can pick its own device.
 *
 *   GICC_SKIP_DWQ_INIT=1 GICC_PROXY_ENABLED=1 MPICH_GPU_SUPPORT_ENABLED=1 \
 *     srun --jobid=$JOBID --overlap -n 2 --gpu-bind=none \
 *     ./examples/proxy/bench_pingpong --mode=proxy
 */

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"

#define MPI_TAG 100

// Sizes in bytes. Sweep covers latency-bound (small) through bandwidth-bound
// (large). Keep the largest below the buffer cap below.
static const size_t kSizes[] = {
    8,
    64,
    256,
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
};
static constexpr int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);

// Outer iterations per size. Each outer iter measures BATCH_PER_OUTER msgs
// and reports their mean. We then take the median across NUM_OUTER samples.
static constexpr int NUM_OUTER = 21;
// Inner batch — how many puts/sends per single MPI_Wtime window. Bigger
// amortizes per-iter constant overhead (kernel launch / MPI_Wtime jitter).
static constexpr int BATCH_PER_OUTER = 50;

// How many warmup iterations before timed run.
static constexpr int NUM_WARMUP = 10;

// Buffer registered for both proxy fi_write and MPI_Send.
static constexpr size_t kBufBytes = 32 * 1024 * 1024;

static const char* fmt_size(size_t s, char* buf) {
    if (s < 1024)
        snprintf(buf, 32, "%zuB", s);
    else if (s < 1024 * 1024)
        snprintf(buf, 32, "%zuKB", s / 1024);
    else
        snprintf(buf, 32, "%zuMB", s / (1024 * 1024));
    return buf;
}

// Inner kernel for proxy mode.
// per_msg_quiet=true:  loop N times of (put_no_db; quiet) — measures
//                      end-to-end per-message latency (each msg fully drains
//                      through the proxy + libfabric CQ before the next one).
// per_msg_quiet=false: loop N times of put_no_db, then ONE quiet at the end —
//                      lets the proxy + NIC pipeline N submissions and
//                      measures effective per-message cost under steady-state
//                      throughput (much closer to what e.g. MPI_Isend+Waitall
//                      would measure).
__global__ void proxy_send_kernel(gicc::DeviceCtx* ctx, int peer,
                                  int buf_idx, size_t bytes, int n,
                                  bool per_msg_quiet) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < n; ++i) {
        gicc::put_no_db(ctx, peer, buf_idx, /*dst_off=*/0,
                        buf_idx, /*src_off=*/0, bytes);
        if (per_msg_quiet) gicc::quiet(ctx);
    }
    if (!per_msg_quiet) gicc::quiet(ctx);
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv) {
    // Force line-buffered stdout so per-size progress is visible while a
    // run is in flight (default fully buffered when stdout is not a tty).
    setvbuf(stdout, nullptr, _IOLBF, 0);

    // --mode=mpi or --mode=proxy ; --quick = 3 sizes / 3 outer / 5 batch
    // --per-msg-quiet : (proxy mode) put + quiet per msg = pure latency
    //                   default: N puts + 1 quiet per outer = throughput
    std::string mode = "proxy";
    bool quick = false;
    bool per_msg_quiet = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--mode=", 0) == 0) {
            mode = a.substr(7);
        } else if (a == "--quick") {
            quick = true;
        } else if (a == "--per-msg-quiet") {
            per_msg_quiet = true;
        }
    }
    if (mode != "mpi" && mode != "proxy") {
        fprintf(stderr, "bench_pingpong: --mode must be 'mpi' or 'proxy'\n");
        return 2;
    }

    // gicc::Runtime initializes Bootstrap (MPI) internally; we just use
    // its rank/size and MPI_COMM_WORLD afterwards.
    gicc::Runtime rt;
    int rank = rt.rank();
    int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) {
            fprintf(stderr,
                    "bench_pingpong: need exactly 2 ranks (got %d)\n",
                    nranks);
        }
        return 1;
    }
    int peer = 1 - rank;

    void* d_buf = nullptr;
    if (gpuMalloc(&d_buf, kBufBytes) != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: gpuMalloc failed\n", rank);
        return 3;
    }
    (void)gpuMemset(d_buf, 0xAB, kBufBytes);
    (void)gpuDeviceSynchronize();

    auto bh = rt.register_buffer(d_buf, kBufBytes, /*is_device=*/true);
    rt.exchange();

    // Force the proxy thread to spawn now (lazy init would otherwise
    // happen during the first prepare() inside the timed loop, polluting
    // the first sample).
    if (mode == "proxy") {
        (void)rt.prepare();
    }
    rt.barrier();

    if (rank == 0) {
        const char* gpu_aware = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
        printf("\n=== bench_pingpong (mode=%s) ===\n", mode.c_str());
        printf("ranks=%d  outer_iters=%d  batch_per_outer=%d  warmup=%d\n",
               nranks,
               quick ? 3 : NUM_OUTER,
               quick ? 5 : BATCH_PER_OUTER,
               quick ? 2 : NUM_WARMUP);
        if (mode == "mpi") {
            printf("MPICH_GPU_SUPPORT_ENABLED=%s\n",
                   gpu_aware ? gpu_aware : "(unset)");
        }
        printf("\n%-10s %12s %14s %14s\n",
               "size", "iters_total", "mean_us/msg", "median_us/msg");
        printf("---------------------------------------------------\n");
    }

    // Quick mode: just 3 sizes, fewer iters — for diagnostics.
    int num_sizes = quick ? 3 : kNumSizes;
    int num_outer = quick ? 3 : NUM_OUTER;
    int batch_per_outer = quick ? 5 : BATCH_PER_OUTER;
    int num_warmup = quick ? 2 : NUM_WARMUP;
    if (rank == 0 && quick) {
        printf("(quick mode: %d sizes, %d outer, %d batch, %d warmup)\n",
               num_sizes, num_outer, batch_per_outer, num_warmup);
    }

    for (int s = 0; s < num_sizes; ++s) {
        size_t bytes = kSizes[s];
        if (rank == 0) {
            char sb[32]; fmt_size(bytes, sb);
            printf("  starting size=%s ...\n", sb);
        }

        // --- warmup ---
        if (mode == "mpi") {
            for (int w = 0; w < num_warmup; ++w) {
                if (rank == 0)
                    MPI_Send(d_buf, bytes, MPI_BYTE, peer, MPI_TAG,
                             MPI_COMM_WORLD);
                else
                    MPI_Recv(d_buf, bytes, MPI_BYTE, peer, MPI_TAG,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            // Proxy warmup: rank 0 fires num_warmup small put+quiet ops.
            if (rank == 0) {
                gicc::DeviceCtx* d_ctx = rt.prepare();
                gpuLaunchKernel(proxy_send_kernel, dim3(1), dim3(1), 0, 0,
                                d_ctx, peer, bh.index, bytes, num_warmup,
                                per_msg_quiet);
                (void)gpuDeviceSynchronize();
                rt.reset();
            }
        }
        rt.barrier();

        // --- timed outer loop ---
        std::vector<double> samples;
        samples.reserve(num_outer);

        for (int o = 0; o < num_outer; ++o) {
            rt.barrier();
            double t0 = MPI_Wtime();

            if (mode == "mpi") {
                if (rank == 0) {
                    for (int i = 0; i < batch_per_outer; ++i)
                        MPI_Send(d_buf, bytes, MPI_BYTE, peer, MPI_TAG,
                                 MPI_COMM_WORLD);
                } else {
                    for (int i = 0; i < batch_per_outer; ++i)
                        MPI_Recv(d_buf, bytes, MPI_BYTE, peer, MPI_TAG,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            } else {
                if (rank == 0) {
                    gicc::DeviceCtx* d_ctx = rt.prepare();
                    gpuLaunchKernel(proxy_send_kernel, dim3(1), dim3(1), 0, 0,
                                    d_ctx, peer, bh.index, bytes,
                                    batch_per_outer, per_msg_quiet);
                    (void)gpuDeviceSynchronize();
                    rt.reset();
                }
                // Rank 1: nothing to do for proxy mode (one-sided RDMA).
            }

            double t1 = MPI_Wtime();
            if (rank == 0) {
                samples.push_back((t1 - t0) * 1e6 /
                                  static_cast<double>(BATCH_PER_OUTER));
            }
        }

        if (rank == 0) {
            double sum = 0.0;
            for (double v : samples) sum += v;
            double mean = sum / samples.size();
            double med = median(samples);
            char sb[32];
            fmt_size(bytes, sb);
            printf("%-10s %12d %14.3f %14.3f\n",
                   sb, NUM_OUTER * BATCH_PER_OUTER, mean, med);
            fflush(stdout);
        }
        rt.barrier();
    }

    if (rank == 0) {
        printf("---------------------------------------------------\n");
    }

    rt.barrier();
    (void)gpuFree(d_buf);
    return 0;
}
