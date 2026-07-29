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
#include <sched.h>
#include <dirent.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Print the CPUs that this process (and therefore its proxy worker threads)
// is allowed to run on. Critical for multi-threaded proxy benchmarks: on a
// default --cpus-per-task=1 Slurm config the whole proxy fleet would be
// time-sliced on one core and look like a regression instead of speedup.
static void print_cpu_affinity(const char* tag) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        printf("[cpu-aff] %s: sched_getaffinity failed\n", tag);
        return;
    }
    int n = CPU_COUNT(&mask);
    int first = -1, last = -1;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &mask)) {
            if (first < 0) first = c;
            last = c;
        }
    }
    printf("[cpu-aff] %s: %d CPUs allowed (range %d..%d)\n",
           tag, n, first, last);
    fflush(stdout);
}

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/runtime_helpers.h"  // gicc_runtime_trigger_val

#define MPI_TAG 100

// Sizes in bytes. Sweep covers latency-bound (small) through bandwidth-bound
// (large). Keep the largest below the buffer cap below.
static const size_t kSizes[] = {
    1, 2, 4,                       // tiny: pure latency / NIC min-packet
    8,
    64,
    256,
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    512 * 1024,                    // bridge 256K→1M
    1024 * 1024,
    2 * 1024 * 1024,               // bridge 1M→4M
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
        gicc::put(ctx, peer, buf_idx, /*dst_off=*/0,
                  buf_idx, /*src_off=*/0, bytes);
        if (per_msg_quiet) gicc::quiet(ctx);
    }
    if (!per_msg_quiet) gicc::quiet(ctx);
}

// GET counterpart of proxy_send_kernel. Pulls from peer's buffer at
// offset 0 into our buffer at offset 0. Same pipelining vs per-msg-quiet
// tradeoff as the PUT kernel.
__global__ void proxy_get_kernel(gicc::DeviceCtx* ctx, int peer,
                                 int buf_idx, size_t bytes, int n,
                                 bool per_msg_quiet) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < n; ++i) {
        // get: (source_rank, src_buf=peer_buf, src_off=0,
        //       dst_buf=our_buf, dst_off=0, bytes)
        gicc::get(ctx, peer,
                  buf_idx, /*src_off=*/0,
                  buf_idx, /*dst_off=*/0, bytes);
        if (per_msg_quiet) gicc::quiet(ctx);
    }
    if (!per_msg_quiet) gicc::quiet(ctx);
}

// DWQ mode trigger kernel — pre-staged host enqueues fire when the
// device thread does flush() (lead-thread MMIO write) then quiet()
// (poll completion counter).
__global__ void dwq_flush_quiet_kernel(gicc::DeviceCtx* ctx) {
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

// Multi-lane kernel — one thread per block, each block hashes blockIdx.x
// to a proxy ring index. With NUM_PROXY_THREADS>1 this lets the bench
// measure aggregate throughput across the proxy fleet (the single-block
// kernel above only ever uses ring 0). Each lane does n puts then ONE
// quiet against its own ring.
__global__ void proxy_send_kernel_multi(gicc::DeviceCtx* ctx, int peer,
                                        int buf_idx, size_t bytes, int n) {
    if (threadIdx.x != 0) return;
    int lane = blockIdx.x;  // one lane per block (mod num_proxy_rings)
    for (int i = 0; i < n; ++i) {
        gicc::put(ctx, peer,
                  buf_idx, /*dst_off=*/0,
                  buf_idx, /*src_off=*/0, bytes, /*lane=*/lane);
    }
    gicc::quiet(ctx, /*lane=*/lane);
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
    // --lanes=K       : (proxy mode) launch K parallel kernel blocks, each
    //                   hashed to a proxy ring (block_idx % num_proxy_rings).
    //                   Use with GICC_NUM_PROXY_THREADS=N to measure how
    //                   parallel proxy threads scale aggregate throughput.
    std::string mode = "proxy";
    std::string op   = "put";   // "put" | "get"
    bool quick = false;
    bool per_msg_quiet = false;
    int  lanes = 1;
    int  batch_override = 0;  // 0 = use default BATCH_PER_OUTER
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--mode=", 0) == 0) {
            mode = a.substr(7);
        } else if (a.rfind("--op=", 0) == 0) {
            op = a.substr(5);
        } else if (a == "--quick") {
            quick = true;
        } else if (a == "--per-msg-quiet") {
            per_msg_quiet = true;
        } else if (a.rfind("--lanes=", 0) == 0) {
            lanes = std::atoi(a.substr(8).c_str());
            if (lanes < 1) lanes = 1;
        } else if (a.rfind("--batch=", 0) == 0) {
            batch_override = std::atoi(a.substr(8).c_str());
            if (batch_override < 1) batch_override = 1;
        }
    }
    if (op != "put" && op != "get") {
        fprintf(stderr, "bench_pingpong: --op must be 'put' or 'get'\n");
        return 2;
    }
    // GET is one-sided like PUT but pulls FROM the peer. For mode=mpi
    // there is no direct GET analog (MPI_Get is RMA, not point-to-point).
    if (op == "get" && mode == "mpi") {
        fprintf(stderr,
            "bench_pingpong: --op=get is only valid for proxy/dwq modes\n");
        return 2;
    }
    // GET via proxy needs lanes=1 for now (the multi-lane GET kernel is
    // a separate symbol we have not added). Single-block lane=1 still
    // gives the per-msg-quiet/throughput pipelined timings.
    if (op == "get" && lanes != 1) {
        fprintf(stderr,
            "bench_pingpong: --op=get currently requires --lanes=1\n");
        return 2;
    }
    if (mode != "mpi" && mode != "proxy" && mode != "dwq" && mode != "dwq-tonly") {
        fprintf(stderr,
            "bench_pingpong: --mode must be 'mpi', 'proxy', 'dwq', or "
            "'dwq-tonly' (DWQ with host enqueue OUTSIDE the timing window)\n");
        return 2;
    }

    // gicc::Runtime initializes Bootstrap (MPI) internally; we just use
    // its rank/size and MPI_COMM_WORLD afterwards.
    gicc::Runtime rt;
    // DWQ mode uses the GDA-style host-wait fast path (shared completion
    // counter, mono_total_ops_ accounting). Proxy mode doesn't care.
    if (mode == "dwq" || mode == "dwq-tonly") rt.enable_host_wait_mode();
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

    // Force lazy init (proxy thread spawn / DWQ ctx) before timing.
    if (mode == "proxy" || mode == "dwq" || mode == "dwq-tonly") {
        (void)rt.prepare();
        rt.reset();
    }
    rt.barrier();

    // Run a small burn-in so each proxy worker actually executes once
    // before we probe its placement — last_cpu read from /proc/self/task
    // is the LAST scheduling slot, so threads that have not yet run all
    // report cpu=0 from their constructor.
    if ((mode == "proxy" || mode == "dwq" || mode == "dwq-tonly") && rank == 0) {
        const int burn_n = 64;
        gicc::DeviceCtx* d_ctx = rt.prepare();
        if (mode == "proxy") {
            if (lanes > 1) {
                gpuLaunchKernel(proxy_send_kernel_multi,
                                dim3(lanes), dim3(1), 0, 0,
                                d_ctx, peer, bh.index, /*bytes=*/8, burn_n);
            } else if (op == "put") {
                gpuLaunchKernel(proxy_send_kernel,
                                dim3(1), dim3(1), 0, 0,
                                d_ctx, peer, bh.index, /*bytes=*/8, burn_n,
                                /*per_msg_quiet=*/false);
            } else {
                gpuLaunchKernel(proxy_get_kernel,
                                dim3(1), dim3(1), 0, 0,
                                d_ctx, peer, bh.index, /*bytes=*/8, burn_n,
                                /*per_msg_quiet=*/false);
            }
        } else {
            for (int i = 0; i < burn_n; ++i) {
                if (op == "put") {
                    rt.put(bh, peer, bh.index, 8, 0, 0);
                } else {
                    rt.get(bh, peer, bh.index, 8, 0, 0);
                }
            }
            d_ctx = rt.prepare();
            gpuLaunchKernel(dwq_flush_quiet_kernel, dim3(1), dim3(1), 0, 0, d_ctx);
        }
        (void)gpuDeviceSynchronize();
        rt.reset();
    }
    rt.barrier();

    // After lazy init AND a burn-in, dump the per-thread "last CPU used"
    // from /proc/self/task/<tid>/stat field 39 (processor). Lets us see
    // whether the N proxy threads actually landed on distinct cores.
    if (rank == 0 && mode == "proxy") {
        DIR* d = opendir("/proc/self/task");
        if (d) {
            printf("[cpu-aff] rank0 thread placement (tid → last CPU):\n");
            struct dirent* ent;
            int shown = 0;
            while ((ent = readdir(d)) != nullptr && shown < 32) {
                if (ent->d_name[0] == '.') continue;
                char path[256], comm[64] = "?", stat_buf[1024];
                snprintf(path, sizeof(path), "/proc/self/task/%s/comm", ent->d_name);
                FILE* fc = fopen(path, "r");
                if (fc) { if (fgets(comm, sizeof(comm), fc)) { /* strip \n */
                    size_t L = strlen(comm); if (L && comm[L-1]=='\n') comm[L-1]=0; }
                    fclose(fc); }
                snprintf(path, sizeof(path), "/proc/self/task/%s/stat", ent->d_name);
                FILE* fs = fopen(path, "r");
                int last_cpu = -1;
                if (fs) {
                    if (fgets(stat_buf, sizeof(stat_buf), fs)) {
                        // stat fields are space-separated; processor is field 39
                        // (1-indexed). Skip past ") " for the comm field which
                        // may contain spaces.
                        char* rp = strrchr(stat_buf, ')');
                        if (rp) {
                            int field = 1;  // ')' ends field 2
                            char* tok = rp + 1;
                            while (*tok && field < 39) {
                                if (*tok == ' ') field++;
                                tok++;
                            }
                            last_cpu = atoi(tok);
                        }
                    }
                    fclose(fs);
                }
                printf("    tid=%-7s comm=%-16s last_cpu=%d\n",
                       ent->d_name, comm, last_cpu);
                shown++;
            }
            closedir(d);
            fflush(stdout);
        }
    }

    if (rank == 0) {
        char tag[64];
        snprintf(tag, sizeof(tag), "rank0 (mode=%s)", mode.c_str());
        print_cpu_affinity(tag);
        const char* gpu_aware = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
        printf("\n=== bench_pingpong (mode=%s op=%s) ===\n",
               mode.c_str(), op.c_str());
        printf("ranks=%d  outer_iters=%d  batch_per_outer=%d  warmup=%d  lanes=%d\n",
               nranks,
               quick ? 3 : NUM_OUTER,
               quick ? 5 : BATCH_PER_OUTER,
               quick ? 2 : NUM_WARMUP,
               lanes);
        if (mode == "mpi") {
            printf("MPICH_GPU_SUPPORT_ENABLED=%s\n",
                   gpu_aware ? gpu_aware : "(unset)");
        } else {
            const char* npt = std::getenv("GICC_NUM_PROXY_THREADS");
            printf("GICC_NUM_PROXY_THREADS=%s\n", npt ? npt : "1 (default)");
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
    if (batch_override > 0) {
        batch_per_outer = batch_override;
        if (rank == 0) {
            printf("(batch override: batch_per_outer=%d)\n", batch_per_outer);
        }
    }
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

        // --- correctness setup:
        //   op=put: stamp src on rank 0, wipe dst on rank 1. After PUT,
        //           rank 1 verifies.
        //   op=get: stamp src on rank 1 (data we'll pull), wipe dst on
        //           rank 0. After GET, rank 0 verifies.
        const uint8_t pat_byte = (uint8_t)((s * 17 + 0xA1) & 0xFF);
        const int src_rank_for_check = (op == "put") ? 0 : 1;
        if (rank == src_rank_for_check) {
            (void)gpuMemset(d_buf, pat_byte, std::max<size_t>(bytes, 64));
        } else {
            (void)gpuMemset(d_buf, 0x00, std::max<size_t>(bytes, 64));
        }
        (void)gpuDeviceSynchronize();
        rt.barrier();

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
        } else if (mode == "dwq" || mode == "dwq-tonly") {
            if (rank == 0) {
                for (int w = 0; w < num_warmup; ++w) {
                    for (int i = 0; i < batch_per_outer; ++i) {
                        if (op == "put") {
                            rt.put(bh, peer, bh.index, bytes,
                                         /*src_off=*/0, /*dst_off=*/0);
                        } else {
                            // GET: pull peer's data into our buffer.
                            // local_dst=bh, src_rank=peer, src_buf=peer's idx,
                            // local_off=0, remote_off=0.
                            rt.get(bh, peer, bh.index, bytes,
                                         /*local_off=*/0, /*remote_off=*/0);
                        }
                    }
                    gicc::DeviceCtx* d_ctx = rt.prepare();
                    gpuLaunchKernel(dwq_flush_quiet_kernel,
                                    dim3(1), dim3(1), 0, 0, d_ctx);
                    (void)gpuDeviceSynchronize();
                    rt.reset();
                }
            }
        } else {
            // Proxy warmup: rank 0 fires num_warmup small put+quiet ops.
            if (rank == 0) {
                gicc::DeviceCtx* d_ctx = rt.prepare();
                if (op == "put") {
                    gpuLaunchKernel(proxy_send_kernel, dim3(1), dim3(1), 0, 0,
                                    d_ctx, peer, bh.index, bytes, num_warmup,
                                    per_msg_quiet);
                } else {
                    gpuLaunchKernel(proxy_get_kernel, dim3(1), dim3(1), 0, 0,
                                    d_ctx, peer, bh.index, bytes, num_warmup,
                                    per_msg_quiet);
                }
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

            // dwq-tonly: pre-stage host enqueue BEFORE t0 so the timing
            // window contains only the trigger+complete+reset legs. Lets
            // us answer "is host enqueue the reason DWQ loses at small
            // sizes?" without changing the rest of the harness.
            //
            // Also measure host enqueue cost separately if requested.
            double t_enq_us = 0.0;
            if (mode == "dwq-tonly" && rank == 0) {
                double te0 = MPI_Wtime();
                for (int i = 0; i < batch_per_outer; ++i) {
                    if (op == "put") {
                        rt.put(bh, peer, bh.index, bytes,
                                     /*src_off=*/0, /*dst_off=*/0);
                    } else {
                        rt.get(bh, peer, bh.index, bytes,
                                     /*local_off=*/0, /*remote_off=*/0);
                    }
                }
                t_enq_us = (MPI_Wtime() - te0) * 1e6 / batch_per_outer;
            }

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
            } else if (mode == "dwq" || mode == "dwq-tonly") {
                if (rank == 0) {
                    // dwq: host pre-stages batch_per_outer puts INSIDE
                    // the timing window. dwq-tonly: those same puts were
                    // pre-staged BEFORE t0 above, so this branch only
                    // measures the device-side trigger + completion + reset.
                    if (mode == "dwq") {
                        for (int i = 0; i < batch_per_outer; ++i) {
                            if (op == "put") {
                                rt.put(bh, peer, bh.index, bytes,
                                             /*src_off=*/0, /*dst_off=*/0);
                            } else {
                                rt.get(bh, peer, bh.index, bytes,
                                             /*local_off=*/0, /*remote_off=*/0);
                            }
                        }
                    }
                    gicc::DeviceCtx* d_ctx = rt.prepare();
                    gpuLaunchKernel(dwq_flush_quiet_kernel,
                                    dim3(1), dim3(1), 0, 0, d_ctx);
                    (void)gpuDeviceSynchronize();
                    rt.reset();
                }
            } else {
                if (rank == 0) {
                    gicc::DeviceCtx* d_ctx = rt.prepare();
                    if (lanes > 1) {
                        // Multi-lane mode: K blocks each push to its own
                        // ring (mod num_proxy_rings) then quiet that ring.
                        // Aggregate work = lanes * batch_per_outer puts.
                        // (PUT-only; GET requires --lanes=1.)
                        gpuLaunchKernel(proxy_send_kernel_multi,
                                        dim3(lanes), dim3(1), 0, 0,
                                        d_ctx, peer, bh.index, bytes,
                                        batch_per_outer);
                    } else if (op == "put") {
                        gpuLaunchKernel(proxy_send_kernel,
                                        dim3(1), dim3(1), 0, 0,
                                        d_ctx, peer, bh.index, bytes,
                                        batch_per_outer, per_msg_quiet);
                    } else {
                        gpuLaunchKernel(proxy_get_kernel,
                                        dim3(1), dim3(1), 0, 0,
                                        d_ctx, peer, bh.index, bytes,
                                        batch_per_outer, per_msg_quiet);
                    }
                    (void)gpuDeviceSynchronize();
                    rt.reset();
                }
                // Rank 1: nothing to do for proxy mode (one-sided RDMA).
            }

            double t1 = MPI_Wtime();
            if (rank == 0) {
                // Total messages this outer iter = batch_per_outer * lanes
                // (lanes lanes each issued batch_per_outer puts).
                int msgs_this_outer = batch_per_outer * lanes;
                samples.push_back((t1 - t0) * 1e6 /
                                  static_cast<double>(msgs_this_outer));
                if (mode == "dwq-tonly" && o == num_outer / 2) {
                    // Mid-sample: dump host-enqueue cost (per put_no_db) so
                    // we can see exactly how much fi_control(FI_QUEUE_WORK)
                    // is costing us, separately from trigger+complete.
                    fprintf(stderr, "[dwq-tonly] size=%zu  enq=%.2f us/put  "
                            "tonly=%.2f us/msg\n",
                            bytes, t_enq_us,
                            (t1 - t0) * 1e6 / msgs_this_outer);
                }
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

        // --- correctness verify: the "dst" rank confirms first & last
        //     byte. For PUT that's rank 1; for GET that's rank 0.
        const int dst_rank_for_check = (op == "put") ? 1 : 0;
        if (rank == dst_rank_for_check) {
            uint8_t first_dst = 0xFF, last_dst = 0xFF;
            size_t check_len = std::max<size_t>(bytes, 1);
            (void)gpuMemcpy(&first_dst, d_buf, 1, gpuMemcpyDeviceToHost);
            (void)gpuMemcpy(&last_dst,
                            (char*)d_buf + (check_len - 1), 1,
                            gpuMemcpyDeviceToHost);
            if (first_dst != pat_byte || last_dst != pat_byte) {
                char sb[32]; fmt_size(bytes, sb);
                fprintf(stderr,
                    "[VERIFY-FAIL mode=%s op=%s] size=%s first=0x%02x last=0x%02x "
                    "expected=0x%02x\n",
                    mode.c_str(), op.c_str(), sb, first_dst, last_dst, pat_byte);
                fflush(stderr);
            }
        }
        rt.barrier();
    }

    if (rank == 0) {
        printf("---------------------------------------------------\n");
    }

    // Cumulative enqueue audit. mono_total_ops_ increments inside
    // rt.put_no_db (host_wait_mode path) and gicc_runtime_dwq_enqueue,
    // so for the dwq / dwq-tonly modes the expected total is
    // (NUM_WARMUP + NUM_OUTER) * batch_per_outer * num_sizes. Proxy
    // mode bypasses this counter (puts go through the ring not the
    // host enqueue path), so for proxy the actual value will be near
    // zero — that's expected and not a bug.
    if (rank == 0) {
        uint64_t actual = gicc_runtime_trigger_val(&rt);
        uint64_t expected_per_size =
            (uint64_t)(num_warmup + num_outer) * (uint64_t)batch_per_outer;
        uint64_t expected = expected_per_size * (uint64_t)num_sizes;
        printf("[enqueue-audit mode=%s] mono_total_ops=%lu  "
               "dwq_expected=%lu  match=%s\n",
               mode.c_str(), (unsigned long)actual, (unsigned long)expected,
               (mode == "dwq" || mode == "dwq-tonly")
                   ? ((actual == expected) ? "YES" :
                      (actual > expected ? "MORE" : "LESS"))
                   : "(proxy mode uses ring, expect 0)");
    }

    // Final placement snapshot — by now proxy threads have been hot for
    // seconds, so last_cpu reflects steady-state scheduling. A healthy
    // multi-thread run should show distinct CPU numbers across worker
    // tids (not all clustered on cpu=0).
    if (rank == 0 && mode == "proxy") {
        DIR* d = opendir("/proc/self/task");
        if (d) {
            printf("[cpu-aff] rank0 thread placement AFTER bench:\n");
            struct dirent* ent;
            int shown = 0;
            while ((ent = readdir(d)) != nullptr && shown < 32) {
                if (ent->d_name[0] == '.') continue;
                char path[256], stat_buf[1024], comm[64] = "?";
                snprintf(path, sizeof(path), "/proc/self/task/%s/comm", ent->d_name);
                FILE* fc = fopen(path, "r");
                if (fc) { if (fgets(comm, sizeof(comm), fc)) {
                    size_t L = strlen(comm); if (L && comm[L-1]=='\n') comm[L-1]=0; }
                    fclose(fc); }
                snprintf(path, sizeof(path), "/proc/self/task/%s/stat", ent->d_name);
                FILE* fs = fopen(path, "r");
                int last_cpu = -1;
                if (fs) {
                    if (fgets(stat_buf, sizeof(stat_buf), fs)) {
                        char* rp = strrchr(stat_buf, ')');
                        if (rp) {
                            int field = 1;
                            char* tok = rp + 1;
                            while (*tok && field < 39) {
                                if (*tok == ' ') field++;
                                tok++;
                            }
                            last_cpu = atoi(tok);
                        }
                    }
                    fclose(fs);
                }
                printf("    tid=%-7s comm=%-16s last_cpu=%d\n",
                       ent->d_name, comm, last_cpu);
                shown++;
            }
            closedir(d);
            fflush(stdout);
        }
    }

    rt.barrier();
    (void)gpuFree(d_buf);
    return 0;
}
