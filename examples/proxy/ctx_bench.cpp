/*
 * ctx_bench.cpp - Compiler-context microbenchmarks on the GICC substrate.
 *
 * Question under test: does program context that only a compiler can see
 * change WHICH communication path is fastest, for communication operations
 * that a runtime cannot tell apart (same op, same size, same peer, same
 * topology, same hardware)?
 *
 * Three axes, two ranks, one rank per node:
 *
 *   --exp=reuse     A loop of N puts with a loop-invariant descriptor.
 *                   Configs differ only in WHERE the descriptor staging and
 *                   the kernel launches happen:
 *                     restage      stage 1 descriptor + launch, per iteration
 *                                  (what a context-free lowering must emit:
 *                                   the runtime sees one put at a time)
 *                     hoist        stage all N descriptors before the loop,
 *                                  still one launch per iteration
 *                     hoist-fused  stage all N, then ONE kernel fires them
 *                                  one per iteration with N flushes
 *                   Proxy side: fused (one kernel, N pushes) and per-launch.
 *                   Sweeps N = the compiler-visible trip count.
 *
 *   --exp=batch     Fixed op count, sweep ops-per-trigger B. A runtime sees
 *                   B separate calls; a compiler sees the whole group.
 *
 *   --exp=distance  One put, then D us of independent device compute before
 *                   the completion point. Sweeps D and reports the EXPOSED
 *                   communication cost against a no-comm baseline of the
 *                   same D. Issue-to-first-use distance is invisible to the
 *                   runtime at the moment the put is issued.
 *
 *   --exp=grid      The JOINT configuration space, for offline decider work.
 *                   A workload point is (bytes, K, D): K puts of `bytes`
 *                   issued in one phase, then D us of independent device
 *                   compute, then the phase's completion point.
 *
 *                   The configuration space is CONDITIONAL on the path --
 *                   the two paths do not expose the same knobs, so this is
 *                   not a product space:
 *
 *                     trigger:  B = descriptors released per MMIO trigger
 *                               (K/B triggers, one terminal completion wait)
 *                     proxy:    P = concurrent pushing blocks in the kernel
 *                               L = proxy worker lanes, i.e.
 *                                   GICC_NUM_PROXY_THREADS. Process-wide, so
 *                                   L needs one binary run per value.
 *
 *                   Emits one GRID row per (point, config) cell plus a
 *                   no-communication spin baseline per D, so an offline
 *                   decider can be scored against the per-point oracle.
 *
 * The proxy path and the DWQ/trigger path cannot coexist in one process
 * (enable_host_wait_mode() disables proxy dispatch), so pick one with
 * --path and run the binary twice:
 *
 *   GICC_SKIP_DWQ_INIT=1 GICC_PROXY_ENABLED=1 \
 *     srun -p pci -N 2 -n 2 --ntasks-per-node=1 -c 8 --gpu-bind=none \
 *     ./ctx_bench --path=proxy --exp=reuse
 *   srun ... ./ctx_bench --path=trigger --exp=reuse
 *
 * Rows prefixed "CSV," are machine-readable:
 *   CSV,exp,path,config,bytes,param,samples,median_us,per_op_us,min,max,p25,p75
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

// Registered window. Every op in this bench uses offset 0, so this only has
// to hold the largest message.
static constexpr size_t kBufBytes = 32 * 1024 * 1024;

//----------------------------------------------------------------------------
// Kernels
//----------------------------------------------------------------------------

// Wall-clock spin. wall_clock64() is a constant-rate counter (unlike
// clock64(), which tracks the shader clock), so a tick count converts to a
// fixed wall-time independent of DVFS.
__device__ __forceinline__ void spin_ticks(long long ticks) {
    if (ticks <= 0) return;
    long long t0 = wall_clock64();
    while ((wall_clock64() - t0) < ticks) { }
}

// Proxy: n pushes into the ring, quiet every `quiet_every` ops
// (quiet_every <= 0 => a single quiet after all n).
__global__ void k_proxy_loop(gicc::DeviceCtx* ctx, int peer, int buf,
                             size_t bytes, int n, int quiet_every) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < n; ++i) {
        gicc::put(ctx, peer, buf, /*dst_off=*/0, buf, /*src_off=*/0, bytes);
        if (quiet_every > 0 && ((i + 1) % quiet_every) == 0) gicc::quiet(ctx);
    }
    if (quiet_every <= 0) gicc::quiet(ctx);
}

// Trigger: fire n pre-staged descriptors, one flush each. With
// prepare_delta(1, n) the MMIO add is 1, so flush i releases descriptor i.
__global__ void k_flush_n(gicc::DeviceCtx* ctx, int n) {
    for (int i = 0; i < n; ++i) gicc::flush(ctx);
}

// Same, minus the per-flush __threadfence_system(). gicc::flush() fences
// after every MMIO store so a preceding data write is visible before the
// NIC reads the buffer; back-to-back triggers with no intervening device
// write only need the fence once. Isolates how much of the fused trigger
// loop's cost is the fence rather than the trigger itself.
__global__ void k_flush_n_nofence(gicc::DeviceCtx* ctx, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (ctx->trigger_addr_ == nullptr) return;
    for (int i = 0; i < n; ++i) *ctx->trigger_addr_ = ctx->trigger_val_;
    __threadfence_system();
}

// Launch-cost baseline: same launch structure, no communication.
__global__ void k_empty() { }

// Distance experiment.
__global__ void k_spin(long long ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    spin_ticks(ticks);
}
__global__ void k_proxy_put_spin_quiet(gicc::DeviceCtx* ctx, int peer, int buf,
                                       size_t bytes, int ops, long long ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < ops; ++i)
        gicc::put(ctx, peer, buf, /*dst_off=*/0, buf, /*src_off=*/0, bytes);
    spin_ticks(ticks);
    gicc::quiet(ctx);
}
__global__ void k_flush_spin(gicc::DeviceCtx* ctx, long long ticks) {
    gicc::flush(ctx);
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    spin_ticks(ticks);
}

// Grid experiment.
//
// Trigger side: `groups` triggers, each releasing B pre-staged descriptors,
// then the phase's independent compute. Completion is waited for on the host.
__global__ void k_flush_groups_spin(gicc::DeviceCtx* ctx, int groups,
                                    long long ticks) {
    for (int g = 0; g < groups; ++g) gicc::flush(ctx);
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    spin_ticks(ticks);
}

// Proxy side: gridDim.x blocks share the K pushes, block b using lane b
// (lane_to_ring takes it modulo the fleet size, so P > L round-robins).
// Every block runs the phase's compute, then drains its own lane.
__global__ void k_proxy_blocks_spin(gicc::DeviceCtx* ctx, int peer, int buf,
                                    size_t bytes, int ops, long long ticks) {
    if (threadIdx.x != 0) return;
    const int lane  = blockIdx.x;
    const int base  = ops / gridDim.x;
    const int extra = ops % gridDim.x;
    const int mine  = base + ((int)blockIdx.x < extra ? 1 : 0);
    for (int i = 0; i < mine; ++i)
        gicc::put(ctx, peer, buf, /*dst_off=*/0, buf, /*src_off=*/0, bytes, lane);
    spin_ticks(ticks);
    gicc::quiet(ctx, lane);
}

// ONE thread pushing, spread round-robin over `lanes` rings.
//
// The multi-block kernel above conflates two things: more blocks pushing
// AND more rings being drained. This separates them. It is also the shape
// the pass can actually emit today without any new API -- `lane` is
// already a defaulted argument of put/quiet, so assigning it per call site
// is a constant rewrite, whereas widening the issue guard to several
// blocks is a change to the code the pass generates.
//
// If the bottleneck is the push itself, spreading one thread's pushes over
// more rings buys nothing. If it is the drain, it should.
__global__ void k_proxy_lanes_spin(gicc::DeviceCtx* ctx, int peer, int buf,
                                   size_t bytes, int ops, int lanes,
                                   long long ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < ops; ++i)
        gicc::put(ctx, peer, buf, /*dst_off=*/0, buf, /*src_off=*/0, bytes,
                  i % lanes);
    spin_ticks(ticks);
    for (int l = 0; l < lanes; ++l) gicc::quiet(ctx, l);
}

//----------------------------------------------------------------------------
// Stats
//----------------------------------------------------------------------------

struct Stats {
    double median = 0, min = 0, max = 0, p25 = 0, p75 = 0;
    int    n = 0;
};

static Stats summarize(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    s.n      = (int)n;
    s.median = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    s.min    = v.front();
    s.max    = v.back();
    s.p25    = v[n / 4];
    s.p75    = v[(3 * n) / 4];
    return s;
}

static const char* fmt_size(size_t s, char* buf) {
    if (s < 1024)                snprintf(buf, 32, "%zuB",  s);
    else if (s < 1024 * 1024)    snprintf(buf, 32, "%zuKB", s / 1024);
    else                         snprintf(buf, 32, "%zuMB", s / (1024 * 1024));
    return buf;
}

// Rank pairing. With 2N ranks laid out N per node, rank r pairs with
// r + N, so every pair crosses the node boundary; the first half send and
// the second half receive. N pairs push concurrently, which is what puts
// the NIC under contention -- a different regime from a lone pair, and the
// point of running more than one.
static int g_half = 1;
static inline bool is_sender(int rank) { return rank < g_half; }
static inline int  peer_of(int rank, int nranks) { return (rank + g_half) % nranks; }

// One timed configuration: warmup, then `samples` barrier-delimited runs of
// `body` on every sending rank, timed on rank 0. All ranks enter every
// barrier, so the senders stay in step and the measured window contains the
// concurrent phase rather than one rank's private one.
template <typename F>
static Stats time_config(gicc::Runtime& rt, int rank, int samples, int warmup,
                         F&& body) {
    for (int w = 0; w < warmup; ++w) {
        rt.barrier();
        if (is_sender(rank)) body();
        rt.barrier();
    }
    std::vector<double> v;
    v.reserve(samples);
    for (int s = 0; s < samples; ++s) {
        rt.barrier();
        double t0 = MPI_Wtime();
        if (is_sender(rank)) body();
        double t1 = MPI_Wtime();
        rt.barrier();
        if (rank == 0) v.push_back((t1 - t0) * 1e6);
    }
    return summarize(std::move(v));
}

static void emit(const char* exp, const char* path, const char* config,
                 size_t bytes, long param, int ops_per_sample, const Stats& st) {
    if (st.n == 0) return;
    char sb[32];
    fmt_size(bytes, sb);
    double per_op = ops_per_sample > 0 ? st.median / ops_per_sample : st.median;
    printf("CSV,%s,%s,%s,%zu,%ld,%d,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f\n",
           exp, path, config, bytes, param, st.n,
           st.median, per_op, st.min, st.max, st.p25, st.p75);
    printf("    %-24s %8s param=%-6ld  median=%9.2f us  per-op=%8.3f us"
           "  [%.2f .. %.2f]\n",
           config, sb, param, st.median, per_op, st.min, st.max);
    fflush(stdout);
}

//----------------------------------------------------------------------------
// Correctness: one transfer through the active path, checked by the peer.
//----------------------------------------------------------------------------
static bool verify_path(gicc::Runtime& rt, int rank, int peer,
                        const gicc::Buffer& bh, void* d_buf, size_t bytes,
                        bool is_proxy, uint8_t pat) {
    size_t chk = std::min<size_t>(bytes, 4096);
    if (is_sender(rank)) (void)hipMemset(d_buf, pat, std::max<size_t>(bytes, 64));
    else                 (void)hipMemset(d_buf, 0x00, std::max<size_t>(bytes, 64));
    (void)hipDeviceSynchronize();
    rt.barrier();

    if (is_sender(rank)) {
        gicc::DeviceCtx* ctx;
        if (is_proxy) {
            ctx = rt.prepare();
            hipLaunchKernelGGL(k_proxy_loop, dim3(1), dim3(1), 0, 0,
                               ctx, peer, bh.index, bytes, 1, 0);
        } else {
            rt.put(bh, peer, bh.index, bytes, 0, 0);
            ctx = rt.prepare();
            hipLaunchKernelGGL(k_flush_n, dim3(1), dim3(1), 0, 0, ctx, 1);
        }
        (void)hipDeviceSynchronize();
        rt.reset();
    }
    rt.barrier();

    int ok = 1;
    if (!is_sender(rank)) {
        std::vector<uint8_t> h(chk);
        (void)hipMemcpy(h.data(), d_buf, chk, hipMemcpyDeviceToHost);
        for (size_t i = 0; i < chk; ++i) {
            if (h[i] != pat) { ok = 0; break; }
        }
    }
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return all_ok != 0;
}

// The grid's proxy configurations push from several blocks at once, each
// against its own lane. That is a different device code path from the
// single-block kernel above (concurrent atomic_push, per-lane quiet), so it
// gets its own delivery check before any of it is timed.
static bool verify_multiblock(gicc::Runtime& rt, int rank, int peer,
                              const gicc::Buffer& bh, void* d_buf,
                              int blocks, int ops, uint8_t pat) {
    const size_t bytes = 4096;
    if (is_sender(rank)) (void)hipMemset(d_buf, pat, bytes);
    else                 (void)hipMemset(d_buf, 0x00, bytes);
    (void)hipDeviceSynchronize();
    rt.barrier();

    if (is_sender(rank)) {
        gicc::DeviceCtx* ctx = rt.prepare();
        hipLaunchKernelGGL(k_proxy_blocks_spin, dim3(blocks), dim3(1), 0, 0,
                           ctx, peer, bh.index, bytes, ops, /*ticks=*/0);
        (void)hipDeviceSynchronize();
        rt.reset();
    }
    rt.barrier();

    int ok = 1;
    if (!is_sender(rank)) {
        std::vector<uint8_t> h(bytes);
        (void)hipMemcpy(h.data(), d_buf, bytes, hipMemcpyDeviceToHost);
        for (size_t i = 0; i < bytes; ++i) {
            if (h[i] != pat) { ok = 0; break; }
        }
    }
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return all_ok != 0;
}

//----------------------------------------------------------------------------
// O1 — descriptor reuse / staging hoist
//----------------------------------------------------------------------------
static void run_reuse(gicc::Runtime& rt, int rank, int peer,
                      const gicc::Buffer& bh, const std::vector<size_t>& sizes,
                      const std::vector<int>& Ns, bool is_proxy,
                      int samples, int warmup, double max_stage_bytes) {
    const char* path = is_proxy ? "proxy" : "trigger";

    for (size_t bytes : sizes) {
        for (int N : Ns) {
            // Pre-staging a whole loop puts N transfers in flight at once.
            // Past a few hundred MB the provider's deferred work queue
            // refuses new work, so skip the point for every config rather
            // than compare a staged config against one that could not run.
            if (max_stage_bytes > 0 && (double)bytes * N > max_stage_bytes) {
                if (rank == 0)
                    printf("    (skip bytes=%zu N=%d: %.0f MB in flight "
                           "exceeds the pre-stage budget)\n",
                           bytes, N, (double)bytes * N / (1024 * 1024));
                continue;
            }
            // Launch-structure baselines, no communication. Every timed
            // config below is (launch structure) + (communication), so the
            // matching baseline isolates the communication cost:
            //   restage / hoist / proxy-perlaunch -> base-perlaunch
            //   hoist-fused / proxy-fused         -> base-fused
            emit("reuse", path, "base-perlaunch", bytes, N, N,
                 time_config(rt, rank, samples, warmup, [&] {
                     for (int i = 0; i < N; ++i)
                         hipLaunchKernelGGL(k_empty, dim3(1), dim3(1), 0, 0);
                     (void)hipDeviceSynchronize();
                 }));
            emit("reuse", path, "base-fused", bytes, N, N,
                 time_config(rt, rank, samples, warmup, [&] {
                     hipLaunchKernelGGL(k_empty, dim3(1), dim3(1), 0, 0);
                     (void)hipDeviceSynchronize();
                 }));

            if (is_proxy) {
                // One kernel issues the whole loop; the host is not involved.
                emit("reuse", path, "proxy-fused", bytes, N, N,
                     time_config(rt, rank, samples, warmup, [&] {
                         gicc::DeviceCtx* ctx = rt.prepare();
                         hipLaunchKernelGGL(k_proxy_loop, dim3(1), dim3(1), 0, 0,
                                            ctx, peer, bh.index, bytes, N, 0);
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));

                // One kernel, but each iteration waits for its own transfer
                // (device-side quiet per op) instead of one drain at the end.
                emit("reuse", path, "proxy-fused-perquiet", bytes, N, N,
                     time_config(rt, rank, samples, warmup, [&] {
                         gicc::DeviceCtx* ctx = rt.prepare();
                         hipLaunchKernelGGL(k_proxy_loop, dim3(1), dim3(1), 0, 0,
                                            ctx, peer, bh.index, bytes, N, 1);
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));

                // One launch per iteration, one push each: same launch
                // structure as the trigger configs below.
                emit("reuse", path, "proxy-perlaunch", bytes, N, N,
                     time_config(rt, rank, samples, warmup, [&] {
                         for (int i = 0; i < N; ++i) {
                             gicc::DeviceCtx* ctx = rt.prepare();
                             hipLaunchKernelGGL(k_proxy_loop, dim3(1), dim3(1),
                                                0, 0, ctx, peer, bh.index,
                                                bytes, 1, 0);
                         }
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));
            } else {
                // Context-free lowering: the descriptor is staged inside the
                // loop because the runtime only ever sees the current put.
                emit("reuse", path, "trigger-restage", bytes, N, N,
                     time_config(rt, rank, samples, warmup, [&] {
                         for (int i = 0; i < N; ++i) {
                             rt.put(bh, peer, bh.index, bytes, 0, 0);
                             gicc::DeviceCtx* ctx = rt.prepare();
                             hipLaunchKernelGGL(k_flush_n, dim3(1), dim3(1),
                                                0, 0, ctx, 1);
                         }
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));

                // Staging hoisted out of the loop, launch structure unchanged.
                // Isolates the hoist from the launch-fusion effect below.
                emit("reuse", path, "trigger-hoist", bytes, N, N,
                     time_config(rt, rank, samples, warmup, [&] {
                         for (int i = 0; i < N; ++i)
                             rt.put(bh, peer, bh.index, bytes, 0, 0);
                         for (int i = 0; i < N; ++i) {
                             gicc::DeviceCtx* ctx = rt.prepare_delta(1, 1);
                             hipLaunchKernelGGL(k_flush_n, dim3(1), dim3(1),
                                                0, 0, ctx, 1);
                         }
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));

                // Full compiler form: stage the loop once, then one kernel
                // releases the descriptors one per iteration.
                emit("reuse", path, "trigger-hoist-fused", bytes, N, N,
                     time_config(rt, rank, samples, warmup, [&] {
                         for (int i = 0; i < N; ++i)
                             rt.put(bh, peer, bh.index, bytes, 0, 0);
                         gicc::DeviceCtx* ctx = rt.prepare_delta(1, N);
                         hipLaunchKernelGGL(k_flush_n, dim3(1), dim3(1), 0, 0,
                                            ctx, N);
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));

                // Same, with the redundant per-trigger fence dropped.
                emit("reuse", path, "trigger-hoist-fused-nf", bytes, N, N,
                     time_config(rt, rank, samples, warmup, [&] {
                         for (int i = 0; i < N; ++i)
                             rt.put(bh, peer, bh.index, bytes, 0, 0);
                         gicc::DeviceCtx* ctx = rt.prepare_delta(1, N);
                         hipLaunchKernelGGL(k_flush_n_nofence, dim3(1), dim3(1),
                                            0, 0, ctx, N);
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));

                // Aggregated: the loop's N transfers released by ONE trigger.
                // Legal only when no iteration consumes its transfer before
                // the loop ends — a compiler-visible dependence property.
                emit("reuse", path, "trigger-hoist-batched", bytes, N, N,
                     time_config(rt, rank, samples, warmup, [&] {
                         for (int i = 0; i < N; ++i)
                             rt.put(bh, peer, bh.index, bytes, 0, 0);
                         gicc::DeviceCtx* ctx = rt.prepare_delta(N, 1);
                         hipLaunchKernelGGL(k_flush_n, dim3(1), dim3(1), 0, 0,
                                            ctx, 1);
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));
            }
        }
    }
}

//----------------------------------------------------------------------------
// Phase decomposition — where does a staged trigger phase actually spend
// its time, and which component carries the run-to-run variance?
//   stage  N x rt.put()          host descriptor enqueue (fi_control)
//   fire   prepare + launch + device sync
//   wait   rt.reset()            completion counter + builder recycle
//----------------------------------------------------------------------------
static void run_decompose(gicc::Runtime& rt, int rank, int peer,
                          const gicc::Buffer& bh, const std::vector<size_t>& sizes,
                          const std::vector<int>& Ns, bool is_proxy,
                          int samples, int warmup, double max_stage_bytes) {
    if (is_proxy) return;   // staging is a host-side concept
    for (size_t bytes : sizes) {
        for (int N : Ns) {
            if (max_stage_bytes > 0 && (double)bytes * N > max_stage_bytes) continue;
            std::vector<double> st, fi, wa, tot;
            for (int s = 0; s < samples + warmup; ++s) {
                rt.barrier();
                double t0 = MPI_Wtime();
                if (rank == 0) {
                    for (int i = 0; i < N; ++i)
                        rt.put(bh, peer, bh.index, bytes, 0, 0);
                }
                double t1 = MPI_Wtime();
                if (rank == 0) {
                    gicc::DeviceCtx* ctx = rt.prepare_delta(N, 1);
                    hipLaunchKernelGGL(k_flush_n, dim3(1), dim3(1), 0, 0, ctx, 1);
                    (void)hipDeviceSynchronize();
                }
                double t2 = MPI_Wtime();
                if (rank == 0) rt.reset();
                double t3 = MPI_Wtime();
                rt.barrier();
                if (rank == 0 && s >= warmup) {
                    st.push_back((t1-t0)*1e6); fi.push_back((t2-t1)*1e6);
                    wa.push_back((t3-t2)*1e6); tot.push_back((t3-t0)*1e6);
                }
            }
            emit("decompose", "trigger", "stage", bytes, N, N, summarize(st));
            emit("decompose", "trigger", "fire",  bytes, N, N, summarize(fi));
            emit("decompose", "trigger", "wait",  bytes, N, N, summarize(wa));
            emit("decompose", "trigger", "total", bytes, N, N, summarize(tot));
        }
    }
}

//----------------------------------------------------------------------------
// O2 — batching (ops per trigger)
//----------------------------------------------------------------------------
static void run_batch(gicc::Runtime& rt, int rank, int peer,
                      const gicc::Buffer& bh, const std::vector<size_t>& sizes,
                      const std::vector<int>& Bs, int total_ops, bool is_proxy,
                      int samples, int warmup) {
    const char* path = is_proxy ? "proxy" : "trigger";

    for (size_t bytes : sizes) {
        for (int B : Bs) {
            if (B > total_ops) continue;
            const int groups = total_ops / B;
            const int ops    = groups * B;

            if (is_proxy) {
                emit("batch", path, "proxy", bytes, B, ops,
                     time_config(rt, rank, samples, warmup, [&] {
                         gicc::DeviceCtx* ctx = rt.prepare();
                         hipLaunchKernelGGL(k_proxy_loop, dim3(1), dim3(1), 0, 0,
                                            ctx, peer, bh.index, bytes, ops, B);
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));
            } else {
                emit("batch", path, "trigger", bytes, B, ops,
                     time_config(rt, rank, samples, warmup, [&] {
                         for (int g = 0; g < groups; ++g) {
                             for (int i = 0; i < B; ++i)
                                 rt.put(bh, peer, bh.index, bytes, 0, 0);
                             gicc::DeviceCtx* ctx = rt.prepare();
                             hipLaunchKernelGGL(k_flush_n, dim3(1), dim3(1),
                                                0, 0, ctx, 1);
                         }
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));
            }
        }
    }
}

//----------------------------------------------------------------------------
// O3 — issue-to-first-use distance
//----------------------------------------------------------------------------
static void run_distance(gicc::Runtime& rt, int rank, int peer,
                         const gicc::Buffer& bh, const std::vector<size_t>& sizes,
                         const std::vector<int>& Ds, double ticks_per_us,
                         const std::vector<int>& Ks, bool is_proxy,
                         int samples, int warmup) {
    const char* path = is_proxy ? "proxy" : "trigger";

    for (int ops : Ks) {
    for (int D : Ds) {
        const long long ticks = (long long)(D * ticks_per_us);

        // No-communication baseline for this D. Same kernel structure, so
        // exposed comm cost = (comm config) - (this).
        char spin_cfg[48];
        snprintf(spin_cfg, sizeof(spin_cfg), "spin-only-k%d", ops);
        emit("distance", path, spin_cfg, 0, D, 1,
             time_config(rt, rank, samples, warmup, [&] {
                 hipLaunchKernelGGL(k_spin, dim3(1), dim3(1), 0, 0, ticks);
                 (void)hipDeviceSynchronize();
             }));

        for (size_t bytes : sizes) {
            if (is_proxy) {
                char cfg[48];
                snprintf(cfg, sizeof(cfg), "proxy-k%d", ops);
                emit("distance", path, cfg, bytes, D, 1,
                     time_config(rt, rank, samples, warmup, [&] {
                         gicc::DeviceCtx* ctx = rt.prepare();
                         hipLaunchKernelGGL(k_proxy_put_spin_quiet, dim3(1),
                                            dim3(1), 0, 0, ctx, peer, bh.index,
                                            bytes, ops, ticks);
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));
            } else {
                char cfg[48];
                snprintf(cfg, sizeof(cfg), "trigger-k%d", ops);
                emit("distance", path, cfg, bytes, D, 1,
                     time_config(rt, rank, samples, warmup, [&] {
                         for (int i = 0; i < ops; ++i)
                             rt.put(bh, peer, bh.index, bytes, 0, 0);
                         gicc::DeviceCtx* ctx = rt.prepare();
                         hipLaunchKernelGGL(k_flush_spin, dim3(1), dim3(1),
                                            0, 0, ctx, ticks);
                         (void)hipDeviceSynchronize();
                         rt.reset();
                     }));
            }
        }
    }
    }
}

//----------------------------------------------------------------------------
// Joint configuration space — one row per (workload point, configuration).
//----------------------------------------------------------------------------

static void emit_grid(const char* path, const char* config, size_t bytes,
                      int K, int D, int B, int P, int L, const Stats& st) {
    if (st.n == 0) return;
    // `pairs` is appended last so CSVs collected before it existed still
    // parse; a reader that finds the field missing should assume one pair.
    printf("GRID,%s,%s,%zu,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
           path, config, bytes, K, D, B, P, L,
           st.n, st.median, st.min, st.max, st.p25, st.p75, g_half);
    char sb[32];
    fmt_size(bytes, sb);
    printf("    %-16s %8s K=%-3d D=%-4d  median=%9.2f us  [%.2f .. %.2f]\n",
           config, sb, K, D, st.median, st.min, st.max);
    fflush(stdout);
}

static void run_grid(gicc::Runtime& rt, int rank, int peer,
                     const gicc::Buffer& bh, const std::vector<size_t>& sizes,
                     const std::vector<int>& Ks, const std::vector<int>& Ds,
                     const std::vector<int>& Bs, const std::vector<int>& Ps,
                     int lanes, double ticks_per_us, bool is_proxy,
                     int samples, int warmup, double max_stage_bytes) {
    const char* path = is_proxy ? "proxy" : "trigger";

    for (int D : Ds) {
        const long long ticks = (long long)(D * ticks_per_us);

        // No-communication baseline for this D, same launch structure.
        // Exposed communication cost = (cell median) - (this).
        emit_grid(path, "spin-only", 0, 0, D, 0, 1, lanes,
                  time_config(rt, rank, samples, warmup, [&] {
                      hipLaunchKernelGGL(k_spin, dim3(1), dim3(1), 0, 0, ticks);
                      (void)hipDeviceSynchronize();
                  }));

        for (size_t bytes : sizes) {
            for (int K : Ks) {
                if (max_stage_bytes > 0 && (double)bytes * K > max_stage_bytes) {
                    if (rank == 0)
                        printf("    (skip bytes=%zu K=%d: %.0f MB in flight "
                               "exceeds the pre-stage budget)\n",
                               bytes, K, (double)bytes * K / (1024 * 1024));
                    continue;
                }

                if (is_proxy) {
                    // Knob: how many blocks push concurrently. Splitting K
                    // across blocks spreads the ring CAS contention and, when
                    // P <= L, spreads the work over distinct proxy workers.
                    for (int P : Ps) {
                        if (P > K) continue;
                        char cfg[32];
                        snprintf(cfg, sizeof(cfg), "proxy-P%d-L%d", P, lanes);
                        emit_grid(path, cfg, bytes, K, D, 0, P, lanes,
                                  time_config(rt, rank, samples, warmup, [&] {
                                      gicc::DeviceCtx* ctx = rt.prepare();
                                      hipLaunchKernelGGL(k_proxy_blocks_spin,
                                                         dim3(P), dim3(1), 0, 0,
                                                         ctx, peer, bh.index,
                                                         bytes, K, ticks);
                                      (void)hipDeviceSynchronize();
                                      rt.reset();
                                  }));
                    }
                } else {
                    // Knob: descriptors released per MMIO trigger. Only
                    // divisors of K are measured, because prepare_delta()
                    // fires a uniform number of descriptors per trigger.
                    for (int B : Bs) {
                        if (B > K || (K % B) != 0) continue;
                        const int groups = K / B;
                        char cfg[32];
                        snprintf(cfg, sizeof(cfg), "trig-B%d", B);
                        emit_grid(path, cfg, bytes, K, D, B, 1, 0,
                                  time_config(rt, rank, samples, warmup, [&] {
                                      for (int i = 0; i < K; ++i)
                                          rt.put(bh, peer, bh.index, bytes, 0, 0);
                                      gicc::DeviceCtx* ctx =
                                          rt.prepare_delta(B, groups);
                                      hipLaunchKernelGGL(k_flush_groups_spin,
                                                         dim3(1), dim3(1), 0, 0,
                                                         ctx, groups, ticks);
                                      (void)hipDeviceSynchronize();
                                      rt.reset();
                                  }));
                    }
                }
            }
        }
    }
}

//----------------------------------------------------------------------------

static std::vector<int> parse_ints(const std::string& csv) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < csv.size()) {
        size_t comma = csv.find(',', pos);
        if (comma == std::string::npos) comma = csv.size();
        if (comma > pos) out.push_back(std::atoi(csv.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
    }
    return out;
}

static std::vector<size_t> parse_sizes(const std::string& csv) {
    std::vector<size_t> out;
    for (int v : parse_ints(csv)) if (v > 0) out.push_back((size_t)v);
    return out;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string path = "proxy";
    std::string exp  = "reuse";
    int samples = 21, warmup = 10, total_ops = 64;
    double max_stage_bytes = 256.0 * 1024 * 1024;
    std::vector<int> Ks;
    std::vector<size_t> sizes;
    std::vector<int> Ns, Bs, Ds, Ps;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--path=",    0) == 0) path = a.substr(7);
        else if (a.rfind("--exp=",     0) == 0) exp  = a.substr(6);
        else if (a.rfind("--samples=", 0) == 0) samples = std::atoi(a.c_str() + 10);
        else if (a.rfind("--warmup=",  0) == 0) warmup  = std::atoi(a.c_str() + 9);
        else if (a.rfind("--total-ops=", 0) == 0) total_ops = std::atoi(a.c_str() + 12);
        else if (a.rfind("--ops=", 0) == 0) Ks = parse_ints(a.substr(6));
        else if (a.rfind("--max-stage-mb=", 0) == 0)
            max_stage_bytes = std::atof(a.c_str() + 15) * 1024 * 1024;
        else if (a.rfind("--sizes=",   0) == 0) sizes = parse_sizes(a.substr(8));
        else if (a.rfind("--reuse=",   0) == 0) Ns = parse_ints(a.substr(8));
        else if (a.rfind("--batches=", 0) == 0) Bs = parse_ints(a.substr(10));
        else if (a.rfind("--dists=",   0) == 0) Ds = parse_ints(a.substr(8));
        else if (a.rfind("--blocks=",  0) == 0) Ps = parse_ints(a.substr(9));
        else {
            fprintf(stderr, "ctx_bench: unknown argument '%s'\n", a.c_str());
            return 2;
        }
    }
    if (path != "proxy" && path != "trigger") {
        fprintf(stderr, "ctx_bench: --path must be 'proxy' or 'trigger'\n");
        return 2;
    }
    const bool is_proxy = (path == "proxy");

    const bool is_grid = (exp == "grid");
    if (sizes.empty()) sizes = is_grid ? std::vector<size_t>{256, 4096, 65536, 1048576}
                                       : std::vector<size_t>{8, 256, 4096, 65536, 1048576};
    if (Ns.empty())    Ns    = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    if (Bs.empty())    Bs    = {1, 2, 4, 8, 16, 32, 64};
    if (Ds.empty())    Ds    = is_grid ? std::vector<int>{0, 50, 200}
                                       : std::vector<int>{0, 5, 10, 20, 50, 100, 200, 500, 1000};
    if (Ks.empty())    Ks    = is_grid ? std::vector<int>{4, 16, 64}
                                       : std::vector<int>{1};
    if (Ps.empty())    Ps    = {1, 2, 4, 8};

    gicc::Runtime rt;
    if (!is_proxy) rt.enable_host_wait_mode();
    int rank = rt.rank(), nranks = rt.size();
    if (nranks < 2 || (nranks % 2) != 0) {
        if (rank == 0)
            fprintf(stderr, "ctx_bench: need an even rank count >= 2 (got %d)\n",
                    nranks);
        return 1;
    }
    g_half = nranks / 2;
    int peer = peer_of(rank, nranks);

    void* d_buf = nullptr;
    if (hipMalloc(&d_buf, kBufBytes) != hipSuccess) {
        fprintf(stderr, "rank %d: hipMalloc failed\n", rank);
        return 3;
    }
    (void)hipMemset(d_buf, 0xAB, kBufBytes);
    (void)hipDeviceSynchronize();

    auto bh = rt.register_buffer(d_buf, kBufBytes, /*is_device=*/true);
    rt.exchange();

    // Force lazy init (proxy fleet spawn / DWQ context) outside the timers.
    (void)rt.prepare();
    rt.reset();
    rt.barrier();

    // wall_clock64() rate: query the device, then confirm against the host.
    int wall_khz = 0;
    (void)hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0);
    double ticks_per_us = wall_khz > 0 ? wall_khz / 1000.0 : 100.0;
    if (rank == 0) {
        const long long probe = (long long)(ticks_per_us * 2000.0);  // ~2 ms
        hipLaunchKernelGGL(k_spin, dim3(1), dim3(1), 0, 0, probe);
        (void)hipDeviceSynchronize();
        double t0 = MPI_Wtime();
        hipLaunchKernelGGL(k_spin, dim3(1), dim3(1), 0, 0, probe);
        (void)hipDeviceSynchronize();
        double measured_us = (MPI_Wtime() - t0) * 1e6;
        printf("[calib] wallClockRate=%d kHz -> %.3f ticks/us; %lld ticks "
               "measured %.1f us (implied %.3f ticks/us)\n",
               wall_khz, ticks_per_us, probe, measured_us,
               measured_us > 0 ? probe / measured_us : 0.0);
    }
    MPI_Bcast(&ticks_per_us, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    bool ok = verify_path(rt, rank, peer, bh, d_buf, 4096, is_proxy, 0x5A);
    if (rank == 0) {
        printf("=== ctx_bench path=%s exp=%s samples=%d warmup=%d "
               "pairs=%d ===\n",
               path.c_str(), exp.c_str(), samples, warmup, g_half);
        printf("[verify] 4KB transfer through the %s path: %s\n",
               path.c_str(), ok ? "PASS" : "FAIL");
        printf("CSV,exp,path,config,bytes,param,samples,median_us,per_op_us,"
               "min_us,max_us,p25_us,p75_us\n");
    }
    if (!ok) {
        if (rank == 0) fprintf(stderr, "ctx_bench: verification failed, aborting\n");
        return 4;
    }

    if (exp == "reuse" || exp == "all")
        run_reuse(rt, rank, peer, bh, sizes, Ns, is_proxy, samples, warmup,
                  max_stage_bytes);
    if (exp == "batch" || exp == "all")
        run_batch(rt, rank, peer, bh, sizes, Bs, total_ops, is_proxy, samples, warmup);
    if (exp == "decompose" || exp == "all")
        run_decompose(rt, rank, peer, bh, sizes, Ns, is_proxy, samples, warmup,
                      max_stage_bytes);
    if (exp == "distance" || exp == "all")
        run_distance(rt, rank, peer, bh, sizes, Ds, ticks_per_us, Ks,
                     is_proxy, samples, warmup);
    if (exp == "lanes") {
        int fleet = 1;
        if (const char* e = std::getenv("GICC_NUM_PROXY_THREADS")) {
            int v = std::atoi(e);
            if (v >= 1 && v <= 32) fleet = v;
        }
        if (rank == 0)
            printf("LANES,bytes,ops,lanes,fleet,median_us,min_us\n");
        for (size_t bytes : sizes) {
            for (int ops : Ks) {
                for (int L : Ps) {
                    if (L > ops) continue;
                    Stats st = time_config(rt, rank, samples, warmup, [&] {
                        gicc::DeviceCtx* c = rt.prepare();
                        hipLaunchKernelGGL(k_proxy_lanes_spin, dim3(1), dim3(1),
                                           0, 0, c, peer, bh.index, bytes, ops,
                                           L, (long long)0);
                        (void)hipDeviceSynchronize();
                        rt.reset();
                    });
                    if (rank == 0 && st.n)
                        printf("LANES,%zu,%d,%d,%d,%.3f,%.3f\n",
                               bytes, ops, L, fleet, st.median, st.min);
                }
            }
        }
    }
    if (exp == "grid") {
        int lanes = 1;
        if (const char* e = std::getenv("GICC_NUM_PROXY_THREADS")) {
            int v = std::atoi(e);
            if (v >= 1 && v <= 32) lanes = v;
        }
        if (is_proxy) {
            bool mok = verify_multiblock(rt, rank, peer, bh, d_buf,
                                         /*blocks=*/8, /*ops=*/24, 0x3C);
            if (rank == 0)
                printf("[verify] 8-block / 24-op concurrent push: %s\n",
                       mok ? "PASS" : "FAIL");
            if (!mok) {
                if (rank == 0)
                    fprintf(stderr, "ctx_bench: multi-block delivery failed\n");
                return 4;
            }
        }
        if (rank == 0)
            printf("GRID,path,config,bytes,K,D,B,P,L,samples,median_us,"
                   "min_us,max_us,p25_us,p75_us,pairs\n");
        run_grid(rt, rank, peer, bh, sizes, Ks, Ds, Bs, Ps, lanes,
                 ticks_per_us, is_proxy, samples, warmup, max_stage_bytes);
    }

    rt.barrier();
    if (rank == 0) printf("=== done ===\n");
    (void)hipFree(d_buf);
    return 0;
}
