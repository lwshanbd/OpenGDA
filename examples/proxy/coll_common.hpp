/*
 * coll_common.hpp - shared device kernels + host driver for the GICC
 * collective examples (allreduce_ring.cpp, alltoall.cpp).
 *
 * The whole point of these examples is that ONE source compiles to BOTH
 * transports, selected at compile time exactly like the rest of GICC:
 *
 *   - CPU proxy  (-DGICC_CPU_PROXY): the device kernel pushes each RMA
 *     write into the proxy ring; gicc::quiet() spins until the CPU worker
 *     acks the libfabric CQE. Host-side rt.reset() drains the ring.
 *
 *   - DWQ / GPU-trigger (no -DGICC_CPU_PROXY + rt.enable_host_wait_mode()):
 *     the host pre-stages every RMA write via rt.put() into the CXI
 *     deferred work queue; the kernel's lead thread fires them all with a
 *     single MMIO trigger store (gicc::flush). Host-side rt.reset() spins
 *     on the shared completion counter.
 *
 * Both transports share the same completion model at the call site:
 *   issue puts -> hipDeviceSynchronize() -> rt.reset() -> rt.barrier().
 * After the barrier every rank's RMA writes have landed remotely, so a
 * neighbour may safely read what was written into its buffer. This is the
 * same proven pattern minimod and ASF use.
 */
#pragma once

#include <hip/hip_runtime.h>
#ifdef GICC_CPU_PROXY
#include <hip/hip_cooperative_groups.h>
#endif

#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

namespace gicc_coll {

//----------------------------------------------------------------------------
// Optional watchdog instrumentation (-DGICC_COLL_DEBUG). A monitor thread
// prints, per rank, the last checkpoint reached and a heartbeat counter; if
// the heartbeat stops advancing the offending call site is named. Zero cost
// when GICC_COLL_DEBUG is not defined.
//----------------------------------------------------------------------------
#ifdef GICC_COLL_DEBUG
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdio>
inline std::atomic<long>& dbg_beat() { static std::atomic<long> b{0}; return b; }
inline std::atomic<int>&  dbg_step() { static std::atomic<int>  s{-1}; return s; }
inline const char*&       dbg_stage(){ static const char* s = "init"; return s; }
#define COLL_CKPT(stg, stp) do {                       \
        gicc_coll::dbg_stage() = (stg);                \
        gicc_coll::dbg_step().store((stp));            \
        gicc_coll::dbg_beat().fetch_add(1);            \
    } while (0)
inline void start_watchdog(int rank, int period_s = 2) {
    std::thread([rank, period_s] {
        long last = -1; int stuck = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(period_s));
            long b = gicc_coll::dbg_beat().load();
            if (b == last) {
                stuck += period_s;
                fprintf(stderr,
                    "[WATCHDOG] rank %d STUCK %ds  stage='%s' step=%d beat=%ld\n",
                    rank, stuck, gicc_coll::dbg_stage(),
                    gicc_coll::dbg_step().load(), b);
            } else {
                stuck = 0;
            }
            last = b;
        }
    }).detach();
}
#else
#define COLL_CKPT(stg, stp) ((void)0)
inline void start_watchdog(int /*rank*/, int /*period_s*/ = 2) {}
#endif

#ifdef GICC_CPU_PROXY
// Proxy mode: lead thread pushes one RMA write into the proxy ring, then
// quiet() blocks until the worker reports the CQE (remote write complete).
__global__ void proxy_put_kernel(gicc::DeviceCtx* ctx, int peer,
                                 int dst_buf, size_t dst_off,
                                 int src_buf, size_t src_off, size_t bytes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::put(ctx, peer, dst_buf, dst_off, src_buf, src_off, bytes);
        gicc::quiet(ctx);
    }
}
#endif

#ifdef GICC_CPU_PROXY
// Proxy mode all-to-all: one kernel issues every remote put then a single
// quiet(). Each rank writes its send[j] into peer j's recv[rank].
__global__ void alltoall_put_kernel(gicc::DeviceCtx* ctx, int N, int rank,
                                    int recv_buf, int send_buf,
                                    size_t chunk_bytes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int j = 0; j < N; ++j) {
            if (j == rank) continue;
            gicc::put(ctx, j,
                      recv_buf, (size_t)rank * chunk_bytes,   // peer's recv[rank]
                      send_buf, (size_t)j * chunk_bytes,      // my send[j]
                      chunk_bytes);
        }
        gicc::quiet(ctx);
    }
}
#endif

// DWQ mode: lead thread writes the trigger MMIO once, firing every RMA
// write the host pre-staged via rt.put() before the launch. Compiled in
// both builds (gicc::flush exists in both); only launched on the DWQ path.
__global__ void dwq_flush_kernel(gicc::DeviceCtx* ctx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::flush(ctx);
    }
}

// add_kernel - elementwise data[dst_off + i] += recv[i] for the ring
// reduce-scatter phase. Plain HIP kernel, no GICC involvement.
__global__ void add_kernel(float* data, const float* recv,
                           size_t dst_off, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[dst_off + i] += recv[i];
}

//----------------------------------------------------------------------------
// drain_with_progress - wait for the local device work (flush kernel on the
// null stream + any same-node IPC copy on the IPC stream) to finish WHILE
// pumping the libfabric CQ. Required in DWQ mode: a rank receiving an
// incoming cross-node RMA write must service it (no proxy worker thread
// exists), or its FI_HMEM DMA stalls the SDMA engine and a concurrent
// outgoing IPC gpuMemcpyAsync (>=~32KB) deadlocks. See Runtime::progress().
inline void drain_with_progress(gicc::Runtime& rt) {
    hipStream_t ipc = rt.ipc_stream0();
    for (;;) {
        rt.progress();
        bool busy = (hipStreamQuery(0) == hipErrorNotReady);
        if (ipc && hipStreamQuery(ipc) == hipErrorNotReady) busy = true;
        if (!busy) break;
    }
}

// put_one - issue a single RMA write (this rank -> peer), wait for it to
// complete remotely, and barrier so the result is visible to everyone.
//
// Identical call site for both transports; the #ifdef only changes HOW the
// write is launched. After return: my write has landed on `peer`, and
// (because every rank ran the symmetric step) my own buffer holds whatever
// my upstream neighbour wrote into it.
//----------------------------------------------------------------------------
inline void put_one(gicc::Runtime& rt, int peer,
                    const gicc::Buffer& dst_buf, size_t dst_off,
                    const gicc::Buffer& src_buf, size_t src_off,
                    size_t bytes) {
#ifdef GICC_CPU_PROXY
    gicc::DeviceCtx* d = rt.prepare();
    hipLaunchKernelGGL(proxy_put_kernel, dim3(1), dim3(1), 0, 0,
                       d, peer, dst_buf.index, dst_off,
                       src_buf.index, src_off, bytes);
    COLL_CKPT("proxy:devsync", peer);
    (void)hipDeviceSynchronize();
    COLL_CKPT("proxy:reset", peer);
    rt.reset();
#else
    COLL_CKPT("dwq:rt.put", peer);
    rt.put(src_buf, peer, dst_buf.index, bytes, src_off, dst_off);
    COLL_CKPT("dwq:prepare", peer);
    gicc::DeviceCtx* d = rt.prepare();
    COLL_CKPT("dwq:launch", peer);
    hipLaunchKernelGGL(dwq_flush_kernel, dim3(1), dim3(1), 0, 0, d);
    // CRITICAL: drain the device with libfabric progress interleaved, NOT a
    // bare hipDeviceSynchronize. In DWQ mode there is no proxy worker thread,
    // so if this rank is RECEIVING an incoming cross-node RMA write (even
    // though it only SENT via same-node IPC) it must pump the CQ here, or that
    // write's FI_HMEM DMA stalls the SDMA engine and our own outgoing IPC
    // gpuMemcpyAsync (>=~32KB) deadlocks. See Runtime::progress().
    COLL_CKPT("dwq:devsync", peer);
    drain_with_progress(rt);
    COLL_CKPT("dwq:reset", peer);
    rt.reset();
#endif
    COLL_CKPT("barrier", peer);
    rt.barrier();
    COLL_CKPT("put_one:done", peer);
}

//----------------------------------------------------------------------------
// ring_allreduce - sum-reduce `N*elems_per_chunk` floats in d_data in place.
// d_recv / recv_buf is a one-chunk scratch buffer. Buffers must already be
// registered + exchanged. Works identically under both transports.
//----------------------------------------------------------------------------
inline void ring_allreduce(gicc::Runtime& rt,
                           const gicc::Buffer& data_buf, float* d_data,
                           const gicc::Buffer& recv_buf, float* d_recv,
                           int elems_per_chunk) {
    const int    N           = rt.size();
    const int    rank        = rt.rank();
    const size_t chunk_bytes = (size_t)elems_per_chunk * sizeof(float);
    const int    next        = (rank + 1) % N;
    const int    threads     = 256;
    const int    blocks      = (elems_per_chunk + threads - 1) / threads;
    (void)d_recv;

    for (int s = 0; s < N - 1; ++s) {
        const int send_chunk = (rank - s + N) % N;
        const int recv_chunk = (rank - 1 - s + N) % N;
        COLL_CKPT("RS:put_one", s);
        put_one(rt, next, recv_buf, 0, data_buf,
                (size_t)send_chunk * chunk_bytes, chunk_bytes);
        COLL_CKPT("RS:add", s);
        hipLaunchKernelGGL(add_kernel, dim3(blocks), dim3(threads), 0, 0,
                           d_data, d_recv,
                           (size_t)recv_chunk * elems_per_chunk, elems_per_chunk);
        (void)hipDeviceSynchronize();
        COLL_CKPT("RS:barrier", s);
        rt.barrier();
    }
    for (int s = 0; s < N - 1; ++s) {
        const int send_chunk = (rank + 1 - s + 2 * N) % N;
        COLL_CKPT("AG:put_one", s);
        put_one(rt, next, data_buf, (size_t)send_chunk * chunk_bytes,
                data_buf, (size_t)send_chunk * chunk_bytes, chunk_bytes);
    }
}

//----------------------------------------------------------------------------
// alltoall_run - exchange `elems_per_chunk` ints to every peer. send[j] ->
// peer j's recv[rank]; local self-chunk copied directly. Single round.
//----------------------------------------------------------------------------
inline void alltoall_run(gicc::Runtime& rt,
                         const gicc::Buffer& send_buf, int* d_send,
                         const gicc::Buffer& recv_buf, int* d_recv,
                         int elems_per_chunk) {
    const int    N           = rt.size();
    const int    rank        = rt.rank();
    const size_t chunk_bytes = (size_t)elems_per_chunk * sizeof(int);

    (void)hipMemcpy(d_recv + (size_t)rank * elems_per_chunk,
                    d_send + (size_t)rank * elems_per_chunk,
                    chunk_bytes, hipMemcpyDeviceToDevice);
#ifdef GICC_CPU_PROXY
    gicc::DeviceCtx* d = rt.prepare();
    hipLaunchKernelGGL(alltoall_put_kernel, dim3(1), dim3(1), 0, 0,
                       d, N, rank, recv_buf.index, send_buf.index, chunk_bytes);
    (void)hipDeviceSynchronize();
    rt.reset();
#else
    for (int j = 0; j < N; ++j) {
        if (j == rank) continue;
        rt.put(send_buf, j, recv_buf.index, chunk_bytes,
               (size_t)j * chunk_bytes, (size_t)rank * chunk_bytes);
    }
    gicc::DeviceCtx* d = rt.prepare();
    hipLaunchKernelGGL(dwq_flush_kernel, dim3(1), dim3(1), 0, 0, d);
    drain_with_progress(rt);   // pump CQ so incoming RMA can't deadlock our IPC
    rt.reset();
#endif
    rt.barrier();
}

#ifdef GICC_CPU_PROXY
//============================================================================
// FUSED proxy ring all-reduce. Each phase runs entirely inside ONE kernel:
// the N-1 dependent steps are sequenced on-device (device put/quiet + a
// neighbour-set flag), so there is NO per-step host sync / barrier — only a
// single host barrier between the reduce-scatter and all-gather phases.
//
// Synchronisation: after the lead thread's put lands remotely (quiet = CQE),
// it writes a flag into the SAME neighbour's flag buffer. The downstream rank
// spins on its flag[s] (host-pinned, so the NIC write is PCIe-coherent for the
// GPU) before consuming the chunk. Flags are zeroed + barriered before launch.
//
// Distinct recv slot + distinct flag index per step => no buffer reuse race,
// so a one-directional forward signal is sufficient (no back-ACK needed).
//============================================================================

// Reduce-scatter: send chunk to next.recv[s], flag it, wait my flag[s], add.
__global__ void rs_fused_kernel(gicc::DeviceCtx* ctx,
                                int data_idx, int recv_idx, int flag_idx,
                                int one_idx, int N, int rank, int chunk,
                                float* data, const float* recv,
                                volatile unsigned int* flag) {
    const int    next = (rank + 1) % N;
    const int    tid  = threadIdx.x, nt = blockDim.x;
    const size_t cb   = (size_t)chunk * sizeof(float);
    for (int s = 0; s < N - 1; ++s) {
        const int sc = (rank - s + N) % N;
        const int rc = (rank - 1 - s + N) % N;
        if (tid == 0) {
            gicc::put(ctx, next, recv_idx, (size_t)s * cb,
                      data_idx, (size_t)sc * cb, cb);
            gicc::quiet(ctx);
            gicc::put(ctx, next, flag_idx, (size_t)s * sizeof(unsigned int),
                      one_idx, 0, sizeof(unsigned int));
            gicc::quiet(ctx);
            while (flag[s] == 0u) { __builtin_amdgcn_s_sleep(1); }
            __threadfence_system();
        }
        __syncthreads();
        for (int i = tid; i < chunk; i += nt)
            data[(size_t)rc * chunk + i] += recv[(size_t)s * chunk + i];
        __syncthreads();
    }
}

// All-gather: write finalised chunk straight into next.data[sc], flag it,
// wait my flag so the slot is filled before forwarding it next step.
__global__ void ag_fused_kernel(gicc::DeviceCtx* ctx,
                                int data_idx, int flag_idx, int one_idx,
                                int N, int rank, int chunk,
                                volatile unsigned int* flag) {
    const int    next = (rank + 1) % N;
    const size_t cb   = (size_t)chunk * sizeof(float);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int s = 0; s < N - 1; ++s) {
            const int sc = (rank + 1 - s + 2 * N) % N;
            gicc::put(ctx, next, data_idx, (size_t)sc * cb,
                      data_idx, (size_t)sc * cb, cb);
            gicc::quiet(ctx);
            gicc::put(ctx, next, flag_idx,
                      (size_t)(N - 1 + s) * sizeof(unsigned int),
                      one_idx, 0, sizeof(unsigned int));
            gicc::quiet(ctx);
            while (flag[N - 1 + s] == 0u) { __builtin_amdgcn_s_sleep(1); }
            __threadfence_system();
        }
    }
}

// Host driver for the fused proxy all-reduce. flag buffer must be host-pinned
// (registered) of >= 2*(N-1) uint; recv >= (N-1)*chunk floats; one_buf any
// registered 4-byte nonzero source.
inline void ring_allreduce_fused(gicc::Runtime& rt,
                                 const gicc::Buffer& data_buf, float* d_data,
                                 const gicc::Buffer& recv_buf, float* d_recv,
                                 const gicc::Buffer& flag_buf, unsigned int* d_flag,
                                 const gicc::Buffer& one_buf,
                                 int chunk) {
    const int N = rt.size();
    const int rank = rt.rank();

    (void)hipMemset(d_flag, 0, (size_t)2 * (N - 1) * sizeof(unsigned int));
    (void)hipDeviceSynchronize();
    rt.barrier();   // flags zeroed everywhere before any neighbour signals

    gicc::DeviceCtx* d = rt.prepare();
    hipLaunchKernelGGL(rs_fused_kernel, dim3(1), dim3(256), 0, 0,
                       d, data_buf.index, recv_buf.index, flag_buf.index,
                       one_buf.index, N, rank, chunk,
                       d_data, d_recv, (volatile unsigned int*)d_flag);
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();   // reduce-scatter fully drained before all-gather overwrites

    d = rt.prepare();
    hipLaunchKernelGGL(ag_fused_kernel, dim3(1), dim3(1), 0, 0,
                       d, data_buf.index, flag_buf.index, one_buf.index,
                       N, rank, chunk, (volatile unsigned int*)d_flag);
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();
}

//============================================================================
// COOPERATIVE fused ring all-reduce. Both phases run in ONE cooperative
// kernel (grid-wide sync between comm and the reduction), which fixes two
// limitations of rs_fused/ag_fused:
//   1. the reduction now spans the WHOLE GPU (every block/thread), instead of
//      a single block — the single-block add crippled large messages.
//   2. reduce-scatter and all-gather are fused with NO host barrier / reset /
//      relaunch between phases (one launch per all-reduce).
// The lead thread (block 0, thread 0) drives the proxy put/quiet + neighbour
// flag; all blocks grid.sync() around each comm + reduction.
//============================================================================
namespace cg = cooperative_groups;

__global__ void coop_allreduce_kernel(gicc::DeviceCtx* ctx,
                                      int data_idx, int recv_idx, int flag_idx,
                                      int one_idx, int N, int rank, int chunk,
                                      float* data, const float* recv,
                                      volatile unsigned int* flag) {
    cg::grid_group grid = cg::this_grid();
    const bool   lead = (blockIdx.x == 0 && threadIdx.x == 0);
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const int    next = (rank + 1) % N;
    const size_t cb   = (size_t)chunk * sizeof(float);

    // ---- reduce-scatter ----
    for (int s = 0; s < N - 1; ++s) {
        const int sc = (rank - s + N) % N;
        const int rc = (rank - 1 - s + N) % N;
        if (lead) {
            gicc::put(ctx, next, recv_idx, (size_t)s * cb,
                      data_idx, (size_t)sc * cb, cb);
            gicc::quiet(ctx);
            gicc::put(ctx, next, flag_idx, (size_t)s * sizeof(unsigned int),
                      one_idx, 0, sizeof(unsigned int));
            gicc::quiet(ctx);
            while (flag[s] == 0u) { __builtin_amdgcn_s_sleep(1); }
            __threadfence_system();
        }
        grid.sync();
        for (size_t i = gtid; i < (size_t)chunk; i += nthr)
            data[(size_t)rc * chunk + i] += recv[(size_t)s * chunk + i];
        grid.sync();
    }
    // ---- all-gather (same kernel, no host barrier) ----
    for (int s = 0; s < N - 1; ++s) {
        const int sc = (rank + 1 - s + 2 * N) % N;
        if (lead) {
            gicc::put(ctx, next, data_idx, (size_t)sc * cb,
                      data_idx, (size_t)sc * cb, cb);
            gicc::quiet(ctx);
            gicc::put(ctx, next, flag_idx,
                      (size_t)(N - 1 + s) * sizeof(unsigned int),
                      one_idx, 0, sizeof(unsigned int));
            gicc::quiet(ctx);
            while (flag[N - 1 + s] == 0u) { __builtin_amdgcn_s_sleep(1); }
            __threadfence_system();
        }
        grid.sync();
    }
}

// Host driver for the cooperative all-reduce. Same buffer contract as
// ring_allreduce_fused. Grid sized to full occupancy (cooperative launch
// requires every block co-resident).
inline void ring_allreduce_coop(gicc::Runtime& rt,
                                const gicc::Buffer& data_buf, float* d_data,
                                const gicc::Buffer& recv_buf, float* d_recv,
                                const gicc::Buffer& flag_buf, unsigned int* d_flag,
                                const gicc::Buffer& one_buf,
                                int chunk) {
    const int N    = rt.size();
    const int rank = rt.rank();

    static int grid_blocks = 0;
    const int  block_threads = 256;
    if (grid_blocks == 0) {
        int per_sm = 0, n_sm = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (const void*)coop_allreduce_kernel, block_threads, 0);
        (void)hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount,
                                    rt.gpu_id());
        grid_blocks = (per_sm > 0 && n_sm > 0) ? per_sm * n_sm : 1;
    }

    (void)hipMemset(d_flag, 0, (size_t)2 * (N - 1) * sizeof(unsigned int));
    (void)hipDeviceSynchronize();
    rt.barrier();

    gicc::DeviceCtx* d = rt.prepare();
    int   data_idx = data_buf.index, recv_idx = recv_buf.index;
    int   flag_idx = flag_buf.index, one_idx = one_buf.index;
    int   n = N, r = rank, c = chunk;
    volatile unsigned int* fp = (volatile unsigned int*)d_flag;
    void* params[] = {&d, &data_idx, &recv_idx, &flag_idx, &one_idx,
                      &n, &r, &c, &d_data, &d_recv, &fp};
    (void)hipLaunchCooperativeKernel((const void*)coop_allreduce_kernel,
                                     dim3(grid_blocks), dim3(block_threads),
                                     params, 0, 0);
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();
}

//============================================================================
// PIPELINED cooperative ring all-reduce. Splits each ring step's chunk into P
// segments and issues segment p+1's transfer (a non-blocking proxy push)
// BEFORE reducing segment p, so the NIC streams seg p+1 while the GPU adds
// seg p — hiding the add + grid-syncs (and per-step dead time) under
// transfer. Data->flag ordering is preserved (quiet between a segment's data
// and its flag), so it stays correct without provider WAW-ordering
// assumptions. Reduce-scatter is pipelined (where the add overlap lives);
// all-gather stays one-shot per step (no add to overlap).
//
// MEASURED RESULT (negative, kept on purpose): this is ~1.3-2x SLOWER than
// ring_allreduce_coop at every size on Tioga. Reason: enabling the overlap
// requires a SEPARATE cross-rank arrival signal per segment (data->flag,
// quiet-ordered), so P segments cost P x the signaling round-trips per ring
// step. Here the reduction (add) is only ~us while a proxy flag round-trip is
// ~6us, so the signaling we add to enable the overlap costs far more than the
// add it hides. Pipelining only wins when per-segment compute >> per-segment
// signal cost; a put/quiet+flag ring all-reduce is signaling/latency-bound,
// not bandwidth-bound. The right lever for the small-msg regime is FEWER
// steps (recursive-doubling), not finer pipelining. Use ring_allreduce_coop.
// flag layout: [0 .. (N-1)*P) reduce-scatter (per step,segment);
//              [(N-1)*P .. (N-1)*P+(N-1)) all-gather (per step).
//============================================================================
__global__ void coop_allreduce_pipe_kernel(gicc::DeviceCtx* ctx,
                                           int data_idx, int recv_idx,
                                           int flag_idx, int one_idx,
                                           int N, int rank, int chunk, int P,
                                           float* data, const float* recv,
                                           volatile unsigned int* flag) {
    cg::grid_group grid = cg::this_grid();
    const bool   lead = (blockIdx.x == 0 && threadIdx.x == 0);
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const int    next = (rank + 1) % N;
    const int    seg  = chunk / P;                 // base segment length
    const size_t fb   = sizeof(unsigned int);

    // segment [off,len) in elements for segment p
    #define SEG_OFF(p) ((size_t)(p) * seg)
    #define SEG_LEN(p) ((p) == P - 1 ? (size_t)chunk - (size_t)(p) * seg : (size_t)seg)

    // ---- reduce-scatter, pipelined over P segments ----
    for (int s = 0; s < N - 1; ++s) {
        const int    sc     = (rank - s + N) % N;
        const int    rc     = (rank - 1 - s + N) % N;
        const int    rsbase = s * P;
        // prime segment 0
        if (lead) {
            gicc::put(ctx, next, recv_idx, ((size_t)s * chunk + SEG_OFF(0)) * sizeof(float),
                      data_idx, ((size_t)sc * chunk + SEG_OFF(0)) * sizeof(float),
                      SEG_LEN(0) * sizeof(float));
            gicc::quiet(ctx);
            gicc::put(ctx, next, flag_idx, (size_t)(rsbase + 0) * fb, one_idx, 0, fb);
            gicc::quiet(ctx);
        }
        for (int p = 0; p < P; ++p) {
            if (lead && p + 1 < P) {
                // non-blocking push of seg p+1's data (proxy transfers it
                // while we reduce seg p below)
                gicc::put(ctx, next, recv_idx,
                          ((size_t)s * chunk + SEG_OFF(p + 1)) * sizeof(float),
                          data_idx,
                          ((size_t)sc * chunk + SEG_OFF(p + 1)) * sizeof(float),
                          SEG_LEN(p + 1) * sizeof(float));
            }
            if (lead) {
                while (flag[rsbase + p] == 0u) { __builtin_amdgcn_s_sleep(1); }
                __threadfence_system();
            }
            grid.sync();
            const size_t off = (size_t)rc * chunk + SEG_OFF(p);
            const size_t roff = (size_t)s * chunk + SEG_OFF(p);
            const size_t len = SEG_LEN(p);
            for (size_t i = gtid; i < len; i += nthr)
                data[off + i] += recv[roff + i];
            grid.sync();
            if (lead && p + 1 < P) {
                gicc::quiet(ctx);   // seg p+1 data has landed remotely
                gicc::put(ctx, next, flag_idx, (size_t)(rsbase + p + 1) * fb,
                          one_idx, 0, fb);
                gicc::quiet(ctx);
            }
        }
    }
    // ---- all-gather (one-shot per step; no add to overlap) ----
    const int agbase = (N - 1) * P;
    const size_t cb = (size_t)chunk * sizeof(float);
    for (int s = 0; s < N - 1; ++s) {
        const int sc = (rank + 1 - s + 2 * N) % N;
        if (lead) {
            gicc::put(ctx, next, data_idx, (size_t)sc * cb, data_idx,
                      (size_t)sc * cb, cb);
            gicc::quiet(ctx);
            gicc::put(ctx, next, flag_idx, (size_t)(agbase + s) * fb, one_idx, 0, fb);
            gicc::quiet(ctx);
            while (flag[agbase + s] == 0u) { __builtin_amdgcn_s_sleep(1); }
            __threadfence_system();
        }
        grid.sync();
    }
    #undef SEG_OFF
    #undef SEG_LEN
}

inline void ring_allreduce_pipe(gicc::Runtime& rt,
                                const gicc::Buffer& data_buf, float* d_data,
                                const gicc::Buffer& recv_buf, float* d_recv,
                                const gicc::Buffer& flag_buf, unsigned int* d_flag,
                                const gicc::Buffer& one_buf,
                                int chunk, int P = 4) {
    const int N    = rt.size();
    const int rank = rt.rank();
    if (P < 1) P = 1;
    if (P > chunk) P = chunk;          // can't have more segments than elements

    static int grid_blocks = 0;
    const int  block_threads = 256;
    if (grid_blocks == 0) {
        int per_sm = 0, n_sm = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (const void*)coop_allreduce_pipe_kernel, block_threads, 0);
        (void)hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount,
                                    rt.gpu_id());
        grid_blocks = (per_sm > 0 && n_sm > 0) ? per_sm * n_sm : 1;
    }

    const size_t n_flags = (size_t)(N - 1) * P + (size_t)(N - 1);
    (void)hipMemset(d_flag, 0, n_flags * sizeof(unsigned int));
    (void)hipDeviceSynchronize();
    rt.barrier();

    gicc::DeviceCtx* d = rt.prepare();
    int   data_idx = data_buf.index, recv_idx = recv_buf.index;
    int   flag_idx = flag_buf.index, one_idx = one_buf.index;
    int   n = N, r = rank, c = chunk, p = P;
    volatile unsigned int* fp = (volatile unsigned int*)d_flag;
    void* params[] = {&d, &data_idx, &recv_idx, &flag_idx, &one_idx,
                      &n, &r, &c, &p, &d_data, &d_recv, &fp};
    (void)hipLaunchCooperativeKernel((const void*)coop_allreduce_pipe_kernel,
                                     dim3(grid_blocks), dim3(block_threads),
                                     params, 0, 0);
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();
}
#endif  // GICC_CPU_PROXY

// transport_name - for banner printing.
inline const char* transport_name() {
#ifdef GICC_CPU_PROXY
    return "CPU-PROXY";
#else
    return "DWQ (GPU-trigger)";
#endif
}

} // namespace gicc_coll
