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
#include <chrono>
#include <cstdlib>
#include <cstdio>

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

// Issue one chunk as `splits` independent sub-puts WITHOUT quieting between
// them, so the proxy keeps `splits` fi_writes in flight (raises pipeline depth
// = puts-per-quiet, the one CPU-side lever with teeth on cross-node bandwidth).
// Still ONE quiet + ONE flag per ring step afterward -> no extra signaling.
__device__ inline void put_split(gicc::DeviceCtx* ctx, int next,
                                 int dst_idx, size_t dst_off,
                                 int src_idx, size_t src_off,
                                 size_t bytes, int splits) {
    if (splits <= 1) {
        gicc::put(ctx, next, dst_idx, dst_off, src_idx, src_off, bytes);
        return;
    }
    const size_t sub = bytes / splits;
    for (int k = 0; k < splits; ++k) {
        const size_t off = (size_t)k * sub;
        const size_t len = (k == splits - 1) ? (bytes - off) : sub;
        gicc::put(ctx, next, dst_idx, dst_off + off, src_idx, src_off + off, len);
    }
}

__global__ void coop_allreduce_kernel(gicc::DeviceCtx* ctx,
                                      int data_idx, int recv_idx, int flag_idx,
                                      int one_idx, int N, int rank, int chunk,
                                      int splits,
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
            put_split(ctx, next, recv_idx, (size_t)s * cb,
                      data_idx, (size_t)sc * cb, cb, splits);
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
            put_split(ctx, next, data_idx, (size_t)sc * cb,
                      data_idx, (size_t)sc * cb, cb, splits);
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
                                int chunk, int splits = 1) {
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
    if (splits < 1) splits = 1;

    (void)hipMemset(d_flag, 0, (size_t)2 * (N - 1) * sizeof(unsigned int));
    (void)hipDeviceSynchronize();
    rt.barrier();

    gicc::DeviceCtx* d = rt.prepare();
    int   data_idx = data_buf.index, recv_idx = recv_buf.index;
    int   flag_idx = flag_buf.index, one_idx = one_buf.index;
    int   n = N, r = rank, c = chunk, sp = splits;
    volatile unsigned int* fp = (volatile unsigned int*)d_flag;
    void* params[] = {&d, &data_idx, &recv_idx, &flag_idx, &one_idx,
                      &n, &r, &c, &sp, &d_data, &d_recv, &fp};
    (void)hipLaunchCooperativeKernel((const void*)coop_allreduce_kernel,
                                     dim3(grid_blocks), dim3(block_threads),
                                     params, 0, 0);
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();
}

//============================================================================
// LOCALITY-AWARE cooperative ring all-reduce. The big coop-vs-MPI gap was that
// proxy mode runs EVERY ring link over the NIC, including the 6-of-8 intra-node
// links. Here, for a SAME-NODE neighbour we copy the chunk straight into its
// IPC-mapped buffer over xGMI (all threads, no NIC); only CROSS-NODE links use
// the proxy. The completion flag still goes via the proxy (host-pinned, cheap
// poll) in both cases, so the receive side is uniform. ctx->peer_ipc_base /
// ipc_n_bufs come from the runtime (DeviceCtx, set in prepare()).
//============================================================================
__global__ void coop_allreduce_loc_kernel(gicc::DeviceCtx* ctx,
                                          int data_idx, int recv_idx, int flag_idx,
                                          int one_idx, int N, int rank, int chunk,
                                          float* data, float* recv,
                                          volatile unsigned int* flag) {
    cg::grid_group grid = cg::this_grid();
    const bool   lead = (blockIdx.x == 0 && threadIdx.x == 0);
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const int    next = (rank + 1) % N;
    const size_t cb   = (size_t)chunk * sizeof(float);
    const int    nb   = ctx->ipc_n_bufs;
    void** ipc = ctx->peer_ipc_base;
    // same-node peer's recv buffer (RS) / data buffer (AG), or null if cross-node
    float* peer_recv = (ipc && nb) ? (float*)ipc[(size_t)next * nb + recv_idx] : nullptr;
    float* peer_data = (ipc && nb) ? (float*)ipc[(size_t)next * nb + data_idx] : nullptr;

    // ---- reduce-scatter ----
    for (int s = 0; s < N - 1; ++s) {
        const int sc = (rank - s + N) % N;
        const int rc = (rank - 1 - s + N) % N;
        if (peer_recv) {                                   // intra-node: xGMI
            for (size_t i = gtid; i < (size_t)chunk; i += nthr)
                peer_recv[(size_t)s * chunk + i] = data[(size_t)sc * chunk + i];
        } else if (lead) {                                 // cross-node: proxy
            gicc::put(ctx, next, recv_idx, (size_t)s * cb,
                      data_idx, (size_t)sc * cb, cb);
            gicc::quiet(ctx);
        }
        grid.sync();
        if (lead) {
            __threadfence_system();   // publish all blocks' xGMI writes to peer
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
    // ---- all-gather ----
    for (int s = 0; s < N - 1; ++s) {
        const int sc = (rank + 1 - s + 2 * N) % N;
        if (peer_data) {                                   // intra-node: xGMI
            for (size_t i = gtid; i < (size_t)chunk; i += nthr)
                peer_data[(size_t)sc * chunk + i] = data[(size_t)sc * chunk + i];
        } else if (lead) {                                 // cross-node: proxy
            gicc::put(ctx, next, data_idx, (size_t)sc * cb,
                      data_idx, (size_t)sc * cb, cb);
            gicc::quiet(ctx);
        }
        grid.sync();
        if (lead) {
            __threadfence_system();
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

inline void ring_allreduce_coop_loc(gicc::Runtime& rt,
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
            &per_sm, (const void*)coop_allreduce_loc_kernel, block_threads, 0);
        (void)hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount,
                                    rt.gpu_id());
        grid_blocks = (per_sm > 0 && n_sm > 0) ? per_sm * n_sm : 1;
    }

    (void)hipMemset(d_flag, 0, (size_t)2 * (N - 1) * sizeof(unsigned int));
    (void)hipDeviceSynchronize();
    rt.barrier();

    gicc::DeviceCtx* d = rt.prepare();
    int data_idx = data_buf.index, recv_idx = recv_buf.index;
    int flag_idx = flag_buf.index, one_idx = one_buf.index;
    int n = N, r = rank, c = chunk;
    volatile unsigned int* fp = (volatile unsigned int*)d_flag;
    void* params[] = {&d, &data_idx, &recv_idx, &flag_idx, &one_idx,
                      &n, &r, &c, &d_data, &d_recv, &fp};
    (void)hipLaunchCooperativeKernel((const void*)coop_allreduce_loc_kernel,
                                     dim3(grid_blocks), dim3(block_threads),
                                     params, 0, 0);
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();
}

//============================================================================
// HIERARCHICAL all-reduce (2 nodes, P ranks/node, block layout). Avoids the
// single-NIC cross-node funnel of a flat ring:
//   Phase 1  intra-node reduce-scatter ring over P (xGMI) -> each local rank
//            holds its node's partial sum of slice fp=(lp+1)%P (size count/P).
//   Phase 2  inter-node exchange: every local rank swaps its slice fp with the
//            same-position rank on the other node and sums -> ALL P ranks cross
//            simultaneously => all NICs busy, and only count/P crosses per rank
//            (total ~count vs ~2*count for the flat ring).
//   Phase 3  intra-node all-gather ring over P (xGMI) -> everyone gets all
//            global slices.
// Only phase 2 touches the NIC; phases 1/3 are pure xGMI. flag layout:
// [0,P-1) RS, [P-1] inter-node, [P,2P-1) AG.  Requires N == 2*P (K=2 nodes).
//============================================================================
__global__ void coop_allreduce_hier_kernel(gicc::DeviceCtx* ctx,
                                           int data_idx, int recv_idx, int flag_idx,
                                           int one_idx, int N, int rank, int count,
                                           int P, float* data, float* recv,
                                           volatile unsigned int* flag) {
    cg::grid_group grid = cg::this_grid();
    const bool   lead = (blockIdx.x == 0 && threadIdx.x == 0);
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const int    nb   = ctx->ipc_n_bufs;
    void** ipc = ctx->peer_ipc_base;
    const int lp        = rank % P;
    const int node_base = rank - lp;
    const int lnext     = node_base + (lp + 1) % P;     // same-node ring successor
    const int partner   = (rank + P) % N;               // K=2 inter-node peer
    const int slice     = count / P;
    const size_t sb     = (size_t)slice * sizeof(float);
    float* ln_recv = (ipc && nb) ? (float*)ipc[(size_t)lnext * nb + recv_idx] : nullptr;
    float* ln_data = (ipc && nb) ? (float*)ipc[(size_t)lnext * nb + data_idx] : nullptr;

    // ---- Phase 1: intra-node reduce-scatter ring over P (xGMI) ----
    for (int s = 0; s < P - 1; ++s) {
        const int sc = (lp - s + P) % P;
        const int rc = (lp - 1 - s + P) % P;
        if (ln_recv) {
            for (size_t i = gtid; i < (size_t)slice; i += nthr)
                ln_recv[(size_t)s * slice + i] = data[(size_t)sc * slice + i];
        } else if (lead) {
            gicc::put(ctx, lnext, recv_idx, (size_t)s * sb, data_idx, (size_t)sc * sb, sb);
            gicc::quiet(ctx);
        }
        grid.sync();
        if (lead) {
            __threadfence_system();
            gicc::put(ctx, lnext, flag_idx, (size_t)s * sizeof(unsigned int),
                      one_idx, 0, sizeof(unsigned int));
            gicc::quiet(ctx);
            while (flag[s] == 0u) { __builtin_amdgcn_s_sleep(1); }
            __threadfence_system();
        }
        grid.sync();
        for (size_t i = gtid; i < (size_t)slice; i += nthr)
            data[(size_t)rc * slice + i] += recv[(size_t)s * slice + i];
        grid.sync();
    }
    const int fp = (lp + 1) % P;   // slice this rank now owns (node-local sum)

    // ---- Phase 2: inter-node exchange of slice fp (proxy, all ranks parallel) ----
    if (partner != rank) {
        const int xs = P - 1;      // recv slot reserved for the inter-node slice
        if (lead) {
            gicc::put(ctx, partner, recv_idx, (size_t)xs * sb,
                      data_idx, (size_t)fp * sb, sb);
            gicc::quiet(ctx);
            gicc::put(ctx, partner, flag_idx, (size_t)(P - 1) * sizeof(unsigned int),
                      one_idx, 0, sizeof(unsigned int));
            gicc::quiet(ctx);
            while (flag[P - 1] == 0u) { __builtin_amdgcn_s_sleep(1); }
            __threadfence_system();
        }
        grid.sync();
        for (size_t i = gtid; i < (size_t)slice; i += nthr)
            data[(size_t)fp * slice + i] += recv[(size_t)xs * slice + i];
        grid.sync();
    }

    // ---- Phase 3: intra-node all-gather ring over P (xGMI) ----
    for (int s = 0; s < P - 1; ++s) {
        const int sc = (lp + 1 - s + 2 * P) % P;
        if (ln_data) {
            for (size_t i = gtid; i < (size_t)slice; i += nthr)
                ln_data[(size_t)sc * slice + i] = data[(size_t)sc * slice + i];
        } else if (lead) {
            gicc::put(ctx, lnext, data_idx, (size_t)sc * sb, data_idx, (size_t)sc * sb, sb);
            gicc::quiet(ctx);
        }
        grid.sync();
        if (lead) {
            __threadfence_system();
            gicc::put(ctx, lnext, flag_idx, (size_t)(P + s) * sizeof(unsigned int),
                      one_idx, 0, sizeof(unsigned int));
            gicc::quiet(ctx);
            while (flag[P + s] == 0u) { __builtin_amdgcn_s_sleep(1); }
            __threadfence_system();
        }
        grid.sync();
    }
}

// Host driver. count = total elements (must be divisible by P = ranks/node).
// recv >= count floats; flag >= 2*P uints (host-pinned); requires N == 2*P.
inline void ring_allreduce_hier(gicc::Runtime& rt,
                                const gicc::Buffer& data_buf, float* d_data,
                                const gicc::Buffer& recv_buf, float* d_recv,
                                const gicc::Buffer& flag_buf, unsigned int* d_flag,
                                const gicc::Buffer& one_buf,
                                int count, int P) {
    const int N    = rt.size();
    const int rank = rt.rank();
    static int grid_blocks = 0;
    const int  block_threads = 256;
    if (grid_blocks == 0) {
        int per_sm = 0, n_sm = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (const void*)coop_allreduce_hier_kernel, block_threads, 0);
        (void)hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount,
                                    rt.gpu_id());
        grid_blocks = (per_sm > 0 && n_sm > 0) ? per_sm * n_sm : 1;
    }

    (void)hipMemset(d_flag, 0, (size_t)2 * P * sizeof(unsigned int));
    (void)hipDeviceSynchronize();
    rt.barrier();

    gicc::DeviceCtx* d = rt.prepare();
    int data_idx = data_buf.index, recv_idx = recv_buf.index;
    int flag_idx = flag_buf.index, one_idx = one_buf.index;
    int n = N, r = rank, c = count, p = P;
    volatile unsigned int* fp = (volatile unsigned int*)d_flag;
    void* params[] = {&d, &data_idx, &recv_idx, &flag_idx, &one_idx,
                      &n, &r, &c, &p, &d_data, &d_recv, &fp};
    (void)hipLaunchCooperativeKernel((const void*)coop_allreduce_hier_kernel,
                                     dim3(grid_blocks), dim3(block_threads),
                                     params, 0, 0);
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();
}

//============================================================================
// DIRECT hierarchical all-reduce (2 nodes, P ranks/node). Replaces the intra
// RING (2(P-1) serial steps, each a grid.sync + proxy flag) with DIRECT
// all-pairs xGMI passes that need NO per-step barrier and NO per-step flag:
//   * direct reduce-scatter: each rank sums slice lp over all P same-node ranks
//     by reading their data[lp] over xGMI. Race-free with NO cross-rank sync —
//     each rank only WRITES its own slice and READS peers' OTHER slices, which
//     they never write.
//   * inter-node exchange of slice lp (proxy, all ranks parallel -> all NICs).
//   * direct all-gather: each rank copies every other slice from its same-node
//     owner over xGMI. Needs all phase-2 done first -> ONE host barrier between
//     the two kernels (replaces 14 per-step flags).
// Net: 2 grid.syncs + 1 cross flag total, vs the ring's ~42 grid.syncs + 15
// flags. Requires N == 2*P, block layout, peer IPC table in DeviceCtx.
//============================================================================

// Kernel 1: direct intra reduce-scatter (no sync) + inter-node exchange.
__global__ void hier_direct_rs_cross_kernel(gicc::DeviceCtx* ctx,
                                            int data_idx, int recv_idx, int flag_idx,
                                            int one_idx, int N, int rank, int count,
                                            int P, float* data, float* recv,
                                            volatile unsigned int* flag) {
    cg::grid_group grid = cg::this_grid();
    const bool   lead = (blockIdx.x == 0 && threadIdx.x == 0);
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const int    nb   = ctx->ipc_n_bufs;
    void** ipc = ctx->peer_ipc_base;
    const int lp        = rank % P;
    const int node_base = rank - lp;
    const int partner   = (rank + P) % N;
    const int slice     = count / P;
    const size_t base   = (size_t)lp * slice;     // my owned slice

    // ---- direct intra reduce-scatter: data[base+e] = sum over node of slice lp
    for (size_t e = gtid; e < (size_t)slice; e += nthr) {
        float acc = data[base + e];
        for (int q = 0; q < P; ++q) {
            if (q == lp) continue;
            const float* pq = (ipc && nb) ? (const float*)ipc[(size_t)(node_base + q) * nb + data_idx] : nullptr;
            if (pq) acc += pq[base + e];
        }
        data[base + e] = acc;
    }
    grid.sync();

    // ---- inter-node exchange of slice lp (proxy; all P ranks cross at once)
    if (partner != rank) {
        const size_t sb = (size_t)slice * sizeof(float);
        if (lead) {
            gicc::put(ctx, partner, recv_idx, 0, data_idx, base * sizeof(float), sb);
            gicc::quiet(ctx);
            gicc::put(ctx, partner, flag_idx, 0, one_idx, 0, sizeof(unsigned int));
            gicc::quiet(ctx);
            while (flag[0] == 0u) { __builtin_amdgcn_s_sleep(1); }
            flag[0] = 0u;                 // re-arm for next call (no per-call memset)
            __threadfence_system();
        }
        grid.sync();
        for (size_t e = gtid; e < (size_t)slice; e += nthr)
            data[base + e] += recv[e];
        grid.sync();
    }
}

// Kernel 2: direct intra all-gather. Each rank copies every other slice from
// its same-node owner over xGMI (owners hold the global slice after kernel 1 +
// the host barrier). No flags, no sync — pure parallel gather.
__global__ void hier_direct_ag_kernel(gicc::DeviceCtx* ctx, int data_idx,
                                      int N, int rank, int count, int P,
                                      float* data) {
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const int    nb   = ctx->ipc_n_bufs;
    void** ipc = ctx->peer_ipc_base;
    const int lp        = rank % P;
    const int node_base = rank - lp;
    const int slice     = count / P;
    for (int p = 0; p < P; ++p) {
        if (p == lp) continue;                    // already own slice lp
        const float* po = (ipc && nb) ? (const float*)ipc[(size_t)(node_base + p) * nb + data_idx] : nullptr;
        if (!po) continue;
        const size_t b = (size_t)p * slice;
        for (size_t e = gtid; e < (size_t)slice; e += nthr)
            data[b + e] = po[b + e];
    }
}

inline void ring_allreduce_hier_direct(gicc::Runtime& rt,
                                       const gicc::Buffer& data_buf, float* d_data,
                                       const gicc::Buffer& recv_buf, float* d_recv,
                                       const gicc::Buffer& flag_buf, unsigned int* d_flag,
                                       const gicc::Buffer& one_buf,
                                       int count, int P) {
    const int N = rt.size();
    const int rank = rt.rank();
    static int gb1 = 0;
    const int bt = 256;
    if (gb1 == 0) {
        int per_sm = 0, n_sm = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (const void*)hier_direct_rs_cross_kernel, bt, 0);
        (void)hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount, rt.gpu_id());
        gb1 = (per_sm > 0 && n_sm > 0) ? per_sm * n_sm : 1;
    }

    static int prof = (std::getenv("GICC_HDIR_PROF") != nullptr) ? 1 : 0;
    using clk = std::chrono::high_resolution_clock;
    auto us = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    clk::time_point t0, t1, t2, t3, t4, t5;
    if (prof) t0 = clk::now();

    // No per-call flag memset/barrier: the receiver re-arms flag[0]=0 after
    // consuming it in kernel 1, and the end barrier orders that before the next
    // call's cross-write. Caller does a ONE-TIME memset of d_flag at setup.
    if (prof) t1 = clk::now();

    // kernel 1: direct RS + inter-node exchange (cooperative)
    gicc::DeviceCtx* d = rt.prepare();
    int data_idx = data_buf.index, recv_idx = recv_buf.index;
    int flag_idx = flag_buf.index, one_idx = one_buf.index;
    int n = N, r = rank, c = count, p = P;
    volatile unsigned int* fp = (volatile unsigned int*)d_flag;
    void* p1[] = {&d, &data_idx, &recv_idx, &flag_idx, &one_idx,
                  &n, &r, &c, &p, &d_data, &d_recv, &fp};
    (void)hipLaunchCooperativeKernel((const void*)hier_direct_rs_cross_kernel,
                                     dim3(gb1), dim3(bt), p1, 0, 0);
    (void)hipDeviceSynchronize();
    if (prof) t2 = clk::now();   // K1 (RS+cross) done
    rt.reset();
    rt.barrier();                       // all ranks' global slices ready
    if (prof) t3 = clk::now();   // inter-kernel ceremony done

    // kernel 2: direct all-gather (plain kernel, no flags/sync)
    d = rt.prepare();
    const int blk = 512;
    hipLaunchKernelGGL(hier_direct_ag_kernel, dim3(blk), dim3(bt), 0, 0,
                       d, data_idx, n, r, c, p, d_data);
    (void)hipDeviceSynchronize();
    if (prof) t4 = clk::now();   // K2 (AG) done
    rt.reset();
    rt.barrier();
    if (prof) {
        t5 = clk::now();
        if (rank == 0)
            fprintf(stderr, "[hdir prof %dB] start=%.1f K1(rs+cross)=%.1f "
                    "mid-cer=%.1f K2(ag)=%.1f end-cer=%.1f total=%.1f us\n",
                    count * 4, us(t0, t1), us(t1, t2), us(t2, t3),
                    us(t3, t4), us(t4, t5), us(t0, t5));
    }
}

// Size-selected hierarchical all-reduce: DIRECT (latency-optimal) for small/mid,
// RING (bandwidth-optimal) for large — the same algorithm-by-size switch NCCL/MPI
// use. Crossover ~8MB total (measured: 1MB direct wins, 16MB ring wins). Both
// take the same buffers (recv >= count, flag host-pinned >= 2P).
inline void ring_allreduce_best(gicc::Runtime& rt,
                                const gicc::Buffer& data_buf, float* d_data,
                                const gicc::Buffer& recv_buf, float* d_recv,
                                const gicc::Buffer& flag_buf, unsigned int* d_flag,
                                const gicc::Buffer& one_buf, int count, int P) {
    if ((size_t)count * sizeof(float) >= (8u << 20))
        ring_allreduce_hier(rt, data_buf, d_data, recv_buf, d_recv,
                            flag_buf, d_flag, one_buf, count, P);
    else
        ring_allreduce_hier_direct(rt, data_buf, d_data, recv_buf, d_recv,
                                   flag_buf, d_flag, one_buf, count, P);
}

//============================================================================
// DOUBLE BINARY TREE all-reduce (Sanders-Speck-Träff / NCCL "dtree").
//
// A single binary tree wastes half the bandwidth: its leaves only ever SEND
// (they have no children to receive from), and half a binary tree's nodes are
// leaves. The double-tree fix builds TWO complementary in-order binary trees
// whose internal/leaf roles are swapped (for even N, T1 is the mirror of T0:
// T0's internal nodes are T1's leaves and vice-versa). The data is split in
// half; half A is reduce-broadcast over T0, half B over T1. Every rank is then
// internal in exactly one tree and a leaf in the other, so each rank drives a
// full-bandwidth subtree for one half while only forwarding the other — total
// per-rank traffic is balanced and the critical path is ~2*log2(N) steps
// (vs the ring's 2*(N-1)). This is NCCL's default large-allreduce algorithm.
//
// Topology is integer arithmetic over rank ids (computed host-side, passed in).
// Communication auto-routes per edge: a same-node peer is written directly over
// xGMI (peer_ipc_base non-null); a cross-node peer goes through the proxy. The
// arrival flag always goes through the proxy (host-pinned, cheap coherent poll),
// matching coop_allreduce_loc_kernel.
//
// Flag layout (per rank, host-pinned): [0,1] T0 reduce from child slot 0/1,
// [2,3] T1 reduce from child slot 0/1, [4] T0 broadcast from parent, [5] T1
// broadcast from parent. recv buffer: 2 slots of count/2 (a rank parents in at
// most ONE tree, so the two child slots never collide across trees).
// Flags are re-armed in-kernel (receiver zeroes after consuming), so the caller
// memsets the flag buffer ONCE at setup. Requires even `count`.
//
// STABILITY: the caller MUST rt.set_ipc_fastpath(false). dt_send_half sends the
// arrival flag via gicc::put even for same-node edges; with the IPC fast path on
// that becomes an SDMA hipMemcpyAsync that contends with concurrent cross-node
// proxy RMA and intermittently deadlocks on AMD+CXI. With it off the flat tree
// runs stably in sustained loops (see coll_dtree_test, which sets it).
//============================================================================

// In-order binary tree (NCCL ncclGetBtree): for `rank` among `nranks`, returns
// parent `up` (-1 at root), the two children `d0`/`d1` (-1 if absent), and which
// child slot this rank is of its parent (`ct` in {0,1}, -1 at root).
inline void dt_btree(int nranks, int rank, int& up, int& d0, int& d1, int& ct) {
    int bit;
    for (bit = 1; bit < nranks; bit <<= 1)
        if (bit & rank) break;
    if (rank == 0) {
        up = -1; d0 = -1;
        d1 = (nranks > 1) ? (bit >> 1) : -1;
        ct = -1;
        return;
    }
    up = (rank ^ bit) | (bit << 1);
    if (up >= nranks) up = rank ^ bit;
    ct = (rank < up) ? 0 : 1;
    int lowbit = bit >> 1;
    d0 = (lowbit == 0) ? -1 : rank - lowbit;
    d1 = (lowbit == 0) ? -1 : rank + lowbit;
    while (d1 >= nranks) {            // pull child 1 in-bounds for non-pow2 N
        lowbit >>= 1;
        d1 = (lowbit == 0) ? -1 : rank + lowbit;
    }
}

// Double binary tree (NCCL ncclGetDtree): T0 is the plain in-order tree; T1 is
// its mirror (even N) or +1 shift (odd N), which swaps internal/leaf roles.
inline void dt_dtree(int nranks, int rank,
                     int& up0, int& c0a, int& c0b, int& ct0,
                     int& up1, int& c1a, int& c1b, int& ct1) {
    dt_btree(nranks, rank, up0, c0a, c0b, ct0);
    if (nranks % 2 == 1) {
        int sr = (rank - 1 + nranks) % nranks, u, d0, d1, ct;
        dt_btree(nranks, sr, u, d0, d1, ct);
        up1 = (u  == -1) ? -1 : (u  + 1) % nranks;
        c1a = (d0 == -1) ? -1 : (d0 + 1) % nranks;
        c1b = (d1 == -1) ? -1 : (d1 + 1) % nranks;
        ct1 = ct;
    } else {
        int u, d0, d1, ct;
        dt_btree(nranks, nranks - 1 - rank, u, d0, d1, ct);
        up1 = (u  == -1) ? -1 : nranks - 1 - u;
        c1a = (d0 == -1) ? -1 : nranks - 1 - d0;
        c1b = (d1 == -1) ? -1 : nranks - 1 - d1;
        ct1 = ct;
    }
}

#ifdef GICC_CPU_PROXY
// Send `len` floats from my data[src_off] to `peer`'s buffer `dst_idx`[dst_off]
// then signal peer's flag[flag_slot]. Same-node => xGMI (all threads); else
// proxy (lead only). The peer scalar is grid-uniform so the branch + any
// grid.sync inside it are taken by the whole grid.
__device__ inline void dt_send_half(gicc::DeviceCtx* ctx, cg::grid_group& grid,
                                    int peer, int dst_idx, size_t dst_off,
                                    int src_idx, float* mydata, size_t src_off,
                                    int len, int flag_idx, size_t flag_slot,
                                    int one_idx, bool lead,
                                    size_t gtid, size_t nthr) {
    const int nb = ctx->ipc_n_bufs;
    void** ipc = ctx->peer_ipc_base;
    float* peer_buf = (ipc && nb) ? (float*)ipc[(size_t)peer * nb + dst_idx] : nullptr;
    if (peer_buf) {                                   // intra-node: xGMI
        for (size_t i = gtid; i < (size_t)len; i += nthr)
            peer_buf[dst_off + i] = mydata[src_off + i];
        grid.sync();
        if (lead) {
            __threadfence_system();                   // publish xGMI writes
            gicc::put(ctx, peer, flag_idx, flag_slot * sizeof(unsigned int),
                      one_idx, 0, sizeof(unsigned int));
            gicc::quiet(ctx);
        }
    } else if (lead) {                                // cross-node: proxy
        __threadfence_system();
        gicc::put(ctx, peer, dst_idx, dst_off * sizeof(float),
                  src_idx, src_off * sizeof(float), (size_t)len * sizeof(float));
        gicc::quiet(ctx);                             // data landed before flag
        gicc::put(ctx, peer, flag_idx, flag_slot * sizeof(unsigned int),
                  one_idx, 0, sizeof(unsigned int));
        gicc::quiet(ctx);
    }
}

// Cooperative double-tree all-reduce kernel. Reduce (leaf->root) then broadcast
// (root->leaf), both trees, data split in half (T0:[0,h), T1:[h,count)).
__global__ void dtree_allreduce_kernel(gicc::DeviceCtx* ctx,
                                       int data_idx, int recv_idx, int flag_idx,
                                       int one_idx, int rank, int count,
                                       int up0, int c0a, int c0b, int ct0,
                                       int up1, int c1a, int c1b, int ct1,
                                       float* data, float* recv,
                                       volatile unsigned int* flag) {
    cg::grid_group grid = cg::this_grid();
    const bool   lead = (blockIdx.x == 0 && threadIdx.x == 0);
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const int    h    = count / 2;                    // half size (count even)
    const int    up[2] = {up0, up1};
    const int    ca[2] = {c0a, c1a};
    const int    cb[2] = {c0b, c1b};
    const int    ct[2] = {ct0, ct1};

    // ---- REDUCE (leaf -> root) ----
    for (int t = 0; t < 2; ++t) {
        const int  base   = t * h;                    // this tree's half
        const int  kid[2] = {ca[t], cb[t]};
        for (int k = 0; k < 2; ++k) {
            if (kid[k] < 0) continue;                 // missing child
            if (lead) {
                while (flag[t * 2 + k] == 0u) { __builtin_amdgcn_s_sleep(1); }
                flag[t * 2 + k] = 0u;                 // re-arm
                __threadfence_system();
            }
            grid.sync();
            for (size_t i = gtid; i < (size_t)h; i += nthr)
                data[base + i] += recv[(size_t)k * h + i];
            grid.sync();
        }
        if (up[t] >= 0) {                             // not this tree's root
            grid.sync();                              // adds visible before send
            dt_send_half(ctx, grid, up[t], recv_idx, (size_t)ct[t] * h,
                         data_idx, data, (size_t)base, h,
                         flag_idx, (size_t)(t * 2 + ct[t]), one_idx,
                         lead, gtid, nthr);
        }
    }

    // ---- BROADCAST (root -> leaf) ----
    for (int t = 0; t < 2; ++t) {
        const int base = t * h;
        if (up[t] >= 0 && lead) {                     // non-root waits parent
            while (flag[4 + t] == 0u) { __builtin_amdgcn_s_sleep(1); }
            flag[4 + t] = 0u;
            __threadfence_system();
        }
        grid.sync();                                  // parent/root data visible
        const int kid[2] = {ca[t], cb[t]};
        for (int k = 0; k < 2; ++k) {
            if (kid[k] < 0) continue;
            dt_send_half(ctx, grid, kid[k], data_idx, (size_t)base,
                         data_idx, data, (size_t)base, h,
                         flag_idx, (size_t)(4 + t), one_idx,
                         lead, gtid, nthr);
        }
    }
}

// Kernel-internal timing accumulators (populated when GICC_DTREE_KTIME is set),
// so a caller can report on-device tree time excluding host launch/reset/barrier.
inline double& dtree_kernel_us_acc() { static double a = 0.0; return a; }
inline int&    dtree_kernel_calls()  { static int c = 0;      return c; }

// Host driver for the flat double binary tree all-reduce. count must be even;
// recv >= count floats; flag host-pinned >= 6 uints, memset to 0 ONCE by the
// caller before the first call (re-armed in-kernel thereafter).
inline void allreduce_double_tree(gicc::Runtime& rt,
                                  const gicc::Buffer& data_buf, float* d_data,
                                  const gicc::Buffer& recv_buf, float* d_recv,
                                  const gicc::Buffer& flag_buf, unsigned int* d_flag,
                                  const gicc::Buffer& one_buf, int count) {
    const int N    = rt.size();
    const int rank = rt.rank();
    static int grid_blocks = 0;
    const int  block_threads = 256;
    if (grid_blocks == 0) {
        int per_sm = 0, n_sm = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (const void*)dtree_allreduce_kernel, block_threads, 0);
        (void)hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount,
                                    rt.gpu_id());
        // 50% headroom: requesting the occupancy MAX makes a cooperative
        // grid.sync() deadlock-prone (no slack for all blocks to be resident).
        int spm = (per_sm > 1) ? per_sm / 2 : 1;
        grid_blocks = (spm > 0 && n_sm > 0) ? spm * n_sm : 1;
        if (std::getenv("DTREE_PIPE_GB"))            // debug: override grid size
            grid_blocks = std::atoi(std::getenv("DTREE_PIPE_GB"));
    }

    int up0, c0a, c0b, ct0, up1, c1a, c1b, ct1;
    dt_dtree(N, rank, up0, c0a, c0b, ct0, up1, c1a, c1b, ct1);

    // Adaptive grid: small messages are bound by the per-step proxy round-trip,
    // not compute, so a big cooperative grid only adds grid.sync() cost (~230us
    // at 440 blocks for 64 ranks). Use few blocks below ~256KB (cheap grid.sync),
    // the full grid above (bandwidth). Skipped when grid is env-overridden.
    int launch_gb = grid_blocks;
    if (std::getenv("DTREE_PIPE_GB") == nullptr &&
        (size_t)count * sizeof(float) <= (256u << 10) && grid_blocks > 16)
        launch_gb = 16;

    gicc::DeviceCtx* d = rt.prepare();
    int data_idx = data_buf.index, recv_idx = recv_buf.index;
    int flag_idx = flag_buf.index, one_idx = one_buf.index;
    int r = rank, c = count;
    volatile unsigned int* fp = (volatile unsigned int*)d_flag;
    void* params[] = {&d, &data_idx, &recv_idx, &flag_idx, &one_idx, &r, &c,
                      &up0, &c0a, &c0b, &ct0, &up1, &c1a, &c1b, &ct1,
                      &d_data, &d_recv, &fp};
    // Optional kernel-internal timing (GICC_DTREE_KTIME): isolates the on-device
    // tree+proxy time from the host launch/reset/barrier ceremony, for a fair
    // algorithm comparison vs MPI (which launches no kernel).
    static int ktime = (std::getenv("GICC_DTREE_KTIME") != nullptr) ? 1 : 0;
    hipEvent_t ke0, ke1;
    if (ktime) { (void)hipEventCreate(&ke0); (void)hipEventCreate(&ke1);
                 (void)hipEventRecord(ke0, 0); }
    (void)hipLaunchCooperativeKernel((const void*)dtree_allreduce_kernel,
                                     dim3(launch_gb), dim3(block_threads),
                                     params, 0, 0);
    if (ktime) {
        (void)hipEventRecord(ke1, 0);
        (void)hipEventSynchronize(ke1);
        float ms = 0.f; (void)hipEventElapsedTime(&ms, ke0, ke1);
        dtree_kernel_us_acc() += (double)ms * 1000.0;
        dtree_kernel_calls()  += 1;
        (void)hipEventDestroy(ke0); (void)hipEventDestroy(ke1);
    }
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();
}

//============================================================================
// PIPELINED flat double binary tree (the NCCL technique the plain version
// lacks). The plain tree sends a whole half per edge level-by-level, so the
// critical path is depth*(count/2) of serial transfer. NCCL instead streams the
// buffer in CHUNKS through the tree with no per-chunk barrier: while chunk c is
// being forwarded leaf->root, chunk c+1 is one level behind, so the pipeline
// fills and the critical path drops to ~(count/2) + depth*chunk. Here each
// (child,chunk) gets its own recv slot and flag, so chunks never collide and a
// rank can race ahead on later chunks while the root is still on earlier ones.
// recv = count floats (2 child slots x half); flags = 6*S uints (re-armed
// in-kernel, memset once). Requires count even and (count/2) % S == 0 is NOT
// required (last chunk takes the remainder). Caller MUST set_ipc_fastpath(false).
//
// MEASURED (16 ranks / 2 nodes, S=8, vs the un-pipelined plain tree, both vs MPI):
//   bytes    plain    pipe     |  pipelining ~doubles large-message throughput
//   16MB     0.30x    0.31x    |  crossover ~16MB; below it the per-chunk signal
//   64MB     0.31x    0.56x    |  cost dominates (pipe is WORSE for small msgs,
//   256MB    0.35x    0.72x    |  so prefer plain / S=1 there).
// So pipelining is the right large-message lever (NCCL's chunking) — it still
// trails MPI here because the FLAT tree funnels cross-node traffic through the
// few NICs of the boundary ranks (the hierarchical variant spreads it, but has
// its own hang); a size-aware caller should pick plain<16MB and pipe (S~8) above.
//
// GRID: cooperative launch MUST leave occupancy headroom (see driver) or
// grid.sync() deadlocks. CAVEAT: like all the cooperative tree collectives here,
// sustained back-to-back runs can still intermittently hang in the proxy/quiet
// path under load (a transport-level robustness gap, separate from the grid
// issue and from the algorithm); a single call is reliable and correctness
// always passes.
//============================================================================
__global__ void dtree_allreduce_pipe_kernel(gicc::DeviceCtx* ctx,
                                            int data_idx, int recv_idx, int flag_idx,
                                            int one_idx, int rank, int count, int S,
                                            int up0, int c0a, int c0b, int ct0,
                                            int up1, int c1a, int c1b, int ct1,
                                            float* data, float* recv,
                                            volatile unsigned int* flag) {
    cg::grid_group grid = cg::this_grid();
    const bool   lead = (blockIdx.x == 0 && threadIdx.x == 0);
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const int    h    = count / 2;                    // half size
    const int    ce   = h / S;                        // base chunk elems
    const int    up[2] = {up0, up1};
    const int    ca[2] = {c0a, c1a};
    const int    cb[2] = {c0b, c1b};
    const int    ct[2] = {ct0, ct1};

    // ---- REDUCE (leaf -> root), pipelined over S chunks ----
    for (int t = 0; t < 2; ++t) {
        const int base = t * h;
        const int kid[2] = {ca[t], cb[t]};
        for (int c = 0; c < S; ++c) {
            const int co = c * ce;                    // chunk offset within half
            const int ne = (c == S - 1) ? (h - co) : ce;
            for (int k = 0; k < 2; ++k) {
                if (kid[k] < 0) continue;
                const int fr = (t * 2 + k) * S + c;   // reduce flag slot
                if (lead) {
                    while (flag[fr] == 0u) { __builtin_amdgcn_s_sleep(1); }
                    flag[fr] = 0u;
                    __threadfence_system();
                }
                grid.sync();
                for (size_t i = gtid; i < (size_t)ne; i += nthr)
                    data[base + co + i] += recv[(size_t)k * h + co + i];
                grid.sync();
            }
            if (up[t] >= 0) {
                grid.sync();
                dt_send_half(ctx, grid, up[t], recv_idx, (size_t)ct[t] * h + co,
                             data_idx, data, (size_t)base + co, ne,
                             flag_idx, (size_t)((t * 2 + ct[t]) * S + c), one_idx,
                             lead, gtid, nthr);
            }
        }
    }

    // ---- BROADCAST (root -> leaf), pipelined over S chunks ----
    for (int t = 0; t < 2; ++t) {
        const int base = t * h;
        const int kid[2] = {ca[t], cb[t]};
        for (int c = 0; c < S; ++c) {
            const int co = c * ce;
            const int ne = (c == S - 1) ? (h - co) : ce;
            const int fb = 4 * S + t * S + c;         // broadcast flag slot
            if (up[t] >= 0 && lead) {
                while (flag[fb] == 0u) { __builtin_amdgcn_s_sleep(1); }
                flag[fb] = 0u;
                __threadfence_system();
            }
            grid.sync();
            for (int k = 0; k < 2; ++k) {
                if (kid[k] < 0) continue;
                dt_send_half(ctx, grid, kid[k], data_idx, (size_t)base + co,
                             data_idx, data, (size_t)base + co, ne,
                             flag_idx, (size_t)fb, one_idx, lead, gtid, nthr);
            }
        }
    }
}

// Host driver for the pipelined flat double tree. nchunks S>=1 (S=1 reduces to
// the plain tree). count must be even; recv >= count; flag host-pinned >= 6*S
// uints (memset once by the caller).
inline void allreduce_double_tree_pipe(gicc::Runtime& rt,
                                       const gicc::Buffer& data_buf, float* d_data,
                                       const gicc::Buffer& recv_buf, float* d_recv,
                                       const gicc::Buffer& flag_buf, unsigned int* d_flag,
                                       const gicc::Buffer& one_buf, int count, int nchunks) {
    const int N    = rt.size();
    const int rank = rt.rank();
    int S = nchunks;
    if (S < 1) S = 1;
    if (S > count / 2) S = count / 2;                 // not more chunks than elems
    static int grid_blocks = 0;
    const int  block_threads = 256;
    if (grid_blocks == 0) {
        int per_sm = 0, n_sm = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (const void*)dtree_allreduce_pipe_kernel, block_threads, 0);
        (void)hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount,
                                    rt.gpu_id());
        // Cooperative launch needs ALL blocks co-resident or grid.sync()
        // deadlocks. Requesting the occupancy-API MAX (e.g. 8 blocks/CU = full
        // 2048 threads/CU on gfx90a) is unreliable: with zero headroom some
        // block may never be scheduled and the whole grid hangs. Leave 50%
        // headroom — these collectives are HBM/NIC-bound, so fewer blocks costs
        // ~nothing. Env override for experiments.
        int spm = (per_sm > 1) ? per_sm / 2 : 1;
        grid_blocks = (spm > 0 && n_sm > 0) ? spm * n_sm : 1;
        if (std::getenv("DTREE_PIPE_GB"))
            grid_blocks = std::atoi(std::getenv("DTREE_PIPE_GB"));
    }

    int up0, c0a, c0b, ct0, up1, c1a, c1b, ct1;
    dt_dtree(N, rank, up0, c0a, c0b, ct0, up1, c1a, c1b, ct1);

    gicc::DeviceCtx* d = rt.prepare();
    int data_idx = data_buf.index, recv_idx = recv_buf.index;
    int flag_idx = flag_buf.index, one_idx = one_buf.index;
    int r = rank, c = count, s = S;
    volatile unsigned int* fp = (volatile unsigned int*)d_flag;
    void* params[] = {&d, &data_idx, &recv_idx, &flag_idx, &one_idx, &r, &c, &s,
                      &up0, &c0a, &c0b, &ct0, &up1, &c1a, &c1b, &ct1,
                      &d_data, &d_recv, &fp};
    (void)hipLaunchCooperativeKernel((const void*)dtree_allreduce_pipe_kernel,
                                     dim3(grid_blocks), dim3(block_threads),
                                     params, 0, 0);
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();
}

//============================================================================
// HIERARCHICAL double binary tree all-reduce (K nodes, P ranks/node). Keeps the
// winning intra-node xGMI structure of ring_allreduce_hier_direct and replaces
// the inter-node phase with a double binary tree OVER THE K NODES (one tree per
// same-local-position group, all P groups concurrent). This is the variant that
// generalizes the 2-node hier to K>=4 nodes: the flat ring's K-1 serial cross
// steps (or the K=2-only single exchange) become a 2*log2(K)-depth tree.
// the inter-node phase with a double binary tree OVER THE K NODES (one tree per
// same-local-position group, all P groups concurrent). This is the variant that
// generalizes the 2-node hier to K>=4 nodes: the flat ring's K-1 serial cross
// steps (or the K=2-only single exchange) become a 2*log2(K)-depth tree.
//
//   Phase 1  intra-node DIRECT reduce-scatter (xGMI): rank lp sums slice lp over
//            all P same-node ranks -> owns the node-local partial of slice lp.
//   Phase 2  inter-node double tree on slice lp across the K nodes (proxy). The
//            virtual ranks are node ids; physical peer = vrank*P + lp. slice is
//            split in half, half0 over tree T0, half1 over T1 (role-swapped),
//            so every node drives a full-bandwidth subtree for one half.
//   Phase 3  intra-node DIRECT all-gather (xGMI, reuses hier_direct_ag_kernel).
//
// At K=2 the inter-node tree degenerates to a single balanced exchange (each
// node is root of one half, leaf of the other) -> same result as hier_direct,
// no real tree depth. The tree only earns its log(K) advantage at K>=4 nodes.
// NOTE: like the flat tree, phase 2 is NOT pipelined, so at K>=4 it pays a
// depth*slice critical-path cost on large messages (chunk phase 2 to fix).
// Requires count % P == 0 and (count/P) even. flag host-pinned >= 6 uints
// (memset once); recv >= count floats.
//
// STABILITY: the caller MUST rt.set_ipc_fastpath(false) (the same-node flag
// gicc::put otherwise routes through the IPC/SDMA fast path and contends with
// cross-node proxy RMA -> AMD+CXI SDMA deadlock). With that set, the flat tree
// is stable in sustained loops. This hierarchical variant has a RESIDUAL
// intermittent deadlock in kernel 1's cooperative proxy path that persists even
// with ipc_fastpath off (a single call is always reliable - correctness passes
// every size every run - but sustained back-to-back timing intermittently hangs
// in k1, never k2; the protocol is deadlock-free and correctness never fails,
// so it is a runtime/proxy-level race in the cooperative + interleaved-grid.sync
// + proxy-quiet pattern, not an algorithm bug). Because the tree has NO perf
// benefit at K=2 (it is depth-1, strictly worse than the fused `hier`), the
// coll_dtree_test timing path for this collective is opt-in (DTREE_HIER_TIME);
// the residual race is future work for the K>=4 regime where the tree helps.
//============================================================================

// Fused phase 1 + 2 (cooperative): intra-node direct reduce-scatter, then the
// inter-node double tree on slice lp across the K nodes. Fusing them into ONE
// cooperative kernel (grid.sync between) mirrors hier_direct's RS+cross kernel
// and keeps the driver at 2 kernels / 2 resets — the proven-stable structure
// (a 3rd reset/proxy-drain cycle races the CPU proxy ring). The inter-node tree
// is over VIRTUAL ranks = node ids; physical peer = vrank*P + lp.
__global__ void dtree_hier_rs_tree_kernel(gicc::DeviceCtx* ctx,
                                          int data_idx, int recv_idx, int flag_idx,
                                          int one_idx, int rank, int P, int count,
                                          int up0, int c0a, int c0b, int ct0,
                                          int up1, int c1a, int c1b, int ct1,
                                          float* data, float* recv,
                                          volatile unsigned int* flag) {
    cg::grid_group grid = cg::this_grid();
    const bool   lead = (blockIdx.x == 0 && threadIdx.x == 0);
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const int    nb   = ctx->ipc_n_bufs;
    void** ipc = ctx->peer_ipc_base;
    const int    lp        = rank % P;
    const int    node_base = rank - lp;
    const int    slice     = count / P;
    const int    hh        = slice / 2;               // half of my slice
    const size_t sbase     = (size_t)lp * slice;      // my slice region
    const int    up[2] = {up0, up1};
    const int    ca[2] = {c0a, c1a};
    const int    cb[2] = {c0b, c1b};
    const int    ct[2] = {ct0, ct1};

    // ---- intra-node direct reduce-scatter (xGMI; reads peers' originals) ----
    for (size_t e = gtid; e < (size_t)slice; e += nthr) {
        float acc = data[sbase + e];
        for (int q = 0; q < P; ++q) {
            if (q == lp) continue;
            const float* pq = (ipc && nb)
                ? (const float*)ipc[(size_t)(node_base + q) * nb + data_idx] : nullptr;
            if (pq) acc += pq[sbase + e];
        }
        data[sbase + e] = acc;
    }
    grid.sync();                                      // node-local slice ready

    // ---- inter-node double tree REDUCE (leaf node -> root node) ----
    for (int t = 0; t < 2; ++t) {
        const size_t base = sbase + (size_t)t * hh;
        const int    kid[2] = {ca[t], cb[t]};
        for (int k = 0; k < 2; ++k) {
            if (kid[k] < 0) continue;
            if (lead) {
                while (flag[t * 2 + k] == 0u) { __builtin_amdgcn_s_sleep(1); }
                flag[t * 2 + k] = 0u;
                __threadfence_system();
            }
            grid.sync();
            for (size_t i = gtid; i < (size_t)hh; i += nthr)
                data[base + i] += recv[(size_t)k * hh + i];
            grid.sync();
        }
        if (up[t] >= 0) {
            grid.sync();
            dt_send_half(ctx, grid, up[t] * P + lp, recv_idx, (size_t)ct[t] * hh,
                         data_idx, data, base, hh,
                         flag_idx, (size_t)(t * 2 + ct[t]), one_idx,
                         lead, gtid, nthr);
        }
    }

    // ---- BROADCAST (root node -> leaf node) ----
    for (int t = 0; t < 2; ++t) {
        const size_t base = sbase + (size_t)t * hh;
        if (up[t] >= 0 && lead) {
            while (flag[4 + t] == 0u) { __builtin_amdgcn_s_sleep(1); }
            flag[4 + t] = 0u;
            __threadfence_system();
        }
        grid.sync();
        const int kid[2] = {ca[t], cb[t]};
        for (int k = 0; k < 2; ++k) {
            if (kid[k] < 0) continue;
            dt_send_half(ctx, grid, kid[k] * P + lp, data_idx, base,
                         data_idx, data, base, hh,
                         flag_idx, (size_t)(4 + t), one_idx,
                         lead, gtid, nthr);
        }
    }
}

inline void allreduce_double_tree_hier(gicc::Runtime& rt,
                                       const gicc::Buffer& data_buf, float* d_data,
                                       const gicc::Buffer& recv_buf, float* d_recv,
                                       const gicc::Buffer& flag_buf, unsigned int* d_flag,
                                       const gicc::Buffer& one_buf, int count, int P) {
    const int N    = rt.size();
    const int rank = rt.rank();
    const int K    = N / P;                           // number of nodes
    const int nid  = rank / P;                        // my node id (virtual rank)

    static int gb_tree = 0;
    const int  bt = 256;
    if (gb_tree == 0) {
        int per_sm = 0, n_sm = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (const void*)dtree_hier_rs_tree_kernel, bt, 0);
        (void)hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount, rt.gpu_id());
        gb_tree = (per_sm > 0 && n_sm > 0) ? per_sm * n_sm : 1;
        if (std::getenv("DTREE_PIPE_GB"))            // debug: override grid size
            gb_tree = std::atoi(std::getenv("DTREE_PIPE_GB"));
    }

    int data_idx = data_buf.index, recv_idx = recv_buf.index;
    int flag_idx = flag_buf.index, one_idx = one_buf.index;

    static int dbg = (std::getenv("GICC_DTREEH_DBG") != nullptr) ? 1 : 0;
    #define DTH_CKPT(msg) do { if (dbg && rank == 0) \
        fprintf(stderr, "[dtreeh %d B] " msg "\n", count * 4); } while (0)

    // Kernel 1 (cooperative): intra-node direct reduce-scatter + inter-node
    // double tree over K nodes. 2 kernels / 2 resets total (the proven-stable
    // hier_direct structure; a 3rd reset/proxy-drain cycle races the ring).
    int up0, c0a, c0b, ct0, up1, c1a, c1b, ct1;
    dt_dtree(K, nid, up0, c0a, c0b, ct0, up1, c1a, c1b, ct1);
    DTH_CKPT("k1(rs+tree) launch");
    gicc::DeviceCtx* d = rt.prepare();
    int rk = rank, Px = P, c = count;
    volatile unsigned int* fp = (volatile unsigned int*)d_flag;
    void* p1[] = {&d, &data_idx, &recv_idx, &flag_idx, &one_idx, &rk, &Px, &c,
                  &up0, &c0a, &c0b, &ct0, &up1, &c1a, &c1b, &ct1,
                  &d_data, &d_recv, &fp};
    (void)hipLaunchCooperativeKernel((const void*)dtree_hier_rs_tree_kernel,
                                     dim3(gb_tree), dim3(bt), p1, 0, 0);
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();                                     // all global slices ready
    DTH_CKPT("k1 done");

    // Kernel 2: intra-node direct all-gather (reuses hier_direct_ag_kernel).
    d = rt.prepare();
    hipLaunchKernelGGL(hier_direct_ag_kernel, dim3(512), dim3(bt), 0, 0,
                       d, data_idx, N, rank, count, P, d_data);
    (void)hipDeviceSynchronize();
    rt.reset();
    rt.barrier();
    DTH_CKPT("k2(ag) done");
    #undef DTH_CKPT
}
#endif  // GICC_CPU_PROXY

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

//============================================================================
// LL (low-latency) PACKET ring all-reduce — the NCCL/UCCL trick.
//
// The arrival flag is packed INTO the data line (16B = {f0bits,flag,f1bits,
// flag}), so one RMA write carries data+flag and the receiver detects arrival
// by polling the flag in the SAME line — NO separate, ordered flag message
// (which is what made ring_allreduce_pipe a net loss). Relies on 8-byte write
// atomicity (each (data,flag) pair) so a matching flag implies valid data.
// Flags are a per-call MONOTONIC base (no per-iteration re-zeroing); distinct
// recv slot per (phase,step) so stale lines carry an older (smaller) flag the
// receiver simply never matches. Cost: 2x wire bytes (LL) — for SMALL msgs.
//
// MEASURED RESULT (2026-06-03, 8 ranks/2 nodes Tioga): CORRECT at every size
// (the flag-in-data poll works cross-node on CXI), but ~12x SLOWER than
// ring_allreduce_coop (8KB: 8441us vs 686us). This is an AMD platform issue,
// NOT the algorithm: NCCL's LL is fast on NVIDIA because a volatile global load
// there is cheap AND system-coherent, so polling NIC-written HBM is ~free.
//
// I tried FOUR poll variants to make it cheap+coherent on gfx90a, none worked:
//   (a) plain volatile uint64 load  -> stale L2, spins to eviction (8441us)
//   (b) __hip_atomic_load SYSTEM     -> coherent but serializes mem system, even slower
//   (c) __builtin_nontemporal_load (glc+slc, L2 bypass) -> still ~12x (in use here)
//   (d) pkt_recv in HOST-PINNED memory (poll over PCIe, like coop's flags) -> still slow
// So the bottleneck is NOT simply poll-coherence (host-pinned recv didn't fix
// it). The residual ~600us/step is structural to this LL path (per-step pack +
// extra grid.sync + full-GPU poll-vs-proxy interaction) and was not isolated.
// By contrast ring_allreduce_coop polls ONE tiny host-pinned flag/step (cheap)
// and reduces straight from HBM -> that design fits AMD; NCCL's flag-in-HBM-data
// fits NVIDIA. CONCLUSION: ring_allreduce_coop remains the best GICC collective
// on AMD; LL flag-in-data ports correctly but needs deeper AMD-specific work to
// be competitive. Kept as a correctness-validated prototype of the NCCL/UCCL
// approach (see reference/nccl, reference/uccl).
//============================================================================
struct LLPkt { unsigned int d0, f0, d1, f1; };   // 16B: 2 floats + flag x2

__device__ __forceinline__
bool ll_read(const LLPkt* p, unsigned int flag, float& a, float& b) {
    // Two 8-byte loads, each holds one (data,flag) pair contiguously, so a
    // matching flag-half implies the data-half of the SAME write. Use a
    // SYSTEM-scope atomic load: on AMD (gfx90a) a plain volatile global load is
    // served from L2, which the NIC's HBM write does not invalidate, so the
    // poll would spin until the line is naturally evicted (~hundreds of us).
    // The system-scope load is coherent with the NIC write.
    // Non-temporal load: on gfx90a this emits a load with the glc+slc cache
    // bits, bypassing L1+L2 to read HBM directly -> it observes the NIC's HBM
    // write immediately (coherent) AND is a single cheap load (unlike a
    // serializing system-scope atomic, and unlike a plain volatile load which
    // hits stale L2 and spins to eviction).
    const unsigned long long* q =
        reinterpret_cast<const unsigned long long*>(p);
    unsigned long long lo = __builtin_nontemporal_load(q);
    unsigned long long hi = __builtin_nontemporal_load(q + 1);
    if ((unsigned)(lo >> 32) != flag || (unsigned)(hi >> 32) != flag) return false;
    a = __uint_as_float((unsigned)lo);
    b = __uint_as_float((unsigned)hi);
    return true;
}

// Both phases in one cooperative kernel. pkt_recv has 2*(N-1) slots, each
// (chunk/2) lines: RS uses slots [0,N-1), AG uses [N-1,2N-2).
__global__ void coop_allreduce_ll_kernel(gicc::DeviceCtx* ctx,
                                         int data_idx, int pktsend_idx,
                                         int pktrecv_idx, int N, int rank,
                                         int chunk, unsigned int flag_base,
                                         float* data, LLPkt* pkt_send,
                                         LLPkt* pkt_recv) {
    cg::grid_group grid = cg::this_grid();
    const bool   lead = (blockIdx.x == 0 && threadIdx.x == 0);
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    const int    next = (rank + 1) % N;
    const size_t lines = (size_t)chunk / 2;        // 2 floats per line
    const size_t line_bytes = sizeof(LLPkt);

    // ---- reduce-scatter ----
    for (int s = 0; s < N - 1; ++s) {
        const int sc = (rank - s + N) % N;
        const int rc = (rank - 1 - s + N) % N;
        const unsigned int flag = flag_base + (unsigned)s + 1u;
        for (size_t l = gtid; l < lines; l += nthr) {
            pkt_send[l].d0 = __float_as_uint(data[(size_t)sc * chunk + 2 * l]);
            pkt_send[l].f0 = flag;
            pkt_send[l].d1 = __float_as_uint(data[(size_t)sc * chunk + 2 * l + 1]);
            pkt_send[l].f1 = flag;
        }
        __threadfence_system();                    // packed data visible to NIC
        grid.sync();
        if (lead) {
            gicc::put(ctx, next, pktrecv_idx, (size_t)s * lines * line_bytes,
                      pktsend_idx, 0, lines * line_bytes);
            gicc::quiet(ctx);                       // pkt_send reusable next step
        }
        LLPkt* slot = pkt_recv + (size_t)s * lines;
        for (size_t l = gtid; l < lines; l += nthr) {
            float a, b;
            while (!ll_read(slot + l, flag, a, b)) { __builtin_amdgcn_s_sleep(1); }
            data[(size_t)rc * chunk + 2 * l]     += a;
            data[(size_t)rc * chunk + 2 * l + 1] += b;
        }
        grid.sync();
    }
    // ---- all-gather (overwrite, no reduce) ----
    for (int s = 0; s < N - 1; ++s) {
        const int sc = (rank + 1 - s + 2 * N) % N;  // chunk I forward
        const int rc = (rank - s + N) % N;          // chunk I receive
        const unsigned int flag = flag_base + (unsigned)(N - 1 + s) + 1u;
        for (size_t l = gtid; l < lines; l += nthr) {
            pkt_send[l].d0 = __float_as_uint(data[(size_t)sc * chunk + 2 * l]);
            pkt_send[l].f0 = flag;
            pkt_send[l].d1 = __float_as_uint(data[(size_t)sc * chunk + 2 * l + 1]);
            pkt_send[l].f1 = flag;
        }
        __threadfence_system();
        grid.sync();
        if (lead) {
            gicc::put(ctx, next, pktrecv_idx,
                      (size_t)(N - 1 + s) * lines * line_bytes,
                      pktsend_idx, 0, lines * line_bytes);
            gicc::quiet(ctx);
        }
        LLPkt* slot = pkt_recv + (size_t)(N - 1 + s) * lines;
        for (size_t l = gtid; l < lines; l += nthr) {
            float a, b;
            while (!ll_read(slot + l, flag, a, b)) { __builtin_amdgcn_s_sleep(1); }
            data[(size_t)rc * chunk + 2 * l]     = a;
            data[(size_t)rc * chunk + 2 * l + 1] = b;
        }
        grid.sync();
    }
}

// Host driver. chunk must be even. pkt_send >= (chunk/2) lines; pkt_recv >=
// 2*(N-1)*(chunk/2) lines. flag_base is monotonic across calls (caller bumps
// by 2*(N-1) each call) so no flag buffer ever needs zeroing.
inline void ring_allreduce_ll(gicc::Runtime& rt,
                              const gicc::Buffer& data_buf, float* d_data,
                              const gicc::Buffer& pktsend_buf, LLPkt* d_pkt_send,
                              const gicc::Buffer& pktrecv_buf, LLPkt* d_pkt_recv,
                              int chunk, unsigned int flag_base) {
    const int N = rt.size();
    const int rank = rt.rank();

    static int grid_blocks = 0;
    const int  block_threads = 256;
    if (grid_blocks == 0) {
        int per_sm = 0, n_sm = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (const void*)coop_allreduce_ll_kernel, block_threads, 0);
        (void)hipDeviceGetAttribute(&n_sm, hipDeviceAttributeMultiprocessorCount,
                                    rt.gpu_id());
        grid_blocks = (per_sm > 0 && n_sm > 0) ? per_sm * n_sm : 1;
    }

    rt.barrier();   // no flag memset needed (monotonic flags); just line up ranks
    gicc::DeviceCtx* d = rt.prepare();
    int data_idx = data_buf.index, ps_idx = pktsend_buf.index, pr_idx = pktrecv_buf.index;
    int n = N, r = rank, c = chunk;
    unsigned int fb = flag_base;
    void* params[] = {&d, &data_idx, &ps_idx, &pr_idx, &n, &r, &c, &fb,
                      &d_data, &d_pkt_send, &d_pkt_recv};
    (void)hipLaunchCooperativeKernel((const void*)coop_allreduce_ll_kernel,
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
