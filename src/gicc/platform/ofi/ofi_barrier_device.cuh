/**
 * ofi_barrier_device.cuh - GPU-triggered dissemination barrier (device side)
 *
 * Provides BarrierCtx and __device__ barrier() for the CXI/OFI backend.
 *
 * The dissemination barrier executes from the GPU via DWQ-triggered RDMA puts.
 * For N ranks it requires ceil(log2(N)) rounds. In round k:
 *   - Rank i writes a signal to rank (i + 2^k) mod N
 *   - Rank i polls for a signal from rank (i - 2^k + N) mod N
 *
 * The GPU triggers pre-queued DWQ puts by writing to per-round MMIO trigger
 * counters, then polls host-visible signal slots for arrival.
 *
 * Call barrier() from exactly one thread (threadIdx.x == 0 && blockIdx.x == 0).
 */
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace gicc {

/// Number of signal slots per round to avoid overwrites between consecutive barriers
constexpr int BARRIER_SIGNAL_SLOTS = 2;

/**
 * GPU-accessible context for the dissemination barrier.
 *
 * Allocated by gicc::Barrier on the host, passed to kernels as a device pointer.
 */
struct BarrierCtx {
    int n_rounds;                          ///< ceil(log2(size))
    int rank;
    int size;
    int n_signal_slots;                    ///< = BARRIER_SIGNAL_SLOTS
    volatile uint64_t* signals;            ///< Signal buffers [n_rounds * n_signal_slots]
    volatile uint64_t** trigger_addrs;     ///< Per-round MMIO trigger addresses
    /// Per-round IPC fast path. peer_signal_bases[k] is the peer's signals
    /// array mapped into our address space (via hipIpcOpenMemHandle), or
    /// nullptr if round k targets an off-node peer and must go through the
    /// CXI DWQ trigger path. When non-null, the kernel writes the signal
    /// directly via `peer_signal_bases[k][k * n_signal_slots + slot]`.
    volatile uint64_t** peer_signal_bases;
    uint64_t dev_seen;                     ///< Last barrier sequence number the
                                           ///< kernel finished. Device-owned:
                                           ///< host initialises to 0 and never
                                           ///< writes it again. Persists across
                                           ///< kernel launches in single-barrier
                                           ///< mode and across barrier() calls
                                           ///< in continuous mode.
    volatile uint64_t* done_counter;       ///< GPU->CPU notification for continuous mode
    volatile uint64_t* ready_counter;      ///< CPU->GPU notification: monotonic
                                           ///< barrier sequence number bumped by
                                           ///< host setup() once per barrier.
    int notify_done;                       ///< Nonzero when host monitor needs done_counter
};

/**
 * GPU device function — execute one dissemination barrier.
 *
 * IMPORTANT: Call from exactly one thread (threadIdx.x == 0 && blockIdx.x == 0).
 *
 * Usage:
 *   __global__ void my_kernel(..., gicc::BarrierCtx* bctx) {
 *       // ... compute ...
 *       if (threadIdx.x == 0 && blockIdx.x == 0)
 *           gicc::barrier(bctx);
 *       __syncthreads();
 *       // ... continue ...
 *   }
 */
__device__ __forceinline__
void barrier(BarrierCtx* ctx) {
    // Wait for the host to bump ready_counter past the last barrier we
    // finished. The new value is this barrier's sequence number — there is
    // no separate expected_signal field, so the host avoids a per-barrier
    // hipMemcpy. dev_seen is device-owned and persists across launches.
    uint64_t seen = ctx->dev_seen;
    while (*ctx->ready_counter <= seen) {}
    uint64_t expected = seen + 1;

    // Orders pre-barrier GPU writes before any barrier signal is published.
    __threadfence_system();

    // Signal slot for this barrier (alternates to avoid overwrites)
    int slot = expected % ctx->n_signal_slots;

    for (int k = 0; k < ctx->n_rounds; k++) {
        int sig_idx = k * ctx->n_signal_slots + slot;

        if (ctx->peer_signal_bases != nullptr
            && ctx->peer_signal_bases[k] != nullptr)
        {
            // IPC fast path: the peer in round k shares our node. Write
            // the signal directly to their signals[] array via the mapping
            // opened at init time — no NIC round-trip.
            //
            // Match NVSHMEM's direct-store wait loop shape: publish the
            // signal, then fence on the sender side so the peer can observe
            // the HIP IPC store promptly. Do not make the receiver's hot
            // spin loop an acquire load; NVSHMEM uses volatile polling here.
            ctx->peer_signal_bases[k][sig_idx] = expected;
            __threadfence_system();
        } else {
            // Remote: ring the per-round CXI trigger counter so libfabric
            // fires the DWQ op queued by the host's setup().
            *ctx->trigger_addrs[k] = expected;
            // The trigger counter is an MMIO mapping. Keep this fence so the
            // doorbell reaches the NIC before this thread starts waiting.
            __threadfence_system();
        }

        // Wait for signal from peer. Use < (not !=) because a fast peer in
        // continuous mode can overtake us and overwrite the slot with a
        // larger threshold before we observe ours; libfabric preserves
        // FIFO per endpoint and thresholds are monotonically increasing,
        // so any value >= expected means the peer has reached at least
        // this point.
        while ((int64_t)(ctx->signals[sig_idx] - expected) < 0) {}
#ifdef GICC_OFI_BARRIER_STRICT_FENCES
        __threadfence_system();
#endif
    }

    // Publish completion of this barrier so subsequent barrier() calls (in
    // either continuous mode or a later kernel launch) wait for the *next*
    // ready_counter bump, not this one.
    ctx->dev_seen = expected;

    if (ctx->notify_done) {
        // Continuous mode needs a GPU->CPU completion signal for the monitor
        // thread. Single-barrier mode is synchronized by the kernel launch and
        // OFI completion counters, so avoid this host-visible atomic/fence.
        atomicAdd((unsigned long long*)ctx->done_counter, 1ULL);
        __threadfence_system();
    }
}

/**
 * Simple test kernel that just calls barrier().
 */
__global__ void barrier_kernel(BarrierCtx* ctx) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    gicc::barrier(ctx);
}

} // namespace gicc
