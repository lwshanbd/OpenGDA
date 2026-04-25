/**
 * gicc_barrier.hpp - GPU-triggered Dissemination Barrier (host side)
 *
 * Implements a dissemination barrier using DWQ (Deferred Work Queue).
 * The barrier executes entirely from the GPU via RDMA puts.
 *
 * Algorithm:
 *   For N ranks, requires ceil(log2(N)) rounds.
 *   In round k (0 to n_rounds-1):
 *     - Rank i sends signal to rank (i + 2^k) mod N
 *     - Rank i receives signal from rank (i - 2^k + N) mod N
 *
 * Usage Mode 1: Single barrier per kernel launch
 *   gicc::Runtime rt;
 *   gicc::Barrier barrier(rt.fabric());
 *   barrier.init();
 *
 *   for (int iter = 0; iter < N; iter++) {
 *       barrier.setup();
 *       my_kernel<<<...>>>(args, barrier.device_ctx());
 *       hipDeviceSynchronize();
 *       barrier.reset();
 *   }
 *
 *   barrier.finalize();
 *
 * Usage Mode 2: Multiple barriers in single kernel (continuous mode)
 *   gicc::Barrier barrier(rt.fabric());
 *   barrier.init();
 *
 *   barrier.start_continuous(100);
 *   my_kernel<<<...>>>(args, barrier.device_ctx());
 *   barrier.wait_continuous();
 *
 *   barrier.finalize();
 */
#pragma once

#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

#include "fabric.hpp"
#include "dwq_work_builder.hpp"
#include "gicc/platform/ofi/ofi_barrier_device.cuh"

namespace gicc {

class Barrier {
public:
    static constexpr int MAX_ROUNDS = 12;  // up to 4096 ranks

    explicit Barrier(Fabric& comm_)
        : comm_(comm_),
          barrier_count_(0),
          n_rounds_(0),
          d_signals_(nullptr), mr_signals_(nullptr),
          d_trigger_addrs_(nullptr), d_context_(nullptr),
          h_signal_values_{nullptr, nullptr},
          mr_signal_values_{nullptr, nullptr},
          h_done_counter_(nullptr),
          h_ready_counter_(nullptr)
    {
        // ceil(log2(size))
        int sz = comm_.size();
        while ((1 << n_rounds_) < sz) n_rounds_++;

        if (n_rounds_ > MAX_ROUNDS) {
            fprintf(stderr, "Rank %d: too many ranks (%d) for barrier\n",
                    comm_.rank(), sz);
            exit(1);
        }

        allocate_signals();
        exchange_addresses();
        create_counters();
        create_dwq_pool();
        setup_device_context();
    }

    ~Barrier() {
        finalize();
    }

    Barrier(const Barrier&) = delete;
    Barrier& operator=(const Barrier&) = delete;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /** Spawn the monitor thread. Required for continuous mode; optional for
     *  single-barrier mode. Safe to call exactly zero or one time. */
    void init() {
        if (thread_running_.load()) return;

        thread_running_ = true;
        ever_active_ = false;
        continuous_mode_ = false;
        target_barrier_count_ = 0;

        monitor_thread_ = std::thread(&Barrier::monitor_loop, this);
    }

    /** Stop the monitor thread (if running) and release every resource the
     *  constructor allocated. Idempotent — safe to call repeatedly or to call
     *  exactly once via the destructor when init() was never invoked. */
    void finalize() {
        // Stop the monitor thread, if one was ever spawned.
        if (thread_running_.load()) {
            thread_running_ = false;
            continuous_mode_ = false;
            if (monitor_thread_.joinable()) monitor_thread_.join();
        }

        // Every release below is guarded by a nullptr check and clears the
        // pointer, so the whole function is idempotent whether or not init()
        // ran and whether or not finalize() has been called before.

        // DWQ ops (BUILDER_PIPELINE per round)
        for (auto& pair : dwq_ops_)
            for (auto* op : pair) delete op;
        dwq_ops_.clear();

        // Counters
        for (auto& cp : counter_pairs_)
            comm_.fabric->destroy_counter_pair(cp);
        counter_pairs_.clear();

        // Device context
        if (d_context_)           hipFree(d_context_);
        if (d_trigger_addrs_)     hipFree(d_trigger_addrs_);
        if (d_peer_signal_bases_) hipFree(d_peer_signal_bases_);
        d_context_           = nullptr;
        d_trigger_addrs_     = nullptr;
        d_peer_signal_bases_ = nullptr;

        // IPC mappings to peer signal arrays (one per local round).
        for (auto* p : peer_mapped_bases_) {
            if (p) (void)hipIpcCloseMemHandle((void*)p);
        }
        peer_mapped_bases_.clear();
        round_is_local_.clear();

        // Signal source buffers (double-buffered)
        for (int i = 0; i < N_SIGNAL_BUFFERS; i++) {
            delete mr_signal_values_[i];
            mr_signal_values_[i] = nullptr;
            if (h_signal_values_[i]) hipHostFree(h_signal_values_[i]);
            h_signal_values_[i] = nullptr;
        }
        delete mr_signals_;
        mr_signals_ = nullptr;
        if (d_signals_) hipFree(d_signals_);
        d_signals_ = nullptr;
        have_ipc_handle_ = false;

        if (h_done_counter_)  hipHostFree((void*)h_done_counter_);
        if (h_ready_counter_) hipHostFree((void*)h_ready_counter_);
        h_done_counter_ = nullptr;
        h_ready_counter_ = nullptr;
    }

    // =========================================================================
    // Single-barrier mode
    // =========================================================================

    /** Queue DWQ operations for the next barrier and bump ready_counter so
     *  the kernel can advance one more step.
     *
     *  Indexing keys off `queued_count_` (head of the sliding window), not
     *  `barrier_count_` (tail). In single-barrier mode the user calls this
     *  once per kernel launch and the head/tail stay one apart, exactly the
     *  pre-Plan-B behaviour. In continuous mode `start_continuous` and
     *  `monitor_loop` may call this several times back-to-back to keep up to
     *  `PREFETCH_DEPTH` barriers queued ahead of the kernel; each call
     *  consumes a different rotating slot, so payload buffers and DWQ
     *  builders never collide.
     */
    void setup() {
        uint64_t threshold = ++queued_count_;

        // Single-rank case: no peers, no DWQ ops. Just bump ready_counter so
        // the device-side barrier() loop sees a new sequence number and
        // returns. dev_seen is kernel-owned; host never touches it.
        if (n_rounds_ == 0) {
            __atomic_add_fetch(h_ready_counter_, 1, __ATOMIC_SEQ_CST);
            return;
        }

        int buf_idx = (threshold - 1) % N_SIGNAL_BUFFERS;
        uint64_t* h_sig = h_signal_values_[buf_idx];
        MemoryRegion* mr_sig = mr_signal_values_[buf_idx];

        *h_sig = threshold;

        int slot        = threshold % BARRIER_SIGNAL_SLOTS;
        int builder_idx = (threshold - 1) % BUILDER_PIPELINE;

        // Slot-reuse wait. The slot we're about to fill last held the WQE for
        // threshold = (threshold - PREFETCH_DEPTH). In steady continuous mode
        // that send completed long ago, so this loop is a no-op; on the first
        // PREFETCH_DEPTH barriers of a run there is no prior WQE to wait for.
        if (threshold > BUILDER_PIPELINE) {
            uint64_t prev_threshold = threshold - BUILDER_PIPELINE;
            for (int k = 0; k < n_rounds_; k++) {
                if (round_is_local_[k]) continue;
                while (fi_cntr_read(counter_pairs_[k].completion_cntr) < prev_threshold)
                    fi_cq_read(comm_.fabric->cq, NULL, 0);
            }
        }

        for (int k = 0; k < n_rounds_; k++) {
            // Local round: the kernel will write the peer's slot directly
            // via the IPC-mapped pointer — nothing for the host to queue.
            if (round_is_local_[k]) continue;

            int peer = (comm_.rank() + (1 << k)) % comm_.size();

            int sig_idx = k * BARRIER_SIGNAL_SLOTS + slot;
            uint64_t remote_offset = sig_idx * sizeof(uint64_t);
            uint64_t remote_addr = comm_.is_virt_addr_mode()
                ? (remote_signal_addrs_[k] + remote_offset)
                : remote_offset;

            dwq_ops_[k][builder_idx]->queue_rma_write(
                comm_.fabric->domain,
                comm_.fabric->ep,
                h_sig,
                mr_sig->desc,
                sizeof(uint64_t),
                comm_.av_addrs[peer],
                remote_addr,
                remote_signal_keys_[k],
                counter_pairs_[k].trigger_cntr,
                counter_pairs_[k].completion_cntr,
                threshold);
        }

        // Signal GPU that DWQ ops are ready
        __atomic_add_fetch(h_ready_counter_, 1, __ATOMIC_SEQ_CST);
    }

    /** Wait for barrier completion on CPU side. */
    void wait_completion() {
        uint64_t expected = barrier_count_ + 1;
        for (int k = 0; k < n_rounds_; k++) {
            if (round_is_local_[k]) continue;  // no counter for local rounds
            while (fi_cntr_read(counter_pairs_[k].completion_cntr) < expected)
                fi_cq_read(comm_.fabric->cq, NULL, 0);
        }
    }

    /** Reset for next barrier. Call after wait_completion(). */
    void reset() {
        // DwqWorkBuilders are pooled in pairs in the ctor and ping-pong across
        // barriers. The synchronous fi_cntr wait that used to live here moved
        // into setup(): we only wait when we are about to reuse a slot, which
        // in steady state is already complete. Just bump the counter and
        // drain the CQ to let libfabric release completed WQEs.
        barrier_count_++;

        // fi_cq_read returns the number of completions read (> 0) or a
        // negative error code (e.g. -FI_EAGAIN) when the CQ is empty.
        while (fi_cq_read(comm_.fabric->cq, NULL, 0) > 0) { }
    }

    // =========================================================================
    // Continuous mode: multiple barriers in single kernel
    // =========================================================================

    /** Start continuous mode for num_barriers barriers.
     *
     *  Sequence-number ownership: the kernel owns dev_seen and increments it
     *  inside barrier(). The host owns ready_counter and bumps it once per
     *  barrier in setup(). No host->device handshake on a "next threshold"
     *  field is needed, so single-barrier and continuous modes use the same
     *  setup() path.
     */
    void start_continuous(uint64_t num_barriers) {
        ever_active_ = true;
        target_barrier_count_ = barrier_count_ + num_barriers;

        // Single-barrier mode does not update done_counter. Rebase it before
        // enabling continuous GPU->CPU notifications so monitor_loop's
        // gpu_done > barrier_count_ test remains valid after mixed usage.
        __atomic_store_n(const_cast<uint64_t*>(h_done_counter_), barrier_count_,
                         __ATOMIC_RELEASE);
        set_notify_done(true);

        continuous_mode_ = true;

        // Pre-queue up to PREFETCH_DEPTH barriers ahead of the kernel. On
        // entry the kernel sees ready_counter = queued_count_, dev_seen = 0,
        // and can run that many barriers back-to-back without waiting on the
        // host. The monitor thread refills the window one barrier per
        // detected GPU completion so the depth stays roughly constant.
        uint64_t prefetch = num_barriers < (uint64_t)PREFETCH_DEPTH
                              ? num_barriers
                              : (uint64_t)PREFETCH_DEPTH;
        for (uint64_t i = 0; i < prefetch; i++) setup();
    }

    /** Wait for all continuous barriers to complete. */
    void wait_continuous() {
        while (barrier_count_ < target_barrier_count_.load())
            std::this_thread::yield();
        continuous_mode_ = false;
        set_notify_done(false);
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    BarrierCtx* device_ctx() const { return d_context_; }
    uint64_t    count()      const { return barrier_count_; }

private:
    // Sliding-window depth. The host keeps up to PREFETCH_DEPTH DWQ ops queued
    // ahead of the GPU at all times in continuous mode, so the kernel never
    // has to round-trip to host memory between barriers in steady state. All
    // three rotating resources (host payload buffers, device signal slots,
    // DwqWorkBuilders) share this period so they alternate in lock-step.
    //
    // Constraint: BARRIER_SIGNAL_SLOTS in ofi_barrier_device.cuh must equal
    // this value — peers index our signals[] by `expected % SLOTS` and we
    // must stay coherent with that.
    static constexpr int PREFETCH_DEPTH    = 8;
    static constexpr int N_SIGNAL_BUFFERS  = PREFETCH_DEPTH;
    static_assert(BARRIER_SIGNAL_SLOTS == PREFETCH_DEPTH,
                  "BARRIER_SIGNAL_SLOTS must equal Barrier::PREFETCH_DEPTH");

    Fabric& comm_;
    int      n_rounds_;
    uint64_t barrier_count_;     ///< completed barriers (tail of window)
    uint64_t queued_count_ = 0;  ///< barriers queued to libfabric (head)
                                 ///< queued_count_ - barrier_count_ ≤ PREFETCH_DEPTH

    // Signal buffers (host-pinned, RDMA-registered)
    uint64_t*      d_signals_;
    MemoryRegion*  mr_signals_;

    // Remote signal metadata per round
    std::vector<uint64_t> remote_signal_addrs_;
    std::vector<uint64_t> remote_signal_keys_;

    // Per-round counter pairs (reused across barriers)
    std::vector<FabricDwqContext::CounterPair> counter_pairs_;

    // Trigger address array for kernel
    volatile uint64_t** d_trigger_addrs_;

    // IPC fast path. peer_mapped_bases_[k] is the peer-in-round-k's d_signals_
    // base mapped into our address space, or nullptr for remote rounds.
    // d_peer_signal_bases_ is the same array copied to device memory so the
    // kernel can index it via BarrierCtx::peer_signal_bases.
    std::vector<volatile uint64_t*>  peer_mapped_bases_;
    volatile uint64_t**              d_peer_signal_bases_ = nullptr;
    std::vector<bool>                round_is_local_;
    bool                             have_ipc_handle_ = false;
    hipIpcMemHandle_t                ipc_handle_{};

    // Device context
    BarrierCtx  h_context_;
    BarrierCtx* d_context_;

    // Double-buffered signal source
    uint64_t*     h_signal_values_[N_SIGNAL_BUFFERS];
    MemoryRegion* mr_signal_values_[N_SIGNAL_BUFFERS];

    // Sliding-window pool of DWQ work builders. Outer dim is per-round; inner
    // dim is PREFETCH_DEPTH slots. setup() picks a slot by
    // queued_count_ % PREFETCH_DEPTH so that the K most recently queued
    // barriers each have their own persistent fi_deferred_work struct. The
    // OFI completion-counter wait runs once per slot reuse (depth lag), which
    // in steady state is a no-op because the WQE we are reusing finished long
    // ago.
    static constexpr int BUILDER_PIPELINE = PREFETCH_DEPTH;
    std::vector<std::array<DwqWorkBuilder*, BUILDER_PIPELINE>> dwq_ops_;

    // Host-visible counters
    volatile uint64_t* h_done_counter_;
    volatile uint64_t* h_ready_counter_;

    // Monitor thread
    std::thread          monitor_thread_;
    std::atomic<bool>    thread_running_{false};
    std::atomic<bool>    ever_active_{false};
    std::atomic<bool>    continuous_mode_{false};
    std::atomic<uint64_t> target_barrier_count_{0};

    // =========================================================================
    // Monitor thread
    // =========================================================================

    void monitor_loop() {
        while (thread_running_.load(std::memory_order_relaxed)) {
            if (!continuous_mode_.load(std::memory_order_relaxed)) {
                if (!ever_active_.load(std::memory_order_relaxed))
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                else
                    std::this_thread::yield();
                continue;
            }

            fi_cq_read(comm_.fabric->cq, NULL, 0);

            // Acquire ordering pairs with the GPU's atomicAdd +
            // __threadfence_system() at the tail of barrier(). volatile alone
            // would not force the host to observe the new value.
            uint64_t gpu_done = __atomic_load_n(
                const_cast<uint64_t*>(h_done_counter_), __ATOMIC_ACQUIRE);
            if (gpu_done > barrier_count_) {
                reset();

                // Refill the sliding window: keep up to PREFETCH_DEPTH
                // barriers queued ahead of barrier_count_, capped by the
                // remaining barriers in this continuous run.
                uint64_t target = target_barrier_count_.load(std::memory_order_relaxed);
                while (queued_count_ < target &&
                       (queued_count_ - barrier_count_) < (uint64_t)PREFETCH_DEPTH) {
                    setup();
                }

                if (barrier_count_ >= target)
                    continuous_mode_ = false;
            }
        }
    }

    // =========================================================================
    // Initialization helpers
    // =========================================================================

    void allocate_signals() {
        // ready_counter is always needed: device-side barrier() spins on it
        // even when there are zero rounds, and setup() always bumps it.
        // done_counter is used only when continuous mode enables notify_done.
        hipHostMalloc((void**)&h_done_counter_, sizeof(uint64_t), hipHostMallocDefault);
        *h_done_counter_ = 0;

        hipHostMalloc((void**)&h_ready_counter_, sizeof(uint64_t), hipHostMallocDefault);
        *h_ready_counter_ = 0;

        // Skip signal/source buffers entirely when size <= 1 (n_rounds_ == 0):
        // there are no peers to write to, and MemoryRegion rejects zero-length
        // regions.
        if (n_rounds_ == 0) return;

        // d_signals_ is now device memory so peers can open it via
        // hipIpcGetMemHandle and write directly from a kernel on the same
        // node. The remote DWQ path works with device memory too (MR with
        // is_device=true), so the signal path is the same for both flavours
        // of peer.
        size_t signals_size = n_rounds_ * BARRIER_SIGNAL_SLOTS * sizeof(uint64_t);
        hipMalloc(&d_signals_, signals_size);
        hipMemset((void*)d_signals_, 0, signals_size);

        mr_signals_ = new MemoryRegion(
            comm_.fabric->domain, comm_.fabric->ep, comm_.fabric->cxi_info,
            d_signals_, signals_size, true, comm_.gpu_id(), comm_.rank());

        // IPC handle for same-node peers to map into their address space.
        have_ipc_handle_ = false;
        if (hipIpcGetMemHandle(&ipc_handle_, d_signals_) == hipSuccess)
            have_ipc_handle_ = true;

        for (int i = 0; i < N_SIGNAL_BUFFERS; i++) {
            hipHostMalloc(&h_signal_values_[i], sizeof(uint64_t), hipHostMallocDefault);
            *h_signal_values_[i] = 0;
            mr_signal_values_[i] = new MemoryRegion(
                comm_.fabric->domain, comm_.fabric->ep, comm_.fabric->cxi_info,
                h_signal_values_[i], sizeof(uint64_t), false,
                comm_.gpu_id(), comm_.rank());
        }
    }

    void exchange_addresses() {
        if (n_rounds_ == 0) return;  // size <= 1: no peers to exchange with

        int sz = comm_.size();
        int rank = comm_.rank();

        struct Meta {
            uint64_t          base;
            uint64_t          key;
            uint8_t           has_ipc;
            hipIpcMemHandle_t ipc;
        };
        Meta my{};
        my.base     = (uint64_t)d_signals_;
        my.key      = mr_signals_->key;
        my.has_ipc  = have_ipc_handle_ ? 1 : 0;
        my.ipc      = ipc_handle_;

        auto all = comm_.boot.template allgather_fixed<Meta>(my);

        remote_signal_addrs_.resize(n_rounds_);
        remote_signal_keys_.resize(n_rounds_);
        round_is_local_.assign(n_rounds_, false);
        peer_mapped_bases_.assign(n_rounds_, nullptr);

        // Locality map tells us which peer ranks share our node; Bootstrap
        // builds it once per process via MPI_Comm_split_type.
        std::vector<bool> locality = comm_.boot.locality_map();

        for (int k = 0; k < n_rounds_; k++) {
            int peer = (rank + (1 << k)) % sz;
            remote_signal_addrs_[k] = all[peer].base;
            remote_signal_keys_[k]  = all[peer].key;

            if (peer != rank && locality[peer] && all[peer].has_ipc) {
                void* mapped = nullptr;
                if (hipIpcOpenMemHandle(&mapped, all[peer].ipc,
                                        hipIpcMemLazyEnablePeerAccess)
                    == hipSuccess)
                {
                    round_is_local_[k]   = true;
                    peer_mapped_bases_[k] = (volatile uint64_t*)mapped;
                }
            }
        }
    }

    void create_counters() {
        counter_pairs_.resize(n_rounds_);
        for (int k = 0; k < n_rounds_; k++)
            counter_pairs_[k] = comm_.fabric->create_counter_pair();
    }

    // Pre-allocate BUILDER_PIPELINE DwqWorkBuilders per round. setup() picks a
    // slot by barrier_count_ % BUILDER_PIPELINE; the slot is reused only after
    // the OFI completion counter has caught up, which setup() now waits for
    // explicitly when needed.
    void create_dwq_pool() {
        dwq_ops_.resize(n_rounds_);
        for (int k = 0; k < n_rounds_; k++)
            for (int b = 0; b < BUILDER_PIPELINE; b++)
                dwq_ops_[k][b] = new DwqWorkBuilder(comm_.rank());
    }

    void setup_device_context() {
        // Skip the trigger-address array when there are no rounds — hipMalloc
        // with size 0 is implementation-defined and the kernel never indexes
        // trigger_addrs when n_rounds == 0.
        if (n_rounds_ > 0) {
            hipMalloc(&d_trigger_addrs_, n_rounds_ * sizeof(uint64_t*));
            std::vector<volatile uint64_t*> h_trig(n_rounds_);
            for (int k = 0; k < n_rounds_; k++) {
                // Local rounds won't dereference trigger_addrs[k], but keep
                // the entry pointing at the real counter_pair to avoid
                // dangling reads if the kernel ever walked it unguarded.
                h_trig[k] = counter_pairs_[k].dev_trigger_cntr;
            }
            hipMemcpy(d_trigger_addrs_, h_trig.data(),
                      n_rounds_ * sizeof(uint64_t*), hipMemcpyHostToDevice);

            // IPC mapped bases per round (nullptr for remote rounds). The
            // kernel uses a non-null entry to take the IPC fast path.
            hipMalloc(&d_peer_signal_bases_, n_rounds_ * sizeof(uint64_t*));
            hipMemcpy(d_peer_signal_bases_, peer_mapped_bases_.data(),
                      n_rounds_ * sizeof(uint64_t*), hipMemcpyHostToDevice);
        }

        h_context_.n_rounds          = n_rounds_;
        h_context_.rank              = comm_.rank();
        h_context_.size              = comm_.size();
        h_context_.n_signal_slots    = BARRIER_SIGNAL_SLOTS;
        h_context_.signals           = d_signals_;
        h_context_.trigger_addrs     = d_trigger_addrs_;
        h_context_.peer_signal_bases = d_peer_signal_bases_;
        h_context_.dev_seen          = 0;
        h_context_.done_counter      = h_done_counter_;
        h_context_.ready_counter     = h_ready_counter_;
        h_context_.notify_done       = 0;

        hipMalloc(&d_context_, sizeof(BarrierCtx));
        hipMemcpy(d_context_, &h_context_, sizeof(BarrierCtx),
                  hipMemcpyHostToDevice);
    }

    void set_notify_done(bool enabled) {
        h_context_.notify_done = enabled ? 1 : 0;
        if (d_context_) {
            hipMemcpy(&d_context_->notify_done, &h_context_.notify_done,
                      sizeof(h_context_.notify_done), hipMemcpyHostToDevice);
        }
    }
};

} // namespace gicc
