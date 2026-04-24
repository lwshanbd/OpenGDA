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

        // DWQ ops
        for (auto* op : dwq_ops_) delete op;
        dwq_ops_.clear();
        clear_continuous_prequeue();

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

    /** Queue DWQ operations for the next barrier. */
    void setup() {
        uint64_t threshold = barrier_count_ + 1;

        // Single-rank case: no peers, no DWQ ops. Still bump ready_counter so
        // the device-side barrier() loop terminates. Host only writes
        // expected_signal in single-barrier mode — in continuous mode the
        // kernel owns that field (same rule as the n_rounds_ > 0 path below).
        if (n_rounds_ == 0) {
            if (!continuous_mode_.load(std::memory_order_relaxed)) {
                h_context_.expected_signal = threshold;
                if (d_context_) {
                    hipMemcpy(&d_context_->expected_signal,
                              &h_context_.expected_signal,
                              sizeof(uint64_t), hipMemcpyHostToDevice);
                }
            }
            __atomic_add_fetch(h_ready_counter_, 1, __ATOMIC_SEQ_CST);
            return;
        }

        int buf_idx = barrier_count_ % N_SIGNAL_BUFFERS;
        uint64_t* h_sig = h_signal_values_[buf_idx];
        MemoryRegion* mr_sig = mr_signal_values_[buf_idx];

        *h_sig = threshold;

        // Update expected_signal on device (single mode only)
        if (!continuous_mode_.load(std::memory_order_relaxed)) {
            h_context_.expected_signal = threshold;
            hipMemcpy(&d_context_->expected_signal, &h_context_.expected_signal,
                      sizeof(uint64_t), hipMemcpyHostToDevice);
        }

        for (int k = 0; k < n_rounds_; k++) {
            // Local round: the kernel will write the peer's slot directly
            // via the IPC-mapped pointer — nothing for the host to queue.
            if (round_is_local_[k]) continue;

            // Reuse the pre-allocated DwqWorkBuilder for round k.
            queue_round_write(k, threshold, h_sig, mr_sig->desc, dwq_ops_[k]);
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
        // DwqWorkBuilders are pooled in the ctor and reused across barriers.
        // Nothing to free here — setup() will repopulate their fields.

        barrier_count_++;

        uint64_t expected = barrier_count_;
        for (int k = 0; k < n_rounds_; k++) {
            if (round_is_local_[k]) continue;
            while (fi_cntr_read(counter_pairs_[k].completion_cntr) < expected)
                fi_cq_read(comm_.fabric->cq, NULL, 0);
        }

        // Drain any remaining completions so libfabric releases DWQ resources
        // promptly. fi_cq_read returns the number of completions read (> 0)
        // or a negative error code (e.g. -FI_EAGAIN) when the CQ is empty.
        while (fi_cq_read(comm_.fabric->cq, NULL, 0) > 0) { }
    }

    // =========================================================================
    // Continuous mode: multiple barriers in single kernel
    // =========================================================================

    /** Start continuous mode for num_barriers barriers.
     *
     *  expected_signal ownership contract:
     *    - single-barrier mode: the host writes expected_signal = threshold
     *      on every setup() call; the kernel only reads it.
     *    - continuous mode: the host writes expected_signal exactly once
     *      here at start_continuous (the first barrier's threshold), then
     *      continuous_mode_ is turned on and subsequent setup() calls skip
     *      the write. The kernel is expected to increment expected_signal
     *      itself after each barrier() call for the rest of the run.
     */
    void start_continuous(uint64_t num_barriers) {
        ever_active_ = true;
        target_barrier_count_ = barrier_count_ + num_barriers;
        clear_continuous_prequeue();

        // Seed expected_signal for the first barrier before flipping into
        // continuous mode, so setup() will skip the write (kernel owns the
        // field from here on). d_context_ is unconditionally allocated by
        // setup_device_context() regardless of n_rounds_, so the same guard
        // applies as in the single-barrier path in setup() — no n_rounds_
        // clause needed.
        h_context_.expected_signal = barrier_count_ + 1;
        if (d_context_) {
            hipMemcpy(&d_context_->expected_signal, &h_context_.expected_signal,
                      sizeof(uint64_t), hipMemcpyHostToDevice);
        }

        prequeue_continuous(num_barriers);
        continuous_mode_ = true;
    }

    /** Wait for all continuous barriers to complete. */
    void wait_continuous() {
        while (barrier_count_ < target_barrier_count_.load())
            std::this_thread::yield();
        continuous_mode_ = false;
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    BarrierCtx* device_ctx() const { return d_context_; }
    uint64_t    count()      const { return barrier_count_; }

private:
    static constexpr int N_SIGNAL_BUFFERS = 2;

    Fabric& comm_;
    int      n_rounds_;
    uint64_t barrier_count_;

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

    // Pending DWQ ops
    std::vector<DwqWorkBuilder*> dwq_ops_;

    // Continuous-mode prequeue. NVSHMEM's in-kernel barrier does not wait for
    // the host to rearm each epoch. For OFI/CXI DWQ, emulate that by queuing
    // every remote-round work item for the requested continuous run up front.
    std::vector<DwqWorkBuilder*> continuous_dwq_ops_;
    uint64_t*     h_continuous_signal_values_ = nullptr;
    MemoryRegion* mr_continuous_signal_values_ = nullptr;
    std::atomic<bool> continuous_prequeued_{false};

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

                if (barrier_count_ < target_barrier_count_.load(std::memory_order_relaxed)
                    && !continuous_prequeued_.load(std::memory_order_relaxed))
                {
                    setup();
                } else if (barrier_count_ >=
                           target_barrier_count_.load(std::memory_order_relaxed)) {
                    continuous_mode_ = false;
                }
            }
        }
    }

    void queue_round_write(int k, uint64_t threshold, uint64_t* src, void* desc,
                           DwqWorkBuilder* op) {
        int peer = (comm_.rank() + (1 << k)) % comm_.size();
        int slot = threshold % BARRIER_SIGNAL_SLOTS;
        int sig_idx = k * BARRIER_SIGNAL_SLOTS + slot;
        uint64_t remote_offset = sig_idx * sizeof(uint64_t);
        uint64_t remote_addr = comm_.is_virt_addr_mode()
            ? (remote_signal_addrs_[k] + remote_offset)
            : remote_offset;

        op->queue_rma_write(
            comm_.fabric->domain,
            comm_.fabric->ep,
            src,
            desc,
            sizeof(uint64_t),
            comm_.av_addrs[peer],
            remote_addr,
            remote_signal_keys_[k],
            counter_pairs_[k].trigger_cntr,
            counter_pairs_[k].completion_cntr,
            threshold);
    }

    void clear_continuous_prequeue() {
        for (auto* op : continuous_dwq_ops_) delete op;
        continuous_dwq_ops_.clear();

        delete mr_continuous_signal_values_;
        mr_continuous_signal_values_ = nullptr;

        if (h_continuous_signal_values_) hipHostFree(h_continuous_signal_values_);
        h_continuous_signal_values_ = nullptr;
        continuous_prequeued_.store(false, std::memory_order_relaxed);
    }

    void prequeue_continuous(uint64_t num_barriers) {
        if (num_barriers == 0) return;

        if (n_rounds_ == 0) {
            __atomic_add_fetch(h_ready_counter_, num_barriers, __ATOMIC_SEQ_CST);
            continuous_prequeued_.store(true, std::memory_order_relaxed);
            return;
        }

        hipHostMalloc(&h_continuous_signal_values_,
                      num_barriers * sizeof(uint64_t),
                      hipHostMallocDefault);
        for (uint64_t i = 0; i < num_barriers; i++)
            h_continuous_signal_values_[i] = barrier_count_ + i + 1;

        mr_continuous_signal_values_ = new MemoryRegion(
            comm_.fabric->domain, comm_.fabric->ep, comm_.fabric->cxi_info,
            h_continuous_signal_values_, num_barriers * sizeof(uint64_t),
            false, comm_.gpu_id(), comm_.rank());

        uint64_t remote_rounds = 0;
        for (int k = 0; k < n_rounds_; k++) {
            if (!round_is_local_[k]) remote_rounds++;
        }
        continuous_dwq_ops_.reserve(num_barriers * remote_rounds);

        for (uint64_t i = 0; i < num_barriers; i++) {
            uint64_t threshold = barrier_count_ + i + 1;
            for (int k = 0; k < n_rounds_; k++) {
                if (round_is_local_[k]) continue;

                auto* op = new DwqWorkBuilder(comm_.rank());
                queue_round_write(k, threshold,
                                  &h_continuous_signal_values_[i],
                                  mr_continuous_signal_values_->desc,
                                  op);
                continuous_dwq_ops_.push_back(op);
            }
        }

        // Let the kernel advance through the whole continuous run without
        // waiting for the monitor thread to reset/setup each barrier.
        __atomic_add_fetch(h_ready_counter_, num_barriers, __ATOMIC_SEQ_CST);
        continuous_prequeued_.store(true, std::memory_order_relaxed);
    }

    // =========================================================================
    // Initialization helpers
    // =========================================================================

    void allocate_signals() {
        // Ready/done counters are always needed — the device-side barrier()
        // spins on ready_counter even when there are zero rounds, and
        // setup() always bumps it.
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

    // Pre-allocate one DwqWorkBuilder per round. queue_rma_write() re-populates
    // every field on each setup() call, and reset() only returns once libfabric
    // has signalled completion, so reuse across barriers is safe.
    void create_dwq_pool() {
        dwq_ops_.reserve(n_rounds_);
        for (int k = 0; k < n_rounds_; k++)
            dwq_ops_.push_back(new DwqWorkBuilder(comm_.rank()));
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
        h_context_.expected_signal   = 0;
        h_context_.done_counter      = h_done_counter_;
        h_context_.ready_counter     = h_ready_counter_;

        hipMalloc(&d_context_, sizeof(BarrierCtx));
        hipMemcpy(d_context_, &h_context_, sizeof(BarrierCtx),
                  hipMemcpyHostToDevice);
    }
};

} // namespace gicc
