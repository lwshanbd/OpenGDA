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

        // Counters
        for (auto& cp : counter_pairs_)
            comm_.fabric->destroy_counter_pair(cp);
        counter_pairs_.clear();

        // Device context
        if (d_context_)       hipFree(d_context_);
        if (d_trigger_addrs_) hipFree(d_trigger_addrs_);
        d_context_ = nullptr;
        d_trigger_addrs_ = nullptr;

        // Signal source buffers (double-buffered)
        for (int i = 0; i < N_SIGNAL_BUFFERS; i++) {
            delete mr_signal_values_[i];
            mr_signal_values_[i] = nullptr;
            if (h_signal_values_[i]) hipHostFree(h_signal_values_[i]);
            h_signal_values_[i] = nullptr;
        }
        delete mr_signals_;
        mr_signals_ = nullptr;
        if (d_signals_) hipHostFree(d_signals_);
        d_signals_ = nullptr;

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

        int slot = threshold % BARRIER_SIGNAL_SLOTS;

        for (int k = 0; k < n_rounds_; k++) {
            int peer = (comm_.rank() + (1 << k)) % comm_.size();

            auto* dwq = new DwqWorkBuilder(comm_.rank());

            int sig_idx = k * BARRIER_SIGNAL_SLOTS + slot;
            uint64_t remote_offset = sig_idx * sizeof(uint64_t);
            uint64_t remote_addr = comm_.is_virt_addr_mode()
                ? (remote_signal_addrs_[k] + remote_offset)
                : remote_offset;

            dwq->queue_rma_write(
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

            dwq_ops_.push_back(dwq);
        }

        // Signal GPU that DWQ ops are ready
        __atomic_add_fetch(h_ready_counter_, 1, __ATOMIC_SEQ_CST);
    }

    /** Wait for barrier completion on CPU side. */
    void wait_completion() {
        uint64_t expected = barrier_count_ + 1;
        for (int k = 0; k < n_rounds_; k++) {
            while (fi_cntr_read(counter_pairs_[k].completion_cntr) < expected)
                fi_cq_read(comm_.fabric->cq, NULL, 0);
        }
    }

    /** Reset for next barrier. Call after wait_completion(). */
    void reset() {
        for (auto* op : dwq_ops_) delete op;
        dwq_ops_.clear();

        barrier_count_++;

        uint64_t expected = barrier_count_;
        for (int k = 0; k < n_rounds_; k++) {
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

        // Seed expected_signal for the first barrier before flipping into
        // continuous mode, so setup() will skip the write (kernel owns the
        // field from here on).
        h_context_.expected_signal = barrier_count_ + 1;
        if (d_context_ && n_rounds_ > 0) {
            hipMemcpy(&d_context_->expected_signal, &h_context_.expected_signal,
                      sizeof(uint64_t), hipMemcpyHostToDevice);
        }

        continuous_mode_ = true;
        setup();
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

    // Device context
    BarrierCtx  h_context_;
    BarrierCtx* d_context_;

    // Double-buffered signal source
    uint64_t*     h_signal_values_[N_SIGNAL_BUFFERS];
    MemoryRegion* mr_signal_values_[N_SIGNAL_BUFFERS];

    // Pending DWQ ops
    std::vector<DwqWorkBuilder*> dwq_ops_;

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

                if (barrier_count_ < target_barrier_count_.load(std::memory_order_relaxed))
                    setup();
                else
                    continuous_mode_ = false;
            }
        }
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

        size_t signals_size = n_rounds_ * BARRIER_SIGNAL_SLOTS * sizeof(uint64_t);
        hipHostMalloc(&d_signals_, signals_size, hipHostMallocDefault);
        memset((void*)d_signals_, 0, signals_size);

        mr_signals_ = new MemoryRegion(
            comm_.fabric->domain, comm_.fabric->ep, comm_.fabric->cxi_info,
            d_signals_, signals_size, false, comm_.gpu_id(), comm_.rank());

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

        uint64_t my_base = (uint64_t)d_signals_;
        uint64_t my_key  = mr_signals_->key;

        int sz = comm_.size();

        auto all_bases = comm_.boot.template allgather_fixed<uint64_t>(my_base);
        auto all_keys  = comm_.boot.template allgather_fixed<uint64_t>(my_key);

        remote_signal_addrs_.resize(n_rounds_);
        remote_signal_keys_.resize(n_rounds_);

        int rank = comm_.rank();
        for (int k = 0; k < n_rounds_; k++) {
            int peer = (rank + (1 << k)) % sz;
            remote_signal_addrs_[k] = all_bases[peer];
            remote_signal_keys_[k]  = all_keys[peer];
        }
    }

    void create_counters() {
        counter_pairs_.resize(n_rounds_);
        for (int k = 0; k < n_rounds_; k++)
            counter_pairs_[k] = comm_.fabric->create_counter_pair();
    }

    void setup_device_context() {
        // Skip the trigger-address array when there are no rounds — hipMalloc
        // with size 0 is implementation-defined and the kernel never indexes
        // trigger_addrs when n_rounds == 0.
        if (n_rounds_ > 0) {
            hipMalloc(&d_trigger_addrs_, n_rounds_ * sizeof(uint64_t*));

            std::vector<volatile uint64_t*> h_trig(n_rounds_);
            for (int k = 0; k < n_rounds_; k++)
                h_trig[k] = counter_pairs_[k].dev_trigger_cntr;
            hipMemcpy(d_trigger_addrs_, h_trig.data(),
                      n_rounds_ * sizeof(uint64_t*), hipMemcpyHostToDevice);
        }

        h_context_.n_rounds        = n_rounds_;
        h_context_.rank            = comm_.rank();
        h_context_.size            = comm_.size();
        h_context_.n_signal_slots  = BARRIER_SIGNAL_SLOTS;
        h_context_.signals         = d_signals_;
        h_context_.trigger_addrs   = d_trigger_addrs_;
        h_context_.expected_signal = 0;
        h_context_.done_counter    = h_done_counter_;
        h_context_.ready_counter   = h_ready_counter_;

        hipMalloc(&d_context_, sizeof(BarrierCtx));
        hipMemcpy(d_context_, &h_context_, sizeof(BarrierCtx),
                  hipMemcpyHostToDevice);
    }
};

} // namespace gicc
