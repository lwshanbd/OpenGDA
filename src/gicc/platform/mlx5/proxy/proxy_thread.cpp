/*
 * proxy_thread.cpp - CPU proxy worker loop, MLX5/verbs flavour.
 *
 * Port of src/gicc/platform/ofi/proxy/proxy_thread.cpp with the only
 * differences being:
 *   - submit / poll go through ProxyVerbs (ibv_post_send / ibv_poll_cq)
 *     instead of ProxyLibfabric (fi_write / fi_cq_read), and
 *   - retry uses -ENOMEM (verbs' SQ-full signal) instead of -FI_EAGAIN.
 *
 * Keep the two implementations structurally identical so a fix in one
 * is mechanically applicable to the other.
 */
// Suppress the inline __global__ kernels in device_opt.cuh — this is a
// pure host TU; g++ would otherwise try to parse blockIdx / clock64 /
// __syncthreads at namespace scope and fail.
#define GICC_DEVICE_OPT_SUPPRESS_KERNELS 1

#include "proxy_thread.hpp"

#include "gicc/platform/mlx5/mlx5_runtime.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace gicc::mlx5::proxy {

namespace {
constexpr int kSubmitBatch = 32;
constexpr int kCqBatch     = 256;

constexpr auto kShutdownDrainTimeout = std::chrono::seconds(5);
constexpr auto kQuietDrainTimeout    = std::chrono::seconds(30);

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

constexpr size_t ring_idx(uint64_t slot) {
    return static_cast<size_t>(slot)
         & (::gicc::proxy::kProxyRingCapacity - 1);
}
} // anon namespace

ProxyThread::ProxyThread(::gicc::Runtime& rt, int thread_idx)
    : rt_(rt)
{
    ring_host_ = ::gicc::proxy::allocate_d2h_ring_host
                    <::gicc::proxy::kProxyRingCapacity>(&ring_device_);

    // Build the verbs fleet (CQ + per-peer QPs). Reaches into Runtime
    // for ib_context / pd / Bootstrap.
    fleet_ = std::make_unique<ProxyQpFleet>(rt_.proxy_ib_context(),
                                            rt_.proxy_pd(),
                                            rt_.boot(),
                                            thread_idx);
    verbs_ = std::make_unique<ProxyVerbs>(rt_, *fleet_);
}

ProxyThread::~ProxyThread() {
    stop();
    if (ring_host_) {
        ::gicc::proxy::free_d2h_ring_host
            <::gicc::proxy::kProxyRingCapacity>(ring_host_);
        ring_host_   = nullptr;
        ring_device_ = nullptr;
    }
}

void ProxyThread::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true,
                                          std::memory_order_acq_rel)) {
        return;
    }
    thr_ = std::thread(&ProxyThread::main_loop, this);
}

void ProxyThread::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false,
                                          std::memory_order_acq_rel)) {
        return;
    }
    if (thr_.joinable()) {
        thr_.join();
    }
}

void ProxyThread::main_loop() {
    using ::gicc::proxy::TransferCmd;
    using ::gicc::proxy::CmdType;

    while (running_.load(std::memory_order_acquire)) {
        bool stop_submitting = false;

        // 0. Reissue any retry stashed by a prior -ENOMEM.
        if (pending_retry_) {
            int ret = verbs_->submit_write(pending_retry_->cmd,
                                           pending_retry_->slot);
            if (ret == -ENOMEM) {
                stop_submitting = true;
            } else {
                size_t bit = ring_idx(pending_retry_->slot);
                in_flight_.set(bit);
                ++in_flight_count_;
                pending_retry_.reset();
            }
        }

        // 1. Drain ring (bounded for fairness).
        for (int i = 0; !stop_submitting && i < kSubmitBatch; ++i) {
            TransferCmd c;
            uint64_t    slot;
            if (!ring_host_->pop(c, &slot)) break;

            size_t bit = ring_idx(slot);
            if (in_flight_.test(bit)) {
                fprintf(stderr,
                    "ProxyThread(mlx5): pop returned in-flight slot %lu "
                    "(invariant violation)\n", (unsigned long)slot);
                continue;
            }

            switch (c.cmd_type) {
                case CmdType::WRITE: {
                    int ret = verbs_->submit_write(c, slot);
                    if (ret == -ENOMEM) {
                        pending_retry_ = PendingRetry{c, slot};
                        stop_submitting = true;
                        break;
                    }
                    in_flight_.set(bit);
                    ++in_flight_count_;
                    break;
                }
                case CmdType::ATOMIC: {
                    int ret = verbs_->submit_atomic_add(c, slot);
                    if (ret == -ENOMEM) {
                        pending_retry_ = PendingRetry{c, slot};
                        stop_submitting = true;
                        break;
                    }
                    in_flight_.set(bit);
                    ++in_flight_count_;
                    break;
                }
                case CmdType::QUIET:
                    handle_quiet(slot);
                    break;
                default:
                    fprintf(stderr,
                        "ProxyThread(mlx5): unexpected cmd_type %u\n",
                        (unsigned)c.cmd_type);
                    std::abort();
            }
        }

        // 2. Poll CQ.
        Completion comps[kCqBatch];
        int n = verbs_->poll(comps, kCqBatch);
        for (int i = 0; i < n; ++i) {
            uint64_t s = reinterpret_cast<uint64_t>(comps[i].context);
            ring_host_->mark_acked(s);
            size_t bit = ring_idx(s);
            if (in_flight_.test(bit)) {
                in_flight_.reset(bit);
                --in_flight_count_;
            }
        }
        if (n > 0) {
            ring_host_->advance_tail_from_mask();
        } else {
            cpu_relax();
        }
    }

    // Drain residual completions on shutdown, bounded.
    auto drain_deadline = std::chrono::steady_clock::now()
                        + kShutdownDrainTimeout;
    while (in_flight_count_ > 0 &&
           std::chrono::steady_clock::now() < drain_deadline) {
        Completion comps[kCqBatch];
        int n = verbs_->poll(comps, kCqBatch);
        for (int i = 0; i < n; ++i) {
            uint64_t s = reinterpret_cast<uint64_t>(comps[i].context);
            ring_host_->mark_acked(s);
            size_t bit = ring_idx(s);
            if (in_flight_.test(bit)) {
                in_flight_.reset(bit);
                --in_flight_count_;
            }
        }
        if (n > 0) ring_host_->advance_tail_from_mask();
        else      cpu_relax();
    }
    if (in_flight_count_ > 0) {
        fprintf(stderr,
            "ProxyThread(mlx5): shutdown drain timeout, %zu completions "
            "still in flight (leaked)\n", in_flight_count_);
    }
}

void ProxyThread::handle_quiet(uint64_t quiet_slot) {
    size_t target_remaining = in_flight_count_;
    auto deadline = std::chrono::steady_clock::now() + kQuietDrainTimeout;
    while (target_remaining > 0 &&
           std::chrono::steady_clock::now() < deadline) {
        Completion comps[kCqBatch];
        int n = verbs_->poll(comps, kCqBatch);
        for (int i = 0; i < n; ++i) {
            uint64_t s = reinterpret_cast<uint64_t>(comps[i].context);
            ring_host_->mark_acked(s);
            size_t bit = ring_idx(s);
            if (in_flight_.test(bit)) {
                in_flight_.reset(bit);
                --in_flight_count_;
                --target_remaining;
            }
        }
        if (n == 0) cpu_relax();
    }
    if (target_remaining > 0) {
        fprintf(stderr,
            "ProxyThread(mlx5): QUIET drain timeout (slot=%lu, %zu still "
            "in flight)\n", (unsigned long)quiet_slot, target_remaining);
    }
    ring_host_->mark_acked(quiet_slot);
    ring_host_->advance_tail_from_mask();
}

} // namespace gicc::mlx5::proxy
