/*
 * proxy_thread.cpp - CPU proxy worker loop implementation.
 *
 * One worker thread services one D2HRing. Submission and CQ polling share
 * the same loop iteration so a stalled CQ never blocks new submits and a
 * full submit batch never starves the CQ. Bounded batches give fairness;
 * cpu_relax() gives idle-friendly back-off when both sides have nothing
 * to do.
 */
#include "proxy_thread.hpp"

#include "gicc/platform/ofi/ofi_runtime.hpp"

#include <rdma/fi_errno.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace gicc {
namespace proxy {

namespace {
constexpr int kSubmitBatch = 32;
// CQ poll batch — bumped up from 32 (UCCL-EP polls up to 2048). 256 drains
// a typical burst in one syscall and amortizes the per-fi_cq_read overhead.
constexpr int kCqBatch     = 256;

// Bounded drain timeouts. Without these, a dropped completion (peer dead,
// provider stuck) would hang ~ProxyThread() forever, and a hung QUIET would
// deadlock the GPU's quiet() spin. On timeout we log and bail; the lost
// completions are leaked at the libfabric level but the host stays
// responsive.
constexpr auto kShutdownDrainTimeout = std::chrono::seconds(5);
constexpr auto kQuietDrainTimeout    = std::chrono::seconds(30);

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}
} // namespace

ProxyThread::ProxyThread(::gicc::Runtime& rt)
    : rt_(rt)
    , ring_host_(nullptr)
    , ring_device_(nullptr)
    , running_(false)
    , lf_(rt.fabric(), rt)
{
    ring_host_ = allocate_d2h_ring_host<kProxyRingCapacity>(&ring_device_);
}

ProxyThread::~ProxyThread() {
    stop();
    if (ring_host_) {
        free_d2h_ring_host<kProxyRingCapacity>(ring_host_);
        ring_host_   = nullptr;
        ring_device_ = nullptr;
    }
}

void ProxyThread::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true,
                                          std::memory_order_acq_rel)) {
        return;  // already running
    }
    thr_ = std::thread(&ProxyThread::main_loop, this);
}

void ProxyThread::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false,
                                          std::memory_order_acq_rel)) {
        return;  // not running
    }
    if (thr_.joinable()) {
        thr_.join();
    }
}

// Helpers to map a monotonic slot index to its ring-bitset position and
// to mutate in_flight_ + in_flight_count_ together.
namespace {
constexpr size_t ring_idx(uint64_t slot) {
    return static_cast<size_t>(slot) & (kProxyRingCapacity - 1);
}
} // namespace

void ProxyThread::main_loop() {
    while (running_.load(std::memory_order_acquire)) {
        bool stop_submitting = false;

        // 0. Reissue any retry stashed by a prior -FI_EAGAIN. Must clear
        //    before any new pop(), otherwise the failed (cmd, slot) is lost
        //    (pop() already advanced proxy_read_cursor when it was first
        //    popped). At most one PendingRetry exists at a time because the
        //    proxy is single-threaded.
        if (pending_retry_) {
            int ret = lf_.submit_write(pending_retry_->cmd,
                                       pending_retry_->slot);
            if (ret == -FI_EAGAIN) {
                stop_submitting = true;   // still no room; just poll CQ.
            } else {
                size_t bit = ring_idx(pending_retry_->slot);
                in_flight_.set(bit);
                ++in_flight_count_;
                pending_retry_.reset();
            }
        }

        // 1. Drain ring (bounded for fairness).
        //
        // Reverted from a batched flush_batch / FI_MORE refactor (the
        // ProxyLibfabric::submit_writes_batch API is still there for any
        // future provider that benefits from FI_MORE) — on the cxi
        // provider Slingshot uses, the per-op fi_writemsg overhead
        // wiped out the doorbell-amortization win, and the
        // mixed-with-QUIET path introduced a regression in the L4/L5
        // ack ordering. Per-cmd submit_write is simpler and equivalent
        // perf-wise on the systems we measure today.
        for (int i = 0; !stop_submitting && i < kSubmitBatch; ++i) {
            TransferCmd c;
            uint64_t    slot;
            if (!ring_host_->pop(c, &slot)) break;
            // Defensive: pop() advances proxy_read_cursor, so the same slot
            // shouldn't be returned twice. If it ever is, log + skip (don't
            // bail the whole batch — that would starve the next slots).
            size_t bit = ring_idx(slot);
            if (in_flight_.test(bit)) {
                fprintf(stderr,
                        "ProxyThread: pop returned in-flight slot %lu (invariant violation)\n",
                        (unsigned long)slot);
                continue;
            }

            switch (c.cmd_type) {
                case CmdType::WRITE: {
                    int ret = lf_.submit_write(c, slot);
                    if (ret == -FI_EAGAIN) {
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
                            "ProxyThread: unexpected cmd_type %u\n",
                            (unsigned)c.cmd_type);
                    std::abort();
            }
        }

        // 2. Poll CQ.
        Completion comps[kCqBatch];
        int n = lf_.poll(comps, kCqBatch);
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

    // Drain remaining completions on shutdown so we don't leak in-flight ops,
    // bounded by kShutdownDrainTimeout so a dropped completion can't hang
    // the destructor forever.
    auto drain_deadline = std::chrono::steady_clock::now() + kShutdownDrainTimeout;
    while (in_flight_count_ > 0 &&
           std::chrono::steady_clock::now() < drain_deadline) {
        Completion comps[kCqBatch];
        int n = lf_.poll(comps, kCqBatch);
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
    if (in_flight_count_ > 0) {
        fprintf(stderr,
                "ProxyThread: shutdown drain timeout, %zu completions still in flight (leaked)\n",
                in_flight_count_);
    }
}

void ProxyThread::handle_quiet(uint64_t quiet_slot) {
    // Wait for the slots in-flight at QUIET-arrival time to all complete.
    // The proxy is single-threaded, so nothing else can ADD to in_flight_
    // while we are inside this function (we are inside main_loop's submit
    // batch loop and have not yet popped further). So we don't need a
    // snapshot copy — we just remember the count we entered with and
    // wait for it to drain to zero. Bounded by kQuietDrainTimeout so a
    // dropped completion doesn't hang the GPU's quiet() spin forever.
    size_t target_remaining = in_flight_count_;
    auto deadline = std::chrono::steady_clock::now() + kQuietDrainTimeout;
    while (target_remaining > 0 &&
           std::chrono::steady_clock::now() < deadline) {
        Completion comps[kCqBatch];
        int n = lf_.poll(comps, kCqBatch);
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
                "ProxyThread: QUIET drain timeout (slot=%lu, %zu still in flight)\n",
                (unsigned long)quiet_slot, target_remaining);
        // Still ack the QUIET slot below so the GPU's quiet() spin doesn't
        // deadlock. The unfinished puts are now considered lost; production
        // code should surface this as an error to the caller (out of MVP).
    }
    // Ack the QUIET slot itself once the prior submits are quiesced (or on
    // timeout, to release the GPU spin).
    ring_host_->mark_acked(quiet_slot);
    ring_host_->advance_tail_from_mask();
}

} // namespace proxy
} // namespace gicc
