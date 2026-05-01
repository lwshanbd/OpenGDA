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

#include <cstdio>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace gicc {
namespace proxy {

namespace {
constexpr int kSubmitBatch = 32;
constexpr int kCqBatch     = 32;

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

void ProxyThread::main_loop() {
    while (running_.load(std::memory_order_acquire)) {
        // 1. Drain ring (bounded for fairness).
        for (int i = 0; i < kSubmitBatch; ++i) {
            TransferCmd c;
            uint64_t    slot;
            if (!ring_host_->pop(c, &slot)) break;
            // Defensive: pop() advances proxy_read_cursor, so the same slot
            // shouldn't be returned twice. If it ever is, bail this batch.
            if (in_flight_.count(slot)) break;

            switch (c.cmd_type) {
                case CmdType::WRITE: {
                    int ret = lf_.submit_write(c, slot);
                    if (ret == -FI_EAGAIN) {
                        // The slot stays "popped but not in-flight"; we will
                        // not retry it this iteration. Re-issue path will be
                        // added with proper backpressure in a later task.
                        break;
                    }
                    in_flight_.insert(slot);
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
            in_flight_.erase(s);
        }
        if (n > 0) {
            ring_host_->advance_tail_from_mask();
        } else {
            cpu_relax();
        }
    }

    // Drain remaining completions on shutdown so we don't leak in-flight ops.
    while (!in_flight_.empty()) {
        Completion comps[kCqBatch];
        int n = lf_.poll(comps, kCqBatch);
        for (int i = 0; i < n; ++i) {
            uint64_t s = reinterpret_cast<uint64_t>(comps[i].context);
            ring_host_->mark_acked(s);
            in_flight_.erase(s);
        }
        if (n > 0) {
            ring_host_->advance_tail_from_mask();
        } else {
            cpu_relax();
        }
    }
}

void ProxyThread::handle_quiet(uint64_t quiet_slot) {
    // Wait for every currently in-flight slot to ack. Snapshot the set so
    // erasing entries from in_flight_ during the loop doesn't invalidate
    // our termination condition.
    auto target = in_flight_;
    while (!target.empty()) {
        Completion comps[kCqBatch];
        int n = lf_.poll(comps, kCqBatch);
        for (int i = 0; i < n; ++i) {
            uint64_t s = reinterpret_cast<uint64_t>(comps[i].context);
            ring_host_->mark_acked(s);
            in_flight_.erase(s);
            target.erase(s);
        }
        if (n == 0) cpu_relax();
    }
    // Ack the QUIET slot itself once the prior submits are quiesced.
    ring_host_->mark_acked(quiet_slot);
    ring_host_->advance_tail_from_mask();
}

} // namespace proxy
} // namespace gicc
