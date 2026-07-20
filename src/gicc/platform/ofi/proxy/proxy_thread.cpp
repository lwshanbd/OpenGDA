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
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace gicc {
namespace proxy {

namespace {
// Submit/poll batch sizes. Tunable at runtime for cross-node proxy studies:
//   GICC_PROXY_SUBMIT_BATCH  (default 32)  cmds drained+submitted per loop pass
//   GICC_PROXY_CQ_BATCH      (default 256) completions reaped per fi_cq_read
// kCqBatchMax bounds the on-stack completion array; the runtime value is
// clamped into [1, kCqBatchMax].
constexpr int kCqBatchMax = 256;
inline int proxy_env_int(const char* name, int dflt, int lo, int hi) {
    if (const char* v = std::getenv(name)) {
        int x = std::atoi(v);
        if (x >= lo && x <= hi) return x;
    }
    return dflt;
}
const int kSubmitBatch = proxy_env_int("GICC_PROXY_SUBMIT_BATCH", 32, 1, 4096);
const int kCqBatch     = proxy_env_int("GICC_PROXY_CQ_BATCH", 256, 1, kCqBatchMax);

// Bounded drain timeouts. Without these, a dropped completion (peer dead,
// provider stuck) would hang ~ProxyThread() forever, and a hung QUIET would
// deadlock the GPU's quiet() spin. On timeout we log and bail; the lost
// completions are leaked at the libfabric level but the host stays
// responsive.
constexpr auto kShutdownDrainTimeout = std::chrono::seconds(5);
constexpr auto kQuietDrainTimeout    = std::chrono::seconds(30);

// Idle-backoff thresholds. Counted in consecutive empty-poll iterations.
// Below kIdleYield: pure cpu_relax (sub-µs back-off, no wakeup latency).
// Above kIdleYield but below kIdleSleep: std::this_thread::yield (gives up
// the slice to ready siblings on the same core, prevents starving the
// Python main thread doing host work like illum += ...).
// Above kIdleSleep: sleep for a few µs. After ~20ms of pure idle (between
// shots or in finalize) we drop to near-zero CPU, freeing memory bandwidth
// for numpy / zlib on the host.
//
// Tunable for the latency-vs-host-contention trade-off:
//   GICC_PROXY_IDLE_YIELD  (default 1000)   empty iters before yield()
//   GICC_PROXY_IDLE_SLEEP  (default 100000) empty iters before sleep()
//   GICC_PROXY_SLEEP_US    (default 10)     sleep duration (µs) in deep idle
// Bigger thresholds = stay hot longer = lower wake-up latency but more CPU
// stolen from co-located host compute; smaller = friendlier to the host but
// adds wake-up latency to the next op. Pass-/ML-selectable per launch
// scenario (pure-comm kernel wants big; comm overlapped with heavy host
// numpy/zlib wants small).
const int kIdleYield = proxy_env_int("GICC_PROXY_IDLE_YIELD", 1000, 0, 1 << 30);
const int kIdleSleep =
    proxy_env_int("GICC_PROXY_IDLE_SLEEP", 100000, 1, 1 << 30);
const int kSleepUs   = proxy_env_int("GICC_PROXY_SLEEP_US", 10, 1, 100000);

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

// Monotonic nanosecond clock for hot-path profiling.
inline uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

ProxyThread::ProxyThread(::gicc::Runtime& rt, int ep_idx)
    : rt_(rt)
    , ring_host_(nullptr)
    , ring_device_(nullptr)
    , running_(false)
    , lf_(rt.fabric(), rt, ep_idx)
{
    ring_host_ = allocate_d2h_ring_host<kProxyRingCapacity>(&ring_device_);
    if (const char* p = std::getenv("GICC_PROXY_PROFILE")) {
        if (std::atoi(p) != 0) {
            prof_enabled_ = true;
            prof_submit_ns_.assign(kProxyRingCapacity, 0);
        }
    }
}

ProxyThread::~ProxyThread() {
    stop();
    if (prof_enabled_ && prof_op_count_ > 0) {
        double avg_us = (prof_lat_sum_ns_ / (double)prof_op_count_) / 1e3;
        uint64_t poll_total = prof_poll_empty_ + prof_poll_hit_;
        double cq_hit_rate = poll_total ? (100.0 * prof_poll_hit_ / poll_total) : 0.0;
        fprintf(stderr,
            "[proxy-profile] ops=%lu submit_calls=%lu\n"
            "  submit->completion latency (NIC round-trip): "
            "avg=%.3f us min=%.3f us max=%.3f us\n"
            "  max_inflight_depth=%zu (ring_cap=%u)\n"
            "  CQ-poll passes: hit=%lu empty=%lu hit_rate=%.1f%% "
            "(low hit_rate => proxy spins waiting on NIC = NIC-bound)\n",
            (unsigned long)prof_op_count_, (unsigned long)prof_submit_calls_,
            avg_us,
            prof_lat_min_ns_ == ~0ull ? 0.0 : prof_lat_min_ns_ / 1e3,
            prof_lat_max_ns_ / 1e3,
            prof_max_inflight_, (unsigned)kProxyRingCapacity,
            (unsigned long)prof_poll_hit_, (unsigned long)prof_poll_empty_,
            cq_hit_rate);
    }
    if (ring_host_) {
        free_d2h_ring_host<kProxyRingCapacity>(ring_host_);
        ring_host_   = nullptr;
        ring_device_ = nullptr;
    }
}

// Record a completed op: latency = now - submit-stamp for this ring slot.
void ProxyThread::prof_record_completion(size_t bit) {
    uint64_t submit = prof_submit_ns_[bit];
    if (submit == 0) return;  // not a profiled WRITE/READ/ATOMIC (e.g. QUIET)
    uint64_t lat = now_ns() - submit;
    prof_submit_ns_[bit] = 0;
    ++prof_op_count_;
    prof_lat_sum_ns_ += lat;
    if (lat > prof_lat_max_ns_) prof_lat_max_ns_ = lat;
    if (lat < prof_lat_min_ns_) prof_lat_min_ns_ = lat;
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
    // Idle-backoff counter. Reset to 0 whenever we did any real work
    // (submitted, polled a completion, or popped a QUIET). Climbs while
    // ring + CQ stay empty so we can stop hot-spinning between shots.
    int idle_iters = 0;
    while (running_.load(std::memory_order_acquire)) {
        bool stop_submitting = false;
        bool did_work = false;

        // 0. Reissue any retry stashed by a prior -FI_EAGAIN. Must clear
        //    before any new pop(), otherwise the failed (cmd, slot) is lost
        //    (pop() already advanced proxy_read_cursor when it was first
        //    popped). At most one PendingRetry exists at a time because the
        //    proxy is single-threaded.
        if (pending_retry_) {
            int ret;
            switch (pending_retry_->cmd.cmd_type) {
                case CmdType::WRITE:
                    ret = lf_.submit_write(pending_retry_->cmd,
                                           pending_retry_->slot);
                    break;
                case CmdType::READ:
                    ret = lf_.submit_read(pending_retry_->cmd,
                                          pending_retry_->slot);
                    break;
                case CmdType::ATOMIC:
                    ret = lf_.submit_atomic_add(pending_retry_->cmd,
                                                pending_retry_->slot);
                    break;
                default:
                    fprintf(stderr,
                        "ProxyThread: pending retry has unexpected cmd_type %u\n",
                        (unsigned)pending_retry_->cmd.cmd_type);
                    std::abort();
            }
            if (ret == -FI_EAGAIN) {
                stop_submitting = true;   // still no room; just poll CQ.
            } else {
                size_t bit = ring_idx(pending_retry_->slot);
                in_flight_.set(bit);
                ++in_flight_count_;
                if (prof_enabled_) {
                    prof_submit_ns_[bit] = now_ns();
                    ++prof_submit_calls_;
                    if (in_flight_count_ > prof_max_inflight_)
                        prof_max_inflight_ = in_flight_count_;
                }
                pending_retry_.reset();
                did_work = true;
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
            did_work = true;
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
                    if (prof_enabled_) {
                        prof_submit_ns_[bit] = now_ns();
                        ++prof_submit_calls_;
                        if (in_flight_count_ > prof_max_inflight_)
                            prof_max_inflight_ = in_flight_count_;
                    }
                    break;
                }
                case CmdType::READ: {
                    int ret = lf_.submit_read(c, slot);
                    if (ret == -FI_EAGAIN) {
                        pending_retry_ = PendingRetry{c, slot};
                        stop_submitting = true;
                        break;
                    }
                    in_flight_.set(bit);
                    ++in_flight_count_;
                    if (prof_enabled_) {
                        prof_submit_ns_[bit] = now_ns();
                        ++prof_submit_calls_;
                        if (in_flight_count_ > prof_max_inflight_)
                            prof_max_inflight_ = in_flight_count_;
                    }
                    break;
                }
                case CmdType::ATOMIC: {
                    int ret = lf_.submit_atomic_add(c, slot);
                    if (ret == -FI_EAGAIN) {
                        pending_retry_ = PendingRetry{c, slot};
                        stop_submitting = true;
                        break;
                    }
                    in_flight_.set(bit);
                    ++in_flight_count_;
                    if (prof_enabled_) {
                        prof_submit_ns_[bit] = now_ns();
                        ++prof_submit_calls_;
                        if (in_flight_count_ > prof_max_inflight_)
                            prof_max_inflight_ = in_flight_count_;
                    }
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
        Completion comps[kCqBatchMax];
        int n = lf_.poll(comps, kCqBatch);
        for (int i = 0; i < n; ++i) {
            uint64_t s = reinterpret_cast<uint64_t>(comps[i].context);
            ring_host_->mark_acked(s);
            size_t bit = ring_idx(s);
            if (prof_enabled_) prof_record_completion(bit);
            if (in_flight_.test(bit)) {
                in_flight_.reset(bit);
                --in_flight_count_;
            }
        }
        if (prof_enabled_ && in_flight_count_ > 0) {
            // Count poll passes only while we have ops outstanding — that's
            // when "CQ empty" means "waiting on the NIC". hit=got>=1 completion.
            if (n > 0) ++prof_poll_hit_; else ++prof_poll_empty_;
        }
        if (n > 0) {
            ring_host_->advance_tail_from_mask();
            did_work = true;
        }
        if (did_work) {
            idle_iters = 0;
        } else {
            ++idle_iters;
            if (idle_iters < kIdleYield) {
                cpu_relax();
            } else if (idle_iters < kIdleSleep) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(kSleepUs));
            }
        }
    }

    // Drain remaining completions on shutdown so we don't leak in-flight ops,
    // bounded by kShutdownDrainTimeout so a dropped completion can't hang
    // the destructor forever.
    auto drain_deadline = std::chrono::steady_clock::now() + kShutdownDrainTimeout;
    while (in_flight_count_ > 0 &&
           std::chrono::steady_clock::now() < drain_deadline) {
        Completion comps[kCqBatchMax];
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
        Completion comps[kCqBatchMax];
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
