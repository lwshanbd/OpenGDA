// omp_put_signal.cpp - device-side put-with-signal, checked word for word.
//
// Two ranks, each sending to the other. Every iteration runs ONE kernel with
// 2*K teams:
//   - sender team k writes chunk k of its send buffer with a pattern unique to
//     (iteration, rank, chunk, index) and then, from inside the kernel, puts it
//     to the peer with signal slot k = iteration+1;
//   - receiver team k waits for its own slot k to reach iteration+1 and checks
//     chunk k of its receive buffer on the spot.
// The check runs the moment the flag is observed, so a flag that overtook its
// payload shows up as mismatches rather than being hidden by a barrier.
//
// Under DWQ the host stages every chunk's transfer before the kernel
// (ompx_stage_put_signal) and team k's put_signal releases chunk k alone.
// Under the CPU proxy the staging call is a no-op and put_signal pushes the
// transfer onto the ring. The source is identical for both.
//
// After the real run it repeats every size with the flag released BEFORE the
// payload (two slots per chunk). That is the control: the check has to fail
// there, or it could not have seen a reordering in the first place. Then the
// same transfers through the host forms, and a ping-pong, host and device.
//
// Build: bash examples/omp/build_giomp_example.sh examples/omp/omp_put_signal.cpp OUT
// Run  : flux run -N2 -n2 -g1 ./omp_put_signal [--iters=N] [--chunks=K]
//        (GICC_HALO_DWQ=0 selects the CPU proxy)

#include "gicc/omp.h"
#include <omp.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#pragma omp declare target
static inline uint32_t pattern(int it, int rank, int chunk, size_t i) {
    return (uint32_t)it * 0x9E3779B1u ^ (uint32_t)rank * 0x85EBCA77u ^
           (uint32_t)chunk * 0xC2B2AE3Du ^ (uint32_t)i * 0x27D4EB2Fu;
}
#pragma omp end declare target

static const size_t kChunkBytes[] = {4096, 65536, 1 << 20, 4 << 20};
static constexpr size_t kMaxWords = (4u << 20) / sizeof(uint32_t);

// One pass over every chunk size. Returns the words this rank found wrong.
static unsigned long long run(int rank, int peer, int K, int iters, int* base,
                              bool reversed, uint32_t* sbuf, uint32_t* rbuf,
                              uint32_t* dummy) {
    unsigned long long total = 0;
    for (size_t cb : kChunkBytes) {
        const size_t W = cb / sizeof(uint32_t);
        for (int s = 0; s < 2 * K; ++s) ompx_signal_reset(s);
        ompx_barrier();

        unsigned long long bad = 0;
        const double t0 = omp_get_wtime();
        for (int it = *base; it < *base + iters; ++it) {
            const unsigned long long v = (unsigned long long)it + 1;
            for (int k = 0; k < K; ++k) {
                if (reversed) {
                    ompx_stage_put_signal(peer, dummy, dummy, 256, k, v);
                    ompx_stage_put_signal(peer, rbuf + k * W, sbuf + k * W, cb, K + k, v);
                } else {
                    ompx_stage_put_signal(peer, rbuf + k * W, sbuf + k * W, cb, k, v);
                }
            }

            #pragma omp target teams num_teams(2 * K) thread_limit(256) \
                    is_device_ptr(sbuf, rbuf, dummy) map(tofrom: bad) \
                    firstprivate(it, v, W, cb, K, rank, peer, reversed)
            {
                const int t = omp_get_team_num();
                if (t < K) {
                    // Teams finish in the reverse of slot order, so chunk 0
                    // is still being written while later chunks are released.
                    // Only a doorbell per slot keeps chunk 0 from leaving
                    // with them.
                    const double until = omp_get_wtime() + (K - 1 - t) * 20e-6;
                    while (omp_get_wtime() < until) {}
                    uint32_t* c = sbuf + t * W;
                    #pragma omp parallel for
                    for (size_t i = 0; i < W; ++i) c[i] = pattern(it, rank, t, i);
                    if (reversed) {
                        ompx_put_signal(peer, dummy, dummy, 256, t, v);
                        ompx_put_signal(peer, rbuf + t * W, c, cb, K + t, v);
                    } else {
                        ompx_put_signal(peer, rbuf + t * W, c, cb, t, v);
                    }
                } else {
                    const int k = t - K;
                    ompx_signal_wait(k, v);
                    const uint32_t* c = rbuf + k * W;
                    unsigned long long n = 0;
                    #pragma omp parallel for reduction(+ : n)
                    for (size_t i = 0; i < W; ++i) n += c[i] != pattern(it, peer, k, i);
                    if (n) {
                        #pragma omp atomic
                        bad += n;
                    }
                }
            }
            ompx_quiet();
            ompx_barrier();
        }
        const double us = (omp_get_wtime() - t0) * 1e6 / iters;
        *base += iters;
        std::printf("rank %d %-9s %10zu %8d %12llu %10.1f\n", rank,
                    reversed ? "reversed" : "ordered", cb, iters, bad, us);
        total += bad;
    }
    return total;
}

// The host forms: a kernel produces every chunk, the host sends each one with
// ompx_put_signal, and the receiver's host waits for each flag and checks that
// chunk before looking at the next flag.
static unsigned long long run_host(int rank, int peer, int K, int iters, int* base,
                                   uint32_t* sbuf, uint32_t* rbuf) {
    unsigned long long total = 0;
    for (size_t cb : kChunkBytes) {
        const size_t W = cb / sizeof(uint32_t);
        for (int s = 0; s < K; ++s) ompx_signal_reset(s);
        ompx_barrier();
        unsigned long long bad = 0;
        const double t0 = omp_get_wtime();
        for (int it = *base; it < *base + iters; ++it) {
            const unsigned long long v = (unsigned long long)it + 1;
            #pragma omp target teams distribute parallel for collapse(2) \
                    is_device_ptr(sbuf) firstprivate(it, W, K, rank)
            for (int k = 0; k < K; ++k)
                for (size_t i = 0; i < W; ++i) sbuf[k * W + i] = pattern(it, rank, k, i);
            for (int k = 0; k < K; ++k)
                ompx_put_signal(peer, rbuf + k * W, sbuf + k * W, cb, k, v);
            for (int k = 0; k < K; ++k) {
                ompx_signal_wait(k, v);
                unsigned long long n = 0;
                #pragma omp target teams distribute parallel for reduction(+ : n) \
                        is_device_ptr(rbuf) firstprivate(it, W, k, peer)
                for (size_t i = 0; i < W; ++i) n += rbuf[k * W + i] != pattern(it, peer, k, i);
                bad += n;
            }
            ompx_quiet();
            ompx_barrier();
        }
        const double us = (omp_get_wtime() - t0) * 1e6 / iters;
        *base += iters;
        std::printf("rank %d %-9s %10zu %8d %12llu %10.1f\n", rank, "host", cb, iters, bad, us);
        total += bad;
    }
    return total;
}

#pragma omp declare target
static inline uint32_t mark(unsigned long long v, int rank) {
    return (uint32_t)(v << 1) | (uint32_t)rank;
}
#pragma omp end declare target

// Ping-pong on slot 0. Rank 0 sends round i and waits for the reply; rank 1
// waits and replies. The device form runs all N round trips inside one kernel
// -- under DWQ the host stages the N transfers up front and each device
// put_signal releases the next one. The first and last word of every message
// carry the round number, checked on arrival.
static void pingpong(int rank, int peer, uint32_t* buf, int* base,
                     unsigned long long* bad_total) {
    static const size_t kSizes[] = {8, 4096, 65536, 1 << 20};
    const int N = 100;
    if (rank == 0)
        std::printf("\nping-pong (%d round trips, half RTT in us)\n"
                    "%10s %10s %10s %10s\n", N, "bytes", "host", "device", "bad");
    for (size_t bytes : kSizes) {
        const size_t last = bytes / sizeof(uint32_t) - 1;

        ompx_signal_reset(0);
        ompx_barrier();
        int b = *base;
        double t0 = omp_get_wtime();
        for (int i = 1; i <= N; ++i) {
            const unsigned long long v = (unsigned long long)(b + i);
            if (rank == 0) {
                ompx_put_signal(peer, buf, buf, bytes, 0, v);
                ompx_signal_wait(0, v);
            } else {
                ompx_signal_wait(0, v);
                ompx_put_signal(peer, buf, buf, bytes, 0, v);
            }
        }
        const double host_us = (omp_get_wtime() - t0) * 1e6 / N / 2;
        ompx_quiet();
        *base += N;

        ompx_signal_reset(0);
        ompx_barrier();
        b = *base;
        for (int i = 1; i <= N; ++i)
            ompx_stage_put_signal(peer, buf, buf, bytes, 0, (unsigned long long)(b + i));
        ompx_barrier();
        unsigned long long bad = 0;
        t0 = omp_get_wtime();
        #pragma omp target is_device_ptr(buf) map(tofrom: bad) \
                firstprivate(rank, peer, bytes, last, b, N)
        {
            for (int i = 1; i <= N; ++i) {
                const unsigned long long v = (unsigned long long)(b + i);
                if (rank == 1) {
                    ompx_signal_wait(0, v);
                    bad += (buf[0] != mark(v, peer)) + (buf[last] != mark(v, peer));
                }
                buf[0] = buf[last] = mark(v, rank);
                ompx_put_signal(peer, buf, buf, bytes, 0, v);
                if (rank == 0) {
                    ompx_signal_wait(0, v);
                    bad += (buf[0] != mark(v, peer)) + (buf[last] != mark(v, peer));
                }
            }
        }
        const double dev_us = (omp_get_wtime() - t0) * 1e6 / N / 2;
        ompx_quiet();
        ompx_barrier();
        *base += N;
        *bad_total += bad;
        if (rank == 0)
            std::printf("%10zu %10.2f %10.2f %10llu\n", bytes, host_us, dev_us, bad);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int iters = 50;
    int K = 8;
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        if (s.rfind("--iters=", 0) == 0) iters = std::atoi(s.c_str() + 8);
        else if (s.rfind("--chunks=", 0) == 0) K = std::atoi(s.c_str() + 9);
    }
    if (K < 1 || K > 32) {
        std::fprintf(stderr, "--chunks must be in [1, 32]\n");
        return 1;
    }

    ompx_init();
    const int rank = ompx_get_rank_num();
    if (ompx_get_num_ranks() != 2) {
        if (rank == 0) std::fprintf(stderr, "omp_put_signal needs exactly 2 ranks\n");
        ompx_finalize();
        return 1;
    }
    const int peer = rank ^ 1;
    const char* dwq = std::getenv("GICC_HALO_DWQ");
    const bool proxy = std::getenv("GICC_SKIP_DWQ_INIT") || (dwq && std::atoi(dwq) == 0);

    uint32_t* sbuf  = (uint32_t*)ompx_alloc((size_t)K * kMaxWords * sizeof(uint32_t));
    uint32_t* rbuf  = (uint32_t*)ompx_alloc((size_t)K * kMaxWords * sizeof(uint32_t));
    uint32_t* dummy = (uint32_t*)ompx_alloc(256);
    ompx_prepare();

    if (rank == 0) {
        std::printf("omp_put_signal: transport=%s chunks=%d iters=%d\n",
                    proxy ? "proxy" : "dwq", K, iters);
        std::printf("rank - %-9s %10s %8s %12s %10s\n",
                    "order", "chunk_B", "iters", "bad_words", "us/iter");
    }

    int base = 0;    // slot values keep rising from pass to pass
    const unsigned long long ordered = run(rank, peer, K, iters, &base, false,
                                           sbuf, rbuf, dummy);
    const unsigned long long control = run(rank, peer, K, iters, &base, true,
                                           sbuf, rbuf, dummy);
    const unsigned long long host = run_host(rank, peer, K, iters, &base, sbuf, rbuf);
    unsigned long long pp = 0;
    pingpong(rank, peer, rbuf, &base, &pp);

    ompx_barrier();
    // Every real pass must be clean and the control must not be: a control
    // that also comes out clean means the check never had a chance to fail.
    const bool ok = ordered == 0 && host == 0 && pp == 0 && control > 0;
    std::printf("rank %d RESULT %s (device bad=%llu, host bad=%llu, ping-pong bad=%llu, "
                "reversed-control bad=%llu)\n",
                rank, ok ? "PASS" : "FAIL", ordered, host, pp, control);
    ompx_free(dummy);
    ompx_free(rbuf);
    ompx_free(sbuf);
    ompx_finalize();
    return ok ? 0 : 2;
}
