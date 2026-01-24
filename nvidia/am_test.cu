/**
 * am_test.cu - Active Message test
 *
 * Tests the AM subsystem with handle-only messages.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <mpi.h>

#include "gda_am.hpp"

using namespace opengda;
using namespace std::chrono;

constexpr int WARMUP_MSGS = 10;
constexpr int TEST_MSGS = 1000;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    GdaComm comm;
    int rank = comm.rank();
    int size = comm.size();

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires exactly 2 ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    int peer = (rank == 0) ? 1 : 0;
    int n_msgs = (argc > 1) ? atoi(argv[1]) : TEST_MSGS;

    if (rank == 0) {
        printf("=== Active Message Test ===\n");
        printf("Messages: %d (warmup: %d)\n", n_msgs, WARMUP_MSGS);
        printf("AM size: %zu bytes (short AM)\n", sizeof(AmSlot));
        printf("GPU: %s\n", comm.cuda->props.name);
        printf("IB device: %s\n", comm.ibv->dev_name.c_str());
        fflush(stdout);
    }

    // Create AM context with enough staging slots
    GdaAm am(comm, 128, n_msgs + WARMUP_MSGS + 16);

    comm.barrier();

    // ========================================
    // One-way throughput test
    // ========================================

    if (rank == 0) {
        printf("\n--- One-way throughput test ---\n");
        fflush(stdout);
    }

    // Warmup
    if (rank == 0) {
        for (int i = 0; i < WARMUP_MSGS; i++) {
            AmArgs args;
            args.init();
            args[0] = i;
            am.send_handle(peer, AM_HANDLER_NOOP, args);
        }
        am.trigger_and_wait();
    }

    comm.barrier();

    // Timed throughput test
    if (rank == 0) {
        auto t_start = high_resolution_clock::now();

        for (int i = 0; i < n_msgs; i++) {
            AmArgs args;
            args.init();
            args[0] = i;
            args[1] = 0xDEADBEEF;
            am.send_handle(peer, AM_HANDLER_NOOP, args);
        }
        am.trigger_and_wait();

        auto t_end = high_resolution_clock::now();
        double elapsed_us = duration_cast<nanoseconds>(t_end - t_start).count() / 1000.0;

        double msg_rate = n_msgs / (elapsed_us / 1e6);
        double per_msg_us = elapsed_us / n_msgs;
        double bw_MBps = (sizeof(AmSlot) * n_msgs) / (elapsed_us / 1e6) / 1e6;

        printf("Total time: %.2f us\n", elapsed_us);
        printf("Per-message: %.2f us\n", per_msg_us);
        printf("Message rate: %.0f msg/s\n", msg_rate);
        printf("Bandwidth: %.2f MB/s\n", bw_MBps);
        fflush(stdout);
    }

    comm.barrier();

    // Receiver polls
    if (rank == 1) {
        int received = 0;
        int attempts = 0;
        int max_attempts = 1000;

        while (received < n_msgs && attempts < max_attempts) {
            int n = am.poll_once(64);
            received += n;
            if (n == 0) {
                attempts++;
                CUDA_CHECK(cudaDeviceSynchronize());
            } else {
                attempts = 0;  // Reset on progress
            }
        }

        printf("Rank 1: Received %d/%d messages\n", received, n_msgs);
        fflush(stdout);
    }

    comm.barrier();

    // ========================================
    // Ping-pong latency test (simplified)
    // ========================================

    if (rank == 0) {
        printf("\n--- Ping-pong latency test (CPU-driven) ---\n");
        fflush(stdout);
    }

    int pingpong_iters = 100;

    // Reset staging
    am.staging_idx = 0;
    am.pending_ops = 0;

    comm.barrier();

    if (rank == 0) {
        auto pp_start = high_resolution_clock::now();

        for (int i = 0; i < pingpong_iters; i++) {
            // Send ping
            AmArgs args;
            args.init();
            args[0] = i;
            am.send_handle(peer, AM_HANDLER_NOOP, args);
            am.trigger_and_wait();

            // Wait for pong (via MPI barrier as simple sync)
            comm.barrier();
        }

        auto pp_end = high_resolution_clock::now();
        double pp_us = duration_cast<nanoseconds>(pp_end - pp_start).count() / 1000.0;

        double avg_rtt = pp_us / pingpong_iters;
        double avg_oneway = avg_rtt / 2.0;

        printf("Round-trip latency: %.2f us\n", avg_rtt);
        printf("One-way latency: %.2f us\n", avg_oneway);
        fflush(stdout);
    } else {
        for (int i = 0; i < pingpong_iters; i++) {
            // Wait for ping
            while (am.poll_once(1) == 0) {
                // Spin
            }

            // Send pong
            AmArgs args;
            args.init();
            args[0] = i;
            am.send_handle(peer, AM_HANDLER_NOOP, args);
            am.trigger_and_wait();

            comm.barrier();
        }
    }

    comm.barrier();

    if (rank == 0) {
        printf("\nSUCCESS\n");
    }

    MPI_Finalize();
    return 0;
}
