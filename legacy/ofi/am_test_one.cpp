/**
 * am_test_one.cpp - AM test: send 1000 AMs that increment a counter
 *
 * Rank 0 sends 1000 AMs to Rank 1, each executing COUNTER_ADD handler.
 * Uses batched sends with interleaved polling to prevent ring buffer overrun.
 * After polling, Rank 1's counter should be 1000.
 *
 * Also measures latency and bandwidth.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <ctime>

#include <mpi.h>
#include <hip/hip_runtime.h>

#include "gda_am.hpp"
#include "device_affinity.hpp"

using namespace opengda::am;

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

static double timediff_us(const timespec& t_start, const timespec& t_end) {
    return (t_end.tv_sec - t_start.tv_sec) * 1.0e6 +
           (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3;
}

int main(int argc, char** argv) {
    unset_rocr_visible_devices();
    MPI_Init(&argc, &argv);

    GdaComm comm;
    int rank = comm.rank();
    int size = comm.size();

    if (size != 2) {
        if (rank == 0) std::cerr << "Need exactly 2 ranks\n";
        MPI_Finalize();
        return 1;
    }

    int peer = (rank == 0) ? 1 : 0;

    printf("Rank %d: Starting AM test (1000 messages)\n", rank);
    fflush(stdout);

    // Create AM context
    // Ring size = 128 slots, staging pool = 64
    // Batch size limited by both staging pool and ring size
    constexpr int RING_SLOTS = 128;
    constexpr int STAGING_POOL_SIZE = 64;
    constexpr int BATCH_SIZE = 64;  // Must be <= min(STAGING_POOL_SIZE, RING_SLOTS/2) for safety

    GdaAm am(comm, RING_SLOTS, STAGING_POOL_SIZE);

    printf("Rank %d: GdaAm created (ring=%d, staging=%d, batch=%d)\n",
           rank, RING_SLOTS, STAGING_POOL_SIZE, BATCH_SIZE);
    fflush(stdout);

    // Rank 1 allocates a counter on device
    uint64_t* d_counter = nullptr;
    if (rank == 1) {
        HIP_CHECK(hipMalloc(&d_counter, sizeof(uint64_t)));
        HIP_CHECK(hipMemset(d_counter, 0, sizeof(uint64_t)));
        printf("Rank 1: Counter allocated at %p, initial value = 0\n", d_counter);
        fflush(stdout);
    }

    // Exchange counter address: Rank 1 tells Rank 0 where the counter is
    uint64_t counter_addr = (uint64_t)d_counter;
    uint64_t remote_counter_addr = 0;
    MPI_Sendrecv(&counter_addr, 1, MPI_UINT64_T, peer, 0,
                 &remote_counter_addr, 1, MPI_UINT64_T, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank == 0) {
        printf("Rank 0: Remote counter at %p\n", (void*)remote_counter_addr);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Test: Rank 0 sends 1000 AMs to increment Rank 1's counter
    // Using interleaved send/poll to prevent ring overrun
    constexpr int N_MSGS = 1000;
    constexpr int N_BATCHES = (N_MSGS + BATCH_SIZE - 1) / BATCH_SIZE;

    // Short AM: seq(8) + hdr(8) + args(48) = 64 bytes (power of 2)
    constexpr size_t BYTES_PER_AM = AM_SHORT_SIZE;  // 64 bytes

    int sent = 0;
    int processed = 0;

    // Timing variables
    timespec t_start, t_end;
    double total_send_time_us = 0;
    double total_poll_time_us = 0;

    printf("Rank %d: Starting %d batches (batch_size=%d, bytes/AM=%zu)\n",
           rank, N_BATCHES, BATCH_SIZE, BYTES_PER_AM);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t_start);

    for (int batch = 0; batch < N_BATCHES; batch++) {
        // Calculate batch size consistently for both ranks
        int msgs_remaining = N_MSGS - batch * BATCH_SIZE;
        int batch_msgs = (msgs_remaining < BATCH_SIZE) ? msgs_remaining : BATCH_SIZE;

        timespec batch_start, batch_end;

        // Rank 0 sends batch
        if (rank == 0) {
            clock_gettime(CLOCK_MONOTONIC_RAW, &batch_start);

            for (int i = 0; i < batch_msgs; i++) {
                am_args64_t args;
                args.init();
                args[0] = remote_counter_addr;  // Pointer to counter
                args[1] = 1;                     // Value to add

                int ret = am.send_handle(peer, AM_HANDLER_COUNTER_ADD, args);
                if (ret != 0) {
                    printf("Rank 0: send_handle failed at msg %d: %d\n", sent + i, ret);
                    MPI_Finalize();
                    return 1;
                }
            }

            // Trigger and wait for this batch
            am.trigger_and_wait();

            // Fast flush to release DWQ resources
            comm.fast_flush(comm.current_threshold);

            clock_gettime(CLOCK_MONOTONIC_RAW, &batch_end);
            total_send_time_us += timediff_us(batch_start, batch_end);

            sent += batch_msgs;
        }

        // Barrier to sync sender and receiver
        MPI_Barrier(MPI_COMM_WORLD);

        // Rank 1 polls for this batch
        if (rank == 1) {
            clock_gettime(CLOCK_MONOTONIC_RAW, &batch_start);

            int batch_processed = 0;
            int max_retries = 1000;

            while (batch_processed < batch_msgs && max_retries > 0) {
                int p = am.poll_once(batch_msgs);
                batch_processed += p;
                if (p == 0) {
                    usleep(100);
                    max_retries--;
                }
            }

            clock_gettime(CLOCK_MONOTONIC_RAW, &batch_end);
            total_poll_time_us += timediff_us(batch_start, batch_end);

            processed += batch_processed;

            if (batch_processed < batch_msgs) {
                printf("Rank 1: WARNING - batch %d only processed %d/%d\n",
                       batch, batch_processed, batch_msgs);
                fflush(stdout);
            }
        }

        // Barrier before next batch (so sender doesn't overrun)
        MPI_Barrier(MPI_COMM_WORLD);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t_end);
    double total_time_us = timediff_us(t_start, t_end);

    // Final check and timing report
    if (rank == 0) {
        printf("\n=== Rank 0 (Sender) Timing ===\n");
        printf("Total AMs sent: %d\n", sent);
        printf("Total send time: %.2f us\n", total_send_time_us);
        printf("Per-AM send latency: %.2f us\n", total_send_time_us / sent);
        printf("Send throughput: %.2f AM/s\n", sent / (total_send_time_us / 1e6));

        double total_bytes = sent * BYTES_PER_AM;
        double send_bw_MBps = (total_bytes / (1024.0 * 1024.0)) / (total_send_time_us / 1e6);
        printf("Send bandwidth: %.2f MB/s (%.2f Gbps)\n",
               send_bw_MBps, send_bw_MBps * 8 / 1024);
        fflush(stdout);
    }

    if (rank == 1) {
        printf("\n=== Rank 1 (Receiver) Results ===\n");
        printf("Total processed: %d/%d\n", processed, N_MSGS);
        printf("Total poll time: %.2f us\n", total_poll_time_us);
        printf("Per-AM poll latency: %.2f us\n", total_poll_time_us / processed);
        printf("Poll throughput: %.2f AM/s\n", processed / (total_poll_time_us / 1e6));
        fflush(stdout);

        // Check counter value
        uint64_t h_counter = 0;
        HIP_CHECK(hipMemcpy(&h_counter, d_counter, sizeof(uint64_t), hipMemcpyDeviceToHost));

        printf("\nCounter value = %lu (expected %d)\n", h_counter, N_MSGS);

        if (h_counter == (uint64_t)N_MSGS) {
            printf("SUCCESS - all %d AMs executed!\n", N_MSGS);
        } else {
            printf("FAIL - counter=%lu, expected %d\n", h_counter, N_MSGS);
        }

        HIP_CHECK(hipFree(d_counter));
    }

    // Overall timing
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n=== Overall (including barriers) ===\n");
        printf("Total wall time: %.2f us (%.2f ms)\n", total_time_us, total_time_us / 1000);
        printf("End-to-end per-AM: %.2f us\n", total_time_us / N_MSGS);

        double total_bytes = N_MSGS * BYTES_PER_AM;
        double e2e_bw_MBps = (total_bytes / (1024.0 * 1024.0)) / (total_time_us / 1e6);
        printf("End-to-end bandwidth: %.2f MB/s (%.2f Gbps)\n",
               e2e_bw_MBps, e2e_bw_MBps * 8 / 1024);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d: Test complete\n", rank);

    MPI_Finalize();
    return 0;
}
