/**
 * am_test_one.cpp - AM test: send 10 AMs that increment a counter
 *
 * Rank 0 sends 10 AMs to Rank 1, each executing COUNTER_ADD handler.
 * After polling, Rank 1's counter should be 10.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

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

    printf("Rank %d: Starting AM test\n", rank);
    fflush(stdout);

    // Create AM context
    GdaAm am(comm, 128, 16);

    printf("Rank %d: GdaAm created\n", rank);
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

    // Test: Rank 0 sends 10 AMs to increment Rank 1's counter
    constexpr int N_MSGS = 10;

    if (rank == 0) {
        printf("Rank 0: Sending %d AMs (COUNTER_ADD)...\n", N_MSGS);
        fflush(stdout);

        for (int i = 0; i < N_MSGS; i++) {
            am_args64_t args;
            args.init();
            args[0] = remote_counter_addr;  // Pointer to counter
            args[1] = 1;                     // Value to add

            int ret = am.send_handle(peer, AM_HANDLER_COUNTER_ADD, args);
            if (ret != 0) {
                printf("Rank 0: send_handle[%d] failed: %d\n", i, ret);
                MPI_Finalize();
                return 1;
            }
        }

        printf("Rank 0: All sends queued, triggering...\n");
        fflush(stdout);

        am.trigger_and_wait();

        printf("Rank 0: trigger_and_wait completed!\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 1 polls for messages
    if (rank == 1) {
        printf("Rank 1: Polling for %d messages...\n", N_MSGS);
        fflush(stdout);

        int total_processed = 0;
        for (int iter = 0; iter < 100 && total_processed < N_MSGS; iter++) {
            int processed = am.poll_once(N_MSGS);
            total_processed += processed;
            if (processed > 0) {
                printf("Rank 1: poll_once returned %d, total=%d\n", processed, total_processed);
                fflush(stdout);
            }
            if (processed == 0) usleep(10000);
        }

        printf("Rank 1: Total processed: %d/%d\n", total_processed, N_MSGS);
        fflush(stdout);

        // Check counter value
        uint64_t h_counter = 0;
        HIP_CHECK(hipMemcpy(&h_counter, d_counter, sizeof(uint64_t), hipMemcpyDeviceToHost));

        printf("Rank 1: Counter value = %lu (expected %d)\n", h_counter, N_MSGS);
        fflush(stdout);

        if (h_counter == N_MSGS) {
            printf("Rank 1: SUCCESS - all %d AMs executed!\n", N_MSGS);
        } else {
            printf("Rank 1: FAIL - counter=%lu, expected %d\n", h_counter, N_MSGS);
        }

        HIP_CHECK(hipFree(d_counter));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d: Test complete\n", rank);

    MPI_Finalize();
    return 0;
}
