/**
 * am_pingpong.cpp - AM ping-pong latency test (GPU-timed)
 *
 * Optimized timing:
 *   - Each kernel does BATCH_SIZE round-trips internally
 *   - Average within kernel, then average across kernels
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

constexpr int BATCH_SIZE = 10;  // Round-trips per kernel

// =============================================================================
// GPU Kernels for ping-pong
// =============================================================================

/**
 * Initiator kernel: do BATCH_SIZE round-trips, measure total time
 */
__global__ void pingpong_initiator_kernel(
    volatile uint64_t* trigger_addr,
    uint64_t base_threshold,
    am_context_t* am_ctx,
    int peer_rank,
    uint64_t* d_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    am_recv_state_t* recv_state = &am_ctx->recv_states[peer_rank];

    // Start timing for entire batch
    uint64_t t_start = clock64();

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Trigger the send - each AM uses 2 thresholds (body + seq)
        // Threshold for AM i is (i+1)*2
        *trigger_addr = (i + 1) * 2;
        __threadfence_system();

        // Poll for response
        uint64_t expected = recv_state->expected_seq;
        int slot_idx = expected & (recv_state->nslots - 1);
        am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

        while (slot->seq != expected) {
            // Busy wait
        }
        __threadfence_system();
        recv_state->expected_seq = expected + 1;
    }

    // Stop timing
    uint64_t t_end = clock64();
    *d_cycles = t_end - t_start;
}

/**
 * Responder kernel: do BATCH_SIZE round-trips
 */
__global__ void pingpong_responder_kernel(
    volatile uint64_t* trigger_addr,
    uint64_t base_threshold,
    am_context_t* am_ctx,
    int peer_rank)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    am_recv_state_t* recv_state = &am_ctx->recv_states[peer_rank];

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Poll for incoming ping
        uint64_t expected = recv_state->expected_seq;
        int slot_idx = expected & (recv_state->nslots - 1);
        am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

        while (slot->seq != expected) {
            // Busy wait
        }
        __threadfence_system();
        recv_state->expected_seq = expected + 1;

        // Trigger the response - each AM uses 2 thresholds (body + seq)
        *trigger_addr = (i + 1) * 2;
        __threadfence_system();
    }
}

// =============================================================================
// Main
// =============================================================================

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

    // Parse arguments
    int n_batches = (argc > 1) ? atoi(argv[1]) : 100;   // Number of kernel launches
    int warmup_batches = (argc > 2) ? atoi(argv[2]) : 10;

    int total_roundtrips = n_batches * BATCH_SIZE;

    if (rank == 0) {
        printf("AM Ping-Pong Latency Test (GPU-timed, optimized)\n");
        printf("  Batches: %d (warmup: %d)\n", n_batches, warmup_batches);
        printf("  Round-trips per batch: %d\n", BATCH_SIZE);
        printf("  Total round-trips: %d\n", total_roundtrips);
        printf("  AM size: %zu bytes\n", AM_SHORT_SIZE);
        fflush(stdout);
    }

    // Create AM context - need enough staging for BATCH_SIZE AMs
    GdaAm am(comm, 128, BATCH_SIZE + 4);

    // Get GPU clock frequency
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, comm.gpu_id()));
    double clock_rate_khz = props.clockRate;

    if (rank == 0) {
        printf("GPU clock rate: %.0f MHz\n", clock_rate_khz / 1000);
        fflush(stdout);
    }

    // Allocate result buffer
    uint64_t* d_cycles = nullptr;
    HIP_CHECK(hipMalloc(&d_cycles, sizeof(uint64_t)));

    MPI_Barrier(MPI_COMM_WORLD);

    // Run batches
    uint64_t total_cycles = 0;
    int total_batches = warmup_batches + n_batches;

    for (int batch = 0; batch < total_batches; batch++) {
        // Reset staging for this batch
        am.staging_idx = 0;
        am.max_threshold = 0;

        // Stage BATCH_SIZE AMs
        for (int i = 0; i < BATCH_SIZE; i++) {
            am_args_t args;
            args.init();
            args[0] = batch * BATCH_SIZE + i;

            am.send_handle(peer, AM_HANDLER_NOOP, args);
        }
        uint64_t my_threshold = am.max_threshold;

        // Sync before GPU kernel
        MPI_Barrier(MPI_COMM_WORLD);

        // Run GPU kernel
        HIP_CHECK(hipMemset(d_cycles, 0, sizeof(uint64_t)));

        if (rank == 0) {
            hipLaunchKernelGGL(pingpong_initiator_kernel,
                               dim3(1), dim3(1), 0, 0,
                               comm.get_trigger_addr(),
                               0,
                               am.am_ctx.d_context,
                               peer,
                               d_cycles);
        } else {
            hipLaunchKernelGGL(pingpong_responder_kernel,
                               dim3(1), dim3(1), 0, 0,
                               comm.get_trigger_addr(),
                               0,
                               am.am_ctx.d_context,
                               peer);
        }

        HIP_CHECK(hipDeviceSynchronize());

        // Wait for DWQ completion
        comm.wait(my_threshold);
        comm.fast_flush(my_threshold);

        // Collect timing from rank 0 (after warmup)
        if (rank == 0 && batch >= warmup_batches) {
            uint64_t cycles;
            HIP_CHECK(hipMemcpy(&cycles, d_cycles, sizeof(uint64_t), hipMemcpyDeviceToHost));
            total_cycles += cycles;
        }

        // Reset counters for next batch
        comm.reset_counters();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Report results
    if (rank == 0) {
        double total_time_us = (double)total_cycles / clock_rate_khz * 1000.0;
        double avg_rtt_us = total_time_us / total_roundtrips;
        double avg_oneway_us = avg_rtt_us / 2.0;
        double msg_rate = (2.0 * total_roundtrips) / (total_time_us / 1e6);

        printf("\n=== Results (GPU-timed) ===\n");
        printf("Total GPU cycles: %lu\n", total_cycles);
        printf("Total time: %.2f us\n", total_time_us);
        printf("Total round-trips: %d\n", total_roundtrips);
        printf("\n");
        printf("Round-trip latency:  %.2f us\n", avg_rtt_us);
        printf("One-way latency:     %.2f us\n", avg_oneway_us);
        printf("Message rate:        %.0f msg/s\n", msg_rate);
        printf("\nSUCCESS\n");
        fflush(stdout);
    }

    HIP_CHECK(hipFree(d_cycles));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
