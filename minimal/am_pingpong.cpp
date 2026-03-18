/**
 * am_pingpong.cpp - AM latency test (GPU-timed)
 *
 * Tests two modes:
 *   1. ReqRep (Request-Reply): Sender sends Request (64B), Receiver replies with Reply (8B)
 *      - Reply is lightweight: only 8 bytes written to sender's ack buffer
 *   2. ReqReq (Request-Request): Both sides send full Request messages (64B)
 *      - Symmetric path, no lightweight Reply
 *
 * Optimized timing:
 *   - Each kernel does BATCH_SIZE operations internally
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

constexpr int BATCH_SIZE = 10;  // Operations per kernel

// =============================================================================
// GPU Kernels for ReqRep mode (Request-Reply)
// =============================================================================
// ReqRep uses lightweight Reply (8 bytes) instead of full AM (64 bytes)
//
// Initiator:
//   - Triggers Request (64B AM)
//   - Polls local ack buffer (8B) for Reply
//
// Responder:
//   - Polls inbox for Request (64B AM)
//   - Triggers Reply (8B put to sender's ack buffer)

/**
 * ReqRep Initiator: send Request (64B), wait for Reply (8B ack)
 */
__global__ void reqrep_initiator_kernel(
    volatile uint64_t* trigger_addr,
    volatile uint64_t* local_ack,      // Ack buffer where responder writes reply
    uint64_t* d_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t t_start = clock64();

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Trigger the Request (64B AM)
        // Each Request uses 2 thresholds (body + seq)
        *trigger_addr = (i + 1) * 2;
        __threadfence_system();

        // Poll for Reply (8B ack) - wait for ack_value >= (i+1)
        while (*local_ack < (uint64_t)(i + 1)) {
            // Busy wait for lightweight reply
        }
        __threadfence_system();
    }

    uint64_t t_end = clock64();
    *d_cycles = t_end - t_start;
}

/**
 * ReqRep Responder: wait for Request (64B), send Reply (8B)
 */
__global__ void reqrep_responder_kernel(
    volatile uint64_t* trigger_addr,
    am_context_t* am_ctx,
    int peer_rank)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    am_recv_state_t* recv_state = &am_ctx->recv_states[peer_rank];

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Wait for incoming Request (64B AM)
        uint64_t expected = recv_state->expected_seq;
        int slot_idx = expected & (recv_state->nslots - 1);
        am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

        while (slot->seq != expected) {
            // Busy wait for full AM
        }
        __threadfence_system();
        recv_state->expected_seq = expected + 1;

        // Trigger Reply (8B put to sender's ack buffer)
        // Each Reply uses 1 threshold (just the 8B value)
        *trigger_addr = i + 1;  // Reply thresholds: 1, 2, ..., BATCH_SIZE
        __threadfence_system();
    }
}

// =============================================================================
// GPU Kernels for ReqReq mode (Request-Request)
// =============================================================================
// Both sides send full Request (64B AM), symmetric path

/**
 * ReqReq Node A: send Request first, then wait for Request from B
 */
__global__ void reqreq_nodeA_kernel(
    volatile uint64_t* trigger_addr,
    am_context_t* am_ctx,
    int peer_rank,
    uint64_t* d_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    am_recv_state_t* recv_state = &am_ctx->recv_states[peer_rank];

    uint64_t t_start = clock64();

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Send Request to B (64B AM)
        *trigger_addr = (i + 1) * 2;
        __threadfence_system();

        // Wait for Request from B (64B AM)
        uint64_t expected = recv_state->expected_seq;
        int slot_idx = expected & (recv_state->nslots - 1);
        am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

        while (slot->seq != expected) {
            // Busy wait
        }
        __threadfence_system();
        recv_state->expected_seq = expected + 1;
    }

    uint64_t t_end = clock64();
    *d_cycles = t_end - t_start;
}

/**
 * ReqReq Node B: wait for Request from A, then send Request back
 */
__global__ void reqreq_nodeB_kernel(
    volatile uint64_t* trigger_addr,
    am_context_t* am_ctx,
    int peer_rank)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    am_recv_state_t* recv_state = &am_ctx->recv_states[peer_rank];

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Wait for Request from A (64B AM)
        uint64_t expected = recv_state->expected_seq;
        int slot_idx = expected & (recv_state->nslots - 1);
        am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

        while (slot->seq != expected) {
            // Busy wait
        }
        __threadfence_system();
        recv_state->expected_seq = expected + 1;

        // Send Request back to A (64B AM)
        *trigger_addr = (i + 1) * 2;
        __threadfence_system();
    }
}

// =============================================================================
// Test runners
// =============================================================================

void run_reqrep_test(GdaComm& comm, GdaAm& am, int peer,
                     int n_batches, int warmup_batches,
                     double clock_rate_khz, uint64_t* d_cycles)
{
    int rank = comm.rank();
    int total_roundtrips = n_batches * BATCH_SIZE;

    if (rank == 0) {
        printf("\n");
        printf("========================================\n");
        printf("  Mode 1: ReqRep (Request-Reply)\n");
        printf("========================================\n");
        printf("  Request: 64 bytes (full AM)\n");
        printf("  Reply:   8 bytes (ack only)\n");
        printf("  Batches: %d (warmup: %d)\n", n_batches, warmup_batches);
        printf("  Round-trips per batch: %d\n", BATCH_SIZE);
        printf("  Total round-trips: %d\n", total_roundtrips);
        fflush(stdout);
    }

    // Get local ack buffer pointer for this peer
    volatile uint64_t* local_ack = am.am_ctx.h_send_states[peer].local_ack;

    MPI_Barrier(MPI_COMM_WORLD);

    uint64_t total_cycles = 0;
    int total_batches = warmup_batches + n_batches;

    for (int batch = 0; batch < total_batches; batch++) {
        // Reset staging for this batch
        am.staging_idx = 0;
        am.reply_staging_idx = 0;
        am.max_threshold = 0;

        // Reset ack buffer
        uint64_t zero = 0;
        HIP_CHECK(hipMemcpy((void*)local_ack, &zero, sizeof(uint64_t), hipMemcpyHostToDevice));

        if (rank == 0) {
            // Initiator: stage BATCH_SIZE Request AMs (64B each)
            for (int i = 0; i < BATCH_SIZE; i++) {
                am_args_t args;
                args.init();
                args[0] = batch * BATCH_SIZE + i;
                am.send_handle(peer, AM_HANDLER_NOOP, args);
            }
        } else {
            // Responder: stage BATCH_SIZE Reply puts (8B each)
            for (int i = 0; i < BATCH_SIZE; i++) {
                am.reply(peer, i + 1);  // ack_value = 1, 2, 3, ...
            }
        }
        uint64_t my_threshold = am.max_threshold;

        MPI_Barrier(MPI_COMM_WORLD);

        HIP_CHECK(hipMemset(d_cycles, 0, sizeof(uint64_t)));

        if (rank == 0) {
            hipLaunchKernelGGL(reqrep_initiator_kernel,
                               dim3(1), dim3(1), 0, 0,
                               comm.get_trigger_addr(),
                               local_ack,
                               d_cycles);
        } else {
            hipLaunchKernelGGL(reqrep_responder_kernel,
                               dim3(1), dim3(1), 0, 0,
                               comm.get_trigger_addr(),
                               am.am_ctx.d_context,
                               peer);
        }

        HIP_CHECK(hipDeviceSynchronize());

        comm.wait(my_threshold);
        comm.fast_flush(my_threshold);

        if (rank == 0 && batch >= warmup_batches) {
            uint64_t cycles;
            HIP_CHECK(hipMemcpy(&cycles, d_cycles, sizeof(uint64_t), hipMemcpyDeviceToHost));
            total_cycles += cycles;
        }

        comm.reset_counters();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        double total_time_us = (double)total_cycles / clock_rate_khz * 1000.0;
        double avg_rtt_us = total_time_us / total_roundtrips;
        double avg_oneway_us = avg_rtt_us / 2.0;
        double msg_rate = (2.0 * total_roundtrips) / (total_time_us / 1e6);

        printf("\n--- ReqRep Results ---\n");
        printf("Total GPU cycles:    %lu\n", total_cycles);
        printf("Total time:          %.2f us\n", total_time_us);
        printf("Round-trip latency:  %.2f us\n", avg_rtt_us);
        printf("One-way latency:     %.2f us (RTT/2)\n", avg_oneway_us);
        printf("Message rate:        %.0f msg/s\n", msg_rate);
        printf("  (Request=64B, Reply=8B)\n");
        fflush(stdout);
    }
}

void run_reqreq_test(GdaComm& comm, GdaAm& am, int peer,
                     int n_batches, int warmup_batches,
                     double clock_rate_khz, uint64_t* d_cycles)
{
    int rank = comm.rank();
    int total_roundtrips = n_batches * BATCH_SIZE;

    if (rank == 0) {
        printf("\n");
        printf("========================================\n");
        printf("  Mode 2: ReqReq (Request-Request)\n");
        printf("========================================\n");
        printf("  Both directions: 64 bytes (full AM)\n");
        printf("  Batches: %d (warmup: %d)\n", n_batches, warmup_batches);
        printf("  Round-trips per batch: %d\n", BATCH_SIZE);
        printf("  Total round-trips: %d\n", total_roundtrips);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint64_t total_cycles = 0;
    int total_batches = warmup_batches + n_batches;

    for (int batch = 0; batch < total_batches; batch++) {
        // Reset staging for this batch
        am.staging_idx = 0;
        am.max_threshold = 0;

        // Stage BATCH_SIZE AMs (both sides send full Requests)
        for (int i = 0; i < BATCH_SIZE; i++) {
            am_args_t args;
            args.init();
            args[0] = batch * BATCH_SIZE + i;
            am.send_handle(peer, AM_HANDLER_NOOP, args);
        }
        uint64_t my_threshold = am.max_threshold;

        MPI_Barrier(MPI_COMM_WORLD);

        HIP_CHECK(hipMemset(d_cycles, 0, sizeof(uint64_t)));

        if (rank == 0) {
            hipLaunchKernelGGL(reqreq_nodeA_kernel,
                               dim3(1), dim3(1), 0, 0,
                               comm.get_trigger_addr(),
                               am.am_ctx.d_context,
                               peer,
                               d_cycles);
        } else {
            hipLaunchKernelGGL(reqreq_nodeB_kernel,
                               dim3(1), dim3(1), 0, 0,
                               comm.get_trigger_addr(),
                               am.am_ctx.d_context,
                               peer);
        }

        HIP_CHECK(hipDeviceSynchronize());

        comm.wait(my_threshold);
        comm.fast_flush(my_threshold);

        if (rank == 0 && batch >= warmup_batches) {
            uint64_t cycles;
            HIP_CHECK(hipMemcpy(&cycles, d_cycles, sizeof(uint64_t), hipMemcpyDeviceToHost));
            total_cycles += cycles;
        }

        comm.reset_counters();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        double total_time_us = (double)total_cycles / clock_rate_khz * 1000.0;
        double avg_rtt_us = total_time_us / total_roundtrips;
        double avg_oneway_us = avg_rtt_us / 2.0;
        double msg_rate = (2.0 * total_roundtrips) / (total_time_us / 1e6);

        printf("\n--- ReqReq Results ---\n");
        printf("Total GPU cycles:    %lu\n", total_cycles);
        printf("Total time:          %.2f us\n", total_time_us);
        printf("Round-trip latency:  %.2f us\n", avg_rtt_us);
        printf("One-way latency:     %.2f us (RTT/2)\n", avg_oneway_us);
        printf("Message rate:        %.0f msg/s\n", msg_rate);
        printf("  (Both directions=64B)\n");
        fflush(stdout);
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
    int n_batches = (argc > 1) ? atoi(argv[1]) : 100;
    int warmup_batches = (argc > 2) ? atoi(argv[2]) : 10;

    if (rank == 0) {
        printf("============================================================\n");
        printf("         AM Latency Test (GPU-timed)\n");
        printf("============================================================\n");
        printf("AM size: %zu bytes\n", AM_SHORT_SIZE);
        printf("\n");
        printf("ReqRep vs ReqReq:\n");
        printf("  ReqRep: Request(64B) -> Reply(8B ack)\n");
        printf("          Reply is lightweight (only updates ack counter)\n");
        printf("  ReqReq: Request(64B) -> Request(64B)\n");
        printf("          Both directions use full AM slot\n");
        fflush(stdout);
    }

    // Create AM context
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

    // Run both tests
    run_reqrep_test(comm, am, peer, n_batches, warmup_batches, clock_rate_khz, d_cycles);
    run_reqreq_test(comm, am, peer, n_batches, warmup_batches, clock_rate_khz, d_cycles);

    if (rank == 0) {
        printf("\n============================================================\n");
        printf("SUCCESS: Both tests completed\n");
        printf("============================================================\n");
        fflush(stdout);
    }

    HIP_CHECK(hipFree(d_cycles));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
