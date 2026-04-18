/**
 * am_pingpong.cpp - AM latency test (GPU-timed)
 *
 * Tests two modes (ported from legacy/ofi/am_pingpong.cpp, aligned with the
 * Bootstrap-based API used by barrier_test.cpp):
 *   1. ReqRep (Request-Reply): Sender sends Request (64B), Receiver replies
 *                              with lightweight Reply (8B ack only).
 *   2. ReqReq (Request-Request): Both sides send full Request (64B), symmetric.
 *
 * Run: FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 -t 2 \
 *         ./am_pingpong [n_batches] [warmup_batches]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

#include "gicc/bootstrap/bootstrap.hpp"
#include "gicc/platform/ofi/internal/hip_device_context.hpp"
#include "gicc/platform/ofi/internal/fabric.hpp"
#include "gicc/platform/ofi/internal/gda_am.hpp"

using namespace gicc::am;

#define HIP_CHECK(cmd) do {                                                   \
    hipError_t err = cmd;                                                     \
    if (err != hipSuccess) {                                                  \
        fprintf(stderr, "HIP error: %s at %s:%d\n",                           \
                hipGetErrorString(err), __FILE__, __LINE__);                  \
        exit(1);                                                              \
    }                                                                         \
} while (0)

constexpr int BATCH_SIZE = 10;  // Operations per kernel

// =============================================================================
// ReqRep kernels (Request 64B -> Reply 8B ack)
// =============================================================================

__global__ void reqrep_initiator_kernel(
    volatile uint64_t* trigger_addr,
    volatile uint64_t* local_ack,
    uint64_t* d_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t t_start = clock64();

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Request = body put + seq put → 2 thresholds per iter
        *trigger_addr = (i + 1) * 2;
        __threadfence_system();

        while (*local_ack < (uint64_t)(i + 1)) { /* spin */ }
        __threadfence_system();
    }

    uint64_t t_end = clock64();
    *d_cycles = t_end - t_start;
}

__global__ void reqrep_responder_kernel(
    volatile uint64_t* trigger_addr,
    am_context_t* am_ctx,
    int peer_rank)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    am_recv_state_t* recv_state = &am_ctx->recv_states[peer_rank];

    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t expected = recv_state->expected_seq;
        int slot_idx = expected & (recv_state->nslots - 1);
        am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

        while (slot->seq != expected) { /* spin */ }
        __threadfence_system();
        recv_state->expected_seq = expected + 1;

        // Reply = single 8B put → 1 threshold per iter
        *trigger_addr = i + 1;
        __threadfence_system();
    }
}

// =============================================================================
// ReqReq kernels (Request 64B both ways)
// =============================================================================

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
        *trigger_addr = (i + 1) * 2;
        __threadfence_system();

        uint64_t expected = recv_state->expected_seq;
        int slot_idx = expected & (recv_state->nslots - 1);
        am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

        while (slot->seq != expected) { /* spin */ }
        __threadfence_system();
        recv_state->expected_seq = expected + 1;
    }

    uint64_t t_end = clock64();
    *d_cycles = t_end - t_start;
}

__global__ void reqreq_nodeB_kernel(
    volatile uint64_t* trigger_addr,
    am_context_t* am_ctx,
    int peer_rank)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    am_recv_state_t* recv_state = &am_ctx->recv_states[peer_rank];

    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t expected = recv_state->expected_seq;
        int slot_idx = expected & (recv_state->nslots - 1);
        am_slot_t* slot = &recv_state->inbox_slots[slot_idx];

        while (slot->seq != expected) { /* spin */ }
        __threadfence_system();
        recv_state->expected_seq = expected + 1;

        *trigger_addr = (i + 1) * 2;
        __threadfence_system();
    }
}

// =============================================================================
// Test runners
// =============================================================================

static void run_reqrep_test(gicc::Fabric& comm, GdaAm& am, int peer,
                            int n_batches, int warmup_batches,
                            double clock_rate_khz, uint64_t* d_cycles)
{
    int rank = comm.rank();
    int total_roundtrips = n_batches * BATCH_SIZE;

    if (rank == 0) {
        printf("\n========================================\n");
        printf("  Mode 1: ReqRep (Request-Reply)\n");
        printf("========================================\n");
        printf("  Request: 64 bytes (full AM)\n");
        printf("  Reply:   8 bytes (ack only)\n");
        printf("  Batches: %d (warmup: %d)\n", n_batches, warmup_batches);
        printf("  Round-trips per batch: %d\n", BATCH_SIZE);
        printf("  Total round-trips: %d\n", total_roundtrips);
        fflush(stdout);
    }

    volatile uint64_t* local_ack = am.am_ctx.h_send_states[peer].local_ack;

    comm.boot.barrier();

    uint64_t total_cycles = 0;
    int total_batches = warmup_batches + n_batches;

    for (int batch = 0; batch < total_batches; batch++) {
        am.staging_idx = 0;
        am.reply_staging_idx = 0;
        am.max_threshold = 0;

        uint64_t zero = 0;
        HIP_CHECK(hipMemcpy((void*)local_ack, &zero, sizeof(uint64_t),
                            hipMemcpyHostToDevice));

        if (rank == 0) {
            for (int i = 0; i < BATCH_SIZE; i++) {
                am_args_t args; args.init();
                args[0] = batch * BATCH_SIZE + i;
                am.send_handle(peer, AM_HANDLER_NOOP, args);
            }
        } else {
            for (int i = 0; i < BATCH_SIZE; i++) {
                am.reply(peer, i + 1);
            }
        }
        uint64_t my_threshold = am.max_threshold;

        comm.boot.barrier();

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
            HIP_CHECK(hipMemcpy(&cycles, d_cycles, sizeof(uint64_t),
                                hipMemcpyDeviceToHost));
            total_cycles += cycles;
        }

        comm.reset_counters();
    }

    comm.boot.barrier();

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

static void run_reqreq_test(gicc::Fabric& comm, GdaAm& am, int peer,
                            int n_batches, int warmup_batches,
                            double clock_rate_khz, uint64_t* d_cycles)
{
    int rank = comm.rank();
    int total_roundtrips = n_batches * BATCH_SIZE;

    if (rank == 0) {
        printf("\n========================================\n");
        printf("  Mode 2: ReqReq (Request-Request)\n");
        printf("========================================\n");
        printf("  Both directions: 64 bytes (full AM)\n");
        printf("  Batches: %d (warmup: %d)\n", n_batches, warmup_batches);
        printf("  Round-trips per batch: %d\n", BATCH_SIZE);
        printf("  Total round-trips: %d\n", total_roundtrips);
        fflush(stdout);
    }

    comm.boot.barrier();

    uint64_t total_cycles = 0;
    int total_batches = warmup_batches + n_batches;

    for (int batch = 0; batch < total_batches; batch++) {
        am.staging_idx = 0;
        am.max_threshold = 0;

        for (int i = 0; i < BATCH_SIZE; i++) {
            am_args_t args; args.init();
            args[0] = batch * BATCH_SIZE + i;
            am.send_handle(peer, AM_HANDLER_NOOP, args);
        }
        uint64_t my_threshold = am.max_threshold;

        comm.boot.barrier();

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
            HIP_CHECK(hipMemcpy(&cycles, d_cycles, sizeof(uint64_t),
                                hipMemcpyDeviceToHost));
            total_cycles += cycles;
        }

        comm.reset_counters();
    }

    comm.boot.barrier();

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

    gicc::Bootstrap boot;
    gicc::Fabric comm(boot);
    int rank = comm.rank();
    int nranks = comm.size();

    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "am_pingpong needs exactly 2 ranks\n");
        return 1;
    }

    int peer = (rank == 0) ? 1 : 0;

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
        printf("  ReqReq: Request(64B) -> Request(64B)\n");
        fflush(stdout);
    }

    GdaAm am(comm, 128, BATCH_SIZE + 4);

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, comm.gpu_id()));
    double clock_rate_khz = props.clockRate;

    if (rank == 0) {
        printf("GPU clock rate: %.0f MHz\n", clock_rate_khz / 1000);
        fflush(stdout);
    }

    uint64_t* d_cycles = nullptr;
    HIP_CHECK(hipMalloc(&d_cycles, sizeof(uint64_t)));

    run_reqrep_test(comm, am, peer, n_batches, warmup_batches,
                    clock_rate_khz, d_cycles);
    run_reqreq_test(comm, am, peer, n_batches, warmup_batches,
                    clock_rate_khz, d_cycles);

    if (rank == 0) {
        printf("\n============================================================\n");
        printf("SUCCESS: Both tests completed\n");
        printf("============================================================\n");
        fflush(stdout);
    }

    HIP_CHECK(hipFree(d_cycles));
    comm.boot.barrier();
    return 0;
}
