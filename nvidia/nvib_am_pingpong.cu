/**
 * nvib_am_pingpong.cu - AM ping-pong latency test for NVIDIA IB
 *
 * Matches minimal/am_pingpong.cpp structure:
 *   - Each kernel does BATCH_SIZE round-trips internally
 *   - Average within kernel, then average across kernels
 *
 * Build:
 *   cd build && make nvib_am_pingpong
 *
 * Run:
 *   srun -N 2 -n 2 --ntasks-per-node=1 ./nvib_am_pingpong [batches] [warmup]
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <infiniband/verbs.h>

#include "nvib_am_types.hpp"
#include "nvib_am_context.hpp"
#include "nvib_am_device.cuh"
#include "mlx5_devx_qp.hpp"
#include "memory_region.hpp"

using namespace opengda;
using namespace opengda::nvib_am;

#define CUDA_CHECK(cmd) do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

constexpr int BATCH_SIZE = 100;  // Round-trips per kernel (larger batch = better amortization)

// =============================================================================
// Staging buffer for AM sends
// =============================================================================

struct AmStagingBuffer {
    am_slot_t* d_slots;           // Device memory staging slots
    MemoryRegion* mr;             // MR for RDMA
    int num_slots;

    AmStagingBuffer(struct ibv_pd* pd, int nslots, int rank)
        : d_slots(nullptr), mr(nullptr), num_slots(nslots)
    {
        size_t size = nslots * sizeof(am_slot_t);
        CUDA_CHECK(cudaMalloc(&d_slots, size));
        CUDA_CHECK(cudaMemset(d_slots, 0, size));
        mr = new MemoryRegion(pd, d_slots, size, true, rank);
    }

    ~AmStagingBuffer() {
        if (mr) delete mr;
        if (d_slots) cudaFree(d_slots);
    }

    am_slot_t* get_slot(int idx) { return &d_slots[idx]; }
    uint32_t lkey() const { return mr->lkey; }
};

// =============================================================================
// GPU Kernels for ping-pong
// =============================================================================

/**
 * Prepare AM slots on device
 */
__global__ void prepare_am_slots_kernel(
    am_slot_t* staging_slots,
    int batch_size,
    int my_rank,
    uint64_t base_seq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    am_slot_t* slot = &staging_slots[idx];

    // Prepare slot
    slot->seq = base_seq + idx;
    slot->hdr.handler_id = AM_HANDLER_NOOP;
    slot->hdr.flags = AM_FLAG_HANDLE_ONLY;
    slot->hdr.payload_len = 0;
    slot->hdr.src_rank = (uint16_t)my_rank;

    // Clear args
    for (int i = 0; i < 6; i++) {
        slot->args.data[i] = 0;
    }
    slot->args.data[0] = idx;  // Store iteration number
}

/**
 * Initiator kernel: do BATCH_SIZE round-trips, measure total time
 */
__global__ void pingpong_initiator_kernel(
    GdaDeviceStateOpt* gda_state,
    am_slot_t* staging_slots,
    uint32_t staging_lkey,
    uint64_t remote_ring_base,
    uint32_t remote_rkey,
    int nslots,
    uint64_t base_seq,
    am_context_t* am_ctx,
    int peer_rank,
    uint64_t* d_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    am_recv_state_t* recv_state = &am_ctx->recv_states[peer_rank];

    // Start timing for entire batch
    uint64_t t_start = clock64();

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Get staging slot
        am_slot_t* slot = &staging_slots[i];
        uint64_t seq = base_seq + i;

        // Calculate remote slot index
        int slot_idx = seq & (nslots - 1);

        // Send AM using two RDMA WRITEs
        // No need to wait for CQ - AM uses seq field for synchronization
        am_send_short(gda_state, slot, staging_lkey,
                      remote_ring_base, remote_rkey, slot_idx);

        // Poll for response (this implicitly waits for our send to complete
        // because peer can only respond after receiving our message)
        am_wait_one(recv_state);
    }

    // Stop timing
    uint64_t t_end = clock64();
    *d_cycles = t_end - t_start;
}

/**
 * Responder kernel: do BATCH_SIZE round-trips
 */
__global__ void pingpong_responder_kernel(
    GdaDeviceStateOpt* gda_state,
    am_slot_t* staging_slots,
    uint32_t staging_lkey,
    uint64_t remote_ring_base,
    uint32_t remote_rkey,
    int nslots,
    uint64_t base_seq,
    am_context_t* am_ctx,
    int peer_rank)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    am_recv_state_t* recv_state = &am_ctx->recv_states[peer_rank];

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Poll for incoming ping
        am_wait_one(recv_state);

        // Get staging slot for response
        am_slot_t* slot = &staging_slots[i];
        uint64_t seq = base_seq + i;

        // Calculate remote slot index
        int slot_idx = seq & (nslots - 1);

        // Send response
        // No need to wait for CQ - AM uses seq field for synchronization
        am_send_short(gda_state, slot, staging_lkey,
                      remote_ring_base, remote_rkey, slot_idx);
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) fprintf(stderr, "Need exactly 2 ranks\n");
        MPI_Finalize();
        return 1;
    }

    int peer = (rank == 0) ? 1 : 0;

    // Parse arguments
    int n_batches = (argc > 1) ? atoi(argv[1]) : 50;
    int warmup_batches = (argc > 2) ? atoi(argv[2]) : 5;

    int total_roundtrips = n_batches * BATCH_SIZE;

    // Get local rank for GPU selection
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    // Initialize CUDA
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    int gpu_id = local_rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, gpu_id);

    if (rank == 0) {
        printf("NVIB AM Ping-Pong Latency Test\n");
        printf("  Batches: %d (warmup: %d)\n", n_batches, warmup_batches);
        printf("  Round-trips per batch: %d\n", BATCH_SIZE);
        printf("  Total round-trips: %d\n", total_roundtrips);
        printf("  AM size: %zu bytes\n", AM_SHORT_SIZE);
        printf("GPU: %s, clock rate: %.0f MHz\n",
               props.name, props.clockRate / 1000.0);
        fflush(stdout);
    }

    // Open IB device
    int num_devices = 0;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "Rank %d: No IB devices found\n", rank);
        MPI_Finalize();
        return 1;
    }

    // Find mlx5_1 device
    struct ibv_device* target_dev = nullptr;
    for (int i = 0; i < num_devices; i++) {
        const char* name = ibv_get_device_name(dev_list[i]);
        if (name && strcmp(name, "mlx5_1") == 0) {
            target_dev = dev_list[i];
            break;
        }
    }
    if (!target_dev) target_dev = dev_list[0];

    struct ibv_context* ib_ctx = ibv_open_device(target_dev);
    struct ibv_pd* pd = ibv_alloc_pd(ib_ctx);

    // Create DevX QP
    DevxQp* qp = new DevxQp(ib_ctx, pd, rank, 1, 1024, 1024);

    // Create AM context (handles QP connection and address exchange)
    NvibAmContext am_ctx(ib_ctx, pd, qp, AM_DEFAULT_RING_SLOTS);

    // Create staging buffers
    AmStagingBuffer staging(pd, BATCH_SIZE + 4, rank);

    // Get clock frequency
    double clock_rate_khz = props.clockRate;

    // Allocate result buffer
    uint64_t* d_cycles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(uint64_t)));

    MPI_Barrier(MPI_COMM_WORLD);

    // Get peer's send state
    am_peer_send_state_t& peer_state = am_ctx.get_send_state(peer);

    // Run batches
    uint64_t total_cycles = 0;
    int total_batches_to_run = warmup_batches + n_batches;

    // Prepare all staging slots upfront (once)
    {
        int threads = min(BATCH_SIZE, 256);
        int blocks = (BATCH_SIZE + threads - 1) / threads;
        prepare_am_slots_kernel<<<blocks, threads>>>(
            staging.d_slots, BATCH_SIZE, rank, 1);  // Start at seq 1
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Sync before starting all batches
    MPI_Barrier(MPI_COMM_WORLD);

    for (int batch = 0; batch < total_batches_to_run; batch++) {
        // Base sequence for this batch
        uint64_t base_seq = (uint64_t)batch * BATCH_SIZE + 1;

        // Update staging slots sequence numbers on device
        // (reuse the same slots but update seq values)
        int threads = min(BATCH_SIZE, 256);
        int blocks = (BATCH_SIZE + threads - 1) / threads;
        prepare_am_slots_kernel<<<blocks, threads>>>(
            staging.d_slots, BATCH_SIZE, rank, base_seq);

        // No sync here - let it overlap with kernel launch

        if (rank == 0) {
            pingpong_initiator_kernel<<<1, 1>>>(
                am_ctx.get_gda_state(),
                staging.d_slots,
                staging.lkey(),
                peer_state.remote_ring_base,
                peer_state.remote_ring_rkey,
                peer_state.nslots,
                base_seq,
                am_ctx.get_device_context(),
                peer,
                d_cycles);
        } else {
            pingpong_responder_kernel<<<1, 1>>>(
                am_ctx.get_gda_state(),
                staging.d_slots,
                staging.lkey(),
                peer_state.remote_ring_base,
                peer_state.remote_ring_rkey,
                peer_state.nslots,
                base_seq,
                am_ctx.get_device_context(),
                peer);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // Collect timing from rank 0 (after warmup)
        if (rank == 0 && batch >= warmup_batches) {
            uint64_t cycles;
            CUDA_CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total_cycles += cycles;
        }

        // Only barrier every few batches to reduce overhead
        if ((batch + 1) % 10 == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
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

    CUDA_CHECK(cudaFree(d_cycles));

    // Cleanup
    delete qp;
    ibv_dealloc_pd(pd);
    ibv_close_device(ib_ctx);
    ibv_free_device_list(dev_list);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
