/**
 * nvib_am_pingpong_reqrep.cu - ReqRep mode AM ping-pong latency test
 *
 * ReqRep Mode vs ReqReq Mode:
 *   - ReqReq: Both sides send full 64-byte AM (am_send_short)
 *   - ReqRep: Sender sends Request, Receiver sends lightweight 8-byte Reply
 *
 * ReqRep is faster because:
 *   - Reply is only 8 bytes vs 64 bytes
 *   - Reply can be sent immediately in "handler" (no polling overhead)
 *
 * Build:
 *   cd build && make nvib_am_pingpong_reqrep
 *
 * Run:
 *   srun -N 2 -n 2 --ntasks-per-node=1 ./nvib_am_pingpong_reqrep [batches] [warmup]
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

constexpr int BATCH_SIZE = 100;  // Round-trips per kernel

// =============================================================================
// Staging buffers
// =============================================================================

struct AmStagingBuffer {
    am_slot_t* d_slots;
    MemoryRegion* mr;
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

    uint32_t lkey() const { return mr->lkey; }
};

// Ack buffer for receiving replies
struct AckBuffer {
    am_ack_entry_t* d_acks;
    MemoryRegion* mr;
    int num_entries;

    AckBuffer(struct ibv_pd* pd, int nentries, int rank)
        : d_acks(nullptr), mr(nullptr), num_entries(nentries)
    {
        size_t size = nentries * sizeof(am_ack_entry_t);
        CUDA_CHECK(cudaMalloc(&d_acks, size));
        CUDA_CHECK(cudaMemset(d_acks, 0, size));
        mr = new MemoryRegion(pd, d_acks, size, true, rank);
    }

    ~AckBuffer() {
        if (mr) delete mr;
        if (d_acks) cudaFree(d_acks);
    }

    uint64_t addr() const { return (uint64_t)d_acks; }
    uint32_t rkey() const { return mr->rkey; }
    uint32_t lkey() const { return mr->lkey; }
};

// Reply staging buffer array (pre-prepared for fast reply)
struct ReplyBufferArray {
    uint64_t* d_buf;
    MemoryRegion* mr;
    int num_entries;

    ReplyBufferArray(struct ibv_pd* pd, int nentries, int rank)
        : d_buf(nullptr), mr(nullptr), num_entries(nentries)
    {
        size_t size = nentries * sizeof(uint64_t);
        CUDA_CHECK(cudaMalloc(&d_buf, size));
        CUDA_CHECK(cudaMemset(d_buf, 0, size));
        mr = new MemoryRegion(pd, d_buf, size, true, rank);
    }

    ~ReplyBufferArray() {
        if (mr) delete mr;
        if (d_buf) cudaFree(d_buf);
    }

    uint64_t addr() const { return (uint64_t)d_buf; }
    uint32_t lkey() const { return mr->lkey; }
};

// =============================================================================
// GPU Kernels for ReqRep ping-pong
// =============================================================================

/**
 * Prepare Request slots with reply tokens embedded
 */
__global__ void prepare_request_slots_kernel(
    am_slot_t* staging_slots,
    int batch_size,
    int my_rank,
    uint64_t base_seq,
    uint64_t ack_addr,
    uint32_t ack_rkey)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    am_slot_t* slot = &staging_slots[idx];
    uint64_t seq = base_seq + idx;

    // Prepare slot
    slot->seq = seq;
    slot->hdr.handler_id = AM_HANDLER_NOOP;
    slot->hdr.flags = AM_FLAG_HANDLE_ONLY;
    slot->hdr.payload_len = 0;
    slot->hdr.src_rank = (uint16_t)my_rank;

    // Embed reply token in args
    // Ack entry for this iteration is at offset idx * sizeof(am_ack_entry_t)
    slot->args.data[0] = ack_addr + idx * sizeof(am_ack_entry_t);
    slot->args.data[1] = ack_rkey;
    slot->args.data[2] = seq;  // ack_seq = request seq
    slot->args.data[3] = 0;
    slot->args.data[4] = 0;
    slot->args.data[5] = 0;
}

/**
 * Clear ack buffer before each batch
 */
__global__ void clear_ack_buffer_kernel(am_ack_entry_t* acks, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    acks[idx].seq = 0;
}

/**
 * Prepare reply buffer array with ack_seq values
 */
__global__ void prepare_reply_buffer_kernel(uint64_t* reply_buf, int count, uint64_t base_seq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    reply_buf[idx] = base_seq + idx;
}

/**
 * Initiator kernel (ReqRep mode):
 *   - Send Request with reply token
 *   - Wait for lightweight Reply
 */
__global__ void reqrep_initiator_kernel(
    GdaDeviceStateOpt* gda_state,
    am_slot_t* staging_slots,
    uint32_t staging_lkey,
    uint64_t remote_ring_base,
    uint32_t remote_rkey,
    int nslots,
    uint64_t base_seq,
    am_ack_entry_t* ack_entries,
    uint64_t* d_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t t_start = clock64();

    for (int i = 0; i < BATCH_SIZE; i++) {
        am_slot_t* slot = &staging_slots[i];
        uint64_t seq = base_seq + i;
        int slot_idx = seq & (nslots - 1);

        // Send Request (with reply token in args)
        am_send_request(gda_state, slot, staging_lkey,
                        remote_ring_base, remote_rkey, slot_idx);

        // Wait for Reply (just poll 8-byte ack)
        am_wait_reply(&ack_entries[i], seq);
    }

    uint64_t t_end = clock64();
    *d_cycles = t_end - t_start;
}

/**
 * Responder kernel (ReqRep mode) - FAST VERSION:
 *   - Wait for Request
 *   - Send lightweight Reply using pre-prepared buffer (zero-copy)
 */
__global__ void reqrep_responder_kernel(
    GdaDeviceStateOpt* gda_state,
    am_context_t* am_ctx,
    int peer_rank,
    uint64_t reply_buf_base,
    uint32_t reply_lkey)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    am_recv_state_t* recv_state = &am_ctx->recv_states[peer_rank];

    for (int i = 0; i < BATCH_SIZE; i++) {
        // Wait for Request and send Reply using pre-prepared buffer
        am_recv_and_reply_fast(recv_state, gda_state, reply_buf_base, reply_lkey, i);
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

    // Parse arguments
    int n_batches = (argc > 1) ? atoi(argv[1]) : 50;
    int warmup_batches = (argc > 2) ? atoi(argv[2]) : 5;

    // Initialize CUDA
    int gpu_id = 0;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    if (rank == 0) {
        printf("=== NVIB AM Ping-Pong (ReqRep Mode) ===\n");
        printf("Batches: %d (warmup: %d), iterations per batch: %d\n",
               n_batches, warmup_batches, BATCH_SIZE);
        printf("ReqRep: Request=64B, Reply=8B (lightweight)\n\n");
    }

    // Open IB device
    int num_devices;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "Rank %d: No IB devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    struct ibv_context* ib_ctx = ibv_open_device(dev_list[0]);
    ibv_free_device_list(dev_list);
    if (!ib_ctx) {
        fprintf(stderr, "Rank %d: Failed to open IB device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    struct ibv_pd* pd = ibv_alloc_pd(ib_ctx);
    if (!pd) {
        fprintf(stderr, "Rank %d: Failed to allocate PD\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create DevX QP
    DevxQp* qp = new DevxQp(ib_ctx, pd, gpu_id);

    // Initialize AM context
    NvibAmContext am_ctx(ib_ctx, pd, qp, AM_DEFAULT_RING_SLOTS);

    // Create buffers
    AmStagingBuffer staging(pd, BATCH_SIZE, rank);
    AckBuffer ack_buf(pd, BATCH_SIZE, rank);
    ReplyBufferArray reply_buf(pd, BATCH_SIZE, rank);  // Pre-prepared reply array

    // Get peer info
    int peer = (rank == 0) ? 1 : 0;
    am_peer_send_state_t& peer_state = am_ctx.get_send_state(peer);

    // Exchange ack buffer info for ReqRep mode
    struct AckExchange {
        uint64_t addr;
        uint32_t rkey;
    };
    AckExchange my_ack = { ack_buf.addr(), ack_buf.rkey() };
    AckExchange peer_ack;
    MPI_Sendrecv(&my_ack, sizeof(AckExchange), MPI_BYTE, peer, 0,
                 &peer_ack, sizeof(AckExchange), MPI_BYTE, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Allocate cycle counter
    uint64_t* d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(uint64_t)));

    // Run benchmark
    int total_batches_to_run = warmup_batches + n_batches;
    uint64_t total_cycles = 0;

    int threads = min(BATCH_SIZE, 256);
    int blocks = (BATCH_SIZE + threads - 1) / threads;

    // Pre-warmup: prepare first batch BEFORE barrier
    uint64_t base_seq = 1;
    if (rank == 0) {
        clear_ack_buffer_kernel<<<blocks, threads>>>(ack_buf.d_acks, BATCH_SIZE);
        prepare_request_slots_kernel<<<blocks, threads>>>(
            staging.d_slots, BATCH_SIZE, rank, base_seq,
            ack_buf.addr(), ack_buf.rkey());
    } else {
        prepare_reply_buffer_kernel<<<blocks, threads>>>(
            reply_buf.d_buf, BATCH_SIZE, base_seq);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    for (int batch = 0; batch < total_batches_to_run; batch++) {
        base_seq = (uint64_t)batch * BATCH_SIZE + 1;

        if (rank == 0) {
            // Run initiator kernel (data already prepared)
            reqrep_initiator_kernel<<<1, 1>>>(
                am_ctx.get_gda_state(),
                staging.d_slots,
                staging.lkey(),
                peer_state.remote_ring_base,
                peer_state.remote_ring_rkey,
                peer_state.nslots,
                base_seq,
                ack_buf.d_acks,
                d_cycles);

            CUDA_CHECK(cudaDeviceSynchronize());

            // Prepare next batch AFTER this one completes
            if (batch + 1 < total_batches_to_run) {
                uint64_t next_seq = (uint64_t)(batch + 1) * BATCH_SIZE + 1;
                clear_ack_buffer_kernel<<<blocks, threads>>>(ack_buf.d_acks, BATCH_SIZE);
                prepare_request_slots_kernel<<<blocks, threads>>>(
                    staging.d_slots, BATCH_SIZE, rank, next_seq,
                    ack_buf.addr(), ack_buf.rkey());
            }
        } else {
            // Run responder kernel (data already prepared)
            reqrep_responder_kernel<<<1, 1>>>(
                am_ctx.get_gda_state(),
                am_ctx.get_device_context(),
                peer,
                reply_buf.addr(),
                reply_buf.lkey());

            CUDA_CHECK(cudaDeviceSynchronize());

            // Prepare next batch AFTER this one completes
            if (batch + 1 < total_batches_to_run) {
                uint64_t next_seq = (uint64_t)(batch + 1) * BATCH_SIZE + 1;
                prepare_reply_buffer_kernel<<<blocks, threads>>>(
                    reply_buf.d_buf, BATCH_SIZE, next_seq);
            }
        }

        // Collect timing from rank 0 (after warmup)
        if (rank == 0 && batch >= warmup_batches) {
            uint64_t cycles;
            CUDA_CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total_cycles += cycles;
        }

        // Sync occasionally to prevent drift
        if ((batch + 1) % 20 == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Report results
    if (rank == 0) {
        int measurement_batches = n_batches;
        int total_roundtrips = measurement_batches * BATCH_SIZE;

        // Get GPU clock frequency
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));
        double clock_ghz = prop.clockRate / 1e6;

        double total_time_us = (double)total_cycles / (clock_ghz * 1e3);
        double avg_rtt_us = total_time_us / total_roundtrips;
        double one_way_us = avg_rtt_us / 2.0;

        printf("=== Results (ReqRep Mode) ===\n");
        printf("Total roundtrips: %d\n", total_roundtrips);
        printf("Total GPU cycles: %lu\n", total_cycles);
        printf("GPU clock: %.2f GHz\n", clock_ghz);
        printf("Total time: %.2f us\n", total_time_us);
        printf("Average RTT: %.3f us\n", avg_rtt_us);
        printf("One-way latency: %.3f us\n", one_way_us);
        printf("\n");
        printf("Comparison:\n");
        printf("  ReqRep: Request=64B + Reply=8B = 72B per RTT\n");
        printf("  ReqReq: Request=64B + Request=64B = 128B per RTT\n");
        printf("  Savings: 44%% less data transferred\n");
    }

    // Cleanup
    cudaFree(d_cycles);
    delete qp;
    ibv_dealloc_pd(pd);
    ibv_close_device(ib_ctx);

    MPI_Finalize();
    return 0;
}
