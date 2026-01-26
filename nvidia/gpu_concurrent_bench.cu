/**
 * gpu_concurrent_bench.cu - GPU-Triggered Concurrent RDMA Benchmark
 *
 * Similar to minimal/benchmark_runner.hpp:
 *   - N_STREAMS concurrent GPU-triggered RDMA transfers
 *   - Single GPU kernel triggers all transfers
 *   - Measures FULL END-TO-END time including network transfer
 *
 * This measures actual transfer completion, not just WQE posting.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <unistd.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include "mpi_bootstrap.hpp"
#include "memory_region.hpp"
#include "mlx5_devx_qp.hpp"
#include "gda_device_opt.cuh"

using namespace opengda;
using namespace std::chrono;

// Configuration - similar to minimal/
constexpr int N_STREAMS = 32;       // Number of concurrent transfers
constexpr int NUM_ITERATIONS = 20;  // Iterations per size

constexpr size_t TEST_SIZES[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024,
    2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024
};
constexpr int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);
constexpr size_t MAX_SIZE = 16 * 1024 * 1024;  // 16MB per stream

//==============================================================================
// Connection info exchange
//==============================================================================

struct ConnInfo {
    uint32_t qpn;
    uint16_t lid;
    uint8_t  gid[16];
    uint32_t psn;
};

struct BufInfo {
    uint64_t addr;
    uint32_t rkey;
};

//==============================================================================
// GPU Kernel - Concurrent RDMA writes with completion polling
//==============================================================================

/**
 * GPU kernel for concurrent RDMA writes with END-TO-END timing.
 *
 * Sender (rank 0):
 *   1. Record start time
 *   2. Each thread builds WQE and writes data
 *   3. Thread 0 rings doorbell
 *   4. Poll local flag for ACK from receiver
 *   5. Record end time
 *
 * The receiver will write back an ACK when all data arrives.
 */
__global__ void gpu_concurrent_write_kernel(
    GdaDeviceStateOpt* state,
    uint64_t* send_addrs,       // Array of send buffer addresses
    uint32_t* send_lkeys,       // Array of lkeys
    uint64_t remote_base_addr,  // Base of remote buffer
    uint32_t remote_rkey,
    size_t msg_size,
    size_t stream_stride,       // Offset between streams
    int n_streams,
    volatile uint64_t* ack_flag,  // Local flag for ACK from receiver
    uint64_t expected_ack,
    uint64_t* start_clock,
    uint64_t* end_clock)
{
    int stream_id = threadIdx.x;

    __shared__ uint64_t base_wqe_idx;

    // Thread 0 records start time and gets base WQE index
    if (stream_id == 0) {
        *start_clock = clock64();
        base_wqe_idx = gda_load_relaxed_u64(state->prod_idx);
    }
    __syncthreads();

    // Each thread handles one stream
    if (stream_id < n_streams) {
        // Calculate remote address and WQE slot
        uint64_t remote_addr = remote_base_addr + (stream_id * stream_stride);
        uint16_t wqe_slot = (uint16_t)((base_wqe_idx + stream_id) & 0xFFFF);

        // Build WQE
        size_t actual_size = (msg_size < 8) ? 8 : msg_size;
        bool signaled = false;

        gda_build_rdma_write_wqe_opt(
            state, send_addrs[stream_id], send_lkeys[stream_id],
            remote_addr, remote_rkey,
            actual_size, wqe_slot, signaled
        );
    }

    __syncthreads();

    // Thread 0 rings doorbell and waits for ACK
    if (stream_id == 0) {
        uint16_t new_prod = (uint16_t)((base_wqe_idx + n_streams) & 0xFFFF);
        gda_ring_doorbell_bf(state, new_prod);

        // Wait for ACK from receiver (receiver writes back when all data verified)
        while (*ack_flag != expected_ack) {
            // Spin wait
        }

        // Record end time after ACK received
        *end_clock = clock64();
    }
}

/**
 * Receiver kernel - polls for data arrival and sends ACK
 */
__global__ void gpu_receiver_kernel(
    GdaDeviceStateOpt* state,
    volatile uint64_t** recv_flags,  // Flags in recv buffer to check
    int n_streams,
    uint64_t expected_seq,
    uint64_t ack_remote_addr,  // Remote address to write ACK
    uint32_t ack_remote_rkey,
    uint64_t ack_local_addr,
    uint32_t ack_local_lkey,
    uint64_t ack_value)
{
    int stream_id = threadIdx.x;

    // Each thread polls its recv flag
    if (stream_id < n_streams) {
        while (*recv_flags[stream_id] != expected_seq) {
            // Spin wait for data
        }
    }

    __syncthreads();

    // Thread 0 sends ACK back to sender
    if (stream_id == 0) {
        // Write ACK value to local buffer first
        *(volatile uint64_t*)ack_local_addr = ack_value;
        __threadfence_system();

        // Get WQE slot and build RDMA WRITE for ACK
        uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
        uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);

        gda_build_rdma_write_wqe_opt(
            state, ack_local_addr, ack_local_lkey,
            ack_remote_addr, ack_remote_rkey,
            8, wqe_slot, false
        );

        // Ring doorbell
        gda_ring_doorbell_bf(state, (uint16_t)((prod + 1) & 0xFFFF));
    }
}

//==============================================================================
// Helper functions
//==============================================================================

void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

double cycles_to_us(uint64_t cycles, double clock_rate_khz) {
    return cycles / (clock_rate_khz / 1000.0);
}

const char* format_size(size_t size, char* buf) {
    if (size < 1024) {
        snprintf(buf, 32, "%zuB", size);
    } else if (size < 1024 * 1024) {
        snprintf(buf, 32, "%zuKB", size / 1024);
    } else {
        snprintf(buf, 32, "%zuMB", size / (1024 * 1024));
    }
    return buf;
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_size != 2) {
        if (mpi_rank == 0) {
            fprintf(stderr, "This test requires exactly 2 ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    int peer = (mpi_rank == 0) ? 1 : 0;

    // Get local rank for GPU selection
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    // Initialize CUDA
    int num_gpus;
    cuda_check(cudaGetDeviceCount(&num_gpus), "get device count");
    int gpu_id = local_rank % num_gpus;
    cuda_check(cudaSetDevice(gpu_id), "set device");

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, gpu_id);
    double clock_rate_khz = props.clockRate;

    // Open InfiniBand device
    struct ibv_device** dev_list = ibv_get_device_list(nullptr);
    if (!dev_list || !dev_list[0]) {
        fprintf(stderr, "Rank %d: No IB devices found\n", mpi_rank);
        MPI_Finalize();
        return 1;
    }

    struct ibv_context* ctx = ibv_open_device(dev_list[0]);
    struct ibv_pd* pd = ibv_alloc_pd(ctx);

    // Create DevX QP
    DevxQp* devx_qp = new DevxQp(ctx, pd, mpi_rank, 1, 256, 512);

    // Exchange connection info
    ConnInfo my_conn, peer_conn;
    my_conn.qpn = devx_qp->qpn;
    my_conn.lid = 0;
    my_conn.psn = 0;

    union ibv_gid my_gid;
    ibv_query_gid(ctx, 1, 1, &my_gid);
    memcpy(my_conn.gid, &my_gid, 16);

    MPI_Sendrecv(&my_conn, sizeof(ConnInfo), MPI_BYTE, peer, 0,
                 &peer_conn, sizeof(ConnInfo), MPI_BYTE, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Connect QP
    devx_qp->rst2init();
    devx_qp->init2rtr(peer_conn.qpn, peer_conn.lid, peer_conn.gid, peer_conn.psn, 3);
    devx_qp->rtr2rts(my_conn.psn);

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Allocate buffers
    //==========================================================================

    // Per-stream send buffers
    void* d_send_bufs[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cuda_check(cudaMalloc(&d_send_bufs[i], MAX_SIZE), "alloc send buf");
        cuda_check(cudaMemset(d_send_bufs[i], 0, MAX_SIZE), "memset send buf");
    }

    // Shared receive buffer
    void* d_recv_buf;
    cuda_check(cudaMalloc(&d_recv_buf, MAX_SIZE * N_STREAMS), "alloc recv buf");
    cuda_check(cudaMemset(d_recv_buf, 0, MAX_SIZE * N_STREAMS), "memset recv buf");

    // ACK buffer (for receiver to write back to sender)
    uint64_t* d_ack_buf;
    cuda_check(cudaMalloc(&d_ack_buf, sizeof(uint64_t)), "alloc ack buf");
    cuda_check(cudaMemset(d_ack_buf, 0, sizeof(uint64_t)), "memset ack buf");

    // Register memory
    MemoryRegion* send_mrs[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        send_mrs[i] = new MemoryRegion(pd, d_send_bufs[i], MAX_SIZE, true, mpi_rank);
    }
    MemoryRegion* recv_mr = new MemoryRegion(pd, d_recv_buf, MAX_SIZE * N_STREAMS, true, mpi_rank);
    MemoryRegion* ack_mr = new MemoryRegion(pd, d_ack_buf, sizeof(uint64_t), true, mpi_rank);

    // Exchange buffer info (recv buffer and ack buffer)
    struct { BufInfo recv; BufInfo ack; } my_bufs, peer_bufs;
    my_bufs.recv.addr = (uint64_t)d_recv_buf;
    my_bufs.recv.rkey = recv_mr->rkey;
    my_bufs.ack.addr = (uint64_t)d_ack_buf;
    my_bufs.ack.rkey = ack_mr->rkey;

    MPI_Sendrecv(&my_bufs, sizeof(my_bufs), MPI_BYTE, peer, 1,
                 &peer_bufs, sizeof(peer_bufs), MPI_BYTE, peer, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //==========================================================================
    // Setup GPU arrays
    //==========================================================================

    // Send addresses and lkeys arrays
    uint64_t h_send_addrs[N_STREAMS];
    uint32_t h_send_lkeys[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        h_send_addrs[i] = (uint64_t)d_send_bufs[i];
        h_send_lkeys[i] = send_mrs[i]->lkey;
    }
    uint64_t* d_send_addrs;
    uint32_t* d_send_lkeys;
    cuda_check(cudaMalloc(&d_send_addrs, N_STREAMS * sizeof(uint64_t)), "alloc send addrs");
    cuda_check(cudaMalloc(&d_send_lkeys, N_STREAMS * sizeof(uint32_t)), "alloc send lkeys");
    cuda_check(cudaMemcpy(d_send_addrs, h_send_addrs, N_STREAMS * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "copy send addrs");
    cuda_check(cudaMemcpy(d_send_lkeys, h_send_lkeys, N_STREAMS * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "copy send lkeys");

    // Device state
    GdaDeviceStateOpt h_state;
    memset(&h_state, 0, sizeof(h_state));
    h_state.qpn = devx_qp->qpn;
    h_state.nwqes = 1 << devx_qp->log_wq_size;
    h_state.nwqes_mask = h_state.nwqes - 1;
    h_state.wqe_buf = devx_qp->d_wq_buf;
    h_state.wqe_lkey = 0;
    h_state.dbrec = devx_qp->d_dbrec;
    h_state.bf_reg = (volatile uint64_t*)devx_qp->d_uar_reg;
    h_state.prod_idx = devx_qp->d_prod_idx;
    h_state.remote_addr = peer_bufs.recv.addr;
    h_state.remote_rkey = peer_bufs.recv.rkey;

    GdaDeviceStateOpt* d_state;
    cuda_check(cudaMalloc(&d_state, sizeof(GdaDeviceStateOpt)), "alloc state");
    cuda_check(cudaMemcpy(d_state, &h_state, sizeof(GdaDeviceStateOpt),
                          cudaMemcpyHostToDevice), "copy state");

    // Timing
    uint64_t* d_start_clock;
    uint64_t* d_end_clock;
    cuda_check(cudaMalloc(&d_start_clock, sizeof(uint64_t)), "alloc start clock");
    cuda_check(cudaMalloc(&d_end_clock, sizeof(uint64_t)), "alloc end clock");

    // Recv flag pointers array (for receiver polling)
    volatile uint64_t** d_recv_flags;
    cuda_check(cudaMalloc(&d_recv_flags, N_STREAMS * sizeof(uint64_t*)), "alloc recv flags");

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Print header
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=======================================================\n");
        printf("   GPU-Triggered Concurrent RDMA Benchmark\n");
        printf("   (End-to-End: includes network transfer time)\n");
        printf("=======================================================\n");
        printf("GPU: %s (%.2f GHz)\n", props.name, clock_rate_khz / 1e6);
        printf("IB Device: %s\n", ibv_get_device_name(dev_list[0]));
        printf("Concurrent streams: %d\n", N_STREAMS);
        printf("Iterations per size: %d\n", NUM_ITERATIONS);
        printf("-------------------------------------------------------\n\n");

        printf("%-8s  %12s  %12s  %s\n", "Size", "Total(us)", "Per-xfer(us)", "Statistics");
        printf("========  ============  ============  =====================================\n");
        printf("Note: %d concurrent streams, per-xfer = total/%d\n\n", N_STREAMS, N_STREAMS);
        fflush(stdout);
    }

    //==========================================================================
    // Run benchmark
    //==========================================================================

    for (int size_idx = 0; size_idx < NUM_TEST_SIZES; size_idx++) {
        size_t msg_size = TEST_SIZES[size_idx];
        size_t stream_stride = MAX_SIZE;
        size_t flag_offset = (msg_size < 8) ? 0 : (msg_size - 8);

        double iteration_times[NUM_ITERATIONS];
        int successful_iters = 0;

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            uint64_t seq_num = iter + 1;

            // Reset buffers
            if (mpi_rank == 1) {
                cuda_check(cudaMemset(d_recv_buf, 0, MAX_SIZE * N_STREAMS), "clear recv");
            }
            if (mpi_rank == 0) {
                cuda_check(cudaMemset(d_ack_buf, 0, sizeof(uint64_t)), "clear ack");
                for (int i = 0; i < N_STREAMS; i++) {
                    // Set send buffer with pattern, and sequence at flag offset
                    uint8_t pattern = (iter + 0xA0 + i) & 0xFF;
                    cuda_check(cudaMemset(d_send_bufs[i], pattern, msg_size), "set send pattern");
                    // Write sequence number at flag location
                    cuda_check(cudaMemcpy((char*)d_send_bufs[i] + flag_offset, &seq_num,
                                          sizeof(uint64_t), cudaMemcpyHostToDevice), "set seq");
                }
            }
            cuda_check(cudaDeviceSynchronize(), "sync init");

            MPI_Barrier(MPI_COMM_WORLD);

            // Setup recv flag pointers for this iteration
            volatile uint64_t* h_recv_flags[N_STREAMS];
            for (int i = 0; i < N_STREAMS; i++) {
                h_recv_flags[i] = (volatile uint64_t*)((char*)d_recv_buf + i * stream_stride + flag_offset);
            }
            cuda_check(cudaMemcpy(d_recv_flags, h_recv_flags, N_STREAMS * sizeof(uint64_t*),
                                  cudaMemcpyHostToDevice), "copy recv flags");

            // Launch kernels on both ranks
            auto t_start = high_resolution_clock::now();

            if (mpi_rank == 0) {
                // Sender kernel
                gpu_concurrent_write_kernel<<<1, N_STREAMS>>>(
                    d_state,
                    d_send_addrs,
                    d_send_lkeys,
                    peer_bufs.recv.addr,
                    peer_bufs.recv.rkey,
                    msg_size,
                    stream_stride,
                    N_STREAMS,
                    (volatile uint64_t*)d_ack_buf,
                    seq_num,
                    d_start_clock,
                    d_end_clock
                );
            } else {
                // Receiver kernel
                gpu_receiver_kernel<<<1, N_STREAMS>>>(
                    d_state,
                    d_recv_flags,
                    N_STREAMS,
                    seq_num,
                    peer_bufs.ack.addr,  // Write ACK to sender's ack buffer
                    peer_bufs.ack.rkey,
                    (uint64_t)d_ack_buf,  // Use our ack buffer as source
                    ack_mr->lkey,
                    seq_num              // ACK value = seq_num
                );
            }

            cuda_check(cudaDeviceSynchronize(), "kernel sync");

            auto t_end = high_resolution_clock::now();

            if (mpi_rank == 0) {
                // Get GPU-measured time
                uint64_t start_cycles, end_cycles;
                cuda_check(cudaMemcpy(&start_cycles, d_start_clock, sizeof(uint64_t),
                                      cudaMemcpyDeviceToHost), "copy start");
                cuda_check(cudaMemcpy(&end_cycles, d_end_clock, sizeof(uint64_t),
                                      cudaMemcpyDeviceToHost), "copy end");

                double gpu_time_us = cycles_to_us(end_cycles - start_cycles, clock_rate_khz);
                iteration_times[successful_iters++] = gpu_time_us;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Calculate and print statistics (only rank 0)
        if (mpi_rank == 0) {
            // Sort times
            for (int i = 0; i < successful_iters - 1; i++) {
                for (int j = i + 1; j < successful_iters; j++) {
                    if (iteration_times[j] < iteration_times[i]) {
                        double tmp = iteration_times[i];
                        iteration_times[i] = iteration_times[j];
                        iteration_times[j] = tmp;
                    }
                }
            }

            int samples = (successful_iters < 10) ? successful_iters : 10;
            double sum = 0.0;
            for (int i = 0; i < samples; i++) sum += iteration_times[i];
            double avg = sum / samples;
            double per_xfer = avg / N_STREAMS;

            char size_buf[32];
            printf("%-8s  %12.2f  %12.2f  (min=%.2f max=%.2f)\n",
                   format_size(msg_size, size_buf), avg, per_xfer,
                   iteration_times[0], iteration_times[samples - 1]);
            fflush(stdout);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    //==========================================================================
    // Summary
    //==========================================================================

    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("\n=======================================================\n");
        printf("Benchmark complete!\n");
        printf("\nThis benchmark measures END-TO-END time including:\n");
        printf("  - WQE build + doorbell on sender\n");
        printf("  - Network transfer\n");
        printf("  - Data polling on receiver\n");
        printf("  - ACK write back to sender\n");
        printf("=======================================================\n");
        fflush(stdout);
    }

    // Cleanup
    cudaFree(d_state);
    cudaFree(d_start_clock);
    cudaFree(d_end_clock);
    cudaFree(d_send_addrs);
    cudaFree(d_send_lkeys);
    cudaFree(d_recv_flags);
    cudaFree(d_ack_buf);

    for (int i = 0; i < N_STREAMS; i++) {
        delete send_mrs[i];
        cudaFree(d_send_bufs[i]);
    }
    delete recv_mr;
    delete ack_mr;
    cudaFree(d_recv_buf);

    delete devx_qp;
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);

    MPI_Finalize();
    return 0;
}
