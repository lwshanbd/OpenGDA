/**
 * gpu_concurrent_bench.cu - GPU-Triggered Concurrent RDMA Benchmark
 *
 * Similar to minimal/benchmark_runner.hpp:
 *   - N_STREAMS concurrent GPU-triggered RDMA transfers
 *   - Single GPU kernel triggers all transfers
 *   - Measures total time / N_STREAMS = per-transfer time
 *
 * This shows the amortized cost of GPU-triggered RDMA with batching.
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
// GPU Kernel - Concurrent RDMA writes (one-way, measures WQE build + doorbell)
//==============================================================================

/**
 * GPU kernel for concurrent RDMA writes.
 * This kernel measures the time to build N WQEs and ring the doorbell.
 * Completion is verified via MPI_Barrier and data verification on rank 1.
 */
__global__ void gpu_concurrent_write_kernel(
    GdaDeviceStateOpt* state,
    uint64_t* send_addrs,       // Array of send buffer addresses (one per stream)
    uint32_t* send_lkeys,       // Array of lkeys (one per stream)
    uint64_t remote_base_addr,  // Base of remote buffer
    uint32_t remote_rkey,
    size_t msg_size,
    size_t stream_stride,       // Offset between streams in remote buffer
    int n_streams,
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
        // Calculate remote address and WQE slot for this stream
        uint64_t remote_addr = remote_base_addr + (stream_id * stream_stride);
        uint16_t wqe_slot = (uint16_t)((base_wqe_idx + stream_id) & 0xFFFF);

        // Build WQE
        size_t actual_size = (msg_size < 8) ? 8 : msg_size;
        bool signaled = false;  // No signaling needed for this test

        gda_build_rdma_write_wqe_opt(
            state, send_addrs[stream_id], send_lkeys[stream_id],
            remote_addr, remote_rkey,
            actual_size, wqe_slot, signaled
        );
    }

    __syncthreads();

    // Thread 0 rings doorbell for all WQEs and records end time
    if (stream_id == 0) {
        uint16_t new_prod = (uint16_t)((base_wqe_idx + n_streams) & 0xFFFF);
        gda_ring_doorbell_bf(state, new_prod);

        // Record end time after doorbell
        *end_clock = clock64();
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

    // Create DevX QP - need larger WQ for concurrent operations
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

    // Per-stream send buffers (source data)
    void* d_send_bufs[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cuda_check(cudaMalloc(&d_send_bufs[i], MAX_SIZE), "alloc send buf");
        cuda_check(cudaMemset(d_send_bufs[i], 0, MAX_SIZE), "memset send buf");
    }

    // Shared receive buffer (destination for peer's writes)
    void* d_recv_buf;
    cuda_check(cudaMalloc(&d_recv_buf, MAX_SIZE * N_STREAMS), "alloc recv buf");
    cuda_check(cudaMemset(d_recv_buf, 0, MAX_SIZE * N_STREAMS), "memset recv buf");

    // Register memory
    MemoryRegion* send_mrs[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        send_mrs[i] = new MemoryRegion(pd, d_send_bufs[i], MAX_SIZE, true, mpi_rank);
    }
    MemoryRegion* recv_mr = new MemoryRegion(pd, d_recv_buf, MAX_SIZE * N_STREAMS, true, mpi_rank);

    // Exchange buffer info
    BufInfo my_buf, peer_buf;
    my_buf.addr = (uint64_t)d_recv_buf;
    my_buf.rkey = recv_mr->rkey;

    MPI_Sendrecv(&my_buf, sizeof(BufInfo), MPI_BYTE, peer, 1,
                 &peer_buf, sizeof(BufInfo), MPI_BYTE, peer, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //==========================================================================
    // Setup GPU arrays
    //==========================================================================

    // Send addresses array
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
    h_state.remote_addr = peer_buf.addr;
    h_state.remote_rkey = peer_buf.rkey;

    GdaDeviceStateOpt* d_state;
    cuda_check(cudaMalloc(&d_state, sizeof(GdaDeviceStateOpt)), "alloc state");
    cuda_check(cudaMemcpy(d_state, &h_state, sizeof(GdaDeviceStateOpt),
                          cudaMemcpyHostToDevice), "copy state");

    // Timing
    uint64_t* d_start_clock;
    uint64_t* d_end_clock;
    cuda_check(cudaMalloc(&d_start_clock, sizeof(uint64_t)), "alloc start clock");
    cuda_check(cudaMalloc(&d_end_clock, sizeof(uint64_t)), "alloc end clock");

    // Host verification buffer
    uint8_t* h_verify_buf = (uint8_t*)malloc(MAX_SIZE * N_STREAMS);

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Print header
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=======================================================\n");
        printf("   GPU-Triggered Concurrent RDMA Benchmark\n");
        printf("   (NVIDIA + InfiniBand with DevX API)\n");
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
    // Run benchmark for each message size
    //==========================================================================

    int total_errors = 0;

    for (int size_idx = 0; size_idx < NUM_TEST_SIZES; size_idx++) {
        size_t msg_size = TEST_SIZES[size_idx];
        size_t stream_stride = MAX_SIZE;

        double iteration_times[NUM_ITERATIONS];
        int successful_iters = 0;

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Clear recv buffer on rank 1
            if (mpi_rank == 1) {
                cuda_check(cudaMemset(d_recv_buf, 0, MAX_SIZE * N_STREAMS), "clear recv");
            }

            // Initialize send buffers with pattern on rank 0
            if (mpi_rank == 0) {
                for (int i = 0; i < N_STREAMS; i++) {
                    uint8_t pattern = (iter + 0xA0 + i) & 0xFF;
                    cuda_check(cudaMemset(d_send_bufs[i], pattern, msg_size), "set send pattern");
                }
            }
            cuda_check(cudaDeviceSynchronize(), "sync init");

            MPI_Barrier(MPI_COMM_WORLD);

            if (mpi_rank == 0) {
                // Launch kernel - measures WQE build + doorbell time
                auto t_start = high_resolution_clock::now();

                gpu_concurrent_write_kernel<<<1, N_STREAMS>>>(
                    d_state,
                    d_send_addrs,
                    d_send_lkeys,
                    peer_buf.addr,
                    peer_buf.rkey,
                    msg_size,
                    stream_stride,
                    N_STREAMS,
                    d_start_clock,
                    d_end_clock
                );

                cuda_check(cudaDeviceSynchronize(), "kernel sync");

                auto t_end = high_resolution_clock::now();

                // Get GPU-measured time
                uint64_t start_cycles, end_cycles;
                cuda_check(cudaMemcpy(&start_cycles, d_start_clock, sizeof(uint64_t),
                                      cudaMemcpyDeviceToHost), "copy start");
                cuda_check(cudaMemcpy(&end_cycles, d_end_clock, sizeof(uint64_t),
                                      cudaMemcpyDeviceToHost), "copy end");

                double gpu_time_us = cycles_to_us(end_cycles - start_cycles, clock_rate_khz);
                double host_time_us = duration_cast<nanoseconds>(t_end - t_start).count() / 1000.0;

                // Use host time (includes kernel launch + sync)
                iteration_times[successful_iters++] = host_time_us;
            }

            // Wait for all transfers to complete
            MPI_Barrier(MPI_COMM_WORLD);

            // Small delay to ensure NIC has processed all WQEs
            usleep(1000);

            MPI_Barrier(MPI_COMM_WORLD);

            // Verify data on rank 1
            if (mpi_rank == 1) {
                // Copy each stream's data separately (they're at MAX_SIZE stride)
                int errors = 0;
                for (int i = 0; i < N_STREAMS && errors < 5; i++) {
                    uint8_t expected = (iter + 0xA0 + i) & 0xFF;
                    uint8_t* stream_recv = (uint8_t*)d_recv_buf + i * MAX_SIZE;
                    uint8_t* verify_ptr = h_verify_buf;

                    cuda_check(cudaMemcpy(verify_ptr, stream_recv, msg_size,
                                          cudaMemcpyDeviceToHost), "verify copy");

                    for (size_t j = 0; j < msg_size && errors < 5; j++) {
                        if (verify_ptr[j] != expected) {
                            errors++;
                            if (iter == 0 && size_idx == 0) {
                                fprintf(stderr, "Rank 1: Verify error stream %d byte %zu: "
                                        "got 0x%02x expected 0x%02x\n",
                                        i, j, verify_ptr[j], expected);
                            }
                        }
                    }
                }
                total_errors += errors;
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
        printf("\nThis benchmark measures %d concurrent GPU-triggered RDMA\n", N_STREAMS);
        printf("writes, similar to minimal/benchmark_runner.hpp for CXI.\n");
        printf("\nKey metrics:\n");
        printf("  - Total(us): Time for all %d transfers (kernel + doorbell)\n", N_STREAMS);
        printf("  - Per-xfer(us): Amortized time per transfer\n");
        printf("=======================================================\n");
        fflush(stdout);
    }

    if (mpi_rank == 1 && total_errors > 0) {
        fprintf(stderr, "Rank 1: Total verification errors: %d\n", total_errors);
    }

    // Cleanup
    free(h_verify_buf);
    cudaFree(d_state);
    cudaFree(d_start_clock);
    cudaFree(d_end_clock);
    cudaFree(d_send_addrs);
    cudaFree(d_send_lkeys);

    for (int i = 0; i < N_STREAMS; i++) {
        delete send_mrs[i];
        cudaFree(d_send_bufs[i]);
    }
    delete recv_mr;
    cudaFree(d_recv_buf);

    delete devx_qp;
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);

    MPI_Finalize();
    return 0;
}
