/**
 * pingpong.cu - RDMA ping-pong latency test
 *
 * Measures round-trip latency between two nodes using RDMA write.
 * Both CPU-triggered and GPU-timed versions.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <mpi.h>

#include "gda_comm.hpp"

using namespace opengda;
using namespace std::chrono;

constexpr int WARMUP_ITERS = 100;
constexpr int TEST_ITERS = 1000;
constexpr size_t MSG_SIZE = 8;  // Small message for latency test

// GPU kernel for ping-pong with GPU timing
__global__ void pingpong_kernel(
    volatile uint64_t* local_flag,
    volatile uint64_t* trigger_addr,
    uint64_t my_rank,
    int iterations,
    uint64_t* d_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t t_start = clock64();

    for (int i = 0; i < iterations; i++) {
        if (my_rank == 0) {
            // Initiator: signal ready and wait for response
            // Note: In this simplified version, we poll local flag
            // Real implementation would trigger RDMA and poll completion
            while (*local_flag != (uint64_t)(i + 1)) {
                // Busy wait
            }
        } else {
            // Responder: wait for ping, then respond
            while (*local_flag != (uint64_t)(i + 1)) {
                // Busy wait
            }
        }
    }

    uint64_t t_end = clock64();
    *d_cycles = t_end - t_start;
}

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
    size_t msg_size = (argc > 1) ? atol(argv[1]) : MSG_SIZE;
    int iterations = (argc > 2) ? atoi(argv[2]) : TEST_ITERS;

    if (rank == 0) {
        printf("=== RDMA Ping-Pong Latency Test ===\n");
        printf("Message size: %zu bytes\n", msg_size);
        printf("Iterations: %d (warmup: %d)\n", iterations, WARMUP_ITERS);
        printf("GPU: %s\n", comm.cuda->props.name);
        printf("IB device: %s\n", comm.ibv->dev_name.c_str());
        fflush(stdout);
    }

    // Allocate buffers on GPU
    void* d_send_buf = nullptr;
    void* d_recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_buf, msg_size));
    CUDA_CHECK(cudaMalloc(&d_recv_buf, msg_size));
    CUDA_CHECK(cudaMemset(d_send_buf, rank + 1, msg_size));
    CUDA_CHECK(cudaMemset(d_recv_buf, 0, msg_size));

    // Register buffers
    auto send_handle = comm.register_buffer(d_send_buf, msg_size, true);
    auto recv_handle = comm.register_buffer(d_recv_buf, msg_size, true);

    // Exchange buffer info
    comm.exchange_buffer_info(recv_handle, 0);  // Exchange recv buffer info

    comm.barrier();

    // ========================================
    // CPU-triggered ping-pong test
    // ========================================

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (rank == 0) {
            // Send ping
            comm.put(send_handle, peer, 0, msg_size, true);
            comm.wait(1);
            comm.barrier();  // Wait for pong
        } else {
            comm.barrier();  // Wait for ping
            // Send pong
            comm.put(send_handle, peer, 0, msg_size, true);
            comm.wait(1);
        }
    }

    comm.barrier();

    // Timed runs
    auto t_start = high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        if (rank == 0) {
            comm.put(send_handle, peer, 0, msg_size, true);
            comm.wait(1);
            comm.barrier();
        } else {
            comm.barrier();
            comm.put(send_handle, peer, 0, msg_size, true);
            comm.wait(1);
        }
    }

    auto t_end = high_resolution_clock::now();
    double elapsed_us = duration_cast<nanoseconds>(t_end - t_start).count() / 1000.0;

    comm.barrier();

    if (rank == 0) {
        double avg_rtt = elapsed_us / iterations;
        double avg_oneway = avg_rtt / 2.0;
        double msg_rate = (2.0 * iterations) / (elapsed_us / 1e6);
        double bw_gbps = (msg_size * 8.0 * 2.0 * iterations) / (elapsed_us / 1e6) / 1e9;

        printf("\n=== Results (CPU-triggered) ===\n");
        printf("Total time: %.2f us\n", elapsed_us);
        printf("Round-trip latency: %.2f us\n", avg_rtt);
        printf("One-way latency: %.2f us\n", avg_oneway);
        printf("Message rate: %.0f msg/s\n", msg_rate);
        printf("Bandwidth: %.3f Gbps\n", bw_gbps);
        printf("\n");
        fflush(stdout);
    }

    // ========================================
    // Bandwidth test (streaming writes)
    // ========================================

    // Drain any pending completions before BW test
    int drained = comm.poll(100);
    if (drained > 0) {
        printf("Rank %d: Drained %d pending completions before BW test\n", rank, drained);
        fflush(stdout);
    }

    // Reset counters for BW test
    comm.reset_counters();

    comm.barrier();

    // Larger buffer for bandwidth test
    size_t bw_size = 1024 * 1024;  // 1MB
    void* d_bw_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bw_buf, bw_size));
    auto bw_handle = comm.register_buffer(d_bw_buf, bw_size, true);

    // Debug: print buffer info before exchange
    printf("Rank %d: BW buffer local addr=%p, lkey=%u, rkey=%u\n",
           rank, d_bw_buf, bw_handle.mr->lkey, bw_handle.mr->rkey);
    fflush(stdout);

    comm.exchange_buffer_info(bw_handle, 1);

    // Debug: print remote buffer info after exchange
    uint64_t peer_key = ((uint64_t)peer << 32) | 1;
    auto it = comm.remote_info.find(peer_key);
    if (it != comm.remote_info.end()) {
        printf("Rank %d: Peer %d BW buffer remote addr=%lx, rkey=%u\n",
               rank, peer, it->second.addr, it->second.rkey);
    } else {
        printf("Rank %d: ERROR - No remote info for peer %d buf 1\n", rank, peer);
    }
    fflush(stdout);

    comm.barrier();

    int bw_iters = 100;

    // Warmup
    if (rank == 0) {
        for (int i = 0; i < 10; i++) {
            comm.put(bw_handle, peer, 1, bw_size, true);
            comm.wait(1);
        }
    }

    comm.barrier();

    // Timed bandwidth test
    if (rank == 0) {
        auto bw_start = high_resolution_clock::now();

        for (int i = 0; i < bw_iters; i++) {
            comm.put(bw_handle, peer, 1, bw_size, (i == bw_iters - 1));
            if ((i + 1) % 16 == 0 || i == bw_iters - 1) {
                comm.wait(1);  // Wait periodically to avoid overflow
            }
        }

        auto bw_end = high_resolution_clock::now();
        double bw_us = duration_cast<nanoseconds>(bw_end - bw_start).count() / 1000.0;

        double total_bytes = (double)bw_size * bw_iters;
        double bw_GBps = total_bytes / (bw_us / 1e6) / 1e9;
        double bw_Gbps = bw_GBps * 8;

        printf("=== Bandwidth Test ===\n");
        printf("Buffer size: %zu bytes\n", bw_size);
        printf("Iterations: %d\n", bw_iters);
        printf("Total data: %.2f MB\n", total_bytes / 1e6);
        printf("Time: %.2f us\n", bw_us);
        printf("Bandwidth: %.2f GB/s (%.2f Gbps)\n", bw_GBps, bw_Gbps);
        printf("\n");
        fflush(stdout);
    }

    comm.barrier();

    // Cleanup
    CUDA_CHECK(cudaFree(d_send_buf));
    CUDA_CHECK(cudaFree(d_recv_buf));
    CUDA_CHECK(cudaFree(d_bw_buf));

    if (rank == 0) {
        printf("SUCCESS\n");
    }

    MPI_Finalize();
    return 0;
}
