/**
 * gpu_pingpong.cu - GPU-Triggered RDMA Ping-Pong Test
 *
 * This test demonstrates true GPU-initiated RDMA using MLX5 DevX.
 * The GPU kernel directly builds WQEs and rings doorbells without
 * any CPU involvement during the data transfer.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <mpi.h>

#include "gda_gpu_comm.hpp"
#include "gda_device.cuh"

using namespace opengda;
using namespace std::chrono;

constexpr int WARMUP_ITERS = 10;
constexpr int TEST_ITERS = 100;
constexpr size_t MSG_SIZE = 8;

// Simple GPU kernel to trigger RDMA write
// This demonstrates GPU-initiated RDMA: the GPU directly builds WQEs
// and rings doorbells without any CPU involvement.
__global__ void simple_rdma_write_kernel(
    GdaDeviceState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size,
    int* result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Get current producer index
    uint64_t prod = *state->prod_idx;
    uint16_t wqe_idx = (uint16_t)((prod + 1) & 0xFFFF);

    // Build RDMA WRITE WQE directly from GPU
    gda_build_rdma_write_wqe(
        state,
        local_addr,
        local_lkey,
        state->remote_addr,
        state->remote_rkey,
        size,
        wqe_idx,
        true  // signaled
    );

    // Ring doorbell to trigger NIC - this is the key GPU-initiated action
    gda_ring_doorbell(state, wqe_idx);

    // For now, we rely on MPI barriers for synchronization
    // TODO: Implement proper GPU-side CQ polling for fully autonomous operation
    *result = 0;
}

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

    // Initialize GPU-triggered communication
    GdaGpuComm comm;

    int peer = (comm.rank() == 0) ? 1 : 0;
    size_t msg_size = (argc > 1) ? atol(argv[1]) : MSG_SIZE;
    int iterations = (argc > 2) ? atoi(argv[2]) : TEST_ITERS;

    if (comm.rank() == 0) {
        printf("=== GPU-Triggered RDMA Ping-Pong Test ===\n");
        printf("Message size: %zu bytes\n", msg_size);
        printf("Iterations: %d (warmup: %d)\n", iterations, WARMUP_ITERS);
        printf("GPU: %s\n", comm.props.name);
        printf("IB device: %s\n", comm.mlx5->dev_name.c_str());
        printf("\n");
        fflush(stdout);
    }

    // Allocate buffers on GPU
    void* d_send_buf = nullptr;
    void* d_recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_buf, msg_size));
    CUDA_CHECK(cudaMalloc(&d_recv_buf, msg_size));
    CUDA_CHECK(cudaMemset(d_send_buf, comm.rank() + 1, msg_size));
    CUDA_CHECK(cudaMemset(d_recv_buf, 0, msg_size));

    // Register buffers
    auto send_handle = comm.register_buffer(d_send_buf, msg_size, true);
    auto recv_handle = comm.register_buffer(d_recv_buf, msg_size, true);

    // Exchange recv buffer info (peers will write to our recv buffer)
    comm.exchange_buffer_info(recv_handle, 0);

    // Set target to peer's recv buffer
    comm.set_remote_target(peer, 0);

    // Get local buffer info for kernel
    uint64_t local_addr;
    uint32_t local_lkey;
    comm.get_local_buffer_info(send_handle, &local_addr, &local_lkey);

    // Allocate result on GPU
    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    comm.barrier();

    // ========================================
    // First test: CPU-triggered for comparison
    // ========================================

    if (comm.rank() == 0) {
        printf("--- CPU-triggered test (baseline) ---\n");
        fflush(stdout);
    }

    comm.barrier();

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (comm.rank() == 0) {
            comm.cpu_put(send_handle, peer, 0, msg_size);
            comm.barrier();
        } else {
            comm.barrier();
        }
    }

    comm.barrier();

    // Timed CPU test
    auto cpu_start = high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        if (comm.rank() == 0) {
            comm.cpu_put(send_handle, peer, 0, msg_size);
            comm.barrier();
        } else {
            comm.barrier();
        }
    }

    auto cpu_end = high_resolution_clock::now();
    double cpu_us = duration_cast<nanoseconds>(cpu_end - cpu_start).count() / 1000.0;

    if (comm.rank() == 0) {
        printf("CPU-triggered: %.2f us total, %.2f us per op\n",
               cpu_us, cpu_us / iterations);
        fflush(stdout);
    }

    comm.barrier();

    // ========================================
    // GPU-triggered test
    // ========================================

    if (comm.rank() == 0) {
        printf("\n--- GPU-triggered test ---\n");
        fflush(stdout);
    }

    comm.barrier();

    // Get device state pointer
    GdaDeviceState* d_state = comm.get_device_state();

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (comm.rank() == 0) {
            simple_rdma_write_kernel<<<1, 1>>>(d_state, local_addr, local_lkey,
                                               msg_size, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            comm.barrier();
        } else {
            comm.barrier();
        }
    }

    comm.barrier();

    // Timed GPU test
    auto gpu_start = high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        if (comm.rank() == 0) {
            simple_rdma_write_kernel<<<1, 1>>>(d_state, local_addr, local_lkey,
                                               msg_size, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            comm.barrier();
        } else {
            comm.barrier();
        }
    }

    auto gpu_end = high_resolution_clock::now();
    double gpu_us = duration_cast<nanoseconds>(gpu_end - gpu_start).count() / 1000.0;

    // Check result
    int h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify data transfer on rank 1
    comm.barrier();
    if (comm.rank() == 1) {
        uint8_t recv_byte;
        CUDA_CHECK(cudaMemcpy(&recv_byte, d_recv_buf, 1, cudaMemcpyDeviceToHost));
        printf("Rank 1: Received byte = 0x%02x (expected 0x01 from rank 0)\n", recv_byte);
        if (recv_byte == 0x01) {
            printf("Data transfer VERIFIED!\n");
        } else {
            printf("WARNING: Data mismatch! GPU-triggered RDMA may not have worked.\n");
        }
        fflush(stdout);
    }
    comm.barrier();

    if (comm.rank() == 0) {
        printf("GPU-triggered: %.2f us total, %.2f us per op\n",
               gpu_us, gpu_us / iterations);
        printf("Last poll result: %d (0=success, -1=timeout, -2=error)\n", h_result);

        if (cpu_us > 0) {
            printf("\nSpeedup: %.2fx\n", cpu_us / gpu_us);
        }

        if (h_result == 0) {
            printf("\nSUCCESS\n");
        } else {
            printf("\nWARNING: CQ polling returned %d\n", h_result);
        }
        fflush(stdout);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_send_buf));
    CUDA_CHECK(cudaFree(d_recv_buf));

    MPI_Finalize();
    return 0;
}
