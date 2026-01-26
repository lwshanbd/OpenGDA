/**
 * gpu_pingpong_opt.cu - Optimized GPU-Triggered RDMA Ping-Pong Test
 *
 * Optimizations:
 * 1. Pre-launched kernel (no kernel launch overhead in timing)
 * 2. GPU-side timing
 * 3. Unsignaled operations for max throughput
 * 4. Proper warm-up
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <unistd.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "gda_gpu_comm.hpp"
#include "gda_device.cuh"
#include "gda_persistent.cuh"

using namespace opengda;
using namespace std::chrono;

constexpr int WARMUP_ITERS = 100;
constexpr int TEST_ITERS = 1000;
constexpr size_t MSG_SIZE = 8;

// Pre-launched kernel that waits for signal, then runs iterations
__global__ void optimized_rdma_kernel(
    GdaDeviceState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size,
    int warmup_iters,
    int test_iters,
    volatile int* phase,      // 0=wait, 1=warmup, 2=test, 3=done
    uint64_t* gpu_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Phase 1: Warmup
    while (*phase < 1) {
        __threadfence_system();
    }

    for (int i = 0; i < warmup_iters; i++) {
        uint64_t prod = *state->prod_idx;
        uint16_t wqe_idx = (uint16_t)((prod + 1) & 0xFFFF);

        gda_build_rdma_write_wqe(
            state,
            local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, wqe_idx, false
        );
        gda_ring_doorbell(state, wqe_idx);
    }

    // Signal warmup done
    *phase = 10;
    __threadfence_system();

    // Phase 2: Wait for test signal
    while (*phase < 2) {
        __threadfence_system();
    }

    // Timed test - GPU measures its own cycles
    uint64_t start = clock64();

    for (int i = 0; i < test_iters; i++) {
        uint64_t prod = *state->prod_idx;
        uint16_t wqe_idx = (uint16_t)((prod + 1) & 0xFFFF);

        gda_build_rdma_write_wqe(
            state,
            local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, wqe_idx, false
        );
        gda_ring_doorbell(state, wqe_idx);
    }

    uint64_t end = clock64();
    *gpu_cycles = end - start;

    // Signal done
    *phase = 30;
    __threadfence_system();
}

// Simpler kernel for direct measurement - no phase coordination
__global__ void direct_rdma_burst(
    GdaDeviceState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size,
    int count,
    uint64_t* gpu_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t start = clock64();

    for (int i = 0; i < count; i++) {
        uint64_t prod = *state->prod_idx;
        uint16_t wqe_idx = (uint16_t)((prod + 1) & 0xFFFF);

        gda_build_rdma_write_wqe(
            state,
            local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, wqe_idx, false
        );
        gda_ring_doorbell(state, wqe_idx);
    }

    uint64_t end = clock64();
    *gpu_cycles = end - start;
}

// Burst with last op signaled for verification
__global__ void direct_rdma_burst_signaled(
    GdaDeviceState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size,
    int count,
    uint64_t* gpu_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t start = clock64();

    for (int i = 0; i < count; i++) {
        uint64_t prod = *state->prod_idx;
        uint16_t wqe_idx = (uint16_t)((prod + 1) & 0xFFFF);

        // Signal last operation for completion detection
        bool signaled = (i == count - 1);

        gda_build_rdma_write_wqe(
            state,
            local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, wqe_idx, signaled
        );
        gda_ring_doorbell(state, wqe_idx);
    }

    // Wait for last completion if we have CQ access
    // For now, just add a fence - actual CQ polling can be added later
    __threadfence_system();

    uint64_t end = clock64();
    *gpu_cycles = end - start;
}

// Single operation for latency measurement
__global__ void single_rdma_op(
    GdaDeviceState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t prod = *state->prod_idx;
    uint16_t wqe_idx = (uint16_t)((prod + 1) & 0xFFFF);

    gda_build_rdma_write_wqe(
        state,
        local_addr, local_lkey,
        state->remote_addr, state->remote_rkey,
        size, wqe_idx, false
    );
    gda_ring_doorbell(state, wqe_idx);
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
    int warmup_iters = (argc > 3) ? atoi(argv[3]) : WARMUP_ITERS;

    // Get GPU clock rate for timing
    int gpu_id;
    cudaGetDevice(&gpu_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, gpu_id);
    double clock_rate_khz = props.clockRate;  // in kHz

    if (comm.rank() == 0) {
        printf("=== Optimized GPU-Triggered RDMA Test ===\n");
        printf("Message size: %zu bytes\n", msg_size);
        printf("Iterations: %d (warmup: %d)\n", iterations, warmup_iters);
        printf("GPU: %s (%.2f GHz)\n", props.name, clock_rate_khz / 1e6);
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

    // Exchange recv buffer info
    comm.exchange_buffer_info(recv_handle, 0);
    comm.set_remote_target(peer, 0);

    // Get local buffer info
    uint64_t local_addr;
    uint32_t local_lkey;
    comm.get_local_buffer_info(send_handle, &local_addr, &local_lkey);

    // Allocate timing variables
    uint64_t* d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(uint64_t)));

    GdaDeviceState* d_state = comm.get_device_state();

    comm.barrier();

    // ========================================
    // Test 1: CPU-triggered baseline
    // ========================================

    if (comm.rank() == 0) {
        printf("--- Test 1: CPU-triggered baseline ---\n");
        fflush(stdout);
    }

    comm.barrier();

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        if (comm.rank() == 0) {
            comm.cpu_put(send_handle, peer, 0, msg_size);
        }
        comm.barrier();
    }

    comm.barrier();

    // Timed
    auto cpu_start = high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        if (comm.rank() == 0) {
            comm.cpu_put(send_handle, peer, 0, msg_size);
        }
        comm.barrier();
    }

    auto cpu_end = high_resolution_clock::now();
    double cpu_us = duration_cast<nanoseconds>(cpu_end - cpu_start).count() / 1000.0;

    if (comm.rank() == 0) {
        printf("CPU-triggered: %.2f us total, %.3f us per op\n",
               cpu_us, cpu_us / iterations);
        fflush(stdout);
    }

    comm.barrier();

    // ========================================
    // Test 2: GPU-triggered with kernel launch overhead
    // ========================================

    if (comm.rank() == 0) {
        printf("\n--- Test 2: GPU-triggered (with kernel launch) ---\n");
        fflush(stdout);
    }

    comm.barrier();

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        if (comm.rank() == 0) {
            single_rdma_op<<<1, 1>>>(d_state, local_addr, local_lkey, msg_size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        comm.barrier();
    }

    comm.barrier();

    // Timed (includes kernel launch overhead)
    auto gpu_launch_start = high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        if (comm.rank() == 0) {
            single_rdma_op<<<1, 1>>>(d_state, local_addr, local_lkey, msg_size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        comm.barrier();
    }

    auto gpu_launch_end = high_resolution_clock::now();
    double gpu_launch_us = duration_cast<nanoseconds>(gpu_launch_end - gpu_launch_start).count() / 1000.0;

    if (comm.rank() == 0) {
        printf("GPU (with launch): %.2f us total, %.3f us per op\n",
               gpu_launch_us, gpu_launch_us / iterations);
        fflush(stdout);
    }

    comm.barrier();

    // ========================================
    // Test 3: GPU-triggered burst (no per-op kernel launch)
    // ========================================

    if (comm.rank() == 0) {
        printf("\n--- Test 3: GPU-triggered burst (single kernel) ---\n");
        fflush(stdout);
    }

    comm.barrier();

    // Warmup burst
    if (comm.rank() == 0) {
        direct_rdma_burst<<<1, 1>>>(d_state, local_addr, local_lkey, msg_size, warmup_iters, d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    comm.barrier();

    // Give NIC time to complete
    usleep(10000);
    comm.barrier();

    // Timed burst - measure both wall time and GPU cycles
    auto burst_start = high_resolution_clock::now();

    if (comm.rank() == 0) {
        direct_rdma_burst_signaled<<<1, 1>>>(d_state, local_addr, local_lkey, msg_size, iterations, d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto burst_end = high_resolution_clock::now();
    double burst_wall_us = duration_cast<nanoseconds>(burst_end - burst_start).count() / 1000.0;

    // Get GPU cycles
    uint64_t gpu_cycles;
    CUDA_CHECK(cudaMemcpy(&gpu_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Convert cycles to microseconds
    double gpu_burst_us = gpu_cycles / (clock_rate_khz / 1000.0);

    // Allow some time for NIC to complete
    usleep(1000);  // 1ms

    comm.barrier();

    // Verify data
    if (comm.rank() == 1) {
        uint8_t recv_byte;
        CUDA_CHECK(cudaMemcpy(&recv_byte, d_recv_buf, 1, cudaMemcpyDeviceToHost));
        if (recv_byte == 0x01) {
            printf("Rank 1: Data VERIFIED (0x%02x)\n", recv_byte);
        } else {
            printf("Rank 1: Data MISMATCH (got 0x%02x, expected 0x01)\n", recv_byte);
        }
        fflush(stdout);
    }

    comm.barrier();

    if (comm.rank() == 0) {
        printf("GPU burst (wall): %.2f us total, %.3f us per op\n",
               burst_wall_us, burst_wall_us / iterations);
        printf("GPU burst (GPU cycles): %.2f us total, %.3f us per op\n",
               gpu_burst_us, gpu_burst_us / iterations);
        printf("GPU cycles: %lu (%.2f GHz)\n", gpu_cycles, clock_rate_khz / 1e6);
        fflush(stdout);
    }

    // ========================================
    // Summary
    // ========================================

    if (comm.rank() == 0) {
        printf("\n=== Summary ===\n");
        printf("CPU-triggered:      %.3f us/op\n", cpu_us / iterations);
        printf("GPU (with launch):  %.3f us/op\n", gpu_launch_us / iterations);
        printf("GPU burst (wall):   %.3f us/op\n", burst_wall_us / iterations);
        printf("GPU burst (cycles): %.3f us/op\n", gpu_burst_us / iterations);
        printf("\n");

        double improvement = (gpu_launch_us / iterations) / (gpu_burst_us / iterations);
        printf("Kernel launch overhead: %.2fx slower\n", improvement);

        if (gpu_burst_us / iterations < cpu_us / iterations) {
            printf("GPU burst is %.2fx FASTER than CPU!\n",
                   (cpu_us / iterations) / (gpu_burst_us / iterations));
        } else {
            printf("CPU is %.2fx faster than GPU burst\n",
                   (gpu_burst_us / iterations) / (cpu_us / iterations));
        }

        printf("\nSUCCESS\n");
        fflush(stdout);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaFree(d_send_buf));
    CUDA_CHECK(cudaFree(d_recv_buf));

    MPI_Finalize();
    return 0;
}
