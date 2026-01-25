/**
 * gpu_benchmark_opt.cu - Optimized GPU-Triggered RDMA Benchmark
 *
 * Compares performance between:
 *   1. Original implementation (gda_device.cuh)
 *   2. Optimized implementation (gda_device_opt.cuh)
 *
 * Measures:
 *   - Single operation latency
 *   - Burst throughput
 *   - Batched operations
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
#include "gda_device_opt.cuh"

using namespace opengda;
using namespace std::chrono;

// Test configurations
constexpr int WARMUP_ITERS = 100;
constexpr int TEST_ITERS = 1000;
constexpr int BURST_SIZES[] = {1, 8, 32, 128, 512};
constexpr size_t MSG_SIZES[] = {8, 64, 512, 4096, 65536};

//==============================================================================
// Original implementation kernels (for comparison)
//==============================================================================

__global__ void original_burst_kernel(
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
            state, local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, wqe_idx, (i == count - 1)
        );
        gda_ring_doorbell(state, wqe_idx);
    }

    uint64_t end = clock64();
    *gpu_cycles = end - start;
}

//==============================================================================
// Optimized implementation kernels
//==============================================================================

// Single op latency test
__global__ void opt_single_op_kernel(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size,
    int iterations,
    uint64_t* gpu_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t start = clock64();

    for (int i = 0; i < iterations; i++) {
        gda_rdma_write_opt(
            state, local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, false
        );
    }

    uint64_t end = clock64();
    *gpu_cycles = end - start;
}

// Batched burst test (multiple WQEs per doorbell)
__global__ void opt_batched_burst_kernel(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size,
    int count,
    int batch_size,
    uint64_t* gpu_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t start = clock64();

    uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
    int batch_count = 0;

    for (int i = 0; i < count; i++) {
        uint16_t wqe_idx = (uint16_t)((prod + 1 + i) & 0xFFFF);
        bool signaled = (i == count - 1);

        // Build WQE
        gda_build_rdma_write_wqe_opt(
            state, local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, wqe_idx, signaled
        );

        batch_count++;

        // Ring doorbell on batch boundary or last op
        if (batch_count >= batch_size || i == count - 1) {
            gda_ring_doorbell_bf(state, wqe_idx);
            batch_count = 0;
        }
    }

    // Final producer index update
    gda_store_relaxed_u64(state->prod_idx, prod + count);

    uint64_t end = clock64();
    *gpu_cycles = end - start;
}

// Direct comparison: optimized per-op with BlueFlame
__global__ void opt_per_op_bf_kernel(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size,
    int count,
    uint64_t* gpu_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t start = clock64();

    for (int i = 0; i < count; i++) {
        uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
        uint16_t wqe_idx = (uint16_t)((prod + 1) & 0xFFFF);

        gda_build_rdma_write_wqe_opt(
            state, local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, wqe_idx, (i == count - 1)
        );

        gda_ring_doorbell_bf(state, wqe_idx);
    }

    uint64_t end = clock64();
    *gpu_cycles = end - start;
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

//==============================================================================
// Main benchmark
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

    // Initialize communication
    GdaGpuComm comm;
    int peer = (comm.rank() == 0) ? 1 : 0;

    // Get GPU info
    int gpu_id;
    cudaGetDevice(&gpu_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, gpu_id);
    double clock_rate_khz = props.clockRate;

    if (comm.rank() == 0) {
        printf("=======================================================\n");
        printf("     GPU-Triggered RDMA Optimization Benchmark\n");
        printf("=======================================================\n");
        printf("GPU: %s (%.2f GHz)\n", props.name, clock_rate_khz / 1e6);
        printf("IB Device: %s\n", comm.mlx5->dev_name.c_str());
        printf("Warmup: %d, Test: %d iterations\n", WARMUP_ITERS, TEST_ITERS);
        printf("-------------------------------------------------------\n\n");
        fflush(stdout);
    }

    // Allocate test buffers
    size_t max_size = 65536;
    void* d_send_buf = nullptr;
    void* d_recv_buf = nullptr;
    cuda_check(cudaMalloc(&d_send_buf, max_size), "alloc send");
    cuda_check(cudaMalloc(&d_recv_buf, max_size), "alloc recv");
    cuda_check(cudaMemset(d_send_buf, comm.rank() + 1, max_size), "memset send");
    cuda_check(cudaMemset(d_recv_buf, 0, max_size), "memset recv");

    // Register buffers
    auto send_handle = comm.register_buffer(d_send_buf, max_size, true);
    auto recv_handle = comm.register_buffer(d_recv_buf, max_size, true);

    // Exchange buffer info
    comm.exchange_buffer_info(recv_handle, 0);
    comm.set_remote_target(peer, 0);

    // Get local buffer info
    uint64_t local_addr;
    uint32_t local_lkey;
    comm.get_local_buffer_info(send_handle, &local_addr, &local_lkey);

    // Allocate timing variable
    uint64_t* d_cycles;
    cuda_check(cudaMalloc(&d_cycles, sizeof(uint64_t)), "alloc cycles");

    // Get device states
    GdaDeviceState* d_state_orig = comm.get_device_state();

    // Create optimized state from the mlx5 context
    // This adapts the existing QP structures for optimized operations
    GdaDeviceStateOpt h_state_opt;
    h_state_opt.qpn = comm.mlx5->qp->qp_num;
    h_state_opt.nwqes = comm.mlx5->qp_depth;
    h_state_opt.nwqes_mask = comm.mlx5->qp_depth - 1;
    h_state_opt.wqe_buf = comm.mlx5->d_wqe_buf;
    h_state_opt.wqe_lkey = 0;
    // Use GPU-mapped pointers for doorbell and BlueFlame
    h_state_opt.dbrec = comm.mlx5->d_dbrec;
    h_state_opt.bf_reg = comm.mlx5->d_bf_reg;
    h_state_opt.resv_head = nullptr;  // Not used in simple tests
    h_state_opt.ready_head = nullptr;
    h_state_opt.prod_idx = comm.mlx5->d_prod_idx;
    h_state_opt.cqe = (volatile GdaCqe64Opt*)comm.mlx5->d_cqe;
    h_state_opt.ncqes = comm.mlx5->cq_depth;
    h_state_opt.ncqes_mask = comm.mlx5->cq_depth - 1;
    h_state_opt.cq_cons_idx = nullptr;
    h_state_opt.cq_dbrec = nullptr;  // TODO: map CQ dbrec
    h_state_opt.remote_addr = comm.device_state.remote_addr;
    h_state_opt.remote_rkey = comm.device_state.remote_rkey;
    h_state_opt.num_completions = nullptr;
    h_state_opt.batch_size = 32;
    h_state_opt.batch_mask = 31;

    if (comm.rank() == 0) {
        printf("Optimized state:\n");
        printf("  QPN: %u, nwqes: %u\n", h_state_opt.qpn, h_state_opt.nwqes);
        printf("  wqe_buf: %p\n", h_state_opt.wqe_buf);
        printf("  dbrec: %p\n", (void*)h_state_opt.dbrec);
        printf("  bf_reg: %p\n", (void*)h_state_opt.bf_reg);
        printf("  prod_idx: %p\n", (void*)h_state_opt.prod_idx);
        printf("  remote_addr: 0x%lx, rkey: 0x%x\n",
               h_state_opt.remote_addr, h_state_opt.remote_rkey);
        fflush(stdout);
    }

    GdaDeviceStateOpt* d_state_opt;
    cuda_check(cudaMalloc(&d_state_opt, sizeof(GdaDeviceStateOpt)), "alloc opt state");
    cuda_check(cudaMemcpy(d_state_opt, &h_state_opt, sizeof(GdaDeviceStateOpt),
                          cudaMemcpyHostToDevice), "copy opt state");

    comm.barrier();

    //==========================================================================
    // Test 0: Data correctness verification
    //==========================================================================

    if (comm.rank() == 0) {
        printf("=== Test 0: Data Correctness Verification ===\n");
        fflush(stdout);
    }

    // Initialize send buffer with known pattern
    uint8_t send_pattern = 0xAB;
    cuda_check(cudaMemset(d_send_buf, send_pattern, max_size), "memset send pattern");

    // Clear receive buffer on rank 1
    if (comm.rank() == 1) {
        cuda_check(cudaMemset(d_recv_buf, 0, max_size), "clear recv buf");
    }

    comm.barrier();

    // Test 0a: CPU-triggered RDMA (baseline - should work)
    if (comm.rank() == 0) {
        printf("  0a. CPU-triggered RDMA: ");
        fflush(stdout);
        comm.cpu_put(send_handle, peer, 0, 64);
    }

    comm.barrier();
    usleep(10000);
    comm.barrier();

    if (comm.rank() == 1) {
        uint8_t recv_data[64];
        cuda_check(cudaMemcpy(recv_data, d_recv_buf, 64, cudaMemcpyDeviceToHost), "copy recv");
        bool correct = true;
        for (int i = 0; i < 64; i++) {
            if (recv_data[i] != send_pattern) {
                correct = false;
                break;
            }
        }
        printf("%s\n", correct ? "PASSED" : "FAILED");
        fflush(stdout);
    }

    comm.barrier();

    // Clear recv buffer again for GPU test
    if (comm.rank() == 1) {
        cuda_check(cudaMemset(d_recv_buf, 0, max_size), "clear recv buf");
    }

    comm.barrier();

    // Test 0b: Original GPU-triggered RDMA
    if (comm.rank() == 0) {
        printf("  0b. Original GPU-triggered: ");
        fflush(stdout);
        original_burst_kernel<<<1, 1>>>(d_state_orig, local_addr, local_lkey, 64, 1, d_cycles);
        cudaDeviceSynchronize();
    }

    comm.barrier();
    usleep(10000);
    comm.barrier();

    if (comm.rank() == 1) {
        uint8_t recv_data[64];
        cuda_check(cudaMemcpy(recv_data, d_recv_buf, 64, cudaMemcpyDeviceToHost), "copy recv");
        bool correct = true;
        for (int i = 0; i < 64; i++) {
            if (recv_data[i] != send_pattern) {
                correct = false;
                break;
            }
        }
        printf("%s\n", correct ? "PASSED" : "FAILED");
        if (!correct) {
            printf("     First 8 bytes: ");
            for (int i = 0; i < 8; i++) printf("%02x ", recv_data[i]);
            printf("\n");
        }
        fflush(stdout);
    }

    comm.barrier();

    // Clear recv buffer again for optimized GPU test
    if (comm.rank() == 1) {
        cuda_check(cudaMemset(d_recv_buf, 0, max_size), "clear recv buf");
    }

    comm.barrier();

    // Test 0c: Optimized GPU-triggered RDMA
    if (comm.rank() == 0) {
        printf("  0c. Optimized GPU-triggered: ");
        fflush(stdout);
        opt_per_op_bf_kernel<<<1, 1>>>(d_state_opt, local_addr, local_lkey, 64, 1, d_cycles);
        cudaDeviceSynchronize();
    }

    comm.barrier();
    usleep(10000);
    comm.barrier();

    bool verified = false;
    if (comm.rank() == 1) {
        uint8_t recv_data[64];
        cuda_check(cudaMemcpy(recv_data, d_recv_buf, 64, cudaMemcpyDeviceToHost), "copy recv");
        bool correct = true;
        for (int i = 0; i < 64; i++) {
            if (recv_data[i] != send_pattern) {
                correct = false;
                break;
            }
        }
        printf("%s\n", correct ? "PASSED" : "FAILED");
        if (!correct) {
            printf("     First 8 bytes: ");
            for (int i = 0; i < 8; i++) printf("%02x ", recv_data[i]);
            printf("\n");
        }
        verified = correct;
        fflush(stdout);
    }

    // Broadcast verification result
    int verify_result = verified ? 1 : 0;
    MPI_Bcast(&verify_result, 1, MPI_INT, 1, MPI_COMM_WORLD);

    if (comm.rank() == 0) {
        printf("\n");
    }

    comm.barrier();

    //==========================================================================
    // Test 1: Per-operation latency comparison
    //==========================================================================

    if (comm.rank() == 0) {
        printf("=== Test 1: Per-Operation Latency (8 bytes) ===\n");
        printf("%-20s %12s %12s %12s\n",
               "Implementation", "Total (us)", "Per-op (us)", "Cycles/op");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
    }

    comm.barrier();

    // Warmup both
    if (comm.rank() == 0) {
        original_burst_kernel<<<1, 1>>>(d_state_orig, local_addr, local_lkey, 8,
                                         WARMUP_ITERS, d_cycles);
        cudaDeviceSynchronize();

        opt_per_op_bf_kernel<<<1, 1>>>(d_state_opt, local_addr, local_lkey, 8,
                                        WARMUP_ITERS, d_cycles);
        cudaDeviceSynchronize();
    }

    comm.barrier();
    usleep(10000);
    comm.barrier();

    // Test original
    uint64_t orig_cycles = 0;
    if (comm.rank() == 0) {
        original_burst_kernel<<<1, 1>>>(d_state_orig, local_addr, local_lkey, 8,
                                         TEST_ITERS, d_cycles);
        cudaDeviceSynchronize();
        cudaMemcpy(&orig_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    comm.barrier();
    usleep(10000);
    comm.barrier();

    // Test optimized
    uint64_t opt_cycles = 0;
    if (comm.rank() == 0) {
        opt_per_op_bf_kernel<<<1, 1>>>(d_state_opt, local_addr, local_lkey, 8,
                                        TEST_ITERS, d_cycles);
        cudaDeviceSynchronize();
        cudaMemcpy(&opt_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    comm.barrier();

    if (comm.rank() == 0) {
        double orig_us = cycles_to_us(orig_cycles, clock_rate_khz);
        double opt_us = cycles_to_us(opt_cycles, clock_rate_khz);

        printf("%-20s %12.2f %12.3f %12lu\n",
               "Original", orig_us, orig_us / TEST_ITERS, orig_cycles / TEST_ITERS);
        printf("%-20s %12.2f %12.3f %12lu\n",
               "Optimized+BF", opt_us, opt_us / TEST_ITERS, opt_cycles / TEST_ITERS);
        printf("\nSpeedup: %.2fx\n", orig_us / opt_us);
        printf("\n");
        fflush(stdout);
    }

    comm.barrier();

    //==========================================================================
    // Test 2: Batching effect
    //==========================================================================

    if (comm.rank() == 0) {
        printf("=== Test 2: Batching Effect (1000 ops, 8 bytes) ===\n");
        printf("%-12s %12s %12s %12s\n",
               "Batch Size", "Total (us)", "Per-op (us)", "Speedup");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
    }

    comm.barrier();

    // Baseline: per-op doorbell
    uint64_t baseline_cycles = 0;
    if (comm.rank() == 0) {
        opt_per_op_bf_kernel<<<1, 1>>>(d_state_opt, local_addr, local_lkey, 8,
                                        TEST_ITERS, d_cycles);
        cudaDeviceSynchronize();
        cudaMemcpy(&baseline_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    double baseline_us = cycles_to_us(baseline_cycles, clock_rate_khz);

    comm.barrier();

    if (comm.rank() == 0) {
        printf("%-12d %12.2f %12.3f %12s\n",
               1, baseline_us, baseline_us / TEST_ITERS, "1.00x");
    }

    // Test different batch sizes
    for (int batch : BURST_SIZES) {
        if (batch == 1) continue;  // Already tested

        comm.barrier();
        usleep(5000);
        comm.barrier();

        uint64_t batch_cycles = 0;
        if (comm.rank() == 0) {
            opt_batched_burst_kernel<<<1, 1>>>(d_state_opt, local_addr, local_lkey, 8,
                                                TEST_ITERS, batch, d_cycles);
            cudaDeviceSynchronize();
            cudaMemcpy(&batch_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }

        comm.barrier();

        if (comm.rank() == 0) {
            double batch_us = cycles_to_us(batch_cycles, clock_rate_khz);
            printf("%-12d %12.2f %12.3f %12.2fx\n",
                   batch, batch_us, batch_us / TEST_ITERS, baseline_us / batch_us);
        }
    }

    if (comm.rank() == 0) {
        printf("\n");
        fflush(stdout);
    }

    comm.barrier();

    //==========================================================================
    // Test 3: Message size scaling
    //==========================================================================

    if (comm.rank() == 0) {
        printf("=== Test 3: Message Size Scaling (batch=32) ===\n");
        printf("%-12s %12s %12s %15s\n",
               "Size", "Total (us)", "Per-op (us)", "Bandwidth");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
    }

    comm.barrier();

    for (size_t msg_size : MSG_SIZES) {
        comm.barrier();
        usleep(5000);
        comm.barrier();

        uint64_t size_cycles = 0;
        if (comm.rank() == 0) {
            opt_batched_burst_kernel<<<1, 1>>>(d_state_opt, local_addr, local_lkey,
                                                msg_size, TEST_ITERS, 32, d_cycles);
            cudaDeviceSynchronize();
            cudaMemcpy(&size_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }

        comm.barrier();

        if (comm.rank() == 0) {
            double size_us = cycles_to_us(size_cycles, clock_rate_khz);
            double bw_gbps = (msg_size * TEST_ITERS * 8.0) / (size_us * 1000.0);  // Gbps
            printf("%-12zu %12.2f %12.3f %12.2f Gbps\n",
                   msg_size, size_us, size_us / TEST_ITERS, bw_gbps);
        }
    }

    if (comm.rank() == 0) {
        printf("\n");
        fflush(stdout);
    }

    comm.barrier();

    //==========================================================================
    // Test 4: Comparison summary
    //==========================================================================

    if (comm.rank() == 0) {
        printf("=== Test 4: Original vs Optimized Summary ===\n");
        printf("%-12s %15s %15s %12s\n",
               "Size", "Original (us)", "Optimized (us)", "Speedup");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
    }

    for (size_t msg_size : MSG_SIZES) {
        comm.barrier();
        usleep(5000);
        comm.barrier();

        uint64_t orig_size_cycles = 0, opt_size_cycles = 0;

        if (comm.rank() == 0) {
            // Original
            original_burst_kernel<<<1, 1>>>(d_state_orig, local_addr, local_lkey,
                                             msg_size, TEST_ITERS, d_cycles);
            cudaDeviceSynchronize();
            cudaMemcpy(&orig_size_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);

            usleep(5000);

            // Optimized with batching
            opt_batched_burst_kernel<<<1, 1>>>(d_state_opt, local_addr, local_lkey,
                                                msg_size, TEST_ITERS, 32, d_cycles);
            cudaDeviceSynchronize();
            cudaMemcpy(&opt_size_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }

        comm.barrier();

        if (comm.rank() == 0) {
            double orig_us = cycles_to_us(orig_size_cycles, clock_rate_khz);
            double opt_us = cycles_to_us(opt_size_cycles, clock_rate_khz);
            printf("%-12zu %15.3f %15.3f %12.2fx\n",
                   msg_size, orig_us / TEST_ITERS, opt_us / TEST_ITERS, orig_us / opt_us);
        }
    }

    if (comm.rank() == 0) {
        printf("\n=======================================================\n");
        printf("Benchmark complete!\n\n");
        printf("Key optimizations applied:\n");
        printf("  1. L1 cache bypass PTX (st.relaxed.gpu.global.L1::no_allocate)\n");
        printf("  2. BlueFlame doorbell (64-bit UAR write)\n");
        printf("  3. Batched doorbell (multiple WQEs per doorbell)\n");
        printf("  4. Per-32bit atomic WQE writes\n");
        printf("  5. Proper memory ordering (relaxed vs release)\n");
        printf("=======================================================\n");
        fflush(stdout);
    }

    // Cleanup
    cudaFree(d_state_opt);
    cudaFree(d_cycles);
    cudaFree(d_send_buf);
    cudaFree(d_recv_buf);

    MPI_Finalize();
    return 0;
}
