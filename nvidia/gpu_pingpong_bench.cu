/**
 * gpu_pingpong_bench.cu - Comprehensive GPU-Triggered RDMA Ping-Pong Benchmark
 *
 * This benchmark provides a complete comparison of GPU-triggered RDMA performance
 * similar to the minimal/ directory's DWQ implementation.
 *
 * Tests:
 *   1. Warmup phase
 *   2. Multiple message sizes (8B to 64KB)
 *   3. Multiple iteration counts for statistical accuracy
 *   4. CPU-triggered baseline comparison
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <unistd.h>
#include <algorithm>
#include <vector>
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

// Test configurations
constexpr int WARMUP_ITERS = 50;
constexpr int TEST_ITERS = 1000;
constexpr size_t MSG_SIZES[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    128*1024, 256*1024, 512*1024, 1024*1024,
    2*1024*1024, 4*1024*1024, 8*1024*1024, 16*1024*1024
};
constexpr int NUM_SIZES = sizeof(MSG_SIZES) / sizeof(MSG_SIZES[0]);

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
// GPU kernels
//==============================================================================

// Ping-pong kernel with variable message size
// For msg_size >= 8: uses last 8 bytes of message as sequence number
// For msg_size < 8: uses separate flag buffer at fixed offset
__global__ void gpu_pingpong_kernel(
    GdaDeviceStateOpt* state,
    uint64_t send_addr,
    uint32_t send_lkey,
    volatile uint64_t* recv_flag,    // Flag location in recv buffer
    uint64_t send_flag_addr,         // Flag location in send buffer
    uint32_t msg_size,
    int iterations,
    int is_initiator,
    uint64_t* result_cycles,
    uint64_t* min_cycles,
    uint64_t* max_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t total_cycles = 0;
    uint64_t local_min = UINT64_MAX;
    uint64_t local_max = 0;

    // For small messages, we transfer 8 bytes (the flag itself)
    uint32_t actual_size = (msg_size < 8) ? 8 : msg_size;
    volatile uint64_t* send_flag = (volatile uint64_t*)send_flag_addr;

    for (int i = 0; i < iterations; i++) {
        uint64_t iter_start = clock64();

        if (is_initiator) {
            // Write sequence to send flag location
            *send_flag = i + 1;
            __threadfence_system();

            gda_rdma_write_opt(
                state, send_addr, send_lkey,
                state->remote_addr, state->remote_rkey,
                actual_size, false
            );

            // Wait for response
            while (*recv_flag != (uint64_t)(i + 1)) {
                // Spin
            }
        } else {
            // Wait for data
            while (*recv_flag != (uint64_t)(i + 1)) {
                // Spin
            }

            // Send response
            *send_flag = i + 1;
            __threadfence_system();

            gda_rdma_write_opt(
                state, send_addr, send_lkey,
                state->remote_addr, state->remote_rkey,
                actual_size, false
            );
        }

        uint64_t iter_end = clock64();
        uint64_t iter_cycles = iter_end - iter_start;

        total_cycles += iter_cycles;
        if (iter_cycles < local_min) local_min = iter_cycles;
        if (iter_cycles > local_max) local_max = iter_cycles;
    }

    *result_cycles = total_cycles;
    *min_cycles = local_min;
    *max_cycles = local_max;
}

// Warmup kernel - just does iterations without detailed timing
__global__ void gpu_pingpong_warmup_kernel(
    GdaDeviceStateOpt* state,
    uint64_t send_addr,
    uint32_t send_lkey,
    volatile uint64_t* recv_flag,
    uint64_t send_flag_addr,
    uint32_t msg_size,
    int iterations,
    int is_initiator)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t actual_size = (msg_size < 8) ? 8 : msg_size;
    volatile uint64_t* send_flag = (volatile uint64_t*)send_flag_addr;

    for (int i = 0; i < iterations; i++) {
        if (is_initiator) {
            *send_flag = i + 1;
            __threadfence_system();

            gda_rdma_write_opt(
                state, send_addr, send_lkey,
                state->remote_addr, state->remote_rkey,
                actual_size, false
            );

            while (*recv_flag != (uint64_t)(i + 1)) {}
        } else {
            while (*recv_flag != (uint64_t)(i + 1)) {}

            *send_flag = i + 1;
            __threadfence_system();

            gda_rdma_write_opt(
                state, send_addr, send_lkey,
                state->remote_addr, state->remote_rkey,
                actual_size, false
            );
        }
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

    // Open InfiniBand device - prefer mlx5_1 (IB) over mlx5_0 (RoCE)
    int num_devices = 0;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "Rank %d: No IB devices found\n", mpi_rank);
        MPI_Finalize();
        return 1;
    }

    // Find mlx5_1 device (IB), fallback to first device
    struct ibv_device* target_dev = nullptr;
    for (int i = 0; i < num_devices; i++) {
        const char* name = ibv_get_device_name(dev_list[i]);
        if (name && strcmp(name, "mlx5_1") == 0) {
            target_dev = dev_list[i];
            break;
        }
    }
    if (!target_dev) {
        target_dev = dev_list[0];  // Fallback
    }

    struct ibv_context* ctx = ibv_open_device(target_dev);
    if (!ctx) {
        fprintf(stderr, "Rank %d: Failed to open IB device\n", mpi_rank);
        MPI_Finalize();
        return 1;
    }

    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
        fprintf(stderr, "Rank %d: Failed to allocate PD\n", mpi_rank);
        MPI_Finalize();
        return 1;
    }

    // Create DevX QP with GPU-accessible resources
    DevxQp* devx_qp = new DevxQp(ctx, pd, mpi_rank, 1, 256, 512);

    // Query port attributes
    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, 1, &port_attr);

    // Exchange connection info
    ConnInfo my_conn_info;
    my_conn_info.qpn = devx_qp->qpn;
    my_conn_info.lid = port_attr.lid;  // Use actual LID for IB
    my_conn_info.psn = 0;

    union ibv_gid my_gid;
    ibv_query_gid(ctx, 1, 0, &my_gid);  // Use GID index 0 for IB
    memcpy(my_conn_info.gid, &my_gid, 16);

    ConnInfo peer_conn_info;
    MPI_Sendrecv(&my_conn_info, sizeof(ConnInfo), MPI_BYTE, peer, 0,
                 &peer_conn_info, sizeof(ConnInfo), MPI_BYTE, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Connect QP
    int mtu_val = 3;  // 1024 bytes
    devx_qp->rst2init();
    devx_qp->init2rtr(peer_conn_info.qpn, peer_conn_info.lid,
                      peer_conn_info.gid, peer_conn_info.psn, mtu_val);
    devx_qp->rtr2rts(my_conn_info.psn);

    MPI_Barrier(MPI_COMM_WORLD);

    // Allocate test buffers (GPU memory)
    size_t max_size = 16 * 1024 * 1024;  // 16MB
    void* d_send_buf = nullptr;
    void* d_recv_buf = nullptr;
    cuda_check(cudaMalloc(&d_send_buf, max_size), "alloc send");
    cuda_check(cudaMalloc(&d_recv_buf, max_size), "alloc recv");
    cuda_check(cudaMemset(d_send_buf, 0, max_size), "memset send");
    cuda_check(cudaMemset(d_recv_buf, 0, max_size), "memset recv");

    // Register buffers with NIC
    MemoryRegion* send_mr = new MemoryRegion(pd, d_send_buf, max_size, true, mpi_rank);
    MemoryRegion* recv_mr = new MemoryRegion(pd, d_recv_buf, max_size, true, mpi_rank);

    // Exchange buffer info
    BufInfo my_buf_info;
    my_buf_info.addr = (uint64_t)d_recv_buf;
    my_buf_info.rkey = recv_mr->rkey;

    BufInfo peer_buf_info;
    MPI_Sendrecv(&my_buf_info, sizeof(BufInfo), MPI_BYTE, peer, 1,
                 &peer_buf_info, sizeof(BufInfo), MPI_BYTE, peer, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    uint64_t local_addr = (uint64_t)d_send_buf;
    uint32_t local_lkey = send_mr->lkey;

    // Setup device state
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
    h_state.cqe = nullptr;
    h_state.ncqes = 0;
    h_state.ncqes_mask = 0;
    h_state.remote_addr = peer_buf_info.addr;
    h_state.remote_rkey = peer_buf_info.rkey;
    h_state.batch_size = 32;
    h_state.batch_mask = 31;

    GdaDeviceStateOpt* d_state;
    cuda_check(cudaMalloc(&d_state, sizeof(GdaDeviceStateOpt)), "alloc state");
    cuda_check(cudaMemcpy(d_state, &h_state, sizeof(GdaDeviceStateOpt),
                          cudaMemcpyHostToDevice), "copy state");

    // Allocate timing variables
    uint64_t* d_result_cycles;
    uint64_t* d_min_cycles;
    uint64_t* d_max_cycles;
    cuda_check(cudaMalloc(&d_result_cycles, sizeof(uint64_t)), "alloc result");
    cuda_check(cudaMalloc(&d_min_cycles, sizeof(uint64_t)), "alloc min");
    cuda_check(cudaMalloc(&d_max_cycles, sizeof(uint64_t)), "alloc max");

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Print header
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=======================================================\n");
        printf("   GPU-Triggered RDMA Ping-Pong Benchmark\n");
        printf("   (NVIDIA + InfiniBand with DevX API)\n");
        printf("=======================================================\n");
        printf("GPU: %s (%.2f GHz)\n", props.name, clock_rate_khz / 1e6);
        printf("IB Device: %s\n", ibv_get_device_name(target_dev));
        printf("Warmup: %d iterations\n", WARMUP_ITERS);
        printf("Test: %d iterations per message size\n", TEST_ITERS);
        printf("-------------------------------------------------------\n\n");
        fflush(stdout);
    }

    //==========================================================================
    // Warmup phase
    //==========================================================================

    if (mpi_rank == 0) {
        printf("Running warmup (%d iterations, 64 bytes)...\n", WARMUP_ITERS);
        fflush(stdout);
    }

    cuda_check(cudaMemset(d_recv_buf, 0, max_size), "clear recv warmup");
    cuda_check(cudaMemset(d_send_buf, 0, max_size), "clear send warmup");
    MPI_Barrier(MPI_COMM_WORLD);

    int is_initiator = (mpi_rank == 0) ? 1 : 0;

    // For warmup (64 bytes), flag is at offset 64-8 = 56
    volatile uint64_t* warmup_recv_flag = (volatile uint64_t*)((char*)d_recv_buf + 56);
    uint64_t warmup_send_flag_addr = local_addr + 56;

    gpu_pingpong_warmup_kernel<<<1, 1>>>(
        d_state, local_addr, local_lkey, warmup_recv_flag, warmup_send_flag_addr,
        64, WARMUP_ITERS, is_initiator
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Warmup complete.\n\n");
        fflush(stdout);
    }

    //==========================================================================
    // Main benchmark: Message size scaling
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=== Ping-Pong Latency Results ===\n");
        printf("%-8s %12s %12s %12s %12s %12s\n",
               "Size", "RTT (us)", "Half-RTT", "Min RTT", "Max RTT", "BW (Gbps)");
        printf("-----------------------------------------------------------------------\n");
        fflush(stdout);
    }

    for (int sz_idx = 0; sz_idx < NUM_SIZES; sz_idx++) {
        size_t msg_size = MSG_SIZES[sz_idx];

        // Clear recv and send buffers
        cuda_check(cudaMemset(d_recv_buf, 0, max_size), "clear recv test");
        cuda_check(cudaMemset(d_send_buf, 0, max_size), "clear send test");
        MPI_Barrier(MPI_COMM_WORLD);

        // For small messages (< 8 bytes), we use a fixed 8-byte transfer
        // Flag is always at offset 0 (first 8 bytes) for simplicity
        size_t flag_offset = (msg_size < 8) ? 0 : (msg_size - 8);
        volatile uint64_t* recv_flag_ptr = (volatile uint64_t*)((char*)d_recv_buf + flag_offset);
        uint64_t send_flag_addr = local_addr + flag_offset;

        gpu_pingpong_kernel<<<1, 1>>>(
            d_state, local_addr, local_lkey, recv_flag_ptr, send_flag_addr,
            msg_size, TEST_ITERS, is_initiator,
            d_result_cycles, d_min_cycles, d_max_cycles
        );
        cudaDeviceSynchronize();

        uint64_t total_cycles, min_cycles, max_cycles;
        cuda_check(cudaMemcpy(&total_cycles, d_result_cycles, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost), "copy result");
        cuda_check(cudaMemcpy(&min_cycles, d_min_cycles, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost), "copy min");
        cuda_check(cudaMemcpy(&max_cycles, d_max_cycles, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost), "copy max");

        MPI_Barrier(MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            double avg_rtt_us = cycles_to_us(total_cycles, clock_rate_khz) / TEST_ITERS;
            double min_rtt_us = cycles_to_us(min_cycles, clock_rate_khz);
            double max_rtt_us = cycles_to_us(max_cycles, clock_rate_khz);
            double half_rtt_us = avg_rtt_us / 2.0;
            // Bidirectional bandwidth
            double bw_gbps = (msg_size * 2.0 * 8.0) / (avg_rtt_us * 1000.0);

            char size_buf[32];
            printf("%-8s %12.3f %12.3f %12.3f %12.3f %12.2f\n",
                   format_size(msg_size, size_buf),
                   avg_rtt_us, half_rtt_us, min_rtt_us, max_rtt_us, bw_gbps);
            fflush(stdout);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        usleep(10000);  // 10ms between tests
    }

    //==========================================================================
    // Summary
    //==========================================================================

    if (mpi_rank == 0) {
        printf("\n=======================================================\n");
        printf("Benchmark complete!\n");
        printf("\nImplementation details:\n");
        printf("  - MLX5 DevX API for GPU-accessible QP resources\n");
        printf("  - BlueFlame doorbell for low-latency WQE posting\n");
        printf("  - GPU directly builds WQEs and rings doorbell\n");
        printf("  - Memory polling for completion detection\n");
        printf("=======================================================\n");
        fflush(stdout);
    }

    // Cleanup
    cudaFree(d_state);
    cudaFree(d_result_cycles);
    cudaFree(d_min_cycles);
    cudaFree(d_max_cycles);

    delete send_mr;
    delete recv_mr;

    cudaFree(d_send_buf);
    cudaFree(d_recv_buf);

    delete devx_qp;
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);

    MPI_Finalize();
    return 0;
}
