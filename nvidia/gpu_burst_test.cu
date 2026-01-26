/**
 * gpu_burst_test.cu - Pure GPU-Triggered RDMA Burst Test
 *
 * Tests GPU-triggered RDMA only, without mixing CPU-triggered operations.
 * This avoids potential QP state conflicts between CPU and GPU paths.
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

using namespace opengda;
using namespace std::chrono;

constexpr int WARMUP_ITERS = 100;
constexpr int TEST_ITERS = 1000;

// GPU-side burst kernel
__global__ void gpu_burst_write(
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
        uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);
        uint16_t new_prod_idx = (uint16_t)((prod + 1) & 0xFFFF);

        // Build WQE at current slot
        gda_build_rdma_write_wqe(
            state,
            local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, wqe_slot, false  // Not signaled for max speed
        );

        // Ring doorbell with new producer index
        gda_ring_doorbell(state, new_prod_idx);
    }

    // Fence before timing end
    __threadfence_system();

    uint64_t end = clock64();
    *gpu_cycles = end - start;
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

    // Initialize
    GdaGpuComm comm;

    int peer = (comm.rank() == 0) ? 1 : 0;
    int warmup = (argc > 1) ? atoi(argv[1]) : WARMUP_ITERS;
    int iterations = (argc > 2) ? atoi(argv[2]) : TEST_ITERS;

    // Get GPU clock rate
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, comm.gpu_id);
    double clock_rate_khz = props.clockRate;

    if (comm.rank() == 0) {
        printf("=== GPU-Triggered RDMA Burst Test ===\n");
        printf("Warmup: %d, Iterations: %d\n", warmup, iterations);
        printf("GPU: %s (%.2f GHz)\n", props.name, clock_rate_khz / 1e6);
        printf("IB device: %s\n", comm.mlx5->dev_name.c_str());
        printf("\n");
        printf("%12s %12s %12s %12s\n", "Size", "Wall(us)", "GPU(us)", "BW(GB/s)");
        printf("%12s %12s %12s %12s\n", "----", "--------", "-------", "--------");
        fflush(stdout);
    }

    // Test various message sizes
    std::vector<size_t> sizes = {8, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536};

    for (size_t msg_size : sizes) {
        // Allocate buffers
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

        // Allocate timing
        uint64_t* d_cycles;
        CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(uint64_t)));

        GdaDeviceState* d_state = comm.get_device_state();

        comm.barrier();

        // Warmup
        if (comm.rank() == 0) {
            gpu_burst_write<<<1, 1>>>(d_state, local_addr, local_lkey, msg_size, warmup, d_cycles);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Allow NIC to complete warmup
        usleep(10000);
        comm.barrier();

        // Clear recv buffer for verification
        if (comm.rank() == 1) {
            CUDA_CHECK(cudaMemset(d_recv_buf, 0, msg_size));
        }
        comm.barrier();

        // Timed test
        auto wall_start = high_resolution_clock::now();

        if (comm.rank() == 0) {
            gpu_burst_write<<<1, 1>>>(d_state, local_addr, local_lkey, msg_size, iterations, d_cycles);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        auto wall_end = high_resolution_clock::now();
        double wall_us = duration_cast<nanoseconds>(wall_end - wall_start).count() / 1000.0;

        // Get GPU cycles
        uint64_t gpu_cycles;
        CUDA_CHECK(cudaMemcpy(&gpu_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        double gpu_us = gpu_cycles / (clock_rate_khz / 1000.0);

        // Wait for NIC completion
        usleep(10000);
        comm.barrier();

        // Verify
        bool verified = false;
        if (comm.rank() == 1) {
            uint8_t recv_byte;
            CUDA_CHECK(cudaMemcpy(&recv_byte, d_recv_buf, 1, cudaMemcpyDeviceToHost));
            verified = (recv_byte == 0x01);
        }

        // Broadcast verification result
        MPI_Bcast(&verified, 1, MPI_C_BOOL, 1, MPI_COMM_WORLD);

        // Calculate bandwidth
        double total_bytes = (double)msg_size * iterations;
        double gpu_bw_gbps = total_bytes / gpu_us / 1e3;  // GB/s

        if (comm.rank() == 0) {
            printf("%12zu %12.2f %12.2f %12.2f %s\n",
                   msg_size, wall_us / iterations, gpu_us / iterations, gpu_bw_gbps,
                   verified ? "OK" : "FAIL");
            fflush(stdout);
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_cycles));
        CUDA_CHECK(cudaFree(d_send_buf));
        CUDA_CHECK(cudaFree(d_recv_buf));

        comm.barrier();
    }

    if (comm.rank() == 0) {
        printf("\nDone!\n");
    }

    MPI_Finalize();
    return 0;
}
