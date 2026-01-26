/**
 * gpu_debug_test.cu - Debug GPU-Triggered RDMA
 *
 * Single operation with extensive debug output
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "gda_gpu_comm.hpp"
#include "gda_device.cuh"

using namespace opengda;

// Debug kernel that prints WQE details
__global__ void debug_rdma_write(
    GdaDeviceState* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size,
    int* result,
    uint32_t* debug_info)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Print all state
    printf("GPU Debug:\n");
    printf("  qpn = %u\n", state->qpn);
    printf("  nwqes = %u\n", state->nwqes);
    printf("  wqe_buf = %p\n", state->wqe_buf);
    printf("  dbrec = %p\n", state->dbrec);
    printf("  prod_idx ptr = %p\n", state->prod_idx);

    // Read current values
    uint64_t prod = *state->prod_idx;
    uint32_t db_val = *state->dbrec;

    printf("  *prod_idx = %lu\n", prod);
    printf("  *dbrec (before) = 0x%08x\n", db_val);
    printf("  remote_addr = 0x%lx\n", state->remote_addr);
    printf("  remote_rkey = 0x%x\n", state->remote_rkey);
    printf("  local_addr = 0x%lx\n", local_addr);
    printf("  local_lkey = 0x%x\n", local_lkey);
    printf("  size = %u\n", size);

    // WQE goes at slot = current producer index
    uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);
    // New producer index for doorbell
    uint16_t new_prod = (uint16_t)((prod + 1) & 0xFFFF);

    printf("  wqe_slot = %u (where WQE is placed)\n", wqe_slot);
    printf("  new_prod = %u (doorbell value)\n", new_prod);

    // Get WQE pointer - place WQE at current producer slot
    void* wqe_ptr = (void*)((uintptr_t)state->wqe_buf +
                           ((wqe_slot & state->nwqes_mask) << MLX5_SEND_WQE_SHIFT));
    printf("  wqe_ptr = %p\n", wqe_ptr);

    // Build WQE with wqe_idx = slot number
    gda_build_rdma_write_wqe(
        state,
        local_addr, local_lkey,
        state->remote_addr, state->remote_rkey,
        size, wqe_slot, true  // Signaled, use slot as wqe_idx
    );

    // Print WQE contents
    uint32_t* wqe32 = (uint32_t*)wqe_ptr;
    printf("  WQE contents (as uint32):\n");
    for (int i = 0; i < 12; i++) {
        printf("    [%2d] = 0x%08x\n", i, wqe32[i]);
    }

    // Memory fence
    __threadfence_system();

    // Ring doorbell with NEW producer index using release semantics
    uint32_t doorbell_val = gda_htobe32(new_prod & 0xFFFF);
    printf("  Writing doorbell: 0x%08x (new_prod=%u)\n", doorbell_val, new_prod);

    gda_store_release_u32(state->dbrec, doorbell_val);

    // Also try BlueFlame if available
    printf("  bf_reg = %p\n", state->bf_reg);
    if (state->bf_reg) {
        printf("  Writing to BlueFlame register (8-byte format with release)...\n");

        // Following nvshmem: only write 8 bytes (control segment header)
        // Format: opmod(8) | wqe_idx(16) | opcode(8) || qpn(24) | ds(8)
        uint32_t opmod_idx_opcode = gda_htobe32((new_prod & 0xFFFF) << 8);
        uint32_t qpn_ds = gda_htobe32(state->qpn << 8);
        uint64_t bf_val = ((uint64_t)opmod_idx_opcode) | ((uint64_t)qpn_ds << 32);

        printf("  BlueFlame value: 0x%016lx\n", bf_val);
        gda_store_release_u64((volatile uint64_t*)state->bf_reg, bf_val);

        printf("  BlueFlame write done\n");
    }

    // Memory fence again
    __threadfence_system();

    // Update our producer index to new value
    *state->prod_idx = new_prod;

    // Read doorbell back
    __threadfence_system();
    db_val = *state->dbrec;
    printf("  *dbrec (after) = 0x%08x\n", db_val);

    // Store debug info
    debug_info[0] = wqe_slot;
    debug_info[1] = doorbell_val;
    debug_info[2] = db_val;
    debug_info[3] = state->qpn;

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

    GdaGpuComm comm;

    int peer = (comm.rank() == 0) ? 1 : 0;
    size_t msg_size = 8;

    if (comm.rank() == 0) {
        printf("=== GPU Debug Test ===\n\n");
        fflush(stdout);
    }

    // Allocate buffers
    void* d_send_buf = nullptr;
    void* d_recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_buf, msg_size));
    CUDA_CHECK(cudaMalloc(&d_recv_buf, msg_size));
    CUDA_CHECK(cudaMemset(d_send_buf, 0xAB, msg_size));  // Send 0xAB
    CUDA_CHECK(cudaMemset(d_recv_buf, 0x00, msg_size));

    // Register buffers
    auto send_handle = comm.register_buffer(d_send_buf, msg_size, true);
    auto recv_handle = comm.register_buffer(d_recv_buf, msg_size, true);

    // Exchange
    comm.exchange_buffer_info(recv_handle, 0);
    comm.set_remote_target(peer, 0);

    // Get local info
    uint64_t local_addr;
    uint32_t local_lkey;
    comm.get_local_buffer_info(send_handle, &local_addr, &local_lkey);

    // Allocate debug vars
    int* d_result;
    uint32_t* d_debug;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_debug, 16 * sizeof(uint32_t)));

    GdaDeviceState* d_state = comm.get_device_state();

    // Print host-side state
    if (comm.rank() == 0) {
        printf("Host-side info:\n");
        printf("  local_addr = 0x%lx\n", local_addr);
        printf("  local_lkey = 0x%x\n", local_lkey);

        uint64_t remote_addr;
        uint32_t remote_rkey;
        comm.get_remote_buffer_info(peer, 0, &remote_addr, &remote_rkey);
        printf("  remote_addr = 0x%lx\n", remote_addr);
        printf("  remote_rkey = 0x%x\n", remote_rkey);
        printf("\n");
        fflush(stdout);
    }

    comm.barrier();

    // Skip CPU test - it desynchronizes producer index with GPU tracking
    // Instead, we'll do pure GPU-triggered test

    // Test: GPU-triggered DIRECTLY (no CPU operations)
    if (comm.rank() == 0) {
        printf("\nTest 2: GPU-triggered RDMA WRITE...\n");
        fflush(stdout);

        debug_rdma_write<<<1, 1>>>(d_state, local_addr, local_lkey, msg_size, d_result, d_debug);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Get debug info
        uint32_t h_debug[16];
        CUDA_CHECK(cudaMemcpy(h_debug, d_debug, 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        printf("\nHost-side debug check:\n");
        printf("  wqe_slot from kernel = %u\n", h_debug[0]);
        printf("  doorbell_val = 0x%08x\n", h_debug[1]);
        printf("  db read back = 0x%08x\n", h_debug[2]);
        printf("  qpn = %u\n", h_debug[3]);
        fflush(stdout);
    }

    comm.barrier();
    usleep(50000);  // Wait 50ms for NIC

    // Check CQ state after GPU test
    if (comm.rank() == 0) {
        printf("\nCQ state after GPU test:\n");
        struct mlx5_cqe64 {
            uint8_t  rsvd0[46];
            uint16_t wqe_counter;
            uint8_t  signature;
            uint8_t  op_own;
        } __attribute__((packed));

        mlx5_cqe64* cqes = (mlx5_cqe64*)comm.mlx5->cq_ex.buf;
        for (int i = 0; i < 4; i++) {
            printf("  CQE[%d]: op_own=0x%02x, wqe_counter=%u\n",
                   i, cqes[i].op_own, ntohs(cqes[i].wqe_counter));
        }
        printf("\n");
        fflush(stdout);

        // Also poll using ibv_poll_cq
        struct ibv_wc wc;
        int n = ibv_poll_cq(comm.mlx5->cq, 1, &wc);
        if (n > 0) {
            printf("ibv_poll_cq: status=%d (%s), opcode=%d, wr_id=%lu\n",
                   wc.status, ibv_wc_status_str(wc.status), wc.opcode, wc.wr_id);
        } else if (n == 0) {
            printf("ibv_poll_cq: no completions\n");
        } else {
            printf("ibv_poll_cq: error %d\n", n);
        }
        fflush(stdout);
    }
    comm.barrier();

    // Verify GPU test
    if (comm.rank() == 1) {
        uint8_t buf[8];
        CUDA_CHECK(cudaMemcpy(buf, d_recv_buf, msg_size, cudaMemcpyDeviceToHost));
        printf("Rank 1: GPU test received: 0x%02x (expected 0xAB)\n", buf[0]);
        if (buf[0] == 0xAB) {
            printf("GPU-triggered RDMA VERIFIED!\n");
        } else {
            printf("GPU-triggered RDMA FAILED - data not received\n");
        }
        fflush(stdout);
    }

    comm.barrier();

    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_debug));
    CUDA_CHECK(cudaFree(d_send_buf));
    CUDA_CHECK(cudaFree(d_recv_buf));

    MPI_Finalize();
    return 0;
}
