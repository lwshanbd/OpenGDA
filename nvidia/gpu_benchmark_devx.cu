/**
 * gpu_benchmark_devx.cu - GPU-Triggered RDMA Benchmark using DevX API
 *
 * This benchmark uses proper DevX API for GPU-accessible QP resources:
 *   - mlx5dv_devx_alloc_uar() for BlueFlame register
 *   - mlx5dv_devx_umem_reg() for WQE buffer and doorbell record
 *   - cudaHostRegister() with cudaHostRegisterIoMemory for GPU mapping
 *
 * This is the correct approach matching nvshmem's ibgda implementation.
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

// Test configurations
constexpr int WARMUP_ITERS = 100;
constexpr int TEST_ITERS = 1000;
constexpr int BURST_SIZES[] = {1, 8, 32, 128};
constexpr size_t MSG_SIZES[] = {8, 64, 512, 4096, 65536};

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

// Single op latency test
__global__ void gpu_single_op_kernel(
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

// Batched burst test
// WQE indexing: slot i = base_prod + i, doorbell = slot + 1
__global__ void gpu_batched_burst_kernel(
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

    uint64_t base_prod = gda_load_relaxed_u64(state->prod_idx);
    int batch_count = 0;

    for (int i = 0; i < count; i++) {
        // WQE goes to slot base_prod + i
        uint16_t wqe_slot = (uint16_t)((base_prod + i) & 0xFFFF);
        bool signaled = (i == count - 1);

        gda_build_rdma_write_wqe_opt(
            state, local_addr, local_lkey,
            state->remote_addr, state->remote_rkey,
            size, wqe_slot, signaled
        );

        batch_count++;

        if (batch_count >= batch_size || i == count - 1) {
            // Doorbell = next empty slot = wqe_slot + 1
            uint16_t new_prod = (uint16_t)((wqe_slot + 1) & 0xFFFF);
            gda_ring_doorbell_bf(state, new_prod);
            batch_count = 0;
        }
    }

    gda_store_relaxed_u64(state->prod_idx, base_prod + count);

    uint64_t end = clock64();
    *gpu_cycles = end - start;
}

// Data verification kernel - single RDMA write
__global__ void gpu_verify_write_kernel(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint32_t size)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    gda_rdma_write_opt(
        state, local_addr, local_lkey,
        state->remote_addr, state->remote_rkey,
        size, true  // signaled
    );
}

// Ping-pong kernel - true bidirectional GPU-triggered RDMA
// Each iteration: write to remote, wait for remote to write back
__global__ void gpu_pingpong_kernel(
    GdaDeviceStateOpt* state,
    uint64_t send_addr,           // Local send buffer
    uint32_t send_lkey,
    volatile uint64_t* recv_flag, // Local receive flag (GPU memory)
    int iterations,
    int is_initiator,             // Rank 0 starts first
    uint64_t* gpu_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t start = clock64();

    for (int i = 0; i < iterations; i++) {
        if (is_initiator) {
            // Initiator: send first, then wait for response
            // Write sequence number to send buffer, then RDMA write
            *((volatile uint64_t*)send_addr) = i + 1;
            __threadfence_system();

            gda_rdma_write_opt(
                state, send_addr, send_lkey,
                state->remote_addr, state->remote_rkey,
                8, false
            );

            // Wait for response (remote writes to our recv_flag)
            while (*recv_flag != (uint64_t)(i + 1)) {
                // Spin wait
            }
        } else {
            // Responder: wait for data, then send back
            while (*recv_flag != (uint64_t)(i + 1)) {
                // Spin wait
            }

            // Write response
            *((volatile uint64_t*)send_addr) = i + 1;
            __threadfence_system();

            gda_rdma_write_opt(
                state, send_addr, send_lkey,
                state->remote_addr, state->remote_rkey,
                8, false
            );
        }
    }

    uint64_t end = clock64();
    *gpu_cycles = end - start;
}

// Ping-pong kernel with variable message size
__global__ void gpu_pingpong_sized_kernel(
    GdaDeviceStateOpt* state,
    uint64_t send_addr,           // Local send buffer
    uint32_t send_lkey,
    volatile uint64_t* recv_flag, // Local receive flag (in recv buffer)
    uint32_t msg_size,            // Message size to transfer
    int iterations,
    int is_initiator,
    uint64_t* gpu_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t start = clock64();

    for (int i = 0; i < iterations; i++) {
        if (is_initiator) {
            // Write sequence at end of message
            *((volatile uint64_t*)(send_addr + msg_size - 8)) = i + 1;
            __threadfence_system();

            gda_rdma_write_opt(
                state, send_addr, send_lkey,
                state->remote_addr, state->remote_rkey,
                msg_size, false
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
            *((volatile uint64_t*)(send_addr + msg_size - 8)) = i + 1;
            __threadfence_system();

            gda_rdma_write_opt(
                state, send_addr, send_lkey,
                state->remote_addr, state->remote_rkey,
                msg_size, false
            );
        }
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
    if (!ctx) {
        fprintf(stderr, "Rank %d: Failed to open IB device\n", mpi_rank);
        MPI_Finalize();
        return 1;
    }

    const char* dev_name = ibv_get_device_name(dev_list[0]);
    ibv_free_device_list(dev_list);

    // Allocate PD
    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
        fprintf(stderr, "Rank %d: Failed to allocate PD\n", mpi_rank);
        MPI_Finalize();
        return 1;
    }

    // Create DevX QP with GPU-accessible resources
    DevxQp* devx_qp = new DevxQp(ctx, pd, mpi_rank, 1, 256, 512);

    if (mpi_rank == 0) {
        printf("=======================================================\n");
        printf("     GPU-Triggered RDMA Benchmark (DevX API)\n");
        printf("=======================================================\n");
        printf("GPU: %s (%.2f GHz)\n", props.name, clock_rate_khz / 1e6);
        printf("IB Device: %s\n", dev_name);
        printf("Warmup: %d, Test: %d iterations\n", WARMUP_ITERS, TEST_ITERS);
        printf("-------------------------------------------------------\n");
        printf("DevX QP Resources:\n");
        printf("  QPN: %u\n", devx_qp->qpn);
        printf("  h_uar_reg: %p\n", devx_qp->h_uar_reg);
        printf("  d_uar_reg: %p (GPU-accessible BlueFlame)\n", devx_qp->d_uar_reg);
        printf("  h_wq_buf: %p\n", devx_qp->h_wq_buf);
        printf("  d_wq_buf: %p (GPU-accessible WQE buffer)\n", devx_qp->d_wq_buf);
        printf("  h_dbrec: %p\n", (void*)devx_qp->h_dbrec);
        printf("  d_dbrec: %p (GPU-accessible doorbell)\n", (void*)devx_qp->d_dbrec);
        printf("-------------------------------------------------------\n\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Exchange QP connection info
    ConnInfo my_conn_info;
    my_conn_info.qpn = devx_qp->qpn;
    my_conn_info.psn = 0;  // Use 0 for simplicity

    // Query port info
    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, 1, &port_attr);
    my_conn_info.lid = port_attr.lid;

    // Query GID
    union ibv_gid gid;
    ibv_query_gid(ctx, 1, 0, &gid);
    memcpy(my_conn_info.gid, &gid, 16);

    // Exchange connection info
    ConnInfo peer_conn_info;
    MPI_Sendrecv(&my_conn_info, sizeof(ConnInfo), MPI_BYTE, peer, 0,
                 &peer_conn_info, sizeof(ConnInfo), MPI_BYTE, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Get actual MTU from port
    int mtu_val = port_attr.active_mtu;  // 1=256, 2=512, 3=1024, 4=2048, 5=4096

    // Print GID info for debugging
    fprintf(stderr, "Rank %d: my_gid=", mpi_rank);
    for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", my_conn_info.gid[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "Rank %d: peer_gid=", mpi_rank);
    for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", peer_conn_info.gid[i]);
    fprintf(stderr, "\n");

    if (mpi_rank == 0) {
        printf("Connection info:\n");
        printf("  Local: QPN=%u, LID=%u, PSN=%u\n",
               my_conn_info.qpn, my_conn_info.lid, my_conn_info.psn);
        printf("  Remote: QPN=%u, LID=%u, PSN=%u\n",
               peer_conn_info.qpn, peer_conn_info.lid, peer_conn_info.psn);
        printf("  MTU: %d\n", mtu_val);
        fflush(stdout);
    }

    // Connect QP: RST -> INIT -> RTR -> RTS
    devx_qp->rst2init();
    devx_qp->init2rtr(peer_conn_info.qpn, peer_conn_info.lid,
                      peer_conn_info.gid, peer_conn_info.psn, mtu_val);
    devx_qp->rtr2rts(my_conn_info.psn);

    if (mpi_rank == 0) {
        printf("QP connected: local QPN=%u -> remote QPN=%u\n",
               devx_qp->qpn, peer_conn_info.qpn);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Allocate test buffers (GPU memory)
    size_t max_size = 65536;
    void* d_send_buf = nullptr;
    void* d_recv_buf = nullptr;
    cuda_check(cudaMalloc(&d_send_buf, max_size), "alloc send");
    cuda_check(cudaMalloc(&d_recv_buf, max_size), "alloc recv");
    cuda_check(cudaMemset(d_send_buf, mpi_rank + 1, max_size), "memset send");
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

    if (mpi_rank == 0) {
        printf("Buffer info exchanged:\n");
        printf("  Local send: addr=0x%lx, lkey=0x%x\n",
               (uint64_t)d_send_buf, send_mr->lkey);
        printf("  Remote recv: addr=0x%lx, rkey=0x%x\n",
               peer_buf_info.addr, peer_buf_info.rkey);
        printf("-------------------------------------------------------\n\n");
        fflush(stdout);
    }

    uint64_t local_addr = (uint64_t)d_send_buf;
    uint32_t local_lkey = send_mr->lkey;

    // Setup device state using DevX QP's GPU-accessible pointers
    GdaDeviceStateOpt h_state;
    memset(&h_state, 0, sizeof(h_state));
    h_state.qpn = devx_qp->qpn;
    h_state.nwqes = 1 << devx_qp->log_wq_size;
    h_state.nwqes_mask = h_state.nwqes - 1;
    h_state.wqe_buf = devx_qp->d_wq_buf;           // GPU-accessible!
    h_state.wqe_lkey = 0;
    h_state.dbrec = devx_qp->d_dbrec;              // GPU-accessible!
    h_state.bf_reg = (volatile uint64_t*)devx_qp->d_uar_reg;  // GPU-accessible!
    h_state.prod_idx = devx_qp->d_prod_idx;        // GPU-accessible!
    h_state.cqe = nullptr;  // We don't poll CQ in these tests
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

    // Allocate timing variable
    uint64_t* d_cycles;
    cuda_check(cudaMalloc(&d_cycles, sizeof(uint64_t)), "alloc cycles");

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Test 0: Data correctness verification (GPU-triggered)
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=== Test 0: GPU-Triggered Data Verification ===\n");
        fflush(stdout);
    }

    // Initialize send buffer with known pattern
    uint8_t send_pattern = 0xAB;
    cuda_check(cudaMemset(d_send_buf, send_pattern, max_size), "memset send pattern");

    // Clear receive buffer on rank 1
    if (mpi_rank == 1) {
        cuda_check(cudaMemset(d_recv_buf, 0, max_size), "clear recv buf");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // GPU-triggered RDMA WRITE
    if (mpi_rank == 0) {
        printf("  GPU-triggered RDMA (DevX): ");
        fflush(stdout);

        // Debug: Print h_state values before kernel
        printf("\n  Debug - State before kernel:\n");
        printf("    qpn=%u, nwqes=%u, nwqes_mask=%u\n", h_state.qpn, h_state.nwqes, h_state.nwqes_mask);
        printf("    wqe_buf=%p, dbrec=%p, bf_reg=%p\n", h_state.wqe_buf, (void*)h_state.dbrec, (void*)h_state.bf_reg);
        printf("    remote_addr=0x%lx, remote_rkey=0x%x\n", h_state.remote_addr, h_state.remote_rkey);
        printf("    prod_idx ptr=%p, val=%lu\n", (void*)h_state.prod_idx, *devx_qp->h_prod_idx);

        // Debug: Print initial dbrec value
        printf("    dbrec value before = 0x%x\n", *devx_qp->h_dbrec);

        gpu_verify_write_kernel<<<1, 1>>>(d_state, local_addr, local_lkey, 64);
        cudaDeviceSynchronize();

        // Debug: Print dbrec value after kernel
        printf("    dbrec value after = 0x%x\n", *devx_qp->h_dbrec);
        printf("    prod_idx after = %lu\n", *devx_qp->h_prod_idx);

        // Check CQ for completions/errors
        printf("    Checking CQ for completions...\n");
        int cq_polls = 0;
        for (int i = 0; i < 1000; i++) {
            // Read CQE from host-accessible CQ buffer
            uint8_t* cqe_ptr = (uint8_t*)devx_qp->h_cq_buf;
            uint8_t op_own = cqe_ptr[63];  // Last byte has opcode and owner
            uint8_t owner = op_own & 1;
            uint8_t opcode = (op_own >> 4) & 0x0F;

            if (owner == 0) {  // Owner flipped, completion available
                printf("    CQE found: opcode=0x%x, owner=%d\n", opcode, owner);
                if (opcode == 0x0D) {  // Error
                    printf("    ERROR CQE! syndrome byte = 0x%x\n", cqe_ptr[55]);
                }
                cq_polls++;
                break;
            }
            usleep(100);
        }
        if (cq_polls == 0) {
            printf("    No CQE received after polling\n");
        }
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(50000);  // 50ms to let NIC complete
    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 1) {
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
            printf("     First 16 bytes: ");
            for (int i = 0; i < 16; i++) printf("%02x ", recv_data[i]);
            printf("\n");
        }
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("\n");
    }

    //==========================================================================
    // Test 0.5: GPU-Triggered Ping-Pong Latency
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=== Test 0.5: GPU-Triggered Ping-Pong (8 bytes) ===\n");
        printf("%-12s %12s %12s %15s\n",
               "Iterations", "Total (us)", "RTT (us)", "Half-RTT (us)");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
    }

    // For ping-pong, recv_buf is used as the flag buffer
    // Clear recv buffers and sync
    cuda_check(cudaMemset(d_recv_buf, 0, max_size), "clear recv for pingpong");
    MPI_Barrier(MPI_COMM_WORLD);

    // Run ping-pong with different iteration counts
    int pingpong_iters[] = {10, 100, 1000};
    for (int pp_iters : pingpong_iters) {
        // Clear recv buffer
        cuda_check(cudaMemset(d_recv_buf, 0, sizeof(uint64_t)), "clear flag");
        MPI_Barrier(MPI_COMM_WORLD);

        uint64_t pp_cycles = 0;
        int is_initiator = (mpi_rank == 0) ? 1 : 0;

        // Both ranks launch kernels simultaneously
        gpu_pingpong_kernel<<<1, 1>>>(
            d_state,
            local_addr,
            local_lkey,
            (volatile uint64_t*)d_recv_buf,
            pp_iters,
            is_initiator,
            d_cycles
        );
        cudaDeviceSynchronize();

        cuda_check(cudaMemcpy(&pp_cycles, d_cycles, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost), "copy pingpong cycles");

        MPI_Barrier(MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            double pp_us = cycles_to_us(pp_cycles, clock_rate_khz);
            double rtt = pp_us / pp_iters;
            double half_rtt = rtt / 2.0;
            printf("%-12d %12.2f %12.3f %15.3f\n",
                   pp_iters, pp_us, rtt, half_rtt);
            fflush(stdout);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        usleep(10000);  // 10ms between tests
    }

    if (mpi_rank == 0) {
        printf("\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Test 0.6: Ping-Pong with Different Message Sizes
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=== Test 0.6: Ping-Pong Message Size Scaling (100 iters) ===\n");
        printf("%-12s %12s %12s %15s %15s\n",
               "Size", "Total (us)", "RTT (us)", "Half-RTT (us)", "Bandwidth");
        printf("-----------------------------------------------------------------------\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    size_t pp_msg_sizes[] = {8, 64, 256, 1024, 4096, 16384, 65536};
    int pp_size_iters = 100;

    for (size_t msg_sz : pp_msg_sizes) {
        // Clear recv buffer - flag is at offset (msg_sz - 8)
        cuda_check(cudaMemset(d_recv_buf, 0, max_size), "clear for sized pingpong");
        MPI_Barrier(MPI_COMM_WORLD);

        uint64_t pp_cycles = 0;
        int is_initiator = (mpi_rank == 0) ? 1 : 0;

        // Flag is at the end of the message (last 8 bytes)
        volatile uint64_t* flag_ptr = (volatile uint64_t*)((char*)d_recv_buf + msg_sz - 8);

        gpu_pingpong_sized_kernel<<<1, 1>>>(
            d_state,
            local_addr,
            local_lkey,
            flag_ptr,
            msg_sz,
            pp_size_iters,
            is_initiator,
            d_cycles
        );
        cudaDeviceSynchronize();

        cuda_check(cudaMemcpy(&pp_cycles, d_cycles, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost), "copy sized pingpong cycles");

        MPI_Barrier(MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            double pp_us = cycles_to_us(pp_cycles, clock_rate_khz);
            double rtt = pp_us / pp_size_iters;
            double half_rtt = rtt / 2.0;
            // Bandwidth: bytes transferred in both directions / time
            double bw_gbps = (msg_sz * 2.0 * pp_size_iters * 8.0) / (pp_us * 1000.0);
            printf("%-12zu %12.2f %12.3f %15.3f %12.2f Gbps\n",
                   msg_sz, pp_us, rtt, half_rtt, bw_gbps);
            fflush(stdout);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        usleep(10000);
    }

    if (mpi_rank == 0) {
        printf("\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Test 1: Per-operation latency
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=== Test 1: Per-Operation Latency (8 bytes) ===\n");
        printf("%-20s %12s %12s %12s\n",
               "Implementation", "Total (us)", "Per-op (us)", "Cycles/op");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Warmup
    if (mpi_rank == 0) {
        gpu_single_op_kernel<<<1, 1>>>(d_state, local_addr, local_lkey, 8,
                                        WARMUP_ITERS, d_cycles);
        cudaDeviceSynchronize();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(10000);
    MPI_Barrier(MPI_COMM_WORLD);

    // Test
    uint64_t op_cycles = 0;
    if (mpi_rank == 0) {
        gpu_single_op_kernel<<<1, 1>>>(d_state, local_addr, local_lkey, 8,
                                        TEST_ITERS, d_cycles);
        cudaDeviceSynchronize();
        cudaMemcpy(&op_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        double op_us = cycles_to_us(op_cycles, clock_rate_khz);
        printf("%-20s %12.2f %12.3f %12lu\n",
               "DevX+BlueFlame", op_us, op_us / TEST_ITERS, op_cycles / TEST_ITERS);
        printf("\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Test 2: Batching effect
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=== Test 2: Batching Effect (%d ops, 8 bytes) ===\n", TEST_ITERS);
        printf("%-12s %12s %12s %12s\n",
               "Batch Size", "Total (us)", "Per-op (us)", "Speedup");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Baseline: per-op doorbell (batch=1)
    uint64_t baseline_cycles = 0;
    if (mpi_rank == 0) {
        gpu_batched_burst_kernel<<<1, 1>>>(d_state, local_addr, local_lkey, 8,
                                            TEST_ITERS, 1, d_cycles);
        cudaDeviceSynchronize();
        cudaMemcpy(&baseline_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    double baseline_us = cycles_to_us(baseline_cycles, clock_rate_khz);

    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("%-12d %12.2f %12.3f %12s\n",
               1, baseline_us, baseline_us / TEST_ITERS, "1.00x");
    }

    // Test different batch sizes
    for (int batch : BURST_SIZES) {
        if (batch == 1) continue;

        MPI_Barrier(MPI_COMM_WORLD);
        usleep(5000);
        MPI_Barrier(MPI_COMM_WORLD);

        uint64_t batch_cycles = 0;
        if (mpi_rank == 0) {
            gpu_batched_burst_kernel<<<1, 1>>>(d_state, local_addr, local_lkey, 8,
                                                TEST_ITERS, batch, d_cycles);
            cudaDeviceSynchronize();
            cudaMemcpy(&batch_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            double batch_us = cycles_to_us(batch_cycles, clock_rate_khz);
            printf("%-12d %12.2f %12.3f %12.2fx\n",
                   batch, batch_us, batch_us / TEST_ITERS, baseline_us / batch_us);
        }
    }

    if (mpi_rank == 0) {
        printf("\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Test 3: Message size scaling
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=== Test 3: Message Size Scaling (batch=32) ===\n");
        printf("%-12s %12s %12s %15s\n",
               "Size", "Total (us)", "Per-op (us)", "Bandwidth");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t msg_size : MSG_SIZES) {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(5000);
        MPI_Barrier(MPI_COMM_WORLD);

        uint64_t size_cycles = 0;
        if (mpi_rank == 0) {
            gpu_batched_burst_kernel<<<1, 1>>>(d_state, local_addr, local_lkey,
                                                msg_size, TEST_ITERS, 32, d_cycles);
            cudaDeviceSynchronize();
            cudaMemcpy(&size_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            double size_us = cycles_to_us(size_cycles, clock_rate_khz);
            double bw_gbps = (msg_size * TEST_ITERS * 8.0) / (size_us * 1000.0);
            printf("%-12zu %12.2f %12.3f %12.2f Gbps\n",
                   msg_size, size_us, size_us / TEST_ITERS, bw_gbps);
        }
    }

    if (mpi_rank == 0) {
        printf("\n=======================================================\n");
        printf("Benchmark complete!\n\n");
        printf("Using DevX API with GPU-accessible resources:\n");
        printf("  - mlx5dv_devx_alloc_uar() for BlueFlame register\n");
        printf("  - mlx5dv_devx_umem_reg() for WQE buffer & doorbell\n");
        printf("  - cudaHostRegister(cudaHostRegisterIoMemory) for mapping\n");
        printf("=======================================================\n");
        fflush(stdout);
    }

    // Cleanup
    cudaFree(d_state);
    cudaFree(d_cycles);

    delete send_mr;
    delete recv_mr;

    cudaFree(d_send_buf);
    cudaFree(d_recv_buf);

    delete devx_qp;
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);

    MPI_Finalize();
    return 0;
}
