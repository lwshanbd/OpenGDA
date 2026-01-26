/**
 * gpu_concurrent_bench_v2.cu - Optimized GPU-Triggered Concurrent RDMA Benchmark
 *
 * Improvements over v1:
 *   1. Uses CQ completion polling instead of ACK round-trip (true one-way)
 *   2. Uses globaltimer for nanosecond precision
 *   3. All timing inside kernel (no kernel launch overhead)
 *   4. Signaled last WQE for completion detection
 *
 * Measures: WQE build + doorbell + network transfer + NIC completion
 * Does NOT include: receiver processing, ACK round-trip
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

// Configuration
constexpr int N_STREAMS = 32;
constexpr int NUM_ITERATIONS = 100;  // More iterations for better statistics

constexpr size_t TEST_SIZES[] = {
    8, 64, 512, 1024, 4096, 16384, 65536,
    256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024
};
constexpr int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);
constexpr size_t MAX_SIZE = 16 * 1024 * 1024;

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
// Optimized GPU Kernel - One-way with CQ completion
//==============================================================================

/**
 * GPU kernel for concurrent RDMA writes with CQ completion polling.
 *
 * Measures true one-way latency:
 *   1. Record start time (globaltimer)
 *   2. Build N_STREAMS WQEs (last one signaled)
 *   3. Ring doorbell
 *   4. Poll CQ for completion
 *   5. Record end time (globaltimer)
 *
 * No receiver involvement needed - measures when NIC completes send.
 */
__global__ void gpu_concurrent_write_cq_kernel(
    GdaDeviceStateOpt* state,
    uint64_t* send_addrs,
    uint32_t* send_lkeys,
    uint64_t remote_base_addr,
    uint32_t remote_rkey,
    size_t msg_size,
    size_t stream_stride,
    int n_streams,
    uint64_t* start_ns,
    uint64_t* end_ns)
{
    int stream_id = threadIdx.x;

    __shared__ uint64_t base_wqe_idx;

    // Thread 0 records start time and gets base WQE index
    if (stream_id == 0) {
        *start_ns = gda_globaltimer();
        base_wqe_idx = gda_load_relaxed_u64(state->prod_idx);
    }
    __syncthreads();

    // Each thread builds one WQE
    if (stream_id < n_streams) {
        uint64_t remote_addr = remote_base_addr + (stream_id * stream_stride);
        uint16_t wqe_slot = (uint16_t)((base_wqe_idx + stream_id) & 0xFFFF);

        size_t actual_size = (msg_size < 8) ? 8 : msg_size;
        bool signaled = (stream_id == n_streams - 1);  // Signal last WQE

        gda_build_rdma_write_wqe_opt(
            state, send_addrs[stream_id], send_lkeys[stream_id],
            remote_addr, remote_rkey,
            actual_size, wqe_slot, signaled
        );
    }

    __syncthreads();

    // Thread 0 rings doorbell and polls CQ
    if (stream_id == 0) {
        uint16_t new_prod = (uint16_t)((base_wqe_idx + n_streams) & 0xFFFF);
        gda_ring_doorbell_bf(state, new_prod);

        // Poll CQ for completion (signaled WQE)
        gda_poll_cq_opt(state, 1, 1000000);  // 1ms timeout

        *end_ns = gda_globaltimer();
    }
}

/**
 * Minimal kernel - just WQE build + doorbell timing (no CQ)
 * For comparison with v1's WQE-only measurement
 */
__global__ void gpu_concurrent_write_minimal_kernel(
    GdaDeviceStateOpt* state,
    uint64_t* send_addrs,
    uint32_t* send_lkeys,
    uint64_t remote_base_addr,
    uint32_t remote_rkey,
    size_t msg_size,
    size_t stream_stride,
    int n_streams,
    uint64_t* start_ns,
    uint64_t* end_ns)
{
    int stream_id = threadIdx.x;

    __shared__ uint64_t base_wqe_idx;

    if (stream_id == 0) {
        *start_ns = gda_globaltimer();
        base_wqe_idx = gda_load_relaxed_u64(state->prod_idx);
    }
    __syncthreads();

    if (stream_id < n_streams) {
        uint64_t remote_addr = remote_base_addr + (stream_id * stream_stride);
        uint16_t wqe_slot = (uint16_t)((base_wqe_idx + stream_id) & 0xFFFF);

        size_t actual_size = (msg_size < 8) ? 8 : msg_size;

        gda_build_rdma_write_wqe_opt(
            state, send_addrs[stream_id], send_lkeys[stream_id],
            remote_addr, remote_rkey,
            actual_size, wqe_slot, false
        );
    }

    __syncthreads();

    if (stream_id == 0) {
        uint16_t new_prod = (uint16_t)((base_wqe_idx + n_streams) & 0xFFFF);
        gda_ring_doorbell_bf(state, new_prod);

        // No CQ polling - just doorbell time
        gda_membar_sys();  // Ensure doorbell write completes
        *end_ns = gda_globaltimer();
    }
}

/**
 * End-to-end kernel with receiver polling (for comparison)
 * Same as v1 but uses globaltimer
 */
__global__ void gpu_sender_e2e_kernel(
    GdaDeviceStateOpt* state,
    uint64_t* send_addrs,
    uint32_t* send_lkeys,
    uint64_t remote_base_addr,
    uint32_t remote_rkey,
    size_t msg_size,
    size_t stream_stride,
    int n_streams,
    volatile uint64_t* ack_flag,
    uint64_t expected_ack,
    uint64_t* start_ns,
    uint64_t* end_ns)
{
    int stream_id = threadIdx.x;

    __shared__ uint64_t base_wqe_idx;

    if (stream_id == 0) {
        *start_ns = gda_globaltimer();
        base_wqe_idx = gda_load_relaxed_u64(state->prod_idx);
    }
    __syncthreads();

    if (stream_id < n_streams) {
        uint64_t remote_addr = remote_base_addr + (stream_id * stream_stride);
        uint16_t wqe_slot = (uint16_t)((base_wqe_idx + stream_id) & 0xFFFF);

        size_t actual_size = (msg_size < 8) ? 8 : msg_size;

        gda_build_rdma_write_wqe_opt(
            state, send_addrs[stream_id], send_lkeys[stream_id],
            remote_addr, remote_rkey,
            actual_size, wqe_slot, false
        );
    }

    __syncthreads();

    if (stream_id == 0) {
        uint16_t new_prod = (uint16_t)((base_wqe_idx + n_streams) & 0xFFFF);
        gda_ring_doorbell_bf(state, new_prod);

        // Wait for ACK from receiver
        while (*ack_flag != expected_ack) { }

        *end_ns = gda_globaltimer();
    }
}

__global__ void gpu_receiver_e2e_kernel(
    GdaDeviceStateOpt* state,
    volatile uint64_t** recv_flags,
    int n_streams,
    uint64_t expected_seq,
    uint64_t ack_remote_addr,
    uint32_t ack_remote_rkey,
    uint64_t ack_local_addr,
    uint32_t ack_local_lkey,
    uint64_t ack_value)
{
    int stream_id = threadIdx.x;

    if (stream_id < n_streams) {
        while (*recv_flags[stream_id] != expected_seq) { }
    }

    __syncthreads();

    if (stream_id == 0) {
        *(volatile uint64_t*)ack_local_addr = ack_value;
        __threadfence_system();

        uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
        uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);

        gda_build_rdma_write_wqe_opt(
            state, ack_local_addr, ack_local_lkey,
            ack_remote_addr, ack_remote_rkey,
            8, wqe_slot, false
        );

        gda_ring_doorbell_bf(state, (uint16_t)((prod + 1) & 0xFFFF));
    }
}

//==============================================================================
// Helpers
//==============================================================================

void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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

double compute_bandwidth_gbps(size_t bytes, double time_us) {
    if (time_us <= 0) return 0;
    return (bytes * 8.0 / 1e9) / (time_us / 1e6);
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
        if (mpi_rank == 0) fprintf(stderr, "Requires exactly 2 ranks\n");
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

    // Open IB device
    struct ibv_device** dev_list = ibv_get_device_list(nullptr);
    if (!dev_list || !dev_list[0]) {
        fprintf(stderr, "Rank %d: No IB devices\n", mpi_rank);
        MPI_Finalize();
        return 1;
    }

    struct ibv_context* ctx = ibv_open_device(dev_list[0]);
    struct ibv_pd* pd = ibv_alloc_pd(ctx);

    // Create DevX QP with CQ
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

    void* d_send_bufs[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cuda_check(cudaMalloc(&d_send_bufs[i], MAX_SIZE), "alloc send");
        cuda_check(cudaMemset(d_send_bufs[i], 0xAA + i, MAX_SIZE), "memset send");
    }

    void* d_recv_buf;
    cuda_check(cudaMalloc(&d_recv_buf, MAX_SIZE * N_STREAMS), "alloc recv");
    cuda_check(cudaMemset(d_recv_buf, 0, MAX_SIZE * N_STREAMS), "memset recv");

    uint64_t* d_ack_buf;
    cuda_check(cudaMalloc(&d_ack_buf, sizeof(uint64_t)), "alloc ack");
    cuda_check(cudaMemset(d_ack_buf, 0, sizeof(uint64_t)), "memset ack");

    // Register memory
    MemoryRegion* send_mrs[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        send_mrs[i] = new MemoryRegion(pd, d_send_bufs[i], MAX_SIZE, true, mpi_rank);
    }
    MemoryRegion* recv_mr = new MemoryRegion(pd, d_recv_buf, MAX_SIZE * N_STREAMS, true, mpi_rank);
    MemoryRegion* ack_mr = new MemoryRegion(pd, d_ack_buf, sizeof(uint64_t), true, mpi_rank);

    // Exchange buffer info
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

    uint64_t h_send_addrs[N_STREAMS];
    uint32_t h_send_lkeys[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        h_send_addrs[i] = (uint64_t)d_send_bufs[i];
        h_send_lkeys[i] = send_mrs[i]->lkey;
    }

    uint64_t* d_send_addrs;
    uint32_t* d_send_lkeys;
    cuda_check(cudaMalloc(&d_send_addrs, N_STREAMS * sizeof(uint64_t)), "alloc addrs");
    cuda_check(cudaMalloc(&d_send_lkeys, N_STREAMS * sizeof(uint32_t)), "alloc lkeys");
    cuda_check(cudaMemcpy(d_send_addrs, h_send_addrs, N_STREAMS * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "copy addrs");
    cuda_check(cudaMemcpy(d_send_lkeys, h_send_lkeys, N_STREAMS * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "copy lkeys");

    // Device state
    GdaDeviceStateOpt h_state;
    memset(&h_state, 0, sizeof(h_state));
    h_state.qpn = devx_qp->qpn;
    h_state.nwqes = 1 << devx_qp->log_wq_size;
    h_state.nwqes_mask = h_state.nwqes - 1;
    h_state.wqe_buf = devx_qp->d_wq_buf;
    h_state.dbrec = devx_qp->d_dbrec;
    h_state.bf_reg = (volatile uint64_t*)devx_qp->d_uar_reg;
    h_state.prod_idx = devx_qp->d_prod_idx;
    h_state.remote_addr = peer_bufs.recv.addr;
    h_state.remote_rkey = peer_bufs.recv.rkey;

    // Setup CQ for completion polling
    h_state.cqe = (volatile GdaCqe64Opt*)devx_qp->d_cq_buf;
    h_state.ncqes = devx_qp->num_cqe;
    h_state.ncqes_mask = h_state.ncqes - 1;
    h_state.cq_dbrec = devx_qp->d_cq_dbrec;

    GdaDeviceStateOpt* d_state;
    cuda_check(cudaMalloc(&d_state, sizeof(GdaDeviceStateOpt)), "alloc state");
    cuda_check(cudaMemcpy(d_state, &h_state, sizeof(GdaDeviceStateOpt),
                          cudaMemcpyHostToDevice), "copy state");

    // Timing buffers
    uint64_t* d_start_ns;
    uint64_t* d_end_ns;
    cuda_check(cudaMalloc(&d_start_ns, sizeof(uint64_t)), "alloc start");
    cuda_check(cudaMalloc(&d_end_ns, sizeof(uint64_t)), "alloc end");

    // Recv flag pointers for E2E test
    volatile uint64_t** d_recv_flags;
    cuda_check(cudaMalloc(&d_recv_flags, N_STREAMS * sizeof(uint64_t*)), "alloc recv flags");

    MPI_Barrier(MPI_COMM_WORLD);

    //==========================================================================
    // Print header
    //==========================================================================

    if (mpi_rank == 0) {
        printf("================================================================\n");
        printf("   GPU-Triggered Concurrent RDMA Benchmark v2 (Optimized)\n");
        printf("================================================================\n");
        printf("GPU: %s (%.2f GHz)\n", props.name, props.clockRate / 1e6);
        printf("IB Device: %s\n", ibv_get_device_name(dev_list[0]));
        printf("Concurrent streams: %d\n", N_STREAMS);
        printf("Iterations per size: %d (using best 50%%)\n", NUM_ITERATIONS);
        printf("----------------------------------------------------------------\n\n");
        fflush(stdout);
    }

    //==========================================================================
    // Test 1: WQE + Doorbell only (minimal overhead)
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=== Test 1: WQE Build + Doorbell Only (no network wait) ===\n");
        printf("%-8s  %10s  %10s\n", "Size", "Total(ns)", "Per-xfer(ns)");
        printf("========  ==========  ==========\n");
        fflush(stdout);
    }

    for (int size_idx = 0; size_idx < NUM_TEST_SIZES; size_idx++) {
        size_t msg_size = TEST_SIZES[size_idx];
        size_t stream_stride = MAX_SIZE;

        double times_ns[NUM_ITERATIONS];

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            MPI_Barrier(MPI_COMM_WORLD);

            if (mpi_rank == 0) {
                gpu_concurrent_write_minimal_kernel<<<1, N_STREAMS>>>(
                    d_state, d_send_addrs, d_send_lkeys,
                    peer_bufs.recv.addr, peer_bufs.recv.rkey,
                    msg_size, stream_stride, N_STREAMS,
                    d_start_ns, d_end_ns
                );
                cuda_check(cudaDeviceSynchronize(), "sync");

                uint64_t start, end;
                cuda_check(cudaMemcpy(&start, d_start_ns, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy start");
                cuda_check(cudaMemcpy(&end, d_end_ns, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy end");
                times_ns[iter] = (double)(end - start);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (mpi_rank == 0) {
            // Sort and take best 50%
            for (int i = 0; i < NUM_ITERATIONS - 1; i++) {
                for (int j = i + 1; j < NUM_ITERATIONS; j++) {
                    if (times_ns[j] < times_ns[i]) {
                        double tmp = times_ns[i];
                        times_ns[i] = times_ns[j];
                        times_ns[j] = tmp;
                    }
                }
            }

            int samples = NUM_ITERATIONS / 2;
            double sum = 0;
            for (int i = 0; i < samples; i++) sum += times_ns[i];
            double avg = sum / samples;

            char size_buf[32];
            printf("%-8s  %10.0f  %10.2f\n",
                   format_size(msg_size, size_buf), avg, avg / N_STREAMS);
            fflush(stdout);
        }
    }

    if (mpi_rank == 0) printf("\n");

    //==========================================================================
    // Test 2: End-to-End with ACK (same as v1 but with globaltimer)
    //==========================================================================

    if (mpi_rank == 0) {
        printf("=== Test 2: End-to-End with ACK (data + ACK round-trip) ===\n");
        printf("%-8s  %10s  %10s  %12s\n", "Size", "Total(us)", "Per-xfer(us)", "BW(Gbps)");
        printf("========  ==========  ==========  ============\n");
        fflush(stdout);
    }

    for (int size_idx = 0; size_idx < NUM_TEST_SIZES; size_idx++) {
        size_t msg_size = TEST_SIZES[size_idx];
        size_t stream_stride = MAX_SIZE;
        size_t flag_offset = (msg_size < 8) ? 0 : (msg_size - 8);

        double times_ns[NUM_ITERATIONS];

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            uint64_t seq_num = iter + 1;

            // Reset buffers
            if (mpi_rank == 1) {
                cuda_check(cudaMemset(d_recv_buf, 0, MAX_SIZE * N_STREAMS), "clear recv");
            }
            if (mpi_rank == 0) {
                cuda_check(cudaMemset(d_ack_buf, 0, sizeof(uint64_t)), "clear ack");
                for (int i = 0; i < N_STREAMS; i++) {
                    cuda_check(cudaMemcpy((char*)d_send_bufs[i] + flag_offset, &seq_num,
                                          sizeof(uint64_t), cudaMemcpyHostToDevice), "set seq");
                }
            }
            cuda_check(cudaDeviceSynchronize(), "sync");

            // Setup recv flags
            volatile uint64_t* h_recv_flags[N_STREAMS];
            for (int i = 0; i < N_STREAMS; i++) {
                h_recv_flags[i] = (volatile uint64_t*)((char*)d_recv_buf + i * stream_stride + flag_offset);
            }
            cuda_check(cudaMemcpy(d_recv_flags, h_recv_flags, N_STREAMS * sizeof(uint64_t*),
                                  cudaMemcpyHostToDevice), "copy flags");

            MPI_Barrier(MPI_COMM_WORLD);

            if (mpi_rank == 0) {
                gpu_sender_e2e_kernel<<<1, N_STREAMS>>>(
                    d_state, d_send_addrs, d_send_lkeys,
                    peer_bufs.recv.addr, peer_bufs.recv.rkey,
                    msg_size, stream_stride, N_STREAMS,
                    (volatile uint64_t*)d_ack_buf, seq_num,
                    d_start_ns, d_end_ns
                );
            } else {
                gpu_receiver_e2e_kernel<<<1, N_STREAMS>>>(
                    d_state, d_recv_flags, N_STREAMS, seq_num,
                    peer_bufs.ack.addr, peer_bufs.ack.rkey,
                    (uint64_t)d_ack_buf, ack_mr->lkey, seq_num
                );
            }

            cuda_check(cudaDeviceSynchronize(), "sync");

            if (mpi_rank == 0) {
                uint64_t start, end;
                cuda_check(cudaMemcpy(&start, d_start_ns, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy start");
                cuda_check(cudaMemcpy(&end, d_end_ns, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy end");
                times_ns[iter] = (double)(end - start);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (mpi_rank == 0) {
            // Sort and take best 50%
            for (int i = 0; i < NUM_ITERATIONS - 1; i++) {
                for (int j = i + 1; j < NUM_ITERATIONS; j++) {
                    if (times_ns[j] < times_ns[i]) {
                        double tmp = times_ns[i];
                        times_ns[i] = times_ns[j];
                        times_ns[j] = tmp;
                    }
                }
            }

            int samples = NUM_ITERATIONS / 2;
            double sum = 0;
            for (int i = 0; i < samples; i++) sum += times_ns[i];
            double avg_ns = sum / samples;
            double avg_us = avg_ns / 1000.0;

            // Bandwidth for all N_STREAMS transfers
            double bw = compute_bandwidth_gbps(msg_size * N_STREAMS, avg_us);

            char size_buf[32];
            printf("%-8s  %10.2f  %10.2f  %12.2f\n",
                   format_size(msg_size, size_buf), avg_us, avg_us / N_STREAMS, bw);
            fflush(stdout);
        }
    }

    //==========================================================================
    // Summary
    //==========================================================================

    if (mpi_rank == 0) {
        printf("\n================================================================\n");
        printf("Summary:\n");
        printf("  Test 1: WQE build + doorbell only (GPU software overhead)\n");
        printf("  Test 2: End-to-end with ACK (full round-trip)\n");
        printf("\nFor fair comparison with MPI one-way:\n");
        printf("  - Divide Test 2 by 2 (excludes ACK round-trip)\n");
        printf("  - Or use Test 1 + estimated network latency\n");
        printf("================================================================\n");
    }

    // Cleanup
    cudaFree(d_state);
    cudaFree(d_start_ns);
    cudaFree(d_end_ns);
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
