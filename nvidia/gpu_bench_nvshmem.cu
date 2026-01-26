/**
 * gpu_bench_nvshmem.cu - Benchmark using nvshmem-style optimizations
 *
 * Compares original implementation vs nvshmem-style optimizations:
 *   1. Three-tier index management
 *   2. Batch boundary detection
 *   3. Lock-free FIFO ordering
 *   4. atomicMax for doorbell coalescing
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include "mpi_bootstrap.hpp"
#include "memory_region.hpp"
#include "mlx5_devx_qp.hpp"
#include "gda_device_nvshmem.cuh"

using namespace opengda;

constexpr int N_STREAMS = 32;
constexpr int NUM_ITERATIONS = 100;
constexpr size_t TEST_SIZES[] = {8, 64, 512, 1024, 4096, 16384, 65536, 262144, 1048576};
constexpr int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);
constexpr size_t MAX_SIZE = 1024 * 1024;

//==============================================================================
// Connection info
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
// Benchmark kernel using nvshmem-style API
//==============================================================================

__global__ void benchmark_nvshmem_style(
    GdaDeviceQpState* qp,
    uint64_t* local_addrs,
    uint32_t* local_lkeys,
    uint64_t remote_base,
    uint32_t remote_rkey,
    size_t msg_size,
    size_t stride,
    int n_streams,
    uint64_t* start_ns,
    uint64_t* end_ns)
{
    int tid = threadIdx.x;

    __shared__ uint64_t base_wqe_idx;

    // Thread 0 reserves all slots
    if (tid == 0) {
        *start_ns = gda_globaltimer();
        base_wqe_idx = gda_reserve_wqe_slots(qp, n_streams);
    }
    __syncthreads();

    // Each thread builds one WQE
    if (tid < n_streams) {
        uint16_t wqe_idx = (uint16_t)((base_wqe_idx + tid) & 0xFFFF);
        uint64_t remote_addr = remote_base + tid * stride;

        gda_build_rdma_write_wqe(
            qp, local_addrs[tid], local_lkeys[tid],
            remote_addr, remote_rkey,
            (uint32_t)msg_size, wqe_idx, 0
        );
    }

    __syncthreads();

    // Thread 0 submits with single doorbell
    if (tid == 0) {
        gda_submit_requests(qp, base_wqe_idx, n_streams);
        gda_membar_system();
        *end_ns = gda_globaltimer();
    }
}

//==============================================================================
// Original-style kernel for comparison
//==============================================================================

// From gda_device_opt.cuh (simplified)
__device__ __forceinline__ void original_build_wqe(
    void* wqe_buf, uint32_t qpn, uint16_t nwqes_mask,
    uint64_t local_addr, uint32_t local_lkey,
    uint64_t remote_addr, uint32_t remote_rkey,
    uint32_t size, uint16_t wqe_idx)
{
    uint16_t idx = wqe_idx & nwqes_mask;
    volatile uint32_t* wqe = (volatile uint32_t*)((uintptr_t)wqe_buf + (idx << 6));

    // Control segment
    wqe[0] = gda_htobe32((wqe_idx << 8) | 0x08);
    wqe[1] = gda_htobe32((qpn << 8) | 3);
    wqe[2] = 0;
    wqe[3] = 0;

    // Remote address
    uint64_t raddr_be = gda_htobe64(remote_addr);
    wqe[4] = (uint32_t)raddr_be;
    wqe[5] = (uint32_t)(raddr_be >> 32);
    wqe[6] = gda_htobe32(remote_rkey);
    wqe[7] = 0;

    // Data segment
    wqe[8] = gda_htobe32(size);
    wqe[9] = gda_htobe32(local_lkey);
    uint64_t laddr_be = gda_htobe64(local_addr);
    wqe[10] = (uint32_t)laddr_be;
    wqe[11] = (uint32_t)(laddr_be >> 32);
}

__device__ __forceinline__ void original_ring_doorbell(
    volatile uint32_t* dbrec, volatile uint64_t* bf_reg,
    uint32_t qpn, uint16_t prod_idx)
{
    gda_membar_system();
    gda_store_release_u32(dbrec, gda_htobe32(prod_idx));

    uint32_t opmod_idx_opcode = gda_htobe32(prod_idx << 8);
    uint32_t qpn_ds = gda_htobe32(qpn << 8);
    uint64_t bf_val = ((uint64_t)qpn_ds << 32) | opmod_idx_opcode;
    gda_store_release_u64(bf_reg, bf_val);
}

__global__ void benchmark_original_style(
    void* wqe_buf, volatile uint32_t* dbrec, volatile uint64_t* bf_reg,
    volatile uint64_t* prod_idx_ptr,
    uint32_t qpn, uint16_t nwqes_mask,
    uint64_t* local_addrs, uint32_t* local_lkeys,
    uint64_t remote_base, uint32_t remote_rkey,
    size_t msg_size, size_t stride, int n_streams,
    uint64_t* start_ns, uint64_t* end_ns)
{
    int tid = threadIdx.x;

    __shared__ uint64_t base_wqe_idx;

    if (tid == 0) {
        *start_ns = gda_globaltimer();
        base_wqe_idx = gda_load_relaxed_u64(prod_idx_ptr);
    }
    __syncthreads();

    if (tid < n_streams) {
        uint16_t wqe_idx = (uint16_t)((base_wqe_idx + tid) & 0xFFFF);
        uint64_t remote_addr = remote_base + tid * stride;

        original_build_wqe(
            wqe_buf, qpn, nwqes_mask,
            local_addrs[tid], local_lkeys[tid],
            remote_addr, remote_rkey,
            (uint32_t)msg_size, wqe_idx
        );
    }

    __syncthreads();

    if (tid == 0) {
        uint16_t new_prod = (uint16_t)((base_wqe_idx + n_streams) & 0xFFFF);
        original_ring_doorbell(dbrec, bf_reg, qpn, new_prod);
        gda_store_relaxed_u64(prod_idx_ptr, base_wqe_idx + n_streams);
        *end_ns = gda_globaltimer();
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
    if (size < 1024) snprintf(buf, 32, "%zuB", size);
    else if (size < 1024*1024) snprintf(buf, 32, "%zuKB", size/1024);
    else snprintf(buf, 32, "%zuMB", size/(1024*1024));
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
        if (mpi_rank == 0) fprintf(stderr, "Requires 2 ranks\n");
        MPI_Finalize();
        return 1;
    }

    int peer = (mpi_rank == 0) ? 1 : 0;

    // GPU setup
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int num_gpus;
    cuda_check(cudaGetDeviceCount(&num_gpus), "get device count");
    cuda_check(cudaSetDevice(local_rank % num_gpus), "set device");

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, local_rank % num_gpus);

    // IB setup
    struct ibv_device** dev_list = ibv_get_device_list(nullptr);
    struct ibv_context* ctx = ibv_open_device(dev_list[0]);
    struct ibv_pd* pd = ibv_alloc_pd(ctx);

    DevxQp* devx_qp = new DevxQp(ctx, pd, mpi_rank, 1, 256, 512);

    // Connect QP
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

    devx_qp->rst2init();
    devx_qp->init2rtr(peer_conn.qpn, peer_conn.lid, peer_conn.gid, peer_conn.psn, 3);
    devx_qp->rtr2rts(my_conn.psn);

    MPI_Barrier(MPI_COMM_WORLD);

    // Allocate buffers
    void* d_send_bufs[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cuda_check(cudaMalloc(&d_send_bufs[i], MAX_SIZE), "alloc send");
        cuda_check(cudaMemset(d_send_bufs[i], 0xAA + i, MAX_SIZE), "memset");
    }

    void* d_recv_buf;
    cuda_check(cudaMalloc(&d_recv_buf, MAX_SIZE * N_STREAMS), "alloc recv");

    // Register memory
    MemoryRegion* send_mrs[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        send_mrs[i] = new MemoryRegion(pd, d_send_bufs[i], MAX_SIZE, true, mpi_rank);
    }
    MemoryRegion* recv_mr = new MemoryRegion(pd, d_recv_buf, MAX_SIZE * N_STREAMS, true, mpi_rank);

    BufInfo my_buf, peer_buf;
    my_buf.addr = (uint64_t)d_recv_buf;
    my_buf.rkey = recv_mr->rkey;

    MPI_Sendrecv(&my_buf, sizeof(BufInfo), MPI_BYTE, peer, 1,
                 &peer_buf, sizeof(BufInfo), MPI_BYTE, peer, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Setup GPU arrays
    uint64_t h_local_addrs[N_STREAMS];
    uint32_t h_local_lkeys[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        h_local_addrs[i] = (uint64_t)d_send_bufs[i];
        h_local_lkeys[i] = send_mrs[i]->lkey;
    }

    uint64_t* d_local_addrs;
    uint32_t* d_local_lkeys;
    cuda_check(cudaMalloc(&d_local_addrs, N_STREAMS * sizeof(uint64_t)), "alloc addrs");
    cuda_check(cudaMalloc(&d_local_lkeys, N_STREAMS * sizeof(uint32_t)), "alloc lkeys");
    cuda_check(cudaMemcpy(d_local_addrs, h_local_addrs, N_STREAMS * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "copy addrs");
    cuda_check(cudaMemcpy(d_local_lkeys, h_local_lkeys, N_STREAMS * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "copy lkeys");

    // Setup nvshmem-style device state
    // Allocate management variables in GPU memory
    uint64_t* d_resv_head;
    uint64_t* d_ready_head;
    int* d_post_send_lock;
    cuda_check(cudaMalloc(&d_resv_head, sizeof(uint64_t)), "alloc resv_head");
    cuda_check(cudaMalloc(&d_ready_head, sizeof(uint64_t)), "alloc ready_head");
    cuda_check(cudaMalloc(&d_post_send_lock, sizeof(int)), "alloc lock");
    cuda_check(cudaMemset(d_resv_head, 0, sizeof(uint64_t)), "init resv_head");
    cuda_check(cudaMemset(d_ready_head, 0, sizeof(uint64_t)), "init ready_head");
    cuda_check(cudaMemset(d_post_send_lock, 0, sizeof(int)), "init lock");

    GdaDeviceQpState h_qp_state;
    memset(&h_qp_state, 0, sizeof(h_qp_state));
    h_qp_state.qpn = devx_qp->qpn;
    h_qp_state.nwqes = 1 << devx_qp->log_wq_size;
    h_qp_state.nwqes_mask = h_qp_state.nwqes - 1;
    h_qp_state.wqe_buf = devx_qp->d_wq_buf;
    h_qp_state.dbrec = devx_qp->d_dbrec;
    h_qp_state.bf_reg = (volatile uint64_t*)devx_qp->d_uar_reg;
    h_qp_state.resv_head = (volatile uint64_t*)d_resv_head;
    h_qp_state.ready_head = (volatile uint64_t*)d_ready_head;
    h_qp_state.prod_idx = devx_qp->d_prod_idx;
    h_qp_state.post_send_lock = (volatile int*)d_post_send_lock;
    h_qp_state.remote_addr = peer_buf.addr;
    h_qp_state.remote_rkey = peer_buf.rkey;
    h_qp_state.batch_size = GDA_DEFAULT_BATCH_SIZE;
    h_qp_state.batch_mask = ~((uint64_t)(GDA_DEFAULT_BATCH_SIZE - 1));

    GdaDeviceQpState* d_qp_state;
    cuda_check(cudaMalloc(&d_qp_state, sizeof(GdaDeviceQpState)), "alloc qp state");
    cuda_check(cudaMemcpy(d_qp_state, &h_qp_state, sizeof(GdaDeviceQpState),
                          cudaMemcpyHostToDevice), "copy qp state");

    // Timing buffers
    uint64_t* d_start_ns;
    uint64_t* d_end_ns;
    cuda_check(cudaMalloc(&d_start_ns, sizeof(uint64_t)), "alloc start");
    cuda_check(cudaMalloc(&d_end_ns, sizeof(uint64_t)), "alloc end");

    MPI_Barrier(MPI_COMM_WORLD);

    // Print header
    if (mpi_rank == 0) {
        printf("================================================================\n");
        printf("   GPU-Triggered RDMA: Original vs nvshmem-style Benchmark\n");
        printf("================================================================\n");
        printf("GPU: %s\n", props.name);
        printf("Concurrent streams: %d\n", N_STREAMS);
        printf("Iterations: %d\n", NUM_ITERATIONS);
        printf("----------------------------------------------------------------\n\n");

        printf("%-8s  %12s  %12s  %10s\n",
               "Size", "Original(ns)", "nvshmem(ns)", "Speedup");
        printf("========  ============  ============  ==========\n");
        fflush(stdout);
    }

    // Run benchmarks
    for (int size_idx = 0; size_idx < NUM_TEST_SIZES; size_idx++) {
        size_t msg_size = TEST_SIZES[size_idx];
        size_t stride = MAX_SIZE;

        double orig_times[NUM_ITERATIONS];
        double nvshmem_times[NUM_ITERATIONS];

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Reset indices
            cuda_check(cudaMemset(d_resv_head, 0, sizeof(uint64_t)), "reset");
            cuda_check(cudaMemset(d_ready_head, 0, sizeof(uint64_t)), "reset");
            cuda_check(cudaMemset((void*)devx_qp->d_prod_idx, 0, sizeof(uint64_t)), "reset");
            cuda_check(cudaMemset(d_post_send_lock, 0, sizeof(int)), "reset");

            MPI_Barrier(MPI_COMM_WORLD);

            if (mpi_rank == 0) {
                // Original style
                benchmark_original_style<<<1, N_STREAMS>>>(
                    devx_qp->d_wq_buf, devx_qp->d_dbrec,
                    (volatile uint64_t*)devx_qp->d_uar_reg,
                    devx_qp->d_prod_idx,
                    devx_qp->qpn, h_qp_state.nwqes_mask,
                    d_local_addrs, d_local_lkeys,
                    peer_buf.addr, peer_buf.rkey,
                    msg_size, stride, N_STREAMS,
                    d_start_ns, d_end_ns
                );
                cuda_check(cudaDeviceSynchronize(), "sync orig");

                uint64_t start, end;
                cuda_check(cudaMemcpy(&start, d_start_ns, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy");
                cuda_check(cudaMemcpy(&end, d_end_ns, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy");
                orig_times[iter] = (double)(end - start);

                // Reset for nvshmem style
                cuda_check(cudaMemset(d_resv_head, 0, sizeof(uint64_t)), "reset");
                cuda_check(cudaMemset(d_ready_head, 0, sizeof(uint64_t)), "reset");
                cuda_check(cudaMemset((void*)devx_qp->d_prod_idx, 0, sizeof(uint64_t)), "reset");
                cuda_check(cudaMemset(d_post_send_lock, 0, sizeof(int)), "reset");

                // nvshmem style
                benchmark_nvshmem_style<<<1, N_STREAMS>>>(
                    d_qp_state,
                    d_local_addrs, d_local_lkeys,
                    peer_buf.addr, peer_buf.rkey,
                    msg_size, stride, N_STREAMS,
                    d_start_ns, d_end_ns
                );
                cuda_check(cudaDeviceSynchronize(), "sync nvshmem");

                cuda_check(cudaMemcpy(&start, d_start_ns, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy");
                cuda_check(cudaMemcpy(&end, d_end_ns, sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy");
                nvshmem_times[iter] = (double)(end - start);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Statistics
        if (mpi_rank == 0) {
            // Sort
            for (int i = 0; i < NUM_ITERATIONS - 1; i++) {
                for (int j = i + 1; j < NUM_ITERATIONS; j++) {
                    if (orig_times[j] < orig_times[i]) {
                        double tmp = orig_times[i]; orig_times[i] = orig_times[j]; orig_times[j] = tmp;
                    }
                    if (nvshmem_times[j] < nvshmem_times[i]) {
                        double tmp = nvshmem_times[i]; nvshmem_times[i] = nvshmem_times[j]; nvshmem_times[j] = tmp;
                    }
                }
            }

            int samples = NUM_ITERATIONS / 2;
            double orig_avg = 0, nvshmem_avg = 0;
            for (int i = 0; i < samples; i++) {
                orig_avg += orig_times[i];
                nvshmem_avg += nvshmem_times[i];
            }
            orig_avg /= samples;
            nvshmem_avg /= samples;

            char size_buf[32];
            printf("%-8s  %12.0f  %12.0f  %10.2fx\n",
                   format_size(msg_size, size_buf),
                   orig_avg, nvshmem_avg,
                   orig_avg / nvshmem_avg);
            fflush(stdout);
        }
    }

    if (mpi_rank == 0) {
        printf("\n================================================================\n");
        printf("Note: Both measurements are WQE build + doorbell only.\n");
        printf("nvshmem-style adds overhead for index management but provides:\n");
        printf("  - Lock-free FIFO ordering for concurrent submissions\n");
        printf("  - Batching for doorbell coalescing\n");
        printf("  - atomicMax for redundant doorbell elimination\n");
        printf("================================================================\n");
    }

    // Cleanup
    cudaFree(d_qp_state);
    cudaFree(d_resv_head);
    cudaFree(d_ready_head);
    cudaFree(d_post_send_lock);
    cudaFree(d_start_ns);
    cudaFree(d_end_ns);
    cudaFree(d_local_addrs);
    cudaFree(d_local_lkeys);

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
