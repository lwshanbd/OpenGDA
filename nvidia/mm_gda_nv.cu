/**
 * mm_gda_nv.cu - Distributed Matrix Multiplication with CUDA + GPU-Initiated RDMA
 *
 * Uses CUDA for GPU computation and GPU-triggered RDMA (via MLX5 DevX) for
 * inter-rank communication. The GPU directly builds WQEs and rings doorbell
 * to trigger RDMA operations without CPU intervention.
 *
 * Based on minimal/mm_gda_minimal.cpp but using nvidia/ GPU-initiated primitives.
 *
 * Build:
 *   mkdir -p build && cd build && cmake .. && make mm_gda_nv
 *
 * Run:
 *   mpirun -np <npes> ./mm_gda_nv [matrix_size]
 */

#include <iostream>
#include <ctime>
#include <cstring>
#include <unistd.h>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include "mpi_bootstrap.hpp"
#include "memory_region.hpp"
#include "mlx5_devx_qp.hpp"
#include "gda_device_opt.cuh"

using namespace opengda;

// =============================================================================
// Utility macros and functions
// =============================================================================

#define CUDA_CHECK(cmd) do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

void print_matrix(const float* mat, const int Is, const int Js)
{
    for (int i = 0; i < Is; i++) {
        for (int j = 0; j < Js; j++)
            std::cout << mat[i * Js + j] << ' ';
        std::cout << '\n';
    }
}

float timediff_us(const timespec& t_start, const timespec& t_end)
{
    return (t_end.tv_sec - t_start.tv_sec) * 1.0e6 + (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3;
}

// Connection info for exchange
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

// =============================================================================
// CUDA kernel for matrix stripe multiplication
// =============================================================================

/**
 * Computes Cs += As * Bs where:
 *   As: Ns x N (horizontal stripe of A)
 *   Bs: N x Ns (vertical stripe of B, stored as N*Ns)
 *   Cs: Ns x N (horizontal stripe of C, we write to col_offset..col_offset+Ns)
 */
__global__ void matmul_stripe_kernel(
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    float* __restrict__ Cs,
    int N,
    int Ns,
    int col_offset)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (k < N && j < Ns) {
        float b_kj = Bs[k * Ns + j];
        for (int i = 0; i < Ns; i++) {
            atomicAdd(&Cs[i * N + col_offset + j], As[i * N + k] * b_kj);
        }
    }
}

// =============================================================================
// GPU kernel for GPU-triggered RDMA write operation
// =============================================================================

/**
 * GPU kernel that builds WQE and rings doorbell to trigger RDMA write.
 * This is truly GPU-initiated - no CPU involvement.
 *
 * @param state      GPU-accessible device state
 * @param local_addr Source buffer address
 * @param local_lkey Source buffer lkey
 * @param remote_addr Destination buffer address on remote peer
 * @param remote_rkey Destination buffer rkey
 * @param size       Transfer size in bytes
 * @param signaled   Whether to generate completion event
 */
__global__ void gpu_rdma_write_kernel(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    bool signaled)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Get current producer index (this is where we write the WQE)
        uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
        uint16_t wqe_idx = (uint16_t)(prod & 0xFFFF);
        uint16_t new_prod = (uint16_t)((prod + 1) & 0xFFFF);

        // Build WQE at slot wqe_idx
        gda_build_rdma_write_wqe_opt(state, local_addr, local_lkey,
                                      remote_addr, remote_rkey, size,
                                      wqe_idx, signaled);

        // Ring doorbell with new producer index
        gda_ring_doorbell_bf(state, new_prod);
    }
}

/**
 * GPU kernel to wait for RDMA completion by polling the CQ.
 */
__global__ void gpu_wait_completion_kernel(
    GdaDeviceStateOpt* state,
    uint64_t expected_wqe_idx)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Poll CQ until the expected WQE completes
        gda_poll_cq_wqe_counter(state, expected_wqe_idx, 100000000);  // 100ms timeout
    }
}

/**
 * GPU kernel that combines write and wait into one kernel
 * (more efficient for overlapping with computation)
 */
__global__ void gpu_rdma_write_and_wait_kernel(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Get current producer index
        uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
        uint16_t wqe_idx = (uint16_t)(prod & 0xFFFF);
        uint16_t new_prod = (uint16_t)((prod + 1) & 0xFFFF);

        // Build WQE (signaled for completion detection)
        gda_build_rdma_write_wqe_opt(state, local_addr, local_lkey,
                                      remote_addr, remote_rkey, size,
                                      wqe_idx, true);

        // Ring doorbell
        gda_ring_doorbell_bf(state, new_prod);

        // Poll CQ for completion
        gda_poll_cq_wqe_counter(state, new_prod, 100000000);
    }
}

// =============================================================================
// Main program
// =============================================================================

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int mype, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    if (npes < 2) {
        if (mype == 0) {
            std::cerr << "This program requires at least 2 processes\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Get local rank for GPU selection
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    // Initialize CUDA
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    int gpu_id = local_rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, gpu_id));

    // Print GPU assignment
    if (mype < 8 || mype == npes - 1) {
        std::cerr << "Rank " << mype << " (local " << local_rank << ") using GPU "
                  << gpu_id << " (" << props.name << ")\n";
    }

    // Neighbors in the ring
    int left_neighbor = (npes + mype - 1) % npes;
    int right_neighbor = (mype + 1) % npes;

    // Matrix size from command line or default
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    N = (N / npes) * npes;  // Make divisible by npes
    const int Ns = N / npes;
    const size_t stripe_size = N * Ns * sizeof(float);

    if (mype == 0) {
        std::cout << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size << " bytes\n";
        std::cout << "Ring pattern: " << npes << " processes\n";
    }

    // =========================================================================
    // Open IB device
    // =========================================================================
    struct ibv_device** dev_list = ibv_get_device_list(nullptr);
    if (!dev_list || !dev_list[0]) {
        std::cerr << "Rank " << mype << ": No IB devices found\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    struct ibv_context* ctx = ibv_open_device(dev_list[0]);
    if (!ctx) {
        std::cerr << "Rank " << mype << ": Failed to open IB device\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
        std::cerr << "Rank " << mype << ": Failed to allocate PD\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // =========================================================================
    // Create DevX QP for GPU-triggered RDMA (one QP to left neighbor)
    // =========================================================================
    DevxQp* devx_qp = new DevxQp(ctx, pd, mype, 1, 256, 512);

    // Exchange connection info with left neighbor
    ConnInfo my_conn, left_conn;
    my_conn.qpn = devx_qp->qpn;
    my_conn.lid = 0;
    my_conn.psn = 0;

    union ibv_gid my_gid;
    ibv_query_gid(ctx, 1, 1, &my_gid);
    memcpy(my_conn.gid, &my_gid, 16);

    // Each rank sends to left, receives from right
    MPI_Sendrecv(&my_conn, sizeof(ConnInfo), MPI_BYTE, left_neighbor, 0,
                 &left_conn, sizeof(ConnInfo), MPI_BYTE, right_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Connect QP to left neighbor
    devx_qp->rst2init();
    devx_qp->init2rtr(left_conn.qpn, left_conn.lid, left_conn.gid, left_conn.psn, 3);
    devx_qp->rtr2rts(my_conn.psn);

    MPI_Barrier(MPI_COMM_WORLD);

    if (mype == 0) {
        std::cout << "QP connections established\n";
    }

    // =========================================================================
    // Allocate host arrays for initialization
    // =========================================================================
    auto h_As = new float[N * Ns];
    auto h_Bs = new float[N * Ns];
    auto h_Cs = new float[N * Ns];

    // Initialize the matrices on host
    for (int i = 0; i < N * Ns; i++) {
        h_As[i] = (i + mype) % 11 + 7;
        h_Bs[i] = (i + mype) % 13 + 5;
        h_Cs[i] = 0;
    }

    // =========================================================================
    // Allocate device arrays - double buffering for B
    // =========================================================================
    float *d_As, *d_Cs;
    float *d_B[2];  // Double buffer for B stripe
    CUDA_CHECK(cudaMalloc(&d_As, stripe_size));
    CUDA_CHECK(cudaMalloc(&d_Cs, stripe_size));
    CUDA_CHECK(cudaMalloc(&d_B[0], stripe_size));
    CUDA_CHECK(cudaMalloc(&d_B[1], stripe_size));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_As, h_As, stripe_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B[0], h_Bs, stripe_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Cs, h_Cs, stripe_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_B[1], 0, stripe_size));

    // =========================================================================
    // Register memory regions for RDMA
    // =========================================================================
    MemoryRegion* mr_B0 = new MemoryRegion(pd, d_B[0], stripe_size, true, mype);
    MemoryRegion* mr_B1 = new MemoryRegion(pd, d_B[1], stripe_size, true, mype);
    MemoryRegion* mr_B[2] = {mr_B0, mr_B1};

    // Exchange buffer info with neighbors
    // I write to left neighbor's buffer, so I need left's info
    // Right neighbor writes to my buffer, so right needs my info
    BufInfo my_bufs[2], left_bufs[2];
    my_bufs[0].addr = (uint64_t)d_B[0];
    my_bufs[0].rkey = mr_B0->rkey;
    my_bufs[1].addr = (uint64_t)d_B[1];
    my_bufs[1].rkey = mr_B1->rkey;

    MPI_Sendrecv(my_bufs, sizeof(my_bufs), MPI_BYTE, left_neighbor, 1,
                 left_bufs, sizeof(left_bufs), MPI_BYTE, right_neighbor, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // =========================================================================
    // Setup GPU device state for GPU-triggered RDMA
    // =========================================================================
    GdaDeviceStateOpt h_state;
    memset(&h_state, 0, sizeof(h_state));
    h_state.qpn = devx_qp->qpn;
    h_state.nwqes = 1 << devx_qp->log_wq_size;
    h_state.nwqes_mask = h_state.nwqes - 1;
    h_state.wqe_buf = devx_qp->d_wq_buf;
    h_state.dbrec = devx_qp->d_dbrec;
    h_state.bf_reg = (volatile uint64_t*)devx_qp->d_uar_reg;
    h_state.prod_idx = devx_qp->d_prod_idx;
    h_state.batch_size = 1;
    h_state.batch_mask = 0;

    // Setup CQ for completion polling
    h_state.cqe = (volatile GdaCqe64Opt*)devx_qp->d_cq_buf;
    h_state.ncqes = devx_qp->num_cqe;
    h_state.ncqes_mask = h_state.ncqes - 1;
    h_state.cq_dbrec = devx_qp->d_cq_dbrec;

    GdaDeviceStateOpt* d_state;
    CUDA_CHECK(cudaMalloc(&d_state, sizeof(GdaDeviceStateOpt)));
    CUDA_CHECK(cudaMemcpy(d_state, &h_state, sizeof(GdaDeviceStateOpt), cudaMemcpyHostToDevice));

    timespec t0, t1;

    // Make sure all the stripes are initialized
    MPI_Barrier(MPI_COMM_WORLD);

    // Kernel launch configuration for matmul
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (Ns + blockDim.y - 1) / blockDim.y);

    // =========================================================================
    // Warm-up: run one iteration to eliminate first-launch overhead
    // =========================================================================
    {
        // Warmup GPU kernel
        int col_offset = mype * Ns;
        matmul_stripe_kernel<<<gridDim, blockDim>>>(d_As, d_B[0], d_Cs, N, Ns, col_offset);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Reset Cs to zero after warm-up
        CUDA_CHECK(cudaMemset(d_Cs, 0, stripe_size));

        // Warmup RDMA: do one GPU-triggered transfer
        // Note: we write to left neighbor's buf[1]
        gpu_rdma_write_and_wait_kernel<<<1, 1>>>(
            d_state,
            (uint64_t)d_B[0], mr_B[0]->lkey,
            left_bufs[1].addr, left_bufs[1].rkey,
            (uint32_t)stripe_size
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reset producer index for main loop
        *(devx_qp->h_prod_idx) = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Run 10 iterations, skip first 3 (warmup), average last 7
    // =========================================================================
    constexpr int TOTAL_RUNS = 10;
    constexpr int WARMUP_RUNS = 3;
    double times[TOTAL_RUNS];

    for (int run = 0; run < TOTAL_RUNS; run++) {
        // Reset Cs to zero for this run
        CUDA_CHECK(cudaMemset(d_Cs, 0, stripe_size));

        // Restore B[0] with original data (it gets overwritten during ring)
        CUDA_CHECK(cudaMemcpy(d_B[0], h_Bs, stripe_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B[1], h_Bs, stripe_size, cudaMemcpyHostToDevice));

        // Reset producer index
        *(devx_qp->h_prod_idx) = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

        // Main computation loop with ring communication
        for (int s = 0; s < npes; s++) {
            const int block_num = (mype + s) % npes;
            const int cur_buf = s % 2;
            const int next_buf = (s + 1) % 2;

            // Start GPU-triggered RDMA write (non-blocking)
            // Send my current B buffer to left neighbor's next buffer
            gpu_rdma_write_kernel<<<1, 1>>>(
                d_state,
                (uint64_t)d_B[cur_buf], mr_B[cur_buf]->lkey,
                left_bufs[next_buf].addr, left_bufs[next_buf].rkey,
                (uint32_t)stripe_size,
                (s == npes - 1)  // Signal on last iteration for completion
            );

            // Compute while communication is in flight
            int col_offset = block_num * Ns;
            matmul_stripe_kernel<<<gridDim, blockDim>>>(
                d_As, d_B[cur_buf], d_Cs, N, Ns, col_offset
            );

            // Synchronize GPU to ensure compute is done
            CUDA_CHECK(cudaDeviceSynchronize());

            // MPI barrier to synchronize all ranks
            // This ensures the RDMA write from right neighbor has arrived
            MPI_Barrier(MPI_COMM_WORLD);
        }

        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        times[run] = timediff_us(t0, t1);

        if (mype == 0) {
            std::cout << "Run " << run << ": " << times[run] << " us"
                      << (run < WARMUP_RUNS ? " (warmup)" : "") << "\n";
        }
    }

    // Calculate average of last 7 runs
    double sum = 0;
    for (int i = WARMUP_RUNS; i < TOTAL_RUNS; i++) {
        sum += times[i];
    }
    double avg = sum / (TOTAL_RUNS - WARMUP_RUNS);

    if (mype == 0) {
        std::cout << "GPU-triggered RDMA + CUDA average (runs " << WARMUP_RUNS
                  << "-" << (TOTAL_RUNS-1) << "): " << avg << " us\n";
    }

    // =========================================================================
    // Copy result back to host
    // =========================================================================
    CUDA_CHECK(cudaMemcpy(h_Cs, d_Cs, stripe_size, cudaMemcpyDeviceToHost));

    // =========================================================================
    // Print the matrix product (for small matrices)
    // =========================================================================
    if (N < 32) {
        if (mype == 0) {
            auto C = new float[N * N];

            // Copy rank 0's stripe
            for (int i = 0; i < Ns * N; i++)
                C[i] = h_Cs[i];

            // Get other stripes via MPI
            for (int r = 1; r < npes; r++) {
                MPI_Recv(C + r * Ns * N, Ns * N, MPI_FLOAT, r, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            print_matrix(C, N, N);
            delete[] C;
        } else {
            MPI_Send(h_Cs, Ns * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Cleanup
    // =========================================================================
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_B[1]));
    CUDA_CHECK(cudaFree(d_B[0]));
    CUDA_CHECK(cudaFree(d_Cs));
    CUDA_CHECK(cudaFree(d_As));

    delete mr_B1;
    delete mr_B0;
    delete devx_qp;

    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);

    delete[] h_Cs;
    delete[] h_Bs;
    delete[] h_As;

    MPI_Finalize();
    return 0;
}
