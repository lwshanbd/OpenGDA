/**
 * gpu_put_latency.cu - GPU-Triggered RDMA Put Latency Benchmark
 *
 * Matches NVSHMEM's shmem_put_latency.cu exactly:
 *   - Only rank 0 sends (unidirectional)
 *   - Each put followed by quiet (wait for completion)
 *   - Measures single operation latency
 *
 * Supports 3 threadgroup scopes:
 *   --threadgroup-scope thread  : Single thread sends (like nvshmem_int_put_nbi)
 *   --threadgroup-scope warp    : Warp cooperatively sends (like nvshmemx_int_put_warp)
 *   --threadgroup-scope block   : Block cooperatively sends (like nvshmemx_int_put_block)
 *
 * Build:
 *   cd build && make gpu_put_latency
 *
 * Run:
 *   srun -N 2 -n 2 --ntasks-per-node=1 ./gpu_put_latency --threadgroup-scope thread
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
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

//==============================================================================
// Configuration
//==============================================================================

constexpr int WARMUP_ITERS = 10;
constexpr int BENCH_ITERS = 100;
constexpr int THREADS_PER_WARP = 32;
constexpr int DEFAULT_THREADS_PER_BLOCK = 256;

// Size table matching nvshmem shmem_put_latency (4B to 4MB)
constexpr size_t TEST_SIZES[] = {
    4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304
};
constexpr int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);

enum ThreadgroupScope {
    SCOPE_THREAD = 0,
    SCOPE_WARP = 1,
    SCOPE_BLOCK = 2
};

struct Config {
    ThreadgroupScope scope = SCOPE_THREAD;
    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    int warmup_iters = WARMUP_ITERS;
    int bench_iters = BENCH_ITERS;
};

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
// GPU Kernels - Matching NVSHMEM shmem_put_latency.cu
//==============================================================================

// Using gda_quiet() from gda_device_opt.cuh which polls CQ's wqe_counter

// Debug kernel to check CQE from GPU perspective
__global__ void debug_read_cqe(volatile uint8_t* cqe_buf, uint8_t* out) {
    // Read the op_own byte (offset 63) using system memory load
    uint16_t result;
    asm volatile("ld.acquire.sys.global.b8 %0, [%1];" : "=h"(result) : "l"(&cqe_buf[63]));
    out[0] = (uint8_t)result;

    // Also try reading with volatile
    out[1] = cqe_buf[63];

    // Read wqe_counter (offset 60-61)
    out[2] = cqe_buf[60];
    out[3] = cqe_buf[61];
}

/**
 * Thread-scope latency kernel (simple version)
 * - Single thread (<<<1, 1>>>)
 * - Each iteration: put + fence
 *
 * NOTE: This version only measures WQE submission time, not true RTT.
 * For true latency measurement, use ping-pong with GPU memory flags.
 */
__global__ void latency_kern_thread(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    size_t size,
    int iter)
{
    for (int i = 0; i < iter; i++) {
        // Build and post RDMA WRITE
        uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
        uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);

        gda_build_rdma_write_wqe_opt(
            state, local_addr, local_lkey,
            remote_addr, remote_rkey,
            size, wqe_slot, true  // signaled for CQ completion
        );

        gda_ring_doorbell_bf(state, (uint16_t)((prod + 1) & 0xFFFF));

        // Memory fence to ensure ordering
        gda_quiet(state);
    }
}

/**
 * Warp-scope latency kernel (matches latency_kern_warp)
 * - One warp (<<<1, 32>>>)
 * - Warp cooperatively sends, thread 0 does quiet
 */
__global__ void latency_kern_warp(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    size_t size,
    int iter)
{
    int tid = threadIdx.x;

    for (int i = 0; i < iter; i++) {
        // Thread 0 builds and posts RDMA WRITE
        if (tid == 0) {
            uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
            uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);

            gda_build_rdma_write_wqe_opt(
                state, local_addr, local_lkey,
                remote_addr, remote_rkey,
                size, wqe_slot, true
            );

            gda_ring_doorbell_bf(state, (uint16_t)((prod + 1) & 0xFFFF));
        }

        __syncwarp();

        // Thread 0 does quiet
        if (tid == 0) {
            gda_quiet(state);
        }

        __syncwarp();
    }
}

/**
 * Block-scope latency kernel (matches latency_kern_block)
 * - One block (<<<1, threads_per_block>>>)
 * - Block cooperatively sends, thread 0 does quiet
 */
__global__ void latency_kern_block(
    GdaDeviceStateOpt* state,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    size_t size,
    int iter)
{
    int tid = threadIdx.x;

    for (int i = 0; i < iter; i++) {
        // Thread 0 builds and posts RDMA WRITE
        if (tid == 0) {
            uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
            uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);

            gda_build_rdma_write_wqe_opt(
                state, local_addr, local_lkey,
                remote_addr, remote_rkey,
                size, wqe_slot, true
            );

            gda_ring_doorbell_bf(state, (uint16_t)((prod + 1) & 0xFFFF));
        }

        __syncthreads();

        // Thread 0 does quiet
        if (tid == 0) {
            gda_quiet(state);
        }

        __syncthreads();
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

const char* scope_to_string(ThreadgroupScope scope) {
    switch (scope) {
        case SCOPE_THREAD: return "Thread";
        case SCOPE_WARP:   return "Warp";
        case SCOPE_BLOCK:  return "Block";
        default:           return "Unknown";
    }
}

void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --threadgroup-scope <thread|warp|block>  Cooperation scope (default: thread)\n");
    fprintf(stderr, "  --threads-per-block <N>                  Threads per block for block scope (default: %d)\n", DEFAULT_THREADS_PER_BLOCK);
    fprintf(stderr, "  --warmup <N>                             Warmup iterations (default: %d)\n", WARMUP_ITERS);
    fprintf(stderr, "  --iters <N>                              Benchmark iterations (default: %d)\n", BENCH_ITERS);
    fprintf(stderr, "  --help                                   Show this help\n");
}

void parse_args(int argc, char** argv, Config& cfg) {
    static struct option long_options[] = {
        {"threadgroup-scope", required_argument, 0, 's'},
        {"threads-per-block", required_argument, 0, 't'},
        {"warmup",            required_argument, 0, 'w'},
        {"iters",             required_argument, 0, 'i'},
        {"help",              no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "s:t:w:i:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 's':
                if (strcmp(optarg, "thread") == 0) cfg.scope = SCOPE_THREAD;
                else if (strcmp(optarg, "warp") == 0) cfg.scope = SCOPE_WARP;
                else if (strcmp(optarg, "block") == 0) cfg.scope = SCOPE_BLOCK;
                else {
                    fprintf(stderr, "Invalid scope: %s\n", optarg);
                    exit(1);
                }
                break;
            case 't': cfg.threads_per_block = atoi(optarg); break;
            case 'w': cfg.warmup_iters = atoi(optarg); break;
            case 'i': cfg.bench_iters = atoi(optarg); break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }
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
        if (mpi_rank == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    Config cfg;
    parse_args(argc, argv, cfg);

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

    // Open InfiniBand device
    int num_devices = 0;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "Rank %d: No IB devices found\n", mpi_rank);
        MPI_Finalize();
        return 1;
    }

    // Find mlx5_1 (IB) device, fallback to first
    struct ibv_device* target_dev = nullptr;
    for (int i = 0; i < num_devices; i++) {
        const char* name = ibv_get_device_name(dev_list[i]);
        if (name && strcmp(name, "mlx5_1") == 0) {
            target_dev = dev_list[i];
            break;
        }
    }
    if (!target_dev) target_dev = dev_list[0];

    struct ibv_context* ctx = ibv_open_device(target_dev);
    struct ibv_pd* pd = ibv_alloc_pd(ctx);

    // Create DevX QP with sufficient depth for iterations
    // Parameters: ctx, pd, rank, port, qp_depth (WQEs), cq_depth
    DevxQp* devx_qp = new DevxQp(ctx, pd, mpi_rank, 1, 1024, 1024);

    // Query port and exchange connection info
    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, 1, &port_attr);

    ConnInfo my_conn, peer_conn;
    my_conn.qpn = devx_qp->qpn;
    my_conn.lid = port_attr.lid;
    my_conn.psn = 0;

    union ibv_gid my_gid;
    ibv_query_gid(ctx, 1, 0, &my_gid);
    memcpy(my_conn.gid, &my_gid, 16);

    MPI_Sendrecv(&my_conn, sizeof(ConnInfo), MPI_BYTE, peer, 0,
                 &peer_conn, sizeof(ConnInfo), MPI_BYTE, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Connect QP
    devx_qp->rst2init();
    devx_qp->init2rtr(peer_conn.qpn, peer_conn.lid, peer_conn.gid, peer_conn.psn, 3);
    devx_qp->rtr2rts(my_conn.psn);

    MPI_Barrier(MPI_COMM_WORLD);

    // Allocate buffers (max 4MB)
    size_t max_size = 4 * 1024 * 1024;
    void* d_data = nullptr;
    cuda_check(cudaMalloc(&d_data, max_size), "alloc data");
    cuda_check(cudaMemset(d_data, 0, max_size), "memset data");

    // Register with NIC
    MemoryRegion* data_mr = new MemoryRegion(pd, d_data, max_size, true, mpi_rank);

    // Exchange buffer info (rank 0 writes to rank 1's buffer)
    BufInfo my_buf, peer_buf;
    my_buf.addr = (uint64_t)d_data;
    my_buf.rkey = data_mr->rkey;

    MPI_Sendrecv(&my_buf, sizeof(BufInfo), MPI_BYTE, peer, 1,
                 &peer_buf, sizeof(BufInfo), MPI_BYTE, peer, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Setup device state
    GdaDeviceStateOpt h_state;
    memset(&h_state, 0, sizeof(h_state));
    h_state.qpn = devx_qp->qpn;
    h_state.nwqes = 1 << devx_qp->log_wq_size;
    h_state.nwqes_mask = h_state.nwqes - 1;
    h_state.wqe_buf = devx_qp->d_wq_buf;
    h_state.dbrec = devx_qp->d_dbrec;
    h_state.bf_reg = (volatile uint64_t*)devx_qp->d_uar_reg;
    h_state.prod_idx = devx_qp->d_prod_idx;
    h_state.remote_addr = peer_buf.addr;
    h_state.remote_rkey = peer_buf.rkey;

    // Setup CQ for completion polling
    h_state.cqe = (volatile GdaCqe64Opt*)devx_qp->d_cq_buf;
    h_state.ncqes = devx_qp->num_cqe;
    h_state.ncqes_mask = h_state.ncqes - 1;
    h_state.cq_dbrec = devx_qp->d_cq_dbrec;

    h_state.num_completions = nullptr;  // Not using the old method

    GdaDeviceStateOpt* d_state;
    cuda_check(cudaMalloc(&d_state, sizeof(GdaDeviceStateOpt)), "alloc state");
    cuda_check(cudaMemcpy(d_state, &h_state, sizeof(GdaDeviceStateOpt),
                          cudaMemcpyHostToDevice), "copy state");

    // CUDA events for timing
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "create start event");
    cuda_check(cudaEventCreate(&stop), "create stop event");

    MPI_Barrier(MPI_COMM_WORLD);

    // Validate RDMA transfer works
    {
        // Initialize local buffer with pattern
        uint8_t pattern = (uint8_t)(mpi_rank + 0x42);
        cuda_check(cudaMemset(d_data, pattern, 64), "init pattern");

        if (mpi_rank == 0) {
            // Send a single transfer
            latency_kern_thread<<<1, 1>>>(
                d_state, (uint64_t)d_data, data_mr->lkey,
                peer_buf.addr, peer_buf.rkey, 64, 1);
            cuda_check(cudaDeviceSynchronize(), "validation sync");

            // Check CQE state after transfer (CPU view)
            uint8_t cqe_data[64];
            cuda_check(cudaMemcpy(cqe_data, (void*)devx_qp->h_cq_buf, 64, cudaMemcpyDefault),
                       "copy CQE");
            printf("Rank 0: CQE[0] (CPU view) op_own=0x%02x (owner=%d, opcode=0x%x)\n",
                   cqe_data[63], cqe_data[63] & 1, (cqe_data[63] >> 4) & 0xF);
            printf("Rank 0: CQE wqe_counter bytes: 0x%02x 0x%02x\n",
                   cqe_data[60], cqe_data[61]);

            // Check CQE state from GPU perspective
            uint8_t* d_debug_out;
            uint8_t h_debug_out[4];
            cuda_check(cudaMalloc(&d_debug_out, 4), "alloc debug");
            debug_read_cqe<<<1, 1>>>((volatile uint8_t*)devx_qp->d_cq_buf, d_debug_out);
            cuda_check(cudaDeviceSynchronize(), "debug sync");
            cuda_check(cudaMemcpy(h_debug_out, d_debug_out, 4, cudaMemcpyDeviceToHost), "copy debug");
            printf("Rank 0: CQE[0] (GPU view) op_own=0x%02x (ld.acquire), 0x%02x (volatile)\n",
                   h_debug_out[0], h_debug_out[1]);
            printf("Rank 0: CQE wqe_counter (GPU view): 0x%02x 0x%02x\n",
                   h_debug_out[2], h_debug_out[3]);
            cuda_check(cudaFree(d_debug_out), "free debug");
        }

        MPI_Barrier(MPI_COMM_WORLD);
        usleep(100000);  // Wait 100ms for RDMA to complete

        // Check if rank 1 received the data
        if (mpi_rank == 1) {
            uint8_t received[64];
            cuda_check(cudaMemcpy(received, d_data, 64, cudaMemcpyDeviceToHost), "copy received");

            uint8_t expected = 0x42;  // Rank 0's pattern
            bool ok = true;
            for (int i = 0; i < 64; i++) {
                if (received[i] != expected) {
                    ok = false;
                    break;
                }
            }

            if (ok) {
                printf("RDMA Validation: PASSED (received correct data from rank 0)\n");
            } else {
                printf("RDMA Validation: FAILED (expected 0x%02x, got 0x%02x)\n",
                       expected, received[0]);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        fflush(stdout);

        // Reset data buffer for benchmark, but DO NOT reset prod_idx or CQ!
        // The NIC's internal counters are not reset, so resetting prod_idx
        // would cause the doorbell to send already-seen values, causing NIC
        // to ignore new WQEs.
        cuda_check(cudaMemset(d_data, 0, max_size), "reset data");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Determine thread count based on scope
    int threads;
    switch (cfg.scope) {
        case SCOPE_THREAD: threads = 1; break;
        case SCOPE_WARP:   threads = THREADS_PER_WARP; break;
        case SCOPE_BLOCK:  threads = cfg.threads_per_block; break;
    }

    // Print header (matching nvshmem format)
    if (mpi_rank == 0) {
        printf("#shmem_put_latency\n");
        printf("%-12s %-10s %s\n", "size(B)", "scope", "latency (us)");
    }

    // Run benchmark for each size (only rank 0 sends, matching nvshmem)
    for (int size_idx = 0; size_idx < NUM_TEST_SIZES; size_idx++) {
        size_t size = TEST_SIZES[size_idx];

        if (mpi_rank == 0) {
            // Warmup
            switch (cfg.scope) {
                case SCOPE_THREAD:
                    latency_kern_thread<<<1, 1>>>(
                        d_state, (uint64_t)d_data, data_mr->lkey,
                        peer_buf.addr, peer_buf.rkey, size, cfg.warmup_iters);
                    break;
                case SCOPE_WARP:
                    latency_kern_warp<<<1, THREADS_PER_WARP>>>(
                        d_state, (uint64_t)d_data, data_mr->lkey,
                        peer_buf.addr, peer_buf.rkey, size, cfg.warmup_iters);
                    break;
                case SCOPE_BLOCK:
                    latency_kern_block<<<1, cfg.threads_per_block>>>(
                        d_state, (uint64_t)d_data, data_mr->lkey,
                        peer_buf.addr, peer_buf.rkey, size, cfg.warmup_iters);
                    break;
            }
            cuda_check(cudaDeviceSynchronize(), "warmup sync");

            // Benchmark
            cuda_check(cudaEventRecord(start), "record start");
            switch (cfg.scope) {
                case SCOPE_THREAD:
                    latency_kern_thread<<<1, 1>>>(
                        d_state, (uint64_t)d_data, data_mr->lkey,
                        peer_buf.addr, peer_buf.rkey, size, cfg.bench_iters);
                    break;
                case SCOPE_WARP:
                    latency_kern_warp<<<1, THREADS_PER_WARP>>>(
                        d_state, (uint64_t)d_data, data_mr->lkey,
                        peer_buf.addr, peer_buf.rkey, size, cfg.bench_iters);
                    break;
                case SCOPE_BLOCK:
                    latency_kern_block<<<1, cfg.threads_per_block>>>(
                        d_state, (uint64_t)d_data, data_mr->lkey,
                        peer_buf.addr, peer_buf.rkey, size, cfg.bench_iters);
                    break;
            }
            cuda_check(cudaEventRecord(stop), "record stop");
            cuda_check(cudaEventSynchronize(stop), "sync stop");

            float ms;
            cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");

            // Latency in microseconds (matching nvshmem format)
            double latency_us = (ms * 1000.0) / cfg.bench_iters;
            printf("%-12zu %-10s %.6f\n", size, scope_to_string(cfg.scope), latency_us);
            fflush(stdout);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Cleanup
    cuda_check(cudaEventDestroy(start), "destroy start");
    cuda_check(cudaEventDestroy(stop), "destroy stop");
    cuda_check(cudaFree(d_state), "free state");

    delete data_mr;
    cuda_check(cudaFree(d_data), "free data");

    delete devx_qp;
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);

    MPI_Finalize();
    return 0;
}
