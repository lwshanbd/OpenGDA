/**
 * gpu_put_bw.cu - GPU-Triggered RDMA Put Bandwidth Benchmark
 *
 * Similar to NVSHMEM's shmem_put_bw, supports 3 threadgroup scopes:
 *   --threadgroup-scope thread  : Each thread independently sends one chunk
 *   --threadgroup-scope warp    : Each warp cooperatively sends one chunk
 *   --threadgroup-scope block   : Entire block cooperatively sends one chunk
 *
 * Build:
 *   cd build && make gpu_put_bw
 *
 * Run:
 *   srun -N 2 -n 2 --ntasks-per-node=1 ./gpu_put_bw --threadgroup-scope block
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <getopt.h>
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

//==============================================================================
// Configuration
//==============================================================================

constexpr int DEFAULT_BLOCKS = 1;
constexpr int DEFAULT_THREADS = 256;
constexpr int WARMUP_ITERS = 10;
constexpr int BENCH_ITERS = 100;

constexpr size_t MIN_SIZE = 4;                 // 4B (match nvshmem)
constexpr size_t MAX_SIZE = 4 * 1024 * 1024;   // 4MB (match nvshmem)
constexpr int STEP_FACTOR = 2;

enum ThreadgroupScope {
    SCOPE_THREAD = 0,
    SCOPE_WARP = 1,
    SCOPE_BLOCK = 2
};

// Global config
struct Config {
    ThreadgroupScope scope = SCOPE_BLOCK;
    int num_blocks = DEFAULT_BLOCKS;
    int threads_per_block = DEFAULT_THREADS;
    size_t min_size = MIN_SIZE;
    size_t max_size = MAX_SIZE;
    int warmup_iters = WARMUP_ITERS;
    int bench_iters = BENCH_ITERS;
};

// Size table matching nvshmem shmem_put_latency
constexpr size_t TEST_SIZES[] = {
    4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304
};
constexpr int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);

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
// GPU Kernels - Three threadgroup scopes
//==============================================================================

/**
 * Block-scope: Entire block cooperatively sends one chunk per block.
 * - All threads in block collaborate to build WQEs
 * - Thread 0 of each block handles doorbell
 * - Similar to nvshmemx_double_put_nbi_block()
 */
__global__ void bw_block_kernel(
    GdaDeviceStateOpt* state,
    uint64_t local_base,
    uint32_t local_lkey,
    uint64_t remote_base,
    uint32_t remote_rkey,
    size_t total_len,           // Total data size in bytes
    int iter,
    volatile unsigned int* counter_d)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    size_t chunk_size = total_len / nblocks;
    uint64_t local_addr = local_base + bid * chunk_size;
    uint64_t remote_addr = remote_base + bid * chunk_size;

    for (int i = 0; i < iter; i++) {
        // Thread 0 of each block builds WQE and rings doorbell
        if (tid == 0) {
            uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
            uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);

            gda_build_rdma_write_wqe_opt(
                state, local_addr, local_lkey,
                remote_addr, remote_rkey,
                chunk_size, wqe_slot, false
            );

            gda_ring_doorbell_bf(state, (uint16_t)((prod + 1) & 0xFFFF));
        }

        // Synchronize across blocks using counter
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            unsigned int counter = atomicInc((unsigned int*)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != (unsigned int)(i + 1))
                ;
        }
        __syncthreads();
    }

    // Final quiet: wait for all transfers to complete
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        unsigned int counter = atomicInc((unsigned int*)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (iter + 1) - 1)) {
            // Last block to arrive - poll CQ or just fence
            gda_membar_sys();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != (unsigned int)(iter + 1))
            ;
    }
    __syncthreads();
}

/**
 * Warp-scope: Each warp cooperatively sends one chunk.
 * - Each warp handles its own data chunk
 * - Lane 0 of each warp builds WQE and rings doorbell
 * - Similar to nvshmemx_double_put_nbi_warp()
 */
__global__ void bw_warp_kernel(
    GdaDeviceStateOpt* state,
    uint64_t local_base,
    uint32_t local_lkey,
    uint64_t remote_base,
    uint32_t remote_rkey,
    size_t total_len,
    int iter,
    volatile unsigned int* counter_d)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nwarps_per_block = blockDim.x / 32;
    int warpid = tid / 32;
    int laneid = tid % 32;
    int global_warpid = bid * nwarps_per_block + warpid;
    int total_warps = nblocks * nwarps_per_block;

    size_t chunk_per_block = total_len / nblocks;
    size_t chunk_per_warp = chunk_per_block / nwarps_per_block;
    uint64_t local_addr = local_base + global_warpid * chunk_per_warp;
    uint64_t remote_addr = remote_base + global_warpid * chunk_per_warp;

    for (int i = 0; i < iter; i++) {
        // Lane 0 of each warp builds WQE and rings doorbell
        if (laneid == 0) {
            uint64_t prod = gda_load_relaxed_u64(state->prod_idx);
            // Each warp reserves its slot atomically
            uint64_t my_slot = atomicAdd((unsigned long long*)state->prod_idx, 1);
            uint16_t wqe_slot = (uint16_t)(my_slot & 0xFFFF);

            gda_build_rdma_write_wqe_opt(
                state, local_addr, local_lkey,
                remote_addr, remote_rkey,
                chunk_per_warp, wqe_slot, false
            );

            gda_ring_doorbell_bf(state, (uint16_t)((my_slot + 1) & 0xFFFF));
        }

        // Synchronize across blocks
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            unsigned int counter = atomicInc((unsigned int*)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != (unsigned int)(i + 1))
                ;
        }
        __syncthreads();
    }

    // Final quiet
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        unsigned int counter = atomicInc((unsigned int*)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (iter + 1) - 1)) {
            gda_membar_sys();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != (unsigned int)(iter + 1))
            ;
    }
    __syncthreads();
}

/**
 * Thread-scope: Each thread independently sends one chunk.
 * - Each thread handles its own data chunk
 * - Each thread builds WQE and rings doorbell independently
 * - Similar to nvshmem_double_put_nbi()
 */
__global__ void bw_thread_kernel(
    GdaDeviceStateOpt* state,
    uint64_t local_base,
    uint32_t local_lkey,
    uint64_t remote_base,
    uint32_t remote_rkey,
    size_t total_len,
    int iter,
    volatile unsigned int* counter_d)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nthreads_per_block = blockDim.x;
    int global_tid = bid * nthreads_per_block + tid;
    int total_threads = nblocks * nthreads_per_block;

    size_t chunk_per_block = total_len / nblocks;
    size_t chunk_per_thread = chunk_per_block / nthreads_per_block;
    uint64_t local_addr = local_base + global_tid * chunk_per_thread;
    uint64_t remote_addr = remote_base + global_tid * chunk_per_thread;

    for (int i = 0; i < iter; i++) {
        // Each thread independently builds WQE and rings doorbell
        {
            // Atomically reserve WQE slot
            uint64_t my_slot = atomicAdd((unsigned long long*)state->prod_idx, 1);
            uint16_t wqe_slot = (uint16_t)(my_slot & 0xFFFF);

            gda_build_rdma_write_wqe_opt(
                state, local_addr, local_lkey,
                remote_addr, remote_rkey,
                chunk_per_thread, wqe_slot, false
            );

            gda_ring_doorbell_bf(state, (uint16_t)((my_slot + 1) & 0xFFFF));
        }

        // Synchronize across blocks
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            unsigned int counter = atomicInc((unsigned int*)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != (unsigned int)(i + 1))
                ;
        }
        __syncthreads();
    }

    // Final quiet
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        unsigned int counter = atomicInc((unsigned int*)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (iter + 1) - 1)) {
            gda_membar_sys();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != (unsigned int)(iter + 1))
            ;
    }
    __syncthreads();
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
        case SCOPE_THREAD: return "thread";
        case SCOPE_WARP:   return "warp";
        case SCOPE_BLOCK:  return "block";
        default:           return "unknown";
    }
}

void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --threadgroup-scope <thread|warp|block>  Cooperation scope (default: block)\n");
    fprintf(stderr, "  --num-blocks <N>                         Number of blocks (default: %d)\n", DEFAULT_BLOCKS);
    fprintf(stderr, "  --threads-per-block <N>                  Threads per block (default: %d)\n", DEFAULT_THREADS);
    fprintf(stderr, "  --min-size <bytes>                       Minimum message size (default: %zu)\n", MIN_SIZE);
    fprintf(stderr, "  --max-size <bytes>                       Maximum message size (default: %zu)\n", MAX_SIZE);
    fprintf(stderr, "  --warmup <N>                             Warmup iterations (default: %d)\n", WARMUP_ITERS);
    fprintf(stderr, "  --iters <N>                              Benchmark iterations (default: %d)\n", BENCH_ITERS);
    fprintf(stderr, "  --help                                   Show this help\n");
}

void parse_args(int argc, char** argv, Config& cfg) {
    static struct option long_options[] = {
        {"threadgroup-scope", required_argument, 0, 's'},
        {"num-blocks",        required_argument, 0, 'b'},
        {"threads-per-block", required_argument, 0, 't'},
        {"min-size",          required_argument, 0, 'm'},
        {"max-size",          required_argument, 0, 'M'},
        {"warmup",            required_argument, 0, 'w'},
        {"iters",             required_argument, 0, 'i'},
        {"help",              no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "s:b:t:m:M:w:i:h", long_options, nullptr)) != -1) {
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
            case 'b': cfg.num_blocks = atoi(optarg); break;
            case 't': cfg.threads_per_block = atoi(optarg); break;
            case 'm': cfg.min_size = atol(optarg); break;
            case 'M': cfg.max_size = atol(optarg); break;
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
            fprintf(stderr, "This test requires exactly 2 ranks\n");
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

    // Create DevX QP - need enough WQEs for all threads
    int max_wqes = cfg.num_blocks * cfg.threads_per_block * cfg.bench_iters * 2;
    int log_wq_size = 8;  // 256 WQEs minimum
    while ((1 << log_wq_size) < max_wqes && log_wq_size < 14) log_wq_size++;

    DevxQp* devx_qp = new DevxQp(ctx, pd, mpi_rank, 1, log_wq_size, 512);

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

    // Allocate buffers
    void* d_send_buf = nullptr;
    void* d_recv_buf = nullptr;
    cuda_check(cudaMalloc(&d_send_buf, cfg.max_size), "alloc send");
    cuda_check(cudaMalloc(&d_recv_buf, cfg.max_size), "alloc recv");
    cuda_check(cudaMemset(d_send_buf, 0xAA, cfg.max_size), "memset send");
    cuda_check(cudaMemset(d_recv_buf, 0, cfg.max_size), "memset recv");

    // Register with NIC
    MemoryRegion* send_mr = new MemoryRegion(pd, d_send_buf, cfg.max_size, true, mpi_rank);
    MemoryRegion* recv_mr = new MemoryRegion(pd, d_recv_buf, cfg.max_size, true, mpi_rank);

    // Exchange buffer info
    BufInfo my_buf, peer_buf;
    my_buf.addr = (uint64_t)d_recv_buf;
    my_buf.rkey = recv_mr->rkey;

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

    GdaDeviceStateOpt* d_state;
    cuda_check(cudaMalloc(&d_state, sizeof(GdaDeviceStateOpt)), "alloc state");
    cuda_check(cudaMemcpy(d_state, &h_state, sizeof(GdaDeviceStateOpt),
                          cudaMemcpyHostToDevice), "copy state");

    // Allocate counter for cross-block sync
    unsigned int* d_counter;
    cuda_check(cudaMalloc(&d_counter, sizeof(unsigned int) * 2), "alloc counter");

    // CUDA events for timing
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "create start event");
    cuda_check(cudaEventCreate(&stop), "create stop event");

    MPI_Barrier(MPI_COMM_WORLD);

    // Print header
    if (mpi_rank == 0) {
        printf("================================================================\n");
        printf("   GPU-Triggered RDMA Put Bandwidth Benchmark\n");
        printf("================================================================\n");
        printf("GPU: %s\n", props.name);
        printf("IB Device: %s\n", ibv_get_device_name(target_dev));
        printf("Threadgroup scope: %s\n", scope_to_string(cfg.scope));
        printf("Blocks: %d, Threads/block: %d\n", cfg.num_blocks, cfg.threads_per_block);
        printf("Warmup: %d, Iterations: %d\n", cfg.warmup_iters, cfg.bench_iters);
        printf("----------------------------------------------------------------\n\n");

        int units;
        switch (cfg.scope) {
            case SCOPE_THREAD:
                units = cfg.num_blocks * cfg.threads_per_block;
                printf("Thread-scope: %d threads, each sends size/%d bytes\n", units, units);
                break;
            case SCOPE_WARP:
                units = cfg.num_blocks * (cfg.threads_per_block / 32);
                printf("Warp-scope: %d warps, each sends size/%d bytes\n", units, units);
                break;
            case SCOPE_BLOCK:
                units = cfg.num_blocks;
                printf("Block-scope: %d blocks, each sends size/%d bytes\n", units, units);
                break;
        }
        printf("\n");

        printf("%-12s %12s %12s\n", "Size", "Time (ms)", "BW (GB/s)");
        printf("%-12s %12s %12s\n", "------------", "------------", "------------");
        fflush(stdout);
    }

    // Select kernel based on scope
    typedef void (*bw_kernel_t)(GdaDeviceStateOpt*, uint64_t, uint32_t, uint64_t, uint32_t,
                                 size_t, int, volatile unsigned int*);
    bw_kernel_t bw_kernel;
    switch (cfg.scope) {
        case SCOPE_THREAD: bw_kernel = bw_thread_kernel; break;
        case SCOPE_WARP:   bw_kernel = bw_warp_kernel; break;
        case SCOPE_BLOCK:  bw_kernel = bw_block_kernel; break;
    }

    // Run benchmark for each size (using nvshmem-compatible size table)
    for (int size_idx = 0; size_idx < NUM_TEST_SIZES; size_idx++) {
        size_t size = TEST_SIZES[size_idx];

        // Validate size is divisible properly
        int units;
        switch (cfg.scope) {
            case SCOPE_THREAD: units = cfg.num_blocks * cfg.threads_per_block; break;
            case SCOPE_WARP:   units = cfg.num_blocks * (cfg.threads_per_block / 32); break;
            case SCOPE_BLOCK:  units = cfg.num_blocks; break;
        }
        if (size % units != 0 || size < (size_t)units) {
            if (mpi_rank == 0) {
                printf("%-12zu  (skipped - size < %d units or not divisible)\n", size, units);
            }
            continue;
        }

        // Reset counter and prod_idx
        cuda_check(cudaMemset(d_counter, 0, sizeof(unsigned int) * 2), "reset counter");
        uint64_t zero = 0;
        cuda_check(cudaMemcpy((void*)devx_qp->d_prod_idx, &zero, sizeof(uint64_t),
                              cudaMemcpyHostToDevice), "reset prod_idx");

        // Warmup
        bw_kernel<<<cfg.num_blocks, cfg.threads_per_block>>>(
            d_state, (uint64_t)d_send_buf, send_mr->lkey,
            peer_buf.addr, peer_buf.rkey,
            size, cfg.warmup_iters, d_counter
        );
        cuda_check(cudaDeviceSynchronize(), "warmup sync");

        // Reset for benchmark
        cuda_check(cudaMemset(d_counter, 0, sizeof(unsigned int) * 2), "reset counter");
        cuda_check(cudaMemcpy((void*)devx_qp->d_prod_idx, &zero, sizeof(uint64_t),
                              cudaMemcpyHostToDevice), "reset prod_idx");

        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark
        cuda_check(cudaEventRecord(start), "record start");
        bw_kernel<<<cfg.num_blocks, cfg.threads_per_block>>>(
            d_state, (uint64_t)d_send_buf, send_mr->lkey,
            peer_buf.addr, peer_buf.rkey,
            size, cfg.bench_iters, d_counter
        );
        cuda_check(cudaEventRecord(stop), "record stop");
        cuda_check(cudaEventSynchronize(stop), "sync stop");

        float ms;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");

        // Calculate bandwidth: total data / time
        // Total data = size * iterations
        double total_bytes = (double)size * cfg.bench_iters;
        double bw_gbps = total_bytes / (ms * 1e6);  // GB/s

        if (mpi_rank == 0) {
            char size_str[32];
            if (size < 1024) {
                snprintf(size_str, sizeof(size_str), "%zuB", size);
            } else if (size < 1024 * 1024) {
                snprintf(size_str, sizeof(size_str), "%zuKB", size / 1024);
            } else {
                snprintf(size_str, sizeof(size_str), "%zuMB", size / (1024 * 1024));
            }
            printf("%-12s %12.3f %12.2f\n", size_str, ms, bw_gbps);
            fflush(stdout);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Summary
    if (mpi_rank == 0) {
        printf("\n================================================================\n");
        printf("Benchmark complete.\n");
        printf("================================================================\n");
    }

    // Cleanup
    cuda_check(cudaEventDestroy(start), "destroy start");
    cuda_check(cudaEventDestroy(stop), "destroy stop");
    cuda_check(cudaFree(d_counter), "free counter");
    cuda_check(cudaFree(d_state), "free state");

    delete send_mr;
    delete recv_mr;
    cuda_check(cudaFree(d_send_buf), "free send");
    cuda_check(cudaFree(d_recv_buf), "free recv");

    delete devx_qp;
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);

    MPI_Finalize();
    return 0;
}
