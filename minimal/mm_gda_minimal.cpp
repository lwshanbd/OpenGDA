/**
 * mm_gda_minimal.cpp - Distributed Matrix Multiplication with HIP + DWQ
 *
 * Uses HIP for GPU computation and GPU-Direct Async (DWQ) for inter-rank communication.
 * Reimplementation of mm_hip_mpi.cpp using minimal/ DWQ components instead of MPI.
 *
 * Run:
 *   FI_MR_CACHE_MAX_COUNT=0 srun -N <npes> -n <npes> --ntasks-per-node=1 ./mm_gda_minimal [matrix_size]
 */

#include <iostream>
#include <ctime>
#include <cstring>
#include <unistd.h>
#include <vector>

#include <mpi.h>
#include <hip/hip_runtime.h>

#include "gda_comm.hpp"

// =============================================================================
// Utility macros and functions
// =============================================================================

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
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

inline void bytes_to_hex(const uint8_t* in, size_t len, char* out) {
    static const char* h = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        out[2 * i] = h[(in[i] >> 4) & 0xF];
        out[2 * i + 1] = h[in[i] & 0xF];
    }
    out[2 * len] = '\0';
}

inline int hexval(char c) {
    if ('0' <= c && c <= '9') return c - '0';
    if ('a' <= c && c <= 'f') return c - 'a' + 10;
    if ('A' <= c && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline int hex_to_bytes(const char* in, uint8_t* out, size_t outlen) {
    size_t n = strlen(in);
    if (n % 2 != 0 || outlen < n / 2) return -1;
    for (size_t i = 0; i < n; i += 2) {
        int hi = hexval(in[i]);
        int lo = hexval(in[i + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[i / 2] = (uint8_t)((hi << 4) | lo);
    }
    return (int)(n / 2);
}

// =============================================================================
// HIP kernel for matrix stripe multiplication
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
// GPU kernel to trigger DWQ RMA write operation (non-blocking)
// =============================================================================

__global__ void gpu_trigger_only(
    volatile uint64_t* trigger_addr,
    uint64_t trigger_value)
{
    // Just trigger the DWQ operation, don't wait
    *trigger_addr = trigger_value;
}

// =============================================================================
// Main program
// =============================================================================

int main(int argc, char** argv)
{
    // CRITICAL: Unset ROCR_VISIBLE_DEVICES before any initialization
    unset_rocr_visible_devices();

    // Initialize MPI for barrier (much faster than libfabric barrier)
    MPI_Init(&argc, &argv);

    // Initialize GDA communication wrapper
    GdaComm comm;

    int mype = comm.rank();
    int npes = comm.size();

    if (npes < 2) {
        if (mype == 0) {
            std::cerr << "This program requires at least 2 processes\n";
        }
        return 1;
    }

    // Check if neighbors are on same node
    int left_neighbor = (npes + mype - 1) % npes;
    int right_neighbor = (mype + 1) % npes;
    bool* locality_map = comm.pmi.build_locality_map();

    // Allow disabling IPC for benchmarking (set GDA_DISABLE_IPC=1)
    bool use_ipc = (getenv("GDA_DISABLE_IPC") == nullptr);
    bool left_is_local = use_ipc && locality_map[left_neighbor];
    bool right_is_local = use_ipc && locality_map[right_neighbor];

    // Print locality info (only first few ranks)
    if (mype < 8) {
        std::cerr << "Rank " << mype << " on " << comm.pmi.hostname
                  << ": left=" << left_neighbor << (left_is_local ? "(local)" : "(REMOTE)")
                  << ", right=" << right_neighbor << (right_is_local ? "(local)" : "(REMOTE)")
                  << "\n";
    }

    delete[] locality_map;

    // Debug: print GPU assignment
    if (mype < 8 || mype == npes - 1) {
        std::cerr << "Rank " << mype << " (local " << comm.pmi.local_rank << ") using GPU "
                  << comm.gpu_id() << "\n";
    }

    // Matrix size from command line or default
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    N = (N / npes) * npes;  // Make divisible by npes
    const int Ns = N / npes;
    const size_t stripe_size = N * Ns * sizeof(float);

    if (mype == 0)
        std::cout << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size << " bytes\n";

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
    HIP_CHECK(hipMalloc(&d_As, stripe_size));
    HIP_CHECK(hipMalloc(&d_Cs, stripe_size));
    HIP_CHECK(hipMalloc(&d_B[0], stripe_size));
    HIP_CHECK(hipMalloc(&d_B[1], stripe_size));

    // Copy initial data to device
    HIP_CHECK(hipMemcpy(d_As, h_As, stripe_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B[0], h_Bs, stripe_size, hipMemcpyHostToDevice));  // Initial B in buf[0]
    HIP_CHECK(hipMemcpy(d_Cs, h_Cs, stripe_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_B[1], 0, stripe_size));

    // =========================================================================
    // IPC setup for same-node communication
    // =========================================================================
    float* right_d_B[2] = {nullptr, nullptr};  // Pointers to right neighbor's buffers

    // If my LEFT neighbor is local, they need my IPC handles to read from me
    // If my RIGHT neighbor is local, I need their IPC handles to read from them
    {
        hipIpcMemHandle_t my_handles[2];
        char key[PMI2_MAX_KEYLEN];

        // Everyone publishes IPC handles if their left neighbor is local
        // (because left neighbor will read from us)
        if (left_is_local) {
            HIP_CHECK(hipIpcGetMemHandle(&my_handles[0], d_B[0]));
            HIP_CHECK(hipIpcGetMemHandle(&my_handles[1], d_B[1]));

            char ipc_hex[sizeof(my_handles) * 2 + 1];
            bytes_to_hex((uint8_t*)&my_handles, sizeof(my_handles), ipc_hex);
            snprintf(key, sizeof(key), "ipc-%d", mype);
            comm.pmi.kvs_put(key, ipc_hex);
        }

        comm.pmi.barrier();  // KVS needs PMI barrier

        // Get right neighbor's IPC handles if they are local
        if (right_is_local) {
            hipIpcMemHandle_t right_handles[2];
            snprintf(key, sizeof(key), "ipc-%d", right_neighbor);
            char peer_hex[PMI2_MAX_VALLEN];
            comm.pmi.kvs_get(key, peer_hex, sizeof(peer_hex));
            hex_to_bytes(peer_hex, (uint8_t*)&right_handles, sizeof(right_handles));

            // Open right neighbor's buffers
            HIP_CHECK(hipIpcOpenMemHandle((void**)&right_d_B[0], right_handles[0],
                                           hipIpcMemLazyEnablePeerAccess));
            HIP_CHECK(hipIpcOpenMemHandle((void**)&right_d_B[1], right_handles[1],
                                           hipIpcMemLazyEnablePeerAccess));

            if (mype < 8) {
                std::cerr << "Rank " << mype << ": IPC opened right neighbor's buffers\n";
            }
        }
    }

    // =========================================================================
    // Register memory regions for RDMA - both B buffers (for cross-node)
    // =========================================================================
    GdaHandle handle_B0 = comm.register_buffer(d_B[0], stripe_size, true);
    GdaHandle handle_B1 = comm.register_buffer(d_B[1], stripe_size, true);
    GdaHandle* handle_B[2] = {&handle_B0, &handle_B1};

    // =========================================================================
    // Exchange RMA info for BOTH buffers with left neighbor
    // Iteration s: send from buf[s%2] to left's buf[(s+1)%2]
    // =========================================================================
    struct RmaInfo { uint64_t addr; uint64_t key; };
    RmaInfo my_rma_info[2], left_rma_info[2];

    my_rma_info[0].addr = handle_B0.rma_addr;
    my_rma_info[0].key = handle_B0.rma_key;
    my_rma_info[1].addr = handle_B1.rma_addr;
    my_rma_info[1].key = handle_B1.rma_key;

    {
        char rma_hex[128];
        bytes_to_hex((uint8_t*)&my_rma_info, sizeof(my_rma_info), rma_hex);

        char key[PMI2_MAX_KEYLEN];
        snprintf(key, sizeof(key), "rma-%d", mype);
        comm.pmi.kvs_put(key, rma_hex);
        comm.pmi.barrier();  // KVS needs PMI barrier

        // Get left neighbor's RMA info (we write to left's d_Bn)
        int left_rank = (npes + mype - 1) % npes;
        snprintf(key, sizeof(key), "rma-%d", left_rank);
        char peer_hex[PMI2_MAX_VALLEN];
        comm.pmi.kvs_get(key, peer_hex, sizeof(peer_hex));
        hex_to_bytes(peer_hex, (uint8_t*)&left_rma_info, sizeof(left_rma_info));

        // Set remote info for both buffers using buffer index
        comm.set_remote_info_by_index(left_rank, 0, left_rma_info[0].addr, left_rma_info[0].key);
        comm.set_remote_info_by_index(left_rank, 1, left_rma_info[1].addr, left_rma_info[1].key);
    }

    timespec t0, t1;

    // Make sure all the stripes are initialized
    MPI_Barrier(MPI_COMM_WORLD);

    // Kernel launch configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (Ns + blockDim.y - 1) / blockDim.y);

    // Cache trigger address for advanced usage (compute+trigger overlap)
    volatile uint64_t* trigger_addr = comm.get_trigger_addr();
    int left_rank = (npes + mype - 1) % npes;

    // =========================================================================
    // Warm-up: run one iteration to eliminate first-launch overhead
    // =========================================================================
    {
        // Warmup GPU kernel
        int col_offset = mype * Ns;
        hipLaunchKernelGGL(matmul_stripe_kernel, gridDim, blockDim, 0, 0,
                           d_As, d_B[0], d_Cs, N, Ns, col_offset);
        HIP_CHECK(hipDeviceSynchronize());
        // Reset Cs to zero after warm-up
        HIP_CHECK(hipMemset(d_Cs, 0, stripe_size));

        // Warmup DWQ: do one RMA write to prime NIC and libfabric
        uint64_t warmup_threshold = comm.put(handle_B0, left_rank, 0, stripe_size);

        // Trigger and wait for warmup RMA
        comm.trigger(warmup_threshold);
        HIP_CHECK(hipDeviceSynchronize());
        comm.wait(warmup_threshold);

        // Flush DWQ and reset counters for main loop
        comm.flush();
        comm.reset_counters();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Run 10 iterations, skip first 3 (warmup), average last 7
    // =========================================================================
    constexpr int TOTAL_RUNS = 1000;
    constexpr int WARMUP_RUNS = 3;
    double times[TOTAL_RUNS];

    uint64_t global_threshold = 0;  // Continuous threshold across all runs

    for (int run = 0; run < TOTAL_RUNS; run++) {
        // Reset Cs to zero for this run
        HIP_CHECK(hipMemset(d_Cs, 0, stripe_size));

        // Don't reset counters - use continuous thresholds instead

        // Restore B[0] with original data (it gets overwritten during ring)
        HIP_CHECK(hipMemcpy(d_B[0], h_Bs, stripe_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B[1], h_Bs, stripe_size, hipMemcpyHostToDevice));

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

        // Main computation loop with ring communication
        for (int s = 0; s < npes; s++) {
            const int block_num = (mype + s) % npes;
            const int cur_buf = s % 2;
            const int next_buf = (s + 1) % 2;

            // Communication: send d_B[cur_buf] to left, receive into d_B[next_buf] from right
            bool need_dwq = !left_is_local;  // Only need DWQ for cross-node sends

            if (need_dwq) {
                // Cross-node: use DWQ RDMA to write to left neighbor
                // put() returns threshold and auto-increments counter
                uint64_t iter_threshold = comm.put(*handle_B[cur_buf], left_rank,
                                                    next_buf, stripe_size);
                global_threshold = iter_threshold;

                // Trigger DWQ (using trigger_addr for compute overlap)
                hipLaunchKernelGGL(gpu_trigger_only, dim3(1), dim3(1), 0, 0,
                                   trigger_addr, iter_threshold);
            }

            // Same-node receive: copy from right neighbor's buffer via IPC
            if (right_is_local) {
                HIP_CHECK(hipMemcpyAsync(d_B[next_buf], right_d_B[cur_buf],
                                         stripe_size, hipMemcpyDeviceToDevice));
            }

            // Compute while communication is in flight
            int col_offset = block_num * Ns;
            hipLaunchKernelGGL(matmul_stripe_kernel, gridDim, blockDim, 0, 0,
                               d_As, d_B[cur_buf], d_Cs, N, Ns, col_offset);

            HIP_CHECK(hipDeviceSynchronize());

            // Wait for cross-node RMA completion (if needed)
            if (need_dwq) {
                comm.wait(global_threshold);
            }

            // MPI_Barrier(MPI_COMM_WORLD);
            // comm.barrier();
            MPI_Barrier(MPI_COMM_WORLD);
        }

        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        times[run] = timediff_us(t0, t1);

        // Fast flush: aggressively progress CQ to release TLE resources
        // Much faster than flush_dwq() which has sleep(1)
        comm.fast_flush(global_threshold);

        if (mype == 0 && run % 100 == 0) {
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
        std::cout << "DWQ + HIP average (runs " << WARMUP_RUNS << "-" << (TOTAL_RUNS-1)
                  << "): " << avg << " us\n";
    }

    // =========================================================================
    // Copy result back to host
    // =========================================================================
    HIP_CHECK(hipMemcpy(h_Cs, d_Cs, stripe_size, hipMemcpyDeviceToHost));

    // =========================================================================
    // Collect and print the matrix product (for small matrices)
    // =========================================================================
    if (N < 32) {
        // Use PMI KVS to collect results (since we don't have MPI)
        if (mype == 0) {
            auto C = new float[N * N];

            // Copy rank 0's stripe
            for (int i = 0; i < Ns * N; i++)
                C[i] = h_Cs[i];

            // Get other stripes via KVS
            for (int r = 1; r < npes; r++) {
                char key[PMI2_MAX_KEYLEN];
                snprintf(key, sizeof(key), "result-%d", r);
                char* result_hex = new char[2 * stripe_size + 1];
                comm.pmi.kvs_get(key, result_hex, 2 * stripe_size + 1);
                hex_to_bytes(result_hex, (uint8_t*)(C + r * Ns * N), stripe_size);
                delete[] result_hex;
            }

            print_matrix(C, N, N);
            delete[] C;
        } else {
            // Send our stripe to rank 0
            char* result_hex = new char[2 * stripe_size + 1];
            bytes_to_hex((uint8_t*)h_Cs, stripe_size, result_hex);
            char key[PMI2_MAX_KEYLEN];
            snprintf(key, sizeof(key), "result-%d", mype);
            comm.pmi.kvs_put(key, result_hex);
            delete[] result_hex;
        }
        comm.pmi.barrier();  // KVS needs PMI barrier
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Cleanup
    // =========================================================================
    // Close IPC handles first
    if (right_is_local) {
        HIP_CHECK(hipIpcCloseMemHandle(right_d_B[0]));
        HIP_CHECK(hipIpcCloseMemHandle(right_d_B[1]));
    }

    HIP_CHECK(hipFree(d_B[1]));
    HIP_CHECK(hipFree(d_B[0]));
    HIP_CHECK(hipFree(d_Cs));
    HIP_CHECK(hipFree(d_As));

    delete[] h_Cs;
    delete[] h_Bs;
    delete[] h_As;

    MPI_Finalize();
    return 0;
}
