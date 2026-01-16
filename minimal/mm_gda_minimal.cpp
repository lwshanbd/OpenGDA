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

#include <hip/hip_runtime.h>

#include "hip_device_context.hpp"
#include "pmi_session.hpp"
#include "device_affinity.hpp"
#include "fabric_dwq_context.hpp"
#include "memory_region.hpp"
#include "dwq_work_builder.hpp"

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

    // Initialize PMI2 first to get rank
    PmiSession pmi;

    int mype = pmi.rank;
    int npes = pmi.size;

    if (npes < 2) {
        if (mype == 0) {
            std::cerr << "This program requires at least 2 processes\n";
        }
        return 1;
    }

    // Get local rank from environment (SLURM_LOCALID) for GPU selection
    int local_rank = 0;
    const char* local_rank_env = getenv("SLURM_LOCALID");
    if (local_rank_env) {
        local_rank = atoi(local_rank_env);
    } else {
        // Fallback: use global rank mod num_devices
        int num_devices;
        HIP_CHECK(hipGetDeviceCount(&num_devices));
        local_rank = mype % num_devices;
    }

    // Create affinity detector with specified GPU
    DeviceAffinityDetector affinity(local_rank);

    // Initialize GPU with affinity-selected device
    HipDeviceContext hip(affinity.selected_gpu_id);

    // Debug: print GPU assignment
    if (mype < 8 || mype == npes - 1) {
        std::cerr << "Rank " << mype << " (local " << local_rank << ") using GPU "
                  << affinity.selected_gpu_id << "\n";
    }

    // Matrix size from command line or default
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    N = (N / npes) * npes;  // Make divisible by npes
    const int Ns = N / npes;
    const size_t stripe_size = N * Ns * sizeof(float);

    if (mype == 0)
        std::cout << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size << " bytes\n";

    // =========================================================================
    // Initialize Fabric/DWQ Context with affinity-selected CXI
    // =========================================================================
    FabricDwqContext fabric(mype, &affinity);

    // =========================================================================
    // Exchange addresses with all ranks
    // =========================================================================
    {
        char* my_hex = (char*)malloc(2 * fabric.addrlen + 1);
        bytes_to_hex((uint8_t*)fabric.local_addr, fabric.addrlen, my_hex);

        char key[PMI2_MAX_KEYLEN];
        snprintf(key, sizeof(key), "addr-%d", mype);
        pmi.kvs_put(key, my_hex);
        pmi.barrier();

        // Get addresses from left and right neighbors
        int left_rank = (npes + mype - 1) % npes;
        int right_rank = (mype + 1) % npes;

        char left_hex[PMI2_MAX_VALLEN], right_hex[PMI2_MAX_VALLEN];
        snprintf(key, sizeof(key), "addr-%d", left_rank);
        pmi.kvs_get(key, left_hex, sizeof(left_hex));
        snprintf(key, sizeof(key), "addr-%d", right_rank);
        pmi.kvs_get(key, right_hex, sizeof(right_hex));

        // Insert into AV
        uint8_t* left_bin = (uint8_t*)malloc(fabric.addrlen);
        uint8_t* right_bin = (uint8_t*)malloc(fabric.addrlen);
        hex_to_bytes(left_hex, left_bin, fabric.addrlen);
        hex_to_bytes(right_hex, right_bin, fabric.addrlen);

        fi_addr_t left_addr, right_addr;
        if (fi_av_insert(fabric.av, left_bin, 1, &left_addr, 0, NULL) != 1) {
            std::cerr << "Rank " << mype << ": fi_av_insert(left) failed\n";
            exit(1);
        }
        if (fi_av_insert(fabric.av, right_bin, 1, &right_addr, 0, NULL) != 1) {
            std::cerr << "Rank " << mype << ": fi_av_insert(right) failed\n";
            exit(1);
        }

        // Insert own address for self-atomic
        if (fi_av_insert(fabric.av, fabric.local_addr, 1, &fabric.local_addr_in_av, 0, NULL) != 1) {
            std::cerr << "Rank " << mype << ": fi_av_insert(local) failed\n";
            exit(1);
        }

        // Store peer addresses
        fabric.peer_addr = left_addr;  // We send to left neighbor

        free(my_hex);
        free(left_bin);
        free(right_bin);
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
    // Register memory regions for RDMA - both B buffers
    // =========================================================================
    MemoryRegion mr_B0(fabric.domain, fabric.ep, fabric.cxi_info,
                       d_B[0], stripe_size, true, hip.gpu_id, mype);
    MemoryRegion mr_B1(fabric.domain, fabric.ep, fabric.cxi_info,
                       d_B[1], stripe_size, true, hip.gpu_id, mype);
    MemoryRegion* mr_B[2] = {&mr_B0, &mr_B1};

    // =========================================================================
    // Exchange RMA info for BOTH buffers with left neighbor
    // Iteration s: send from buf[s%2] to left's buf[(s+1)%2]
    // =========================================================================
    struct RmaInfo { uint64_t addr; uint64_t key; };
    RmaInfo my_rma_info[2], left_rma_info[2];

    my_rma_info[0].addr = (uint64_t)d_B[0];
    my_rma_info[0].key = mr_B[0]->key;
    my_rma_info[1].addr = (uint64_t)d_B[1];
    my_rma_info[1].key = mr_B[1]->key;

    {
        char rma_hex[128];
        bytes_to_hex((uint8_t*)&my_rma_info, sizeof(my_rma_info), rma_hex);

        char key[PMI2_MAX_KEYLEN];
        snprintf(key, sizeof(key), "rma-%d", mype);
        pmi.kvs_put(key, rma_hex);
        pmi.barrier();

        // Get left neighbor's RMA info (we write to left's d_Bn)
        int left_rank = (npes + mype - 1) % npes;
        snprintf(key, sizeof(key), "rma-%d", left_rank);
        char peer_hex[PMI2_MAX_VALLEN];
        pmi.kvs_get(key, peer_hex, sizeof(peer_hex));
        hex_to_bytes(peer_hex, (uint8_t*)&left_rma_info, sizeof(left_rma_info));
    }

    timespec t0, t1;

    // Make sure all the stripes are initialized
    pmi.barrier();

    // Kernel launch configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (Ns + blockDim.y - 1) / blockDim.y);

    // =========================================================================
    // Warm-up: run one iteration to eliminate first-launch overhead
    // =========================================================================
    {
        int col_offset = mype * Ns;
        hipLaunchKernelGGL(matmul_stripe_kernel, gridDim, blockDim, 0, 0,
                           d_As, d_B[0], d_Cs, N, Ns, col_offset);
        HIP_CHECK(hipDeviceSynchronize());
        // Reset Cs to zero after warm-up
        HIP_CHECK(hipMemset(d_Cs, 0, stripe_size));
    }

    pmi.barrier();
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    // =========================================================================
    // Main computation loop with ring communication
    // Double buffering: iteration s uses buf[s%2], writes to left's buf[(s+1)%2]
    // =========================================================================
    for (int s = 0; s < npes; s++) {
        const int block_num = (mype + s) % npes;
        const uint64_t iter_threshold = s + 1;  // Incremental threshold
        const int cur_buf = s % 2;              // Current buffer to read/send from
        const int next_buf = (s + 1) % 2;       // Buffer to receive into (on my side)

        // Compute target address on left neighbor: left's buf[(s+1)%2]
        uint64_t remote_addr = fabric.is_virt_addr_mode() ? left_rma_info[next_buf].addr : 0;
        uint64_t remote_key = left_rma_info[next_buf].key;

        // Queue DWQ operation: send d_B[cur_buf] to left neighbor's d_B[next_buf]
        DwqWorkBuilder dwq(mype);
        dwq.queue_rma_write(
            fabric.domain, fabric.ep,
            d_B[cur_buf], mr_B[cur_buf]->desc, stripe_size,
            fabric.peer_addr, remote_addr, remote_key,
            fabric.trigger_cntr, fabric.completion_cntr, iter_threshold);

        // Trigger DWQ (non-blocking) - just write to trigger counter
        hipLaunchKernelGGL(gpu_trigger_only, dim3(1), dim3(1), 0, 0,
                           fabric.dev_trigger_cntr, iter_threshold);

        // Compute while DWQ is in flight: Cs += As * B[cur_buf]
        int col_offset = block_num * Ns;
        hipLaunchKernelGGL(matmul_stripe_kernel, gridDim, blockDim, 0, 0,
                           d_As, d_B[cur_buf], d_Cs, N, Ns, col_offset);

        HIP_CHECK(hipDeviceSynchronize());

        // Wait for RMA completion (CPU-side polling)
        while (fi_cntr_read(fabric.completion_cntr) < iter_threshold) {
            fi_cq_read(fabric.cq, NULL, 0);  // Drive progress
        }

        // Barrier to ensure all ranks have completed their sends
        pmi.barrier();

        // No copy needed! Next iteration uses d_B[next_buf] which now has received data
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    if (mype == 0) {
        std::cout << "DWQ + HIP: " << timediff_us(t0, t1) << " us\n";
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
                pmi.kvs_get(key, result_hex, 2 * stripe_size + 1);
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
            pmi.kvs_put(key, result_hex);
            delete[] result_hex;
        }
        pmi.barrier();
    }

    pmi.barrier();

    // =========================================================================
    // Cleanup
    // =========================================================================
    HIP_CHECK(hipFree(d_B[1]));
    HIP_CHECK(hipFree(d_B[0]));
    HIP_CHECK(hipFree(d_Cs));
    HIP_CHECK(hipFree(d_As));

    delete[] h_Cs;
    delete[] h_Bs;
    delete[] h_As;

    return 0;
}
