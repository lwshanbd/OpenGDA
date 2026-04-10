/**
 * test_opengda.cpp - Test the encapsulated opengda.hpp API
 *
 * Two-rank test: Rank 0 puts data to Rank 1 using GPU-triggered RDMA,
 * then verifies correctness.
 *
 * NOTE: Verification uses a GPU kernel to copy data to pinned host memory
 * instead of hipMemcpy, because hipMemcpy(DeviceToHost) for small transfers
 * (<16KB) may read stale L2 cache on AMD GPUs after NIC RDMA writes.
 *
 * Usage: FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 ./test_opengda
 */
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <hip/hip_runtime.h>

#include "opengda.hpp"

__global__ void put_kernel(gda::DeviceCtx* ctx) {
    if (threadIdx.x == 0) {
        gda::trigger_and_wait(ctx);
    }
}

// GPU kernel to copy device memory to pinned host memory for verification.
// GPU kernel reads always see fresh HBM data (kernel launch invalidates L2).
__global__ void verify_copy_kernel(const uint8_t* src, uint8_t* dst, size_t n) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

static void verify_dst(int rank, void* d_dst, size_t size, const char* test_name) {
    if (rank != 1) return;

    // Use GPU kernel copy instead of hipMemcpy to avoid L2 cache staleness
    uint8_t* h_pinned = nullptr;
    (void)hipHostMalloc(&h_pinned, size, hipHostMallocMapped);
    uint8_t* d_pinned = nullptr;
    (void)hipHostGetDevicePointer((void**)&d_pinned, h_pinned, 0);
    hipLaunchKernelGGL(verify_copy_kernel, dim3((size + 255) / 256), dim3(256), 0, 0,
                       (const uint8_t*)d_dst, d_pinned, size);
    (void)hipDeviceSynchronize();

    int errors = 0;
    for (size_t i = 0; i < size; i++) {
        if (h_pinned[i] != 0xAB) errors++;
    }

    if (errors == 0) {
        printf("%s: PASSED - %zu bytes transferred correctly\n", test_name, size);
    } else {
        printf("%s: FAILED - %d/%zu byte errors\n", test_name, errors, size);
    }
    (void)hipHostFree(h_pinned);
}

static void reset_dst(void* d_dst, size_t size) {
    (void)hipMemset(d_dst, 0, size);
    (void)hipDeviceSynchronize();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    gda::Runtime rt;
    int rank = rt.rank();
    int nranks = rt.size();

    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "Need exactly 2 ranks\n");
        MPI_Finalize();
        return 1;
    }

    int peer = 1 - rank;
    constexpr size_t SIZE = 4096;
    constexpr size_t N_PUTS = 4;

    void* d_src = nullptr;
    void* d_dst = nullptr;
    (void)hipMalloc(&d_src, SIZE * N_PUTS);
    (void)hipMalloc(&d_dst, SIZE * N_PUTS);

    if (rank == 0) (void)hipMemset(d_src, 0xAB, SIZE * N_PUTS);
    reset_dst(d_dst, SIZE * N_PUTS);

    auto src_buf = rt.register_buffer(d_src, SIZE * N_PUTS, true);
    auto dst_buf = rt.register_buffer(d_dst, SIZE * N_PUTS, true);
    rt.exchange();
    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Test 1: Single put (4KB)
    // =========================================================================
    if (rank == 0) {
        printf("=== Test 1: Single put (4KB) ===\n");
        rt.put(src_buf, peer, 1, SIZE);
        auto* ctx = rt.prepare();
        hipLaunchKernelGGL(put_kernel, dim3(1), dim3(1), 0, 0, ctx);
        (void)hipDeviceSynchronize();
        rt.reset();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    verify_dst(rank, d_dst, SIZE, "Test 1");
    MPI_Barrier(MPI_COMM_WORLD);
    reset_dst(d_dst, SIZE * N_PUTS);
    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Test 2: Multiple puts (4 x 4KB with offsets)
    // =========================================================================
    if (rank == 0) {
        printf("\n=== Test 2: %zu concurrent puts (4KB each) ===\n", N_PUTS);
        for (size_t i = 0; i < N_PUTS; i++) {
            rt.put(src_buf, peer, 1, SIZE, i * SIZE, i * SIZE);
        }
        auto* ctx = rt.prepare();
        hipLaunchKernelGGL(put_kernel, dim3(1), dim3(1), 0, 0, ctx);
        (void)hipDeviceSynchronize();
        rt.reset();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    verify_dst(rank, d_dst, SIZE * N_PUTS, "Test 2");
    MPI_Barrier(MPI_COMM_WORLD);
    reset_dst(d_dst, SIZE * N_PUTS);
    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Test 3: Large transfer (16KB single put)
    // =========================================================================
    if (rank == 0) {
        printf("\n=== Test 3: Large single put (16KB) ===\n");
        rt.put(src_buf, peer, 1, SIZE * N_PUTS);
        auto* ctx = rt.prepare();
        hipLaunchKernelGGL(put_kernel, dim3(1), dim3(1), 0, 0, ctx);
        (void)hipDeviceSynchronize();
        rt.reset();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    verify_dst(rank, d_dst, SIZE * N_PUTS, "Test 3");
    MPI_Barrier(MPI_COMM_WORLD);

    // Cleanup
    (void)hipFree(d_src);
    (void)hipFree(d_dst);

    if (rank == 0) printf("\nAll tests complete.\n");
    MPI_Finalize();
    return 0;
}
