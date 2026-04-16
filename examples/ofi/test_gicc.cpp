/**
 * test_gicc.cpp - Smoke test for the unified gicc:: API on the libfabric/CXI backend.
 *
 * The user-visible code below is structurally identical to what one would
 * write for the NVIB/mlx5 backend: gicc::Runtime, register_buffer, exchange,
 * put_no_db (host on cxi, device on mlx5), prepare, kernel { flush; quiet; },
 * reset.
 *
 * Run: FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 ./test_gicc
 */
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"

__global__ void put_kernel(gicc::DeviceCtx* ctx) {
    if (threadIdx.x == 0) {
        gicc::flush(ctx);   // CXI: writes trigger MMIO; mlx5: rings BlueFlame doorbell
        gicc::quiet(ctx);   // wait for completion
    }
}

__global__ void verify_copy_kernel(const uint8_t* src, uint8_t* dst, size_t n) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

static void verify_dst(int rank, void* d_dst, size_t size, const char* name) {
    if (rank != 1) return;
    uint8_t* h_pinned = nullptr;
    (void)hipHostMalloc(&h_pinned, size, hipHostMallocMapped);
    uint8_t* d_pinned = nullptr;
    (void)hipHostGetDevicePointer((void**)&d_pinned, h_pinned, 0);
    hipLaunchKernelGGL(verify_copy_kernel,
                       dim3((size + 255) / 256), dim3(256), 0, 0,
                       (const uint8_t*)d_dst, d_pinned, size);
    (void)hipDeviceSynchronize();
    int errors = 0;
    for (size_t i = 0; i < size; i++) if (h_pinned[i] != 0xAB) errors++;
    printf("%s: %s (%zu bytes)\n", name, errors == 0 ? "PASSED" : "FAILED", size);
    (void)hipHostFree(h_pinned);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    gicc::Runtime rt;
    int rank = rt.rank();
    int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "Need exactly 2 ranks\n");
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
    (void)hipMemset(d_dst, 0, SIZE * N_PUTS);
    (void)hipDeviceSynchronize();

    auto src_buf = rt.register_buffer(d_src, SIZE * N_PUTS, true);
    auto dst_buf = rt.register_buffer(d_dst, SIZE * N_PUTS, true);
    rt.exchange();
    rt.boot().barrier();

    if (rank == 0) {
        printf("=== gicc unified API: %zu puts of %zu bytes ===\n", N_PUTS, SIZE);
        for (size_t i = 0; i < N_PUTS; i++) {
            // Host-side queue (CXI). On mlx5 this same call lives inside the kernel.
            rt.put_no_db(src_buf, peer, dst_buf.index, SIZE, i * SIZE, i * SIZE);
        }
        auto* ctx = rt.prepare(peer, dst_buf.index);
        hipLaunchKernelGGL(put_kernel, dim3(1), dim3(1), 0, 0, ctx);
        (void)hipDeviceSynchronize();
        rt.reset();
    }
    rt.boot().barrier();
    verify_dst(rank, d_dst, SIZE * N_PUTS, "gicc put_no_db + flush + quiet");
    rt.boot().barrier();

    (void)hipFree(d_src);
    (void)hipFree(d_dst);
    return 0;
}
