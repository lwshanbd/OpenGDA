/**
 * mm_minimal.cpp - Distributed matmul on the unified gicc::launch API.
 *
 * Same npes-step ring algorithm as before, but expressed in the
 * spec-canonical form: one fused kernel that does
 *   put_no_db (single thread, host-staged on OFI / device WQE on MLX5)
 *     → flush (block-cooperative; on OFI also runs IPC fast-path copies)
 *     → matmul body (overlaps with the in-flight RDMA)
 *     → quiet (block-cooperative wait for RDMA to land).
 *
 * Compared to the previous explicit form, this drops:
 *   - Hand-rolled hipIpcGetMemHandle / OpenMemHandle dance
 *     (rt.exchange() captures and opens IPC handles automatically;
 *      flush() executes the staged IPC copies cooperatively).
 *   - The need_dwq / left_is_local / right_is_local branching
 *     (rt.put_no_db internally chooses IPC fast-path vs DWQ).
 *   - The standalone gicc_trigger_kernel.
 *   - The prepare_trigger + per-op Token + host wait pattern.
 *
 * Run:
 *   FI_MR_CACHE_MAX_COUNT=0 \
 *     srun -N <nodes> -n <ranks> --ntasks-per-node=8 ./mm_minimal <N>
 *
 *   (If GICC_BOOTSTRAP=pmi2, also export PMI_MAX_KVS_ENTRIES=2000.)
 */
#include <iostream>
#include <ctime>
#include <vector>

#include <hip/hip_runtime.h>

#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"

// For unset_rocr_visible_devices() — must run before Bootstrap init on
// Tioga/Flux, otherwise multi-rank-per-node jobs see "invalid device ordinal".
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"

#define HIP_CHECK(cmd) do {                                                    \
    hipError_t err = cmd;                                                      \
    if (err != hipSuccess) {                                                   \
        std::cerr << "HIP error: " << hipGetErrorString(err)                   \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
        exit(1);                                                               \
    }                                                                          \
} while(0)

float timediff_us(const timespec& a, const timespec& b) {
    return (b.tv_sec - a.tv_sec) * 1.0e6f + (b.tv_nsec - a.tv_nsec) / 1.0e3f;
}

// =============================================================================
// Fused matmul + RDMA-forward kernel.
//
// Thread (0,0) of block (0,0) issues the put. flush() and quiet() are called
// unguarded — they internally restrict to block 0 and run block-cooperatively
// (flush executes any IPC fast-path copies; quiet stripes the per-op
// completion poll across block-0 threads).
// =============================================================================
__global__ void matmul_step_kernel(gicc::DeviceCtx* ctx,
                                   const float* __restrict__ As,
                                   const float* __restrict__ Bs,
                                   float* __restrict__ Cs,
                                   int N, int Ns, int col_offset,
                                   int target, int dst_buf,
                                   int src_buf, size_t sz)
{
    // put_no_db must be called from all threads of block 0 (the IPC
    // route uses block-cooperative memcpy_block + __syncthreads). Other
    // blocks early-return inside put_no_db. Whole-buffer copy → both
    // dst_offset and src_offset are 0.
    gicc::put(ctx, target, dst_buf, /*dst_off=*/(size_t)0,
              src_buf, /*src_off=*/(size_t)0, sz);

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (k < N && j < Ns) {
        float b_kj = Bs[k * Ns + j];
        for (int i = 0; i < Ns; i++) {
            atomicAdd(&Cs[i * N + col_offset + j], As[i * N + k] * b_kj);
        }
    }

    gicc::flush(ctx);   // single-thread MMIO trigger for any DWQ-routed ops.
    gicc::quiet(ctx);   // poll DWQ completion slots.
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv)
{
    // CRITICAL: must precede Bootstrap init on Tioga/Flux multi-rank-per-node jobs.
    unset_rocr_visible_devices();

    gicc::Runtime rt;
    int mype = rt.rank();
    int npes = rt.size();
    if (npes < 2) {
        if (mype == 0) std::cerr << "Need at least 2 processes\n";
        return 1;
    }

    int left_neighbor = (npes + mype - 1) % npes;

    if (mype < 8) {
        std::cerr << "Rank " << mype << " gpu " << rt.gpu_id()
                  << ": forwarding to left=" << left_neighbor << "\n";
    }

    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    N = (N / npes) * npes;
    const int Ns = N / npes;
    const size_t stripe_size = (size_t)N * Ns * sizeof(float);

    const int TOTAL_RUNS  = 10;
    const int WARMUP_RUNS = 2;

    if (mype == 0)
        std::cerr << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size
                  << " bytes, " << TOTAL_RUNS << " runs\n";

    auto h_As = new float[N * Ns];
    auto h_Bs = new float[N * Ns];
    auto h_Cs = new float[N * Ns];
    for (int i = 0; i < N * Ns; i++) {
        h_As[i] = (i + mype) % 11 + 7;
        h_Bs[i] = (i + mype) % 13 + 5;
        h_Cs[i] = 0;
    }

    float *d_As, *d_Cs;
    float *d_B[2];
    HIP_CHECK(hipMalloc(&d_As,   stripe_size));
    HIP_CHECK(hipMalloc(&d_Cs,   stripe_size));
    HIP_CHECK(hipMalloc(&d_B[0], stripe_size));
    HIP_CHECK(hipMalloc(&d_B[1], stripe_size));
    HIP_CHECK(hipMemcpy(d_As,   h_As, stripe_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B[0], h_Bs, stripe_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Cs,   h_Cs, stripe_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_B[1], 0, stripe_size));

    auto gB0 = rt.register_buffer(d_B[0], stripe_size, true);
    auto gB1 = rt.register_buffer(d_B[1], stripe_size, true);
    gicc::Buffer* gB[2] = { &gB0, &gB1 };

    rt.exchange();
    rt.boot().barrier();

    timespec t0, t1;
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (Ns + blockDim.y - 1) / blockDim.y);

    // v1.5: dst/src addresses + rkeys are no longer threaded through the
    // launch — Runtime resolves them internally from (peer, dst_buf) and
    // (src_buf, src_offset) tables built at exchange() time.

    // Warm-up.
    {
        const int cur_buf  = 0;
        const int next_buf = 1;
        const int col_offset = mype * Ns;
        gicc::launch<matmul_step_kernel>(rt, gridDim, blockDim,
            d_As, d_B[cur_buf], d_Cs, N, Ns, col_offset,
            left_neighbor, next_buf,
            (int)gB[cur_buf]->lkey,
            (size_t)stripe_size);
        HIP_CHECK(hipDeviceSynchronize());
        rt.reset();
        HIP_CHECK(hipMemset(d_Cs, 0, stripe_size));
    }
    rt.boot().barrier();

    std::vector<double> times(TOTAL_RUNS);

    for (int run = 0; run < TOTAL_RUNS; run++) {
        HIP_CHECK(hipMemset(d_Cs, 0, stripe_size));
        HIP_CHECK(hipMemcpy(d_B[0], h_Bs, stripe_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B[1], h_Bs, stripe_size, hipMemcpyHostToDevice));

        rt.boot().barrier();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

        for (int s = 0; s < npes; s++) {
            const int block_num = (mype + s) % npes;
            const int cur_buf   = s % 2;
            const int next_buf  = (s + 1) % 2;
            const int col_offset = block_num * Ns;

            gicc::launch<matmul_step_kernel>(rt, gridDim, blockDim,
                d_As, d_B[cur_buf], d_Cs, N, Ns, col_offset,
                left_neighbor, next_buf,
                (int)gB[cur_buf]->lkey,
                (size_t)stripe_size);

            HIP_CHECK(hipDeviceSynchronize());
            rt.reset();
            rt.boot().barrier();
        }

        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        times[run] = timediff_us(t0, t1);

        if (mype == 0) {
            std::cout << "Run " << run << ": " << times[run] << " us"
                      << (run < WARMUP_RUNS ? " (warmup)" : "") << "\n";
        }
    }

    double sum = 0;
    for (int i = WARMUP_RUNS; i < TOTAL_RUNS; i++) sum += times[i];
    double avg = sum / (TOTAL_RUNS - WARMUP_RUNS);
    if (mype == 0) {
        std::cout << "gicc::launch average (runs " << WARMUP_RUNS << "-"
                  << (TOTAL_RUNS - 1) << "): " << avg << " us\n";
    }

    HIP_CHECK(hipMemcpy(h_Cs, d_Cs, stripe_size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_B[1]));
    HIP_CHECK(hipFree(d_B[0]));
    HIP_CHECK(hipFree(d_Cs));
    HIP_CHECK(hipFree(d_As));
    delete[] h_Cs; delete[] h_Bs; delete[] h_As;

    return 0;
}
