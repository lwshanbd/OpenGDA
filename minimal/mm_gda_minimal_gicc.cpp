/**
 * mm_gda_minimal_gicc.cpp - Distributed matrix multiplication ported to the
 * unified gicc:: high-level API. Same algorithm and timing methodology as
 * mm_gda_minimal.cpp, but the communication path is built entirely on
 * gicc::Runtime / gicc::Token / gicc::flush / Runtime::wait.
 *
 * Per ring step:
 *   - tok = rt.put_no_db(B[cur], left, next_buf, stripe)   // host-side DWQ enqueue
 *   - ctx = rt.prepare_trigger(tok)
 *   - trigger_kernel<<<1,1>>>(ctx)        // calls gicc::flush(ctx)
 *   - matmul_kernel<<<>>>(...)            // overlaps with the in-flight RDMA
 *   - hipDeviceSynchronize()
 *   - rt.wait(tok)                        // host wait for that specific op
 *
 * Run:
 *   PMI_MAX_KVS_ENTRIES=2000 FI_MR_CACHE_MAX_COUNT=0 \
 *     srun -N <nodes> -n <ranks> --ntasks-per-node=8 ./mm_gda_minimal_gicc <N>
 */
#include <iostream>
#include <ctime>
#include <cstring>
#include <unistd.h>
#include <vector>

#include <mpi.h>
#include <hip/hip_runtime.h>

#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"

// For unset_rocr_visible_devices() — must run before MPI_Init on Tioga/Flux,
// otherwise multi-rank-per-node jobs see "invalid device ordinal".
#include "hip_device_context.hpp"

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
// HIP kernels
// =============================================================================
__global__ void matmul_stripe_kernel(const float* __restrict__ As,
                                     const float* __restrict__ Bs,
                                     float* __restrict__ Cs,
                                     int N, int Ns, int col_offset)
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

// Trigger-only kernel: writes the DWQ trigger MMIO via gicc::flush(ctx).
// The host has already populated ctx->trigger_val_ via prepare_trigger(token).
__global__ void gicc_trigger_kernel(gicc::DeviceCtx* ctx) {
    gicc::flush(ctx);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv)
{
    // CRITICAL: must precede MPI_Init on Tioga/Flux multi-rank-per-node jobs.
    unset_rocr_visible_devices();
    MPI_Init(&argc, &argv);

    gicc::Runtime rt;
    int mype = rt.rank();
    int npes = rt.size();
    if (npes < 2) {
        if (mype == 0) std::cerr << "Need at least 2 processes\n";
        MPI_Finalize();
        return 1;
    }

    int left_neighbor  = (npes + mype - 1) % npes;
    int right_neighbor = (mype + 1) % npes;

    // Detect on-node neighbors via MPI shared communicator (replaces the
    // PMI-based locality map used by the original mm).
    bool* locality_map = nullptr;
    {
        MPI_Comm shm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mype,
                            MPI_INFO_NULL, &shm);
        int shm_size = 0;
        MPI_Comm_size(shm, &shm_size);
        std::vector<int> shm_world_ranks(shm_size);
        MPI_Allgather(&mype, 1, MPI_INT,
                      shm_world_ranks.data(), 1, MPI_INT, shm);
        // Build a global locality map by allgathering shm membership.
        std::vector<int> my_shm_set(npes, 0);
        for (int r : shm_world_ranks) my_shm_set[r] = 1;
        std::vector<int> all_shm(npes * npes, 0);
        MPI_Allgather(my_shm_set.data(), npes, MPI_INT,
                      all_shm.data(),    npes, MPI_INT, MPI_COMM_WORLD);
        locality_map = new bool[npes];
        for (int r = 0; r < npes; r++) locality_map[r] = (all_shm[mype * npes + r] != 0);
        MPI_Comm_free(&shm);
    }

    bool use_ipc = (getenv("GDA_DISABLE_IPC") == nullptr);
    bool left_is_local  = use_ipc && locality_map[left_neighbor];
    bool right_is_local = use_ipc && locality_map[right_neighbor];
    delete[] locality_map;

    if (mype < 8) {
        std::cerr << "Rank " << mype << " gpu " << rt.gpu_id()
                  << ": left=" << left_neighbor << (left_is_local ? "(local)" : "(REMOTE)")
                  << ", right=" << right_neighbor << (right_is_local ? "(local)" : "(REMOTE)")
                  << "\n";
    }

    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    N = (N / npes) * npes;
    const int Ns = N / npes;
    const size_t stripe_size = (size_t)N * Ns * sizeof(float);

    if (mype == 0)
        std::cout << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size << " bytes\n";

    // Host init
    auto h_As = new float[N * Ns];
    auto h_Bs = new float[N * Ns];
    auto h_Cs = new float[N * Ns];
    for (int i = 0; i < N * Ns; i++) {
        h_As[i] = (i + mype) % 11 + 7;
        h_Bs[i] = (i + mype) % 13 + 5;
        h_Cs[i] = 0;
    }

    // Device buffers (double-buffered B)
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

    // Same-node IPC setup (unchanged from baseline; uses MPI Sendrecv for
    // handle exchange instead of PMI KVS so it doesn't collide with
    // gicc::Runtime::exchange()'s key namespace).
    float* right_d_B[2] = {nullptr, nullptr};
    {
        hipIpcMemHandle_t my_handles[2];
        HIP_CHECK(hipIpcGetMemHandle(&my_handles[0], d_B[0]));
        HIP_CHECK(hipIpcGetMemHandle(&my_handles[1], d_B[1]));

        // Send our handles to right neighbor (so they can read from us if
        // they're our left neighbor and we're local). Receive from left
        // neighbor for the symmetric case... actually we want our right
        // neighbor's handles, so do a Sendrecv to the right.
        hipIpcMemHandle_t right_handles[2];
        MPI_Sendrecv(my_handles,    sizeof(my_handles), MPI_BYTE,
                     left_neighbor, 0,
                     right_handles, sizeof(right_handles), MPI_BYTE,
                     right_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (right_is_local) {
            HIP_CHECK(hipIpcOpenMemHandle((void**)&right_d_B[0], right_handles[0],
                                           hipIpcMemLazyEnablePeerAccess));
            HIP_CHECK(hipIpcOpenMemHandle((void**)&right_d_B[1], right_handles[1],
                                           hipIpcMemLazyEnablePeerAccess));
        }
    }

    // Register the two B buffers with gicc::Runtime
    auto gB0 = rt.register_buffer(d_B[0], stripe_size, true);
    auto gB1 = rt.register_buffer(d_B[1], stripe_size, true);
    gicc::Buffer* gB[2] = { &gB0, &gB1 };

    rt.exchange();
    MPI_Barrier(MPI_COMM_WORLD);

    timespec t0, t1;
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (Ns + blockDim.y - 1) / blockDim.y);

    // Warm-up
    {
        int col_offset = mype * Ns;
        hipLaunchKernelGGL(matmul_stripe_kernel, gridDim, blockDim, 0, 0,
                           d_As, d_B[0], d_Cs, N, Ns, col_offset);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemset(d_Cs, 0, stripe_size));

        // Warm-up DWQ: a single put + flush + host wait
        auto wtok = rt.put_no_db(*gB[0], left_neighbor, /*dst_idx=*/0, stripe_size);
        auto* tctx = rt.prepare_trigger(wtok);
        hipLaunchKernelGGL(gicc_trigger_kernel, dim3(1), dim3(1), 0, 0, tctx);
        HIP_CHECK(hipDeviceSynchronize());
        rt.wait(wtok);
        rt.reset();  // drains warmup pending op, resets counters
    }
    MPI_Barrier(MPI_COMM_WORLD);

    constexpr int TOTAL_RUNS  = 1000;
    constexpr int WARMUP_RUNS = 3;
    double times[TOTAL_RUNS];

    for (int run = 0; run < TOTAL_RUNS; run++) {
        HIP_CHECK(hipMemset(d_Cs, 0, stripe_size));
        HIP_CHECK(hipMemcpy(d_B[0], h_Bs, stripe_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B[1], h_Bs, stripe_size, hipMemcpyHostToDevice));

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

        for (int s = 0; s < npes; s++) {
            const int block_num = (mype + s) % npes;
            const int cur_buf   = s % 2;
            const int next_buf  = (s + 1) % 2;

            bool need_dwq = !left_is_local;
            gicc::Token tok{0};
            if (need_dwq) {
                // Queue the cross-node RDMA (host-side DWQ enqueue)
                tok = rt.put_no_db(*gB[cur_buf], left_neighbor, next_buf, stripe_size);

                // Trigger via a tiny GPU kernel — overlaps with matmul below
                auto* tctx = rt.prepare_trigger(tok);
                hipLaunchKernelGGL(gicc_trigger_kernel, dim3(1), dim3(1), 0, 0, tctx);
            }

            // Same-node receive via IPC
            if (right_is_local) {
                HIP_CHECK(hipMemcpyAsync(d_B[next_buf], right_d_B[cur_buf],
                                         stripe_size, hipMemcpyDeviceToDevice));
            }

            // Compute concurrently with the in-flight RDMA
            int col_offset = block_num * Ns;
            hipLaunchKernelGGL(matmul_stripe_kernel, gridDim, blockDim, 0, 0,
                               d_As, d_B[cur_buf], d_Cs, N, Ns, col_offset);

            HIP_CHECK(hipDeviceSynchronize());
            if (need_dwq) {
                rt.wait(tok);
                rt.reset();   // recycle the slot used by this ring step
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        times[run] = timediff_us(t0, t1);

        if (mype == 0 && run % 100 == 0) {
            std::cout << "Run " << run << ": " << times[run] << " us"
                      << (run < WARMUP_RUNS ? " (warmup)" : "") << "\n";
        }
    }

    double sum = 0;
    for (int i = WARMUP_RUNS; i < TOTAL_RUNS; i++) sum += times[i];
    double avg = sum / (TOTAL_RUNS - WARMUP_RUNS);
    if (mype == 0) {
        std::cout << "DWQ + HIP average (runs " << WARMUP_RUNS << "-" << (TOTAL_RUNS - 1)
                  << "): " << avg << " us\n";
    }

    HIP_CHECK(hipMemcpy(h_Cs, d_Cs, stripe_size, hipMemcpyDeviceToHost));

    // Cleanup
    if (right_is_local) {
        HIP_CHECK(hipIpcCloseMemHandle(right_d_B[0]));
        HIP_CHECK(hipIpcCloseMemHandle(right_d_B[1]));
    }
    HIP_CHECK(hipFree(d_B[1]));
    HIP_CHECK(hipFree(d_B[0]));
    HIP_CHECK(hipFree(d_Cs));
    HIP_CHECK(hipFree(d_As));
    delete[] h_Cs; delete[] h_Bs; delete[] h_As;

    MPI_Finalize();
    return 0;
}
