/**
 * mm_hip_mpi.cpp - Distributed Matrix Multiplication with HIP + MPI
 *
 * Uses HIP for GPU computation and MPI for inter-rank communication.
 * Based on mpi_mm.cpp, replacing OpenMP target with HIP kernels.
 *
 * Run:
 *   srun -n <npes> ./mm_hip_mpi
 */

#include <iostream>
#include <utility>
#include <ctime>
#include <cstdlib>

#include <mpi.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
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

/**
 * HIP kernel for matrix stripe multiplication
 * Computes Cb += As * Bs where:
 *   As: Ns x N (horizontal stripe of A)
 *   Bs: N x Ns (vertical stripe of B, stored as N*Ns)
 *   Cb: Ns x Ns block within C stripe
 */
__global__ void matmul_stripe_kernel(
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    float* __restrict__ Cs,
    int N,
    int Ns,
    int col_offset)
{
    // Each thread handles one (k, j) pair
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (k < N && j < Ns) {
        float b_kj = Bs[k * Ns + j];
        for (int i = 0; i < Ns; i++) {
            atomicAdd(&Cs[i * N + col_offset + j], As[i * N + k] * b_kj);
        }
    }
}

int main(int argc, char** argv)
{
    // CRITICAL: Unset ROCR_VISIBLE_DEVICES before any HIP/MPI initialization
    // Flux sets this which causes GPU virtualization issues
    unsetenv("ROCR_VISIBLE_DEVICES");

    MPI_Init(&argc, &argv);

    int mype, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    // Get local rank for GPU selection - try multiple env vars
    int local_rank = -1;
    const char* env_vars[] = {"SLURM_LOCALID", "FLUX_TASK_LOCAL_ID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", NULL};
    for (int i = 0; env_vars[i] && local_rank < 0; i++) {
        const char* val = getenv(env_vars[i]);
        if (val) local_rank = atoi(val);
    }

    // Set GPU device based on local rank
    int num_devices;
    HIP_CHECK(hipGetDeviceCount(&num_devices));

    // Fallback: use global rank mod num_devices if no local rank found
    int gpu_id = (local_rank >= 0) ? (local_rank % num_devices) : (mype % num_devices);
    HIP_CHECK(hipSetDevice(gpu_id));

    // Debug: print GPU assignment
    if (mype < 8 || mype == npes - 1) {
        std::cerr << "Rank " << mype << " using GPU " << gpu_id << " (num_devices=" << num_devices << ")\n";
    }

    // Matrix size from command line or default
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    N = (N / npes) * npes;  // Make divisible by npes
    const int Ns = N / npes;              // Width of the stripes
    const size_t stripe_size = N * Ns * sizeof(float);

    if (mype == 0)
        std::cout << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size << " bytes\n";

    // Host arrays for initialization
    auto h_As = new float[N * Ns];
    auto h_Bs = new float[N * Ns];
    auto h_Cs = new float[N * Ns];

    // Initialize the matrices on host
    for (int i = 0; i < N * Ns; i++) {
        h_As[i] = (i + mype) % 11 + 7;
        h_Bs[i] = (i + mype) % 13 + 5;
        h_Cs[i] = 0;
    }

    // Device arrays
    float *d_As, *d_Bs, *d_Cs, *d_Bn;
    HIP_CHECK(hipMalloc(&d_As, stripe_size));
    HIP_CHECK(hipMalloc(&d_Bs, stripe_size));
    HIP_CHECK(hipMalloc(&d_Cs, stripe_size));
    HIP_CHECK(hipMalloc(&d_Bn, stripe_size));

    // Copy to device
    HIP_CHECK(hipMemcpy(d_As, h_As, stripe_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Bs, h_Bs, stripe_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Cs, h_Cs, stripe_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_Bn, 0, stripe_size));

    timespec t0, t1;

    // Make sure all the stripes are initialized
    MPI_Barrier(MPI_COMM_WORLD);

    // Kernel launch configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (Ns + blockDim.y - 1) / blockDim.y);

    // Warm-up: run one iteration to eliminate first-launch overhead
    {
        int col_offset = mype * Ns;
        hipLaunchKernelGGL(matmul_stripe_kernel, gridDim, blockDim, 0, 0,
                           d_As, d_Bs, d_Cs, N, Ns, col_offset);
        HIP_CHECK(hipDeviceSynchronize());
        // Reset Cs to zero after warm-up
        HIP_CHECK(hipMemset(d_Cs, 0, stripe_size));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int s = 0; s < npes; s++) {
        const int block_num = (mype + s) % npes;

        // Start async send of current Bs to left neighbor
        MPI_Request sreq;
        MPI_Isend(d_Bs, N * Ns, MPI_FLOAT, (npes + mype - 1) % npes, 0, MPI_COMM_WORLD, &sreq);

        // Compute: Cs += As * Bs
        int col_offset = block_num * Ns;
        hipLaunchKernelGGL(matmul_stripe_kernel, gridDim, blockDim, 0, 0,
                           d_As, d_Bs, d_Cs, N, Ns, col_offset);
        HIP_CHECK(hipDeviceSynchronize());

        // Receive next Bs from right neighbor
        MPI_Recv(d_Bn, N * Ns, MPI_FLOAT, (mype + 1) % npes, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Wait(&sreq, MPI_STATUS_IGNORE);

        // Swap Bs and Bn
        std::swap(d_Bs, d_Bn);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    if (mype == 0) {
        std::cout << "MPI + HIP: " << timediff_us(t0, t1) << " us\n";
    }

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_Cs, d_Cs, stripe_size, hipMemcpyDeviceToHost));

    // Collect and print the matrix product
    if (N < 32) {
        if (mype == 0) {
            auto C = new float[N * N];

            for (int i = 0; i < Ns * N; i++)
                C[i] = h_Cs[i];

            for (int i = 1; i < npes; i++)
                MPI_Recv(C + i * Ns * N, N * Ns, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            print_matrix(C, N, N);

            delete[] C;
        } else {
            MPI_Send(h_Cs, N * Ns, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Cleanup
    HIP_CHECK(hipFree(d_Bn));
    HIP_CHECK(hipFree(d_Cs));
    HIP_CHECK(hipFree(d_Bs));
    HIP_CHECK(hipFree(d_As));

    delete[] h_Cs;
    delete[] h_Bs;
    delete[] h_As;

    MPI_Finalize();
    return 0;
}
