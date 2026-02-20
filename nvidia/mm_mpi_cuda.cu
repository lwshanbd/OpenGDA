/**
 * mm_mpi_cuda.cu - Distributed Matrix Multiplication with CUDA-aware MPI
 *
 * Uses CUDA for GPU computation and CUDA-aware MPI for inter-rank communication.
 * MPI directly operates on GPU buffers without explicit memory copies.
 *
 * Build:
 *   nvcc -o mm_mpi_cuda mm_mpi_cuda.cu $(mpicc --showme:compile) $(mpicc --showme:link) -arch=sm_90
 *   Or use CMake with MPI support
 *
 * Run:
 *   mpirun -np <npes> ./mm_mpi_cuda [matrix_size]
 *
 * Note: Requires CUDA-aware MPI (e.g., OpenMPI with CUDA support, MVAPICH2-GDR)
 */

#include <iostream>
#include <ctime>
#include <cstring>
#include <unistd.h>

#include <mpi.h>
#include <cuda_runtime.h>

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
// Main program
// =============================================================================

int main(int argc, char** argv)
{
    // Initialize MPI with threading support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

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

    // Neighbors in the ring
    int left_neighbor = (npes + mype - 1) % npes;
    int right_neighbor = (mype + 1) % npes;

    // Print GPU assignment
    if (mype < 8 || mype == npes - 1) {
        std::cerr << "Rank " << mype << " (local " << local_rank << ") using GPU "
                  << gpu_id << " (" << props.name << ")\n";
    }

    // Matrix size from command line or default
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    N = (N / npes) * npes;  // Make divisible by npes
    const int Ns = N / npes;
    const size_t stripe_size = N * Ns * sizeof(float);
    const int nelems = N * Ns;

    if (mype == 0) {
        std::cout << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size << " bytes\n";
        std::cout << "Ring pattern: " << npes << " ranks\n";
        std::cout << "Using CUDA-aware MPI for GPU buffer communication\n";
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

    // Create CUDA stream for asynchronous operations
    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

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
        matmul_stripe_kernel<<<gridDim, blockDim, 0, compute_stream>>>(
            d_As, d_B[0], d_Cs, N, Ns, col_offset
        );
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        // Reset Cs to zero after warm-up
        CUDA_CHECK(cudaMemset(d_Cs, 0, stripe_size));

        // Warmup MPI: do one sendrecv with GPU buffers
        MPI_Sendrecv(d_B[0], nelems, MPI_FLOAT, left_neighbor, 0,
                     d_B[1], nelems, MPI_FLOAT, right_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

        // Main computation loop with ring communication
        for (int s = 0; s < npes; s++) {
            const int block_num = (mype + s) % npes;
            const int cur_buf = s % 2;
            const int next_buf = (s + 1) % 2;

            // Start non-blocking send/recv using CUDA-aware MPI
            // GPU buffers are passed directly to MPI
            MPI_Request requests[2];

            // Non-blocking send to left neighbor
            MPI_Isend(d_B[cur_buf], nelems, MPI_FLOAT,
                      left_neighbor, s, MPI_COMM_WORLD, &requests[0]);

            // Non-blocking receive from right neighbor
            MPI_Irecv(d_B[next_buf], nelems, MPI_FLOAT,
                      right_neighbor, s, MPI_COMM_WORLD, &requests[1]);

            // Compute while communication is in flight
            int col_offset = block_num * Ns;
            matmul_stripe_kernel<<<gridDim, blockDim, 0, compute_stream>>>(
                d_As, d_B[cur_buf], d_Cs, N, Ns, col_offset
            );

            // Wait for compute to finish (needed before reading result)
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));

            // Wait for MPI communication to complete
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
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
        std::cout << "CUDA-aware MPI + CUDA average (runs " << WARMUP_RUNS
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
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaFree(d_B[1]));
    CUDA_CHECK(cudaFree(d_B[0]));
    CUDA_CHECK(cudaFree(d_Cs));
    CUDA_CHECK(cudaFree(d_As));

    delete[] h_Cs;
    delete[] h_Bs;
    delete[] h_As;

    MPI_Finalize();
    return 0;
}
