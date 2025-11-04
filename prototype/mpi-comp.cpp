/*
 * mpi-comp.cpp - MPI GPU-to-GPU Communication with Computation Benchmark
 *
 * This program measures MPI point-to-point latency with GPU computation workload.
 * Pattern: GPU Kernel1 (compute) -> MPI Transfer -> GPU Kernel2 (compute)
 *
 * Key Features:
 * - Uses AMD GPU memory (HIP/ROCm) with MPI GPU-Direct
 * - Computation workload: SAXPY (y = a*x + y) proportional to transfer size
 * - Tests multiple message sizes: 16KB to 16MB
 * - Runs 20 iterations, reports average of best 10
 * - Requires MPICH_GPU_SUPPORT_ENABLED=1 environment variable
 *
 * Comparison with other tests:
 * - gda-comp.cpp: GPU-driven RDMA (computation inside kernel)
 * - ofi-comp.cpp: CPU-driven RDMA (separate computation kernels)
 * - mpi-comp.cpp: MPI communication (separate computation kernels)
 */

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_HIP(x, msg) do { \
    hipError_t err = (x); \
    if (err != hipSuccess) { \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        fprintf(stderr, "Rank %d: %s failed: %s (%d)\n", rank, msg, hipGetErrorString(err), err); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// =============================================================================
// GPU Computation Kernel - SAXPY operation
// =============================================================================

// GPU computation kernel - SAXPY operation (y = a*x + y)
// Used to add computation workload before and after communication
__global__ void gpu_compute_saxpy(float *x, float *y, float a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Perform SAXPY: y[i] = a * x[i] + y[i]
    // Loop multiple times to increase computation time
    #pragma unroll 1
    for (int iter = 0; iter < 100; iter++) {
      y[idx] = a * x[idx] + y[idx];
    }
  }
}

// =============================================================================
// Test Parameters
// =============================================================================

#define NUM_ITERATIONS 20
#define MPI_TAG 100

const size_t test_sizes[] = {
  16 * 1024,       // 16KB
  32 * 1024,       // 32KB
  64 * 1024,       // 64KB
  128 * 1024,      // 128KB
  256 * 1024,      // 256KB
  512 * 1024,      // 512KB
  1024 * 1024,     // 1MB
  2 * 1024 * 1024, // 2MB
  4 * 1024 * 1024, // 4MB
  8 * 1024 * 1024, // 8MB
  16 * 1024 * 1024, // 16MB
};
const int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

// =============================================================================
// Utility Functions
// =============================================================================

// Format size for display
static const char* format_size(size_t size, char* buf) {
  if (size < 1024) {
    snprintf(buf, 32, "%zuB", size);
  } else if (size < 1024*1024) {
    snprintf(buf, 32, "%zuKB", size/1024);
  } else {
    snprintf(buf, 32, "%zuMB", size/(1024*1024));
  }
  return buf;
}

// =============================================================================
// Main Program
// =============================================================================

int main(int argc, char** argv) {
  int rank, size;
  int provided;

  // Initialize MPI with thread support
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (rank == 0) {
      fprintf(stderr, "This benchmark requires exactly 2 MPI processes\n");
    }
    MPI_Finalize();
    return 1;
  }

  // =============================================================================
  // GPU Initialization
  // =============================================================================
  int device_count;
  CHECK_HIP(hipGetDeviceCount(&device_count), "hipGetDeviceCount");

  // Use rank % device_count for multi-GPU systems
  int gpu_id = rank % device_count;
  CHECK_HIP(hipSetDevice(gpu_id), "hipSetDevice");

  hipDeviceProp_t prop;
  CHECK_HIP(hipGetDeviceProperties(&prop, gpu_id), "hipGetDeviceProperties");

  if (rank == 0) {
    printf("Rank %d: Using GPU %d: %s\n", rank, gpu_id, prop.name);

    // Check GPU-Direct environment variable
    char *gpu_support = getenv("MPICH_GPU_SUPPORT_ENABLED");
    if (gpu_support && atoi(gpu_support) == 1) {
      printf("GPU-Direct: ENABLED (MPICH_GPU_SUPPORT_ENABLED=1)\n");
    } else {
      printf("WARNING: MPICH_GPU_SUPPORT_ENABLED not set to 1\n");
      printf("         GPU-Direct may not be active!\n");
    }
    printf("\n");
  }

  // =============================================================================
  // Allocate GPU Memory
  // =============================================================================
  size_t max_buf_size = test_sizes[num_test_sizes - 1];

  char *d_send_buf = NULL;
  char *d_recv_buf = NULL;

  CHECK_HIP(hipMalloc(&d_send_buf, max_buf_size), "hipMalloc(send)");
  CHECK_HIP(hipMalloc(&d_recv_buf, max_buf_size), "hipMalloc(recv)");

  // Initialize GPU buffers
  CHECK_HIP(hipMemset(d_send_buf, 0xAB, max_buf_size), "hipMemset(send)");
  CHECK_HIP(hipMemset(d_recv_buf, 0, max_buf_size), "hipMemset(recv)");

  // Allocate computation buffers (proportional to max transfer size)
  const size_t max_compute_n = max_buf_size / sizeof(float);
  const size_t max_compute_bytes = max_compute_n * sizeof(float);
  float *d_compute_x = NULL;
  float *d_compute_y = NULL;
  CHECK_HIP(hipMalloc(&d_compute_x, max_compute_bytes), "hipMalloc(compute_x)");
  CHECK_HIP(hipMalloc(&d_compute_y, max_compute_bytes), "hipMalloc(compute_y)");

  // Initialize computation buffers with some data
  CHECK_HIP(hipMemset(d_compute_x, 1, max_compute_bytes), "hipMemset(compute_x)");
  CHECK_HIP(hipMemset(d_compute_y, 0, max_compute_bytes), "hipMemset(compute_y)");

  CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");

  if (rank == 0) {
    printf("GPU buffers allocated:\n");
    printf("  Transfer buffers: %zu bytes\n", max_buf_size);
    printf("  Compute buffers:  %zu bytes\n\n", max_compute_bytes);
  }

  // =============================================================================
  // Latency Benchmark with Computation
  // =============================================================================
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    printf("%-8s  %12s  %s\n", "Size", "Latency(us)", "Statistics");
    printf("========  ============  ===============================================\n");
    printf("Note: Latency is average of best 10/%d iterations\n", NUM_ITERATIONS);
    printf("      Includes GPU computation (SAXPY) before and after transfer\n\n");
  }

  // Determine peer rank
  int peer = (rank == 0) ? 1 : 0;

  // Loop through different message sizes
  for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
    size_t current_size = test_sizes[size_idx];

    // Calculate computation parameters (proportional to transfer size)
    const int compute_n = current_size / sizeof(float);
    const int threads_per_block = 256;
    const int num_blocks = (compute_n + threads_per_block - 1) / threads_per_block;
    const float compute_a = 2.5f;

    // Store all iteration times for statistical analysis
    double iteration_times[NUM_ITERATIONS];
    int successful_iterations = 0;

    MPI_Barrier(MPI_COMM_WORLD);

    // Timed iterations
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);

      double start_time = 0.0;
      if (rank == 0) {
        start_time = MPI_Wtime();
      }

      // === Computation Phase 1: Pre-communication ===
      hipLaunchKernelGGL(gpu_compute_saxpy, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                         d_compute_x, d_compute_y, compute_a, compute_n);
      CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize(compute1)");

      // === Communication Phase: MPI Send/Recv ===
      if (rank == 0) {
        MPI_Send(d_send_buf, current_size, MPI_BYTE, peer, MPI_TAG, MPI_COMM_WORLD);
      } else {
        MPI_Recv(d_recv_buf, current_size, MPI_BYTE, peer, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      // === Computation Phase 2: Post-communication ===
      hipLaunchKernelGGL(gpu_compute_saxpy, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                         d_compute_x, d_compute_y, compute_a, compute_n);
      CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize(compute2)");

      if (rank == 0) {
        double end_time = MPI_Wtime();
        double elapsed = (end_time - start_time) * 1e6; // Convert to microseconds
        iteration_times[successful_iterations] = elapsed;
        successful_iterations++;
      }

      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Print results for this size (rank 0 only)
    if (rank == 0 && successful_iterations > 0) {
      // Sort iteration times in ascending order
      for (int i = 0; i < successful_iterations - 1; i++) {
        for (int j = i + 1; j < successful_iterations; j++) {
          if (iteration_times[j] < iteration_times[i]) {
            double temp = iteration_times[i];
            iteration_times[i] = iteration_times[j];
            iteration_times[j] = temp;
          }
        }
      }

      // Select top 10 fastest iterations (or all if less than 10)
      int samples_to_average = (successful_iterations < 10) ? successful_iterations : 10;
      double sum_best = 0.0;
      for (int i = 0; i < samples_to_average; i++) {
        sum_best += iteration_times[i];
      }
      double avg_best_latency = sum_best / samples_to_average;

      char size_buf[32];
      printf("%-8s  %12.2f  (best %d/%d: min=%.2f max=%.2f)\n",
             format_size(current_size, size_buf),
             avg_best_latency,
             samples_to_average,
             successful_iterations,
             iteration_times[0],
             iteration_times[samples_to_average - 1]);
      fflush(stdout);
    }
  }

  if (rank == 0) {
    printf("\n");
  }

  // =============================================================================
  // Cleanup
  // =============================================================================
  CHECK_HIP(hipFree(d_send_buf), "hipFree(send)");
  CHECK_HIP(hipFree(d_recv_buf), "hipFree(recv)");
  CHECK_HIP(hipFree(d_compute_x), "hipFree(compute_x)");
  CHECK_HIP(hipFree(d_compute_y), "hipFree(compute_y)");

  MPI_Finalize();
  return 0;
}
