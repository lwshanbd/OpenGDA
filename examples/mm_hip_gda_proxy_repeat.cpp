/**
 * mm_hip_gda_proxy_repeat.cpp - Repeated Distributed Matrix Multiplication
 *
 * This version performs N_ITERATIONS of matrix multiplication and records
 * timing statistics. Handles are recreated between iterations to properly
 * reuse DWQ resources.
 *
 * Run:
 *   FI_MR_CACHE_MAX_COUNT=0 srun -n <npes> ./mm_hip_gda_proxy_repeat [matrix_size] [iterations] [nobarrier]
 *   FI_MR_CACHE_MAX_COUNT=0 srun -N 4 -n 16 ./mm_hip_gda_proxy_repeat 512 100
 *   FI_MR_CACHE_MAX_COUNT=0 srun -N 4 -n 32 ./mm_hip_gda_proxy_repeat 512 100 nobarrier
 *
 * The "nobarrier" option disables per-step proxy barrier and relies on
 * double-buffering for correctness. This avoids the window_size < npes
 * limitation but may have correctness issues if network timing is variable.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "gda.h"
#include "gda_barrier_proxy.h"
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t err = cmd;                                                      \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static float timediff_us(const timespec &t_start, const timespec &t_end) {
  return (t_end.tv_sec - t_start.tv_sec) * 1.0e6f +
         (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3f;
}

/**
 * Trigger kernel: one thread triggers the GPU-initiated RDMA op for iteration s
 */
__global__ void gda_trigger_kernel(gda_gpu_handle_t *put_handles, int s) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    gda_gpu_trigger(put_handles[s]);
  }
}

/**
 * Compute kernel: C[:, col_offset:col_offset+Ns] += As * cur_Bs
 */
__global__ void mm_compute_kernel(const float *__restrict__ As,
                                  const float *__restrict__ cur_Bs,
                                  float *__restrict__ Cs, int N, int Ns,
                                  int mype, int npes, int s) {
  __threadfence_system();

  int block_num = (mype + s) % npes;
  int col_offset = block_num * Ns;

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (k < N && j < Ns) {
    float b_kj = cur_Bs[k * Ns + j];
    for (int i = 0; i < Ns; i++) {
      atomicAdd(&Cs[i * N + col_offset + j], As[i * N + k] * b_kj);
    }
  }
}

/**
 * Wait + barrier kernel: wait for completion and perform barrier
 */
__global__ void gda_wait_and_barrier_kernel(gda_gpu_handle_t *put_handles,
                                            gda_proxy_barrier_dev_t *barrier,
                                            int s) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    gda_gpu_wait(put_handles[s]);
    gda_gpu_proxy_barrier_wait(barrier);
  }
}

/**
 * Wait-only kernel: wait for put completion without barrier
 * Used in no-barrier mode where double-buffering provides synchronization
 */
__global__ void gda_wait_only_kernel(gda_gpu_handle_t *put_handles, int s) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    gda_gpu_wait(put_handles[s]);
  }
}

/**
 * Initialize matrix data
 */
__global__ void init_matrix_kernel(float *buf, int size, int mype, int pattern) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (pattern == 0) {
      buf[idx] = (idx + mype) % 11 + 7; // As pattern
    } else if (pattern == 1) {
      buf[idx] = (idx + mype) % 13 + 5; // Bs pattern
    } else {
      buf[idx] = 0; // Cs, Bn pattern (zero)
    }
  }
}

/**
 * Reset matrix to zero
 */
__global__ void zero_matrix_kernel(float *buf, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    buf[idx] = 0.0f;
  }
}

/**
 * Helper class to manage put handles lifecycle
 */
class PutHandleManager {
public:
  PutHandleManager(int npes, float* d_Bs, float* d_Bn,
                   size_t stripe_size, size_t offset_Bs, size_t offset_Bn,
                   int left_neighbor)
    : npes_(npes), d_Bs_(d_Bs), d_Bn_(d_Bn),
      stripe_size_(stripe_size), offset_Bs_(offset_Bs), offset_Bn_(offset_Bn),
      left_neighbor_(left_neighbor), d_put_handles_(nullptr) {
    put_handles_.resize(npes, nullptr);
    gpu_put_handles_.resize(npes);
  }

  ~PutHandleManager() {
    release();
  }

  bool allocate() {
    for (int s = 0; s < npes_; s++) {
      float *src = (s % 2 == 0) ? d_Bs_ : d_Bn_;
      size_t dest_offset = (s % 2 == 0) ? offset_Bn_ : offset_Bs_;

      put_handles_[s] = gda_put(src, stripe_size_, left_neighbor_, dest_offset);
      if (!put_handles_[s]) {
        return false;
      }
      gpu_put_handles_[s] = put_handles_[s]->gpu;
    }

    // Allocate device array if needed
    if (!d_put_handles_) {
      HIP_CHECK(hipMalloc(&d_put_handles_, npes_ * sizeof(gda_gpu_handle_t)));
    }

    // Copy handles to device
    HIP_CHECK(hipMemcpy(d_put_handles_, gpu_put_handles_.data(),
                        npes_ * sizeof(gda_gpu_handle_t), hipMemcpyHostToDevice));
    return true;
  }

  void release() {
    for (int s = 0; s < npes_; s++) {
      if (put_handles_[s]) {
        gda_free(put_handles_[s]);
        put_handles_[s] = nullptr;
      }
    }
  }

  void freeDeviceArray() {
    if (d_put_handles_) {
      HIP_CHECK(hipFree(d_put_handles_));
      d_put_handles_ = nullptr;
    }
  }

  gda_gpu_handle_t* deviceHandles() { return d_put_handles_; }

private:
  int npes_;
  float *d_Bs_, *d_Bn_;
  size_t stripe_size_, offset_Bs_, offset_Bn_;
  int left_neighbor_;
  std::vector<gda_handle_t*> put_handles_;
  std::vector<gda_gpu_handle_t> gpu_put_handles_;
  gda_gpu_handle_t* d_put_handles_;
};

/**
 * Wait for proxy to arm all slots (host-side)
 * This ensures GPU won't stall waiting for slot_state == ARMED
 */
void wait_for_slots_armed(gda_proxy_barrier_dev_t* dev_host, int window_size) {
  // slot_state is in pinned host memory, accessible from CPU
  int max_retries = 1000000;
  for (int retry = 0; retry < max_retries; retry++) {
    bool all_armed = true;
    for (int s = 0; s < window_size; s++) {
      if (__atomic_load_n(&dev_host->slot_state[s], __ATOMIC_ACQUIRE) != GDA_SLOT_ARMED) {
        all_armed = false;
        break;
      }
    }
    if (all_armed) return;

    // Short pause to let proxy thread work
    for (int i = 0; i < 100; i++) {
      __asm__ volatile("pause" ::: "memory");
    }
  }
  // If we get here, something is wrong
  std::cerr << "Warning: Timeout waiting for proxy slots to arm" << std::endl;
}

/**
 * Run one matrix multiplication iteration
 * @param use_barrier: true = use proxy barrier each step, false = wait-only mode
 */
void run_mm_iteration(hipStream_t stream,
                      gda_gpu_handle_t* d_put_handles,
                      gda_proxy_barrier_dev_t* d_barrier,
                      float* d_As, float* d_Bs, float* d_Bn, float* d_Cs,
                      int N, int Ns, int mype, int npes, bool use_barrier) {
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (Ns + blockDim.y - 1) / blockDim.y);

  dim3 oneGrid(1);
  dim3 oneBlock(256);

  auto cur_Bs_ptr = [&](int s) -> float* {
    return (s % 2 == 0) ? d_Bs : d_Bn;
  };

  for (int s = 0; s < npes; s++) {
    hipLaunchKernelGGL(gda_trigger_kernel, oneGrid, oneBlock, 0, stream,
                       d_put_handles, s);

    hipLaunchKernelGGL(mm_compute_kernel, gridDim, blockDim, 0, stream,
                       d_As, cur_Bs_ptr(s), d_Cs, N, Ns, mype, npes, s);

    if (use_barrier) {
      hipLaunchKernelGGL(gda_wait_and_barrier_kernel, oneGrid, oneBlock, 0,
                         stream, d_put_handles, d_barrier, s);
    } else {
      // Wait-only mode: rely on double-buffering for correctness
      // Only barrier at start and end of iteration (outside this function)
      hipLaunchKernelGGL(gda_wait_only_kernel, oneGrid, oneBlock, 0,
                         stream, d_put_handles, s);
    }
  }
}

int main(int argc, char **argv) {
  // Initialize GDA
  if (gda_init() != 0) {
    std::cerr << "Failed to initialize OpenGDA" << std::endl;
    return 1;
  }

  int mype = gda_rank();
  int npes = gda_size();

  // Parse arguments
  int N = (argc > 1) ? atoi(argv[1]) : 512;
  int num_iterations = (argc > 2) ? atoi(argv[2]) : 100;
  bool use_barrier = true;  // Default: use barrier
  if (argc > 3) {
    // argv[3] = "nobarrier" to disable per-step barrier
    use_barrier = (strcmp(argv[3], "nobarrier") != 0);
  }
  N = (N / npes) * npes; // Make divisible by npes

  const int Ns = N / npes;
  const size_t stripe_size = (size_t)N * (size_t)Ns * sizeof(float);
  const size_t total_gpu_needed = 4 * stripe_size;

  if (mype == 0) {
    std::cout << "=== Repeated Matrix Multiplication Benchmark ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Stripe size: " << N << "x" << Ns << " (" << stripe_size << " bytes)" << std::endl;
    std::cout << "Ranks: " << npes << std::endl;
    std::cout << "Iterations: " << num_iterations << std::endl;
    std::cout << "Mode: " << (use_barrier ? "with per-step barrier" : "NO BARRIER (experimental)") << std::endl;
  }

  // Get GPU buffer
  float *gpu_buf = (float *)gda_gpu_buf();
  size_t gpu_buf_size = gda_gpu_buf_size();

  if (!gpu_buf || gpu_buf_size < total_gpu_needed) {
    std::cerr << "Rank " << mype << ": Insufficient GPU buffer. Need "
              << total_gpu_needed << ", have " << gpu_buf_size << std::endl;
    gda_finalize();
    return 1;
  }

  // Layout GPU buffer: [As | Bs | Cs | Bn]
  float *d_As = gpu_buf;
  float *d_Bs = gpu_buf + (size_t)N * (size_t)Ns;
  float *d_Cs = gpu_buf + 2 * (size_t)N * (size_t)Ns;
  float *d_Bn = gpu_buf + 3 * (size_t)N * (size_t)Ns;

  size_t offset_Bs = 1 * stripe_size;
  size_t offset_Bn = 3 * stripe_size;

  int blockSize = 256;
  int numBlocks = (N * Ns + blockSize - 1) / blockSize;

  // Initialize As and Bs (constant across iterations)
  hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                     d_As, N * Ns, mype, 0);
  hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                     d_Bs, N * Ns, mype, 1);
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate proxy barrier (shared across all iterations)
  // Use larger window_size to buffer against proxy thread scheduling delays
  // Each MM iteration uses npes barriers, so window should be >> npes
  int window_size = std::min(64, (int)GDA_PROXY_MAX_WINDOW_SIZE);
  gda_proxy_barrier_t *barrier = gda_proxy_barrier_alloc(window_size);
  if (!barrier) {
    std::cerr << "Rank " << mype << ": Failed to allocate proxy GPU barrier" << std::endl;
    gda_finalize();
    return 1;
  }

  gda_proxy_barrier_dev_t *dev_host = gda_proxy_barrier_get_dev(barrier);
  if (!dev_host) {
    std::cerr << "Rank " << mype << ": Failed to get device context" << std::endl;
    gda_proxy_barrier_free(barrier);
    gda_finalize();
    return 1;
  }

  // Print actual window_size (may be reduced due to DWQ budget)
  if (mype == 0) {
    std::cout << "Proxy barrier: window_size=" << dev_host->window_size
              << ", num_rounds=" << dev_host->num_rounds
              << ", barriers_per_iter=" << npes << std::endl;
    if (dev_host->window_size < npes) {
      std::cout << "WARNING: window_size < npes, GPU may stall waiting for proxy rearm" << std::endl;
    }
  }

  // Copy barrier device context to GPU
  gda_proxy_barrier_dev_t *d_barrier = nullptr;
  HIP_CHECK(hipMalloc(&d_barrier, sizeof(gda_proxy_barrier_dev_t)));
  HIP_CHECK(hipMemcpy(d_barrier, dev_host, sizeof(gda_proxy_barrier_dev_t),
                      hipMemcpyHostToDevice));

  // Create stream
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Storage for timing results
  std::vector<float> iteration_times(num_iterations);
  int left_neighbor = (mype - 1 + npes) % npes;

  // Start proxy thread ONCE - only needed in barrier mode
  gda_barrier();
  if (use_barrier) {
    if (gda_proxy_start(barrier) != 0) {
      std::cerr << "Rank " << mype << ": Failed to start proxy thread" << std::endl;
      gda_proxy_barrier_free(barrier);
      gda_finalize();
      return 1;
    }
  }

  // Warmup iteration
  if (mype == 0) {
    std::cout << "\nWarmup iteration..." << std::endl;
  }

  {
    // Reset Cs and Bn to zero
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_Cs, N * Ns);
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_Bn, N * Ns);
    HIP_CHECK(hipDeviceSynchronize());

    // Reinitialize Bs (it gets overwritten during MM)
    hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_Bs, N * Ns, mype, 1);
    HIP_CHECK(hipDeviceSynchronize());

    // Create handles for warmup
    PutHandleManager handles(npes, d_Bs, d_Bn, stripe_size, offset_Bs, offset_Bn, left_neighbor);
    if (!handles.allocate()) {
      std::cerr << "Rank " << mype << ": Failed to allocate handles for warmup" << std::endl;
      if (use_barrier) gda_proxy_stop(barrier);
      gda_proxy_barrier_free(barrier);
      gda_finalize();
      return 1;
    }

    gda_barrier();

    run_mm_iteration(stream, handles.deviceHandles(), d_barrier,
                     d_As, d_Bs, d_Bn, d_Cs, N, Ns, mype, npes, use_barrier);

    HIP_CHECK(hipStreamSynchronize(stream));

    handles.release();
    handles.freeDeviceArray();
  }

  gda_barrier();

  if (mype == 0) {
    std::cout << "Starting " << num_iterations << " timed iterations..."
              << (use_barrier ? " (with barrier)" : " (NO BARRIER - experimental)")
              << std::endl;
  }

  // Timed iterations
  timespec t_total_start, t_total_end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_start);

  for (int iter = 0; iter < num_iterations; iter++) {
    // Reset Cs and Bn to zero
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_Cs, N * Ns);
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_Bn, N * Ns);
    HIP_CHECK(hipDeviceSynchronize());

    // Reinitialize Bs
    hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_Bs, N * Ns, mype, 1);
    HIP_CHECK(hipDeviceSynchronize());

    // Create fresh handles for this iteration
    PutHandleManager handles(npes, d_Bs, d_Bn, stripe_size, offset_Bs, offset_Bn, left_neighbor);
    if (!handles.allocate()) {
      std::cerr << "Rank " << mype << ": Failed to allocate handles for iter " << iter << std::endl;
      if (use_barrier) gda_proxy_stop(barrier);
      gda_proxy_barrier_free(barrier);
      gda_finalize();
      return 1;
    }

    gda_barrier();

    // Wait for proxy to arm all slots before starting GPU work
    // Wait for proxy slots only if using barrier mode
    if (use_barrier) {
      // This ensures GPU won't stall waiting for slot_state == ARMED
      wait_for_slots_armed(dev_host, dev_host->window_size);
    }

    // Start timing
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    run_mm_iteration(stream, handles.deviceHandles(), d_barrier,
                     d_As, d_Bs, d_Bn, d_Cs, N, Ns, mype, npes, use_barrier);

    HIP_CHECK(hipStreamSynchronize(stream));

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    // Record time
    iteration_times[iter] = timediff_us(t0, t1);

    // Release handles for this iteration
    handles.release();
    handles.freeDeviceArray();

    // Progress indicator
    if (mype == 0 && ((iter + 1) % 10 == 0 || iter == num_iterations - 1)) {
      std::cout << "  Completed " << (iter + 1) << "/" << num_iterations
                << " iterations" << std::endl;
    }
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &t_total_end);

  // Stop proxy thread after ALL iterations complete (only if started)
  gda_proxy_stats_t stats = {0, 0, 0, 0};
  if (use_barrier) {
    gda_proxy_stop(barrier);
    gda_proxy_barrier_get_stats(barrier, &stats);
  }

  // Cleanup
  HIP_CHECK(hipFree(d_barrier));
  HIP_CHECK(hipStreamDestroy(stream));
  gda_proxy_barrier_free(barrier);

  gda_barrier();

  // Compute statistics
  float total_time_us = timediff_us(t_total_start, t_total_end);
  float min_time = *std::min_element(iteration_times.begin(), iteration_times.end());
  float max_time = *std::max_element(iteration_times.begin(), iteration_times.end());
  float sum_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0f);
  float avg_time = sum_time / num_iterations;

  // Compute median
  std::vector<float> sorted_times = iteration_times;
  std::sort(sorted_times.begin(), sorted_times.end());
  float median_time = sorted_times[num_iterations / 2];

  // Compute standard deviation
  float sq_sum = 0.0f;
  for (int i = 0; i < num_iterations; i++) {
    float diff = iteration_times[i] - avg_time;
    sq_sum += diff * diff;
  }
  float std_dev = std::sqrt(sq_sum / num_iterations);

  // Compute percentiles
  float p90 = sorted_times[(int)(num_iterations * 0.90)];
  float p99 = sorted_times[(int)(num_iterations * 0.99)];

  if (mype == 0) {
    std::cout << "\n=== Timing Results ===" << std::endl;
    std::cout << "Total time: " << total_time_us / 1e6 << " s" << std::endl;
    std::cout << "Iterations: " << num_iterations << std::endl;
    std::cout << "\nPer-iteration statistics (us):" << std::endl;
    std::cout << "  Min:    " << min_time << std::endl;
    std::cout << "  Max:    " << max_time << std::endl;
    std::cout << "  Avg:    " << avg_time << std::endl;
    std::cout << "  Median: " << median_time << std::endl;
    std::cout << "  StdDev: " << std_dev << std::endl;
    std::cout << "  P90:    " << p90 << std::endl;
    std::cout << "  P99:    " << p99 << std::endl;
    std::cout << "\nThroughput: " << num_iterations / (total_time_us / 1e6)
              << " iterations/sec" << std::endl;
    std::cout << "\nProxy barrier stats:" << std::endl;
    std::cout << "  Total rearms: " << stats.total_rearms << std::endl;
    std::cout << "  Total polls:  " << stats.queue_polls << std::endl;
    std::cout << "  CQ events:    " << stats.cq_events_drained << std::endl;

    // Print slow iterations for diagnosis
    float threshold = median_time * 2.0f;  // Flag iterations > 2x median
    std::cout << "\n=== Slow Iterations (> 2x median) ===" << std::endl;
    int slow_count = 0;
    for (int i = 0; i < num_iterations; i++) {
      if (iteration_times[i] > threshold) {
        std::cout << "  Iter " << i << ": " << iteration_times[i] << " us ("
                  << (iteration_times[i] / median_time) << "x median)" << std::endl;
        slow_count++;
      }
    }
    if (slow_count == 0) {
      std::cout << "  (none)" << std::endl;
    } else {
      std::cout << "  Total slow iterations: " << slow_count << " / " << num_iterations << std::endl;
    }

    // Print first 10 iteration times to check warmup effect
    std::cout << "\n=== First 10 Iterations (us) ===" << std::endl;
    for (int i = 0; i < std::min(10, num_iterations); i++) {
      std::cout << "  Iter " << i << ": " << iteration_times[i];
      if (iteration_times[i] > threshold) std::cout << " ***SLOW***";
      std::cout << std::endl;
    }
  }

  gda_barrier();
  gda_finalize();
  return 0;
}
