/**
 * mm_hip_gda_proxy_split.cpp - Distributed Matrix Multiplication with HIP + GDA
 * Proxy Barrier (Split-Kernel Version)
 *
 * This version splits the original "big kernel with a for-loop" into 3 kernels
 * per iteration: 1) trigger kernel:    gda_gpu_trigger(put_handles[s]) 2)
 * compute kernel:    Cs += As * cur_Bs  (cur_Bs alternates between d_Bs and
 * d_Bn) 3) wait+barrier kernel: gda_gpu_wait(...) +
 * gda_gpu_proxy_barrier_wait(...)
 *
 * The kernel boundary provides a natural grid-wide synchronization point, so we
 * do NOT need a software grid barrier.
 *
 * Run:
 *   srun -n <npes> ./mm_hip_gda_proxy_split [matrix_size]
 *   srun -n 2 ./mm_hip_gda_proxy_split 4096
 */

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

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

static void print_matrix(const float *mat, const int Is, const int Js) {
  for (int i = 0; i < Is; i++) {
    for (int j = 0; j < Js; j++)
      std::cout << mat[i * Js + j] << ' ';
    std::cout << '\n';
  }
}

static float timediff_us(const timespec &t_start, const timespec &t_end) {
  return (t_end.tv_sec - t_start.tv_sec) * 1.0e6f +
         (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3f;
}

/**
 * 1) Trigger kernel: one thread triggers the GPU-initiated RDMA op for
 * iteration s
 */
__global__ void gda_trigger_kernel(gda_gpu_handle_t *put_handles, int s) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    gda_gpu_trigger(put_handles[s]);
  }
}

/**
 * 2) Compute kernel:
 *   - Ensures RDMA writes are visible before reading (threadfence_system)
 *   - Computes: C[:, col_offset:col_offset+Ns] += As * cur_Bs
 *
 * NOTE: This keeps your original inner loop and atomicAdd semantics (slow but
 * preserves correctness).
 */
__global__ void mm_compute_kernel(const float *__restrict__ As,
                                  const float *__restrict__ cur_Bs,
                                  float *__restrict__ Cs, int N, int Ns,
                                  int mype, int npes, int s) {
  // Make remote RDMA writes visible on this GPU before reading cur_Bs
  __threadfence_system();

  int block_num = (mype + s) % npes;
  int col_offset = block_num * Ns;

  int k = blockIdx.x * blockDim.x + threadIdx.x; // over [0, N)
  int j = blockIdx.y * blockDim.y + threadIdx.y; // over [0, Ns)

  if (k < N && j < Ns) {
    float b_kj = cur_Bs[k * Ns + j];
    for (int i = 0; i < Ns; i++) {
      atomicAdd(&Cs[i * N + col_offset + j], As[i * N + k] * b_kj);
    }
  }
}

/**
 * 3) Wait + cross-rank proxy barrier kernel: one thread waits for completion
 * and performs barrier wait
 */
 #define NIC_SETTLE_DELAY 1000
__global__ void gda_wait_and_barrier_kernel(gda_gpu_handle_t *put_handles,
                                            gda_proxy_barrier_dev_t *barrier,
                                            int s) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    gda_gpu_wait(put_handles[s]);
    gda_gpu_proxy_barrier_wait(barrier);
  }
}

/**
 * HIP kernel to initialize matrix data
 */
__global__ void init_matrix_kernel(float *buf, int size, int mype,
                                   int pattern) {
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
 * Verify local result on each rank without gathering
 *
 * Each rank computes the expected C stripe based on the initialization
 * patterns: As[i, k] = (i * N + k + mype) % 11 + 7 B[k, j] from rank (j / Ns):
 * Bs[k, local_j] = (k * Ns + local_j + b_rank) % 13 + 5 C[i, j] = sum_k A[i, k]
 * * B[k, j]
 */
static bool verify_local_result(float *d_Cs, int N, int Ns, int mype, int npes,
                                bool verbose) {
  // Copy result from GPU
  std::vector<float> h_Cs(N * Ns);
  HIP_CHECK(hipMemcpy(h_Cs.data(), d_Cs, N * Ns * sizeof(float),
                      hipMemcpyDeviceToHost));

  // For large matrices, use sampling verification
  bool use_sampling = (N * Ns > 10000);
  int num_samples = use_sampling ? 1000 : (N * Ns);

  std::vector<int> indices(num_samples);
  if (use_sampling) {
    srand(mype + 42);
    for (int s = 0; s < num_samples; s++) {
      indices[s] = rand() % (N * Ns);
    }
  } else {
    for (int s = 0; s < num_samples; s++) {
      indices[s] = s;
    }
  }

  float max_rel_diff = 0.0f;
  int error_count = 0;
  const float tolerance = 1e-5f;

  for (int ss = 0; ss < num_samples; ss++) {
    int idx = indices[ss];
    int i = idx / N; // Row in [0, Ns)
    int j = idx % N; // Col in [0, N)

    int b_rank = j / Ns;
    int local_j = j % Ns;

    float expected = 0.0f;
    for (int k = 0; k < N; k++) {
      float a_val = (float)(((i * N + k) + mype) % 11 + 7);
      float b_val = (float)(((k * Ns + local_j) + b_rank) % 13 + 5);
      expected += a_val * b_val;
    }

    float actual = h_Cs[idx];
    float diff = std::abs(actual - expected);
    float rel_diff = (expected != 0.0f) ? (diff / std::abs(expected)) : diff;

    if (rel_diff > tolerance) {
      error_count++;
      if (verbose && error_count <= 5) {
        std::cout << "Rank " << mype << ": Mismatch at (" << i << ", " << j
                  << "): "
                  << "expected " << expected << ", got " << actual
                  << " (rel diff: " << rel_diff << ")\n";
      }
    }
    max_rel_diff = std::max(max_rel_diff, rel_diff);
  }

  bool passed = (error_count == 0);
  std::cout << "Rank " << mype << ": Verification "
            << (passed ? "PASSED" : "FAILED") << " ("
            << (num_samples - error_count) << "/" << num_samples << " samples"
            << (use_sampling ? " sampled" : " full")
            << ", max rel diff: " << max_rel_diff << ")\n";

  return passed;
}

int main(int argc, char **argv) {
  // Initialize GDA
  if (gda_init() != 0) {
    std::cerr << "Failed to initialize OpenGDA" << std::endl;
    return 1;
  }

  int mype = gda_rank();
  int npes = gda_size();

  // Matrix size
  int N = (argc > 1) ? atoi(argv[1]) : 4096;
  N = (N / npes) * npes; // Make divisible by npes

  const int Ns = N / npes;
  const size_t stripe_size = (size_t)N * (size_t)Ns * sizeof(float);
  const size_t total_gpu_needed = 4 * stripe_size;

  if (mype == 0) {
    std::cout << "Matrix stripe: " << N << 'x' << Ns << ", " << stripe_size
              << " bytes\n";
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

  // Offsets for RDMA destinations (double buffering)
  size_t offset_Bs = 1 * stripe_size; // Bs lives at this byte offset in gpu_buf
  size_t offset_Bn = 3 * stripe_size; // Bn lives at this byte offset in gpu_buf

  // Initialize matrices on GPU
  int blockSize = 256;
  int numBlocks = (N * Ns + blockSize - 1) / blockSize;

  hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                     d_As, N * Ns, mype, 0);
  hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                     d_Bs, N * Ns, mype, 1);
  hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                     d_Cs, N * Ns, mype, 2);
  hipLaunchKernelGGL(init_matrix_kernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                     d_Bn, N * Ns, mype, 2);
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate proxy barrier
  int window_size = 32;
  gda_proxy_barrier_t *barrier = gda_proxy_barrier_alloc(window_size);
  if (!barrier) {
    std::cerr << "Rank " << mype << ": Failed to allocate proxy GPU barrier"
              << std::endl;
    gda_finalize();
    return 1;
  }

  // Get device context for GPU kernel use
  gda_proxy_barrier_dev_t *dev_host = gda_proxy_barrier_get_dev(barrier);
  if (!dev_host) {
    std::cerr << "Rank " << mype << ": Failed to get device context"
              << std::endl;
    gda_proxy_barrier_free(barrier);
    gda_finalize();
    return 1;
  }

  // Pre-create put handles for all iterations
  int left_neighbor = (mype - 1 + npes) % npes;
  std::vector<gda_handle_t *> put_handles(npes);
  std::vector<gda_gpu_handle_t> gpu_put_handles(npes);

  for (int s = 0; s < npes; s++) {
    // Source alternates due to swapping:
    float *src = (s % 2 == 0) ? d_Bs : d_Bn;

    // Destination alternates to avoid read-write races:
    size_t dest_offset = (s % 2 == 0) ? offset_Bn : offset_Bs;

    put_handles[s] = gda_put(src, stripe_size, left_neighbor, dest_offset);
    if (!put_handles[s]) {
      std::cerr << "Rank " << mype << ": Failed to create put handle " << s
                << std::endl;
      gda_finalize();
      return 1;
    }
    gpu_put_handles[s] = put_handles[s]->gpu;
  }

  // Copy GPU handles array to device
  gda_gpu_handle_t *d_put_handles = nullptr;
  HIP_CHECK(hipMalloc(&d_put_handles, npes * sizeof(gda_gpu_handle_t)));
  HIP_CHECK(hipMemcpy(d_put_handles, gpu_put_handles.data(),
                      npes * sizeof(gda_gpu_handle_t), hipMemcpyHostToDevice));

  // Copy barrier device context to GPU memory
  gda_proxy_barrier_dev_t *d_barrier = nullptr;
  HIP_CHECK(hipMalloc(&d_barrier, sizeof(gda_proxy_barrier_dev_t)));
  HIP_CHECK(hipMemcpy(d_barrier, dev_host, sizeof(gda_proxy_barrier_dev_t),
                      hipMemcpyHostToDevice));

  timespec t0, t1;

  // Synchronize before starting
  gda_barrier();

  // Start proxy thread BEFORE launching kernels
  if (gda_proxy_start(barrier) != 0) {
    std::cerr << "Rank " << mype << ": Failed to start proxy thread"
              << std::endl;
    gda_proxy_barrier_free(barrier);
    gda_finalize();
    return 1;
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

  // Launch geometry for compute kernel (same as your original)
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (Ns + blockDim.y - 1) / blockDim.y);

  // Small kernels: one block is enough; use 256 threads so ROCm has reasonable
  // scheduling
  dim3 oneGrid(1);
  dim3 oneBlock(256);

  auto cur_Bs_ptr = [&](int s) -> float * {
    // Equivalent to swapping inside the big kernel:
    // iteration 0 reads Bs, iteration 1 reads Bn, etc.
    return (s % 2 == 0) ? d_Bs : d_Bn;
  };

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  gda_barrier();

  clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

  for (int s = 0; s < npes; s++) {
    hipLaunchKernelGGL(gda_trigger_kernel, oneGrid, oneBlock, 0, stream,
                       d_put_handles, s);

    hipLaunchKernelGGL(mm_compute_kernel, gridDim, blockDim, 0, stream, d_As,
                       cur_Bs_ptr(s), d_Cs, N, Ns, mype, npes, s);

    hipLaunchKernelGGL(gda_wait_and_barrier_kernel, oneGrid, oneBlock, 0,
                       stream, d_put_handles, d_barrier, s);
  }


  HIP_CHECK(hipStreamSynchronize(stream));
  clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

  // Stop proxy thread
  gda_proxy_stop(barrier);

  // Get proxy stats before cleanup
  gda_proxy_stats_t stats;
  gda_proxy_barrier_get_stats(barrier, &stats);

  // Cleanup
  gda_flush();
  for (int s = 0; s < npes; s++) {
    gda_free(put_handles[s]);
  }
  gda_proxy_barrier_free(barrier);

  HIP_CHECK(hipFree(d_put_handles));
  HIP_CHECK(hipFree(d_barrier));

  gda_barrier();

  if (mype == 0) {
    std::cout << "GDA + HIP (proxy barrier, split kernels): "
              << timediff_us(t0, t1) << " us\n";
    std::cout << "  Barrier iterations: " << npes << "\n";
    std::cout << "  Proxy rearms: " << stats.total_rearms << "\n";
    std::cout << "  Proxy polls: " << stats.queue_polls << "\n";
    std::cout << "  CQ events: " << stats.cq_events_drained << "\n";
  }

  // Verify result on each rank locally
  bool verify_passed = verify_local_result(d_Cs, N, Ns, mype, npes, false);

  // Print matrix for small sizes (debugging)
  if (N <= 16 && mype == 0) {
    auto h_Cs = new float[N * Ns];
    HIP_CHECK(hipMemcpy(h_Cs, d_Cs, stripe_size, hipMemcpyDeviceToHost));
    std::cout << "\nC matrix (rank 0 stripe):\n";
    print_matrix(h_Cs, Ns, N);
    delete[] h_Cs;
  }

  // Collect verification results
  gda_barrier();
  if (!verify_passed) {
    std::cerr << "Rank " << mype << ": VERIFICATION FAILED\n";
  }

  gda_barrier();
  gda_finalize();
  return 0;
}
