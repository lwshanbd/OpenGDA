/**
 * proxy_barrier_test.cpp - Test Dissemination Barrier with CPU Proxy
 *
 * Tests the O(log P) Dissemination barrier implementation with unlimited
 * iterations using CPU proxy thread for DWQ rearm.
 *
 * Run:
 *   srun -N <num_nodes> -n <num_ranks> ./proxy_barrier_test [num_iters] [window_size]
 *   srun -n 2 ./proxy_barrier_test 10000
 *   srun -n 2 ./proxy_barrier_test 10000 32
 */

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include "gda.h"
#include "gda_barrier_proxy.h"

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

/**
 * GPU persistent kernel - runs barrier iterations using dissemination barrier.
 * Uses the gda_gpu_proxy_barrier_wait macro.
 */
__global__ void barrier_kernel(
    gda_proxy_barrier_dev_t* dev,
    int num_iters,
    uint64_t* out_final_epoch,
    uint64_t* out_round_recv,  // [num_rounds] - final d_round_recv values
    uint64_t* out_cycles,
    int* progress_counter
) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    int mype = dev->mype;
    int num_rounds = dev->num_rounds;

    // Progress indicator at start
    printf("GPU[%d]: Kernel started, num_iters=%d, rounds=%d, window=%d\n",
           mype, num_iters, num_rounds, dev->window_size);

    uint64_t start = clock64();

    // Run barrier iterations
    for (int i = 0; i < num_iters; i++) {
        gda_gpu_proxy_barrier_wait(dev);

        // Progress indicator every 1000 iterations (only rank 0)
        if (mype == 0 && (i + 1) % 1000 == 0) {
            atomicAdd(progress_counter, 1);
        }
    }

    uint64_t end = clock64();

    // Output final values
    *out_final_epoch = __atomic_load_n((unsigned long long*)dev->d_epoch, __ATOMIC_ACQUIRE);
    *out_cycles = end - start;

    // Copy d_round_recv values for verification
    for (int k = 0; k < num_rounds; k++) {
        out_round_recv[k] = __atomic_load_n((unsigned long long*)&dev->d_round_recv[k], __ATOMIC_ACQUIRE);
    }

    printf("GPU[%d]: Completed %d iterations! Final epoch=%lu\n",
           mype, num_iters, *out_final_epoch);
}

/**
 * Minimal test kernel - verify GPU can access barrier structure.
 */
__global__ void access_test(gda_proxy_barrier_dev_t* dev, int* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    printf("GPU: Access test - dev=%p\n", (void*)dev);

    // Read basic fields
    results[0] = dev->mype;
    results[1] = dev->npes;
    results[2] = dev->num_rounds;
    results[3] = dev->window_size;

    printf("GPU: mype=%d, npes=%d, rounds=%d, window=%d\n",
           results[0], results[1], results[2], results[3]);

    // Check pointers are accessible
    results[4] = (dev->slot_state != nullptr) ? 1 : 0;
    results[5] = (dev->slot_trigger_addrs != nullptr) ? 1 : 0;
    results[6] = (dev->d_round_recv != nullptr) ? 1 : 0;
    results[7] = (dev->d_slot_done != nullptr) ? 1 : 0;
    results[8] = (dev->d_epoch != nullptr) ? 1 : 0;

    printf("GPU: Pointers - slot_state=%d, triggers=%d, recv=%d, done=%d, epoch=%d\n",
           results[4], results[5], results[6], results[7], results[8]);

    // Try to read slot_state[0]
    if (dev->slot_state) {
        results[9] = __atomic_load_n(&dev->slot_state[0], __ATOMIC_ACQUIRE);
        printf("GPU: slot_state[0]=%d\n", results[9]);
    }

    results[15] = 42;  // Success marker
}

int main(int argc, char** argv)
{
    // Parse arguments
    int num_iters = 10000;      // Default: 10000 iterations
    int window_size = 0;        // Default: use library default (16)

    if (argc > 1) {
        num_iters = atoi(argv[1]);
        if (num_iters <= 0) num_iters = 10000;
    }
    if (argc > 2) {
        window_size = atoi(argv[2]);
    }

    // Initialize GDA
    if (gda_init() != 0) {
        std::cerr << "Failed to initialize OpenGDA" << std::endl;
        return 1;
    }

    int mype = gda_rank();
    int npes = gda_size();

    std::cout << "Rank " << mype << "/" << npes << ": Proxy barrier test - "
              << num_iters << " iterations, window=" << (window_size > 0 ? window_size : 16)
              << std::endl;

    if (npes < 2) {
        std::cerr << "Need at least 2 ranks" << std::endl;
        gda_finalize();
        return 1;
    }

    // Get GPU info
    hipDeviceProp_t prop;
    int device_id;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    double gpu_clock_mhz = prop.clockRate / 1000.0;
    std::cout << "Rank " << mype << ": GPU " << device_id << " - "
              << prop.name << " @ " << gpu_clock_mhz << " MHz" << std::endl;

    // Allocate proxy barrier (uses Dissemination algorithm)
    gda_proxy_barrier_t* pb = gda_proxy_barrier_alloc(window_size);
    if (!pb) {
        std::cerr << "Rank " << mype << ": Failed to allocate proxy barrier" << std::endl;
        gda_finalize();
        return 1;
    }

    // Get device context
    gda_proxy_barrier_dev_t* dev_host = gda_proxy_barrier_get_dev(pb);
    if (!dev_host) {
        std::cerr << "Rank " << mype << ": Failed to get device context" << std::endl;
        gda_proxy_barrier_free(pb);
        gda_finalize();
        return 1;
    }

    int num_rounds = dev_host->num_rounds;
    std::cout << "Rank " << mype << ": Barrier allocated with "
              << num_rounds << " rounds (log2 " << npes << ")" << std::endl;

    // Copy device context to GPU
    gda_proxy_barrier_dev_t* dev_gpu = nullptr;
    HIP_CHECK(hipMalloc(&dev_gpu, sizeof(gda_proxy_barrier_dev_t)));
    HIP_CHECK(hipMemcpy(dev_gpu, dev_host, sizeof(gda_proxy_barrier_dev_t), hipMemcpyHostToDevice));

    // Run access test first
    int* d_access_results;
    HIP_CHECK(hipMalloc(&d_access_results, 16 * sizeof(int)));
    HIP_CHECK(hipMemset(d_access_results, -1, 16 * sizeof(int)));

    std::cout << "Rank " << mype << ": Running access test..." << std::endl;
    hipLaunchKernelGGL(access_test, dim3(1), dim3(1), 0, 0, dev_gpu, d_access_results);
    HIP_CHECK(hipDeviceSynchronize());

    int h_access_results[16];
    HIP_CHECK(hipMemcpy(h_access_results, d_access_results, 16 * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_access_results));

    if (h_access_results[15] != 42) {
        std::cerr << "Rank " << mype << ": Access test FAILED" << std::endl;
        HIP_CHECK(hipFree(dev_gpu));
        gda_proxy_barrier_free(pb);
        gda_finalize();
        return 1;
    }
    std::cout << "Rank " << mype << ": Access test PASSED" << std::endl;

    // Allocate output buffers
    uint64_t* d_final_epoch = nullptr;
    uint64_t* d_round_recv = nullptr;
    uint64_t* d_cycles = nullptr;
    int* d_progress = nullptr;

    HIP_CHECK(hipMalloc(&d_final_epoch, sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&d_round_recv, sizeof(uint64_t) * num_rounds));
    HIP_CHECK(hipMalloc(&d_cycles, sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&d_progress, sizeof(int)));
    HIP_CHECK(hipMemset(d_progress, 0, sizeof(int)));

    // Synchronize before main test
    gda_barrier();
    std::cout << "Rank " << mype << ": Starting proxy thread..." << std::endl;

    // Start proxy thread
    if (gda_proxy_start(pb) != 0) {
        std::cerr << "Rank " << mype << ": Failed to start proxy thread" << std::endl;
        HIP_CHECK(hipFree(dev_gpu));
        gda_proxy_barrier_free(pb);
        gda_finalize();
        return 1;
    }

    // Final sync before kernel launch
    gda_barrier();

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Rank " << mype << ": Launching barrier kernel..." << std::endl;

    // Launch kernel
    hipLaunchKernelGGL(barrier_kernel, dim3(1), dim3(1), 0, 0,
                       dev_gpu, num_iters, d_final_epoch, d_round_recv, d_cycles, d_progress);

    HIP_CHECK(hipDeviceSynchronize());

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Stop proxy thread
    gda_proxy_stop(pb);

    std::cout << "Rank " << mype << ": Kernel completed" << std::endl;

    // Get results
    uint64_t h_final_epoch, h_cycles;
    uint64_t* h_round_recv = new uint64_t[num_rounds];

    HIP_CHECK(hipMemcpy(&h_final_epoch, d_final_epoch, sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_cycles, d_cycles, sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_round_recv, d_round_recv, sizeof(uint64_t) * num_rounds, hipMemcpyDeviceToHost));

    // Get statistics
    gda_proxy_stats_t stats;
    gda_proxy_barrier_get_stats(pb, &stats);

    // Synchronize for ordered output
    gda_barrier();

    // Verify results
    bool pass = true;

    // Check final epoch
    if (h_final_epoch != (uint64_t)num_iters) {
        std::cerr << "Rank " << mype << ": FAIL - final epoch = " << h_final_epoch
                  << ", expected " << num_iters << std::endl;
        pass = false;
    }

    // Check round recv values (each should be >= num_iters)
    for (int k = 0; k < num_rounds; k++) {
        if (h_round_recv[k] < (uint64_t)num_iters) {
            std::cerr << "Rank " << mype << ": FAIL - d_round_recv[" << k << "] = "
                      << h_round_recv[k] << ", expected >= " << num_iters << std::endl;
            pass = false;
        }
    }

    // Compute timing
    double avg_barrier_us = (elapsed_ms * 1000.0) / num_iters;
    double gpu_time_us = (double)h_cycles / gpu_clock_mhz;

    // Print results
    std::cout << "\nRank " << mype << " Results:" << std::endl;
    std::cout << "  Iterations: " << num_iters << std::endl;
    std::cout << "  Window size: " << dev_host->window_size << std::endl;
    std::cout << "  Rounds: " << num_rounds << std::endl;
    std::cout << "  Final epoch: " << h_final_epoch << " (expected: " << num_iters << ")" << std::endl;
    std::cout << "  Round recv:";
    for (int k = 0; k < num_rounds; k++) {
        std::cout << " [" << k << "]=" << h_round_recv[k];
    }
    std::cout << std::endl;
    std::cout << "  Proxy rearms: " << stats.total_rearms << std::endl;
    std::cout << "  Proxy polls: " << stats.queue_polls << std::endl;
    std::cout << "  CQ events: " << stats.cq_events_drained << std::endl;
    std::cout << "  Wall time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "  GPU time: " << (gpu_time_us / 1000.0) << " ms ("
              << (gpu_time_us / num_iters) << " us/barrier)" << std::endl;
    std::cout << "  Avg latency: " << avg_barrier_us << " us/barrier" << std::endl;
    std::cout << "  Throughput: " << (num_iters / elapsed_ms * 1000.0) << " barriers/sec" << std::endl;
    std::cout << "  Status: " << (pass ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    delete[] h_round_recv;
    HIP_CHECK(hipFree(d_final_epoch));
    HIP_CHECK(hipFree(d_round_recv));
    HIP_CHECK(hipFree(d_cycles));
    HIP_CHECK(hipFree(d_progress));
    HIP_CHECK(hipFree(dev_gpu));

    gda_proxy_barrier_free(pb);
    gda_finalize();

    std::cout << "Rank " << mype << ": Done123!" << std::endl;
    return pass ? 0 : 1;
}
