/**
 * barrier_test.cpp - Test GPU-side barrier
 *
 * Simple test to verify GPU barrier works correctly.
 *
 * Run:
 *   srun -n 2 ./barrier_test
 */

#include <iostream>
#include <hip/hip_runtime.h>
#include "gda.h"

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

/**
 * Debug kernel - just test basic operations step by step
 */
__global__ void barrier_debug_kernel(gda_gpu_barrier_t* barrier, int* debug_out)
{
    // Step 0: Verify we can access barrier struct
    debug_out[0] = barrier->mype;
    debug_out[1] = barrier->npes;
    debug_out[2] = barrier->num_phases;

    // Step 1: Read sync_counter
    debug_out[3] = (int)(*barrier->sync_counter);

    // Step 2: Read sync_arr[mype]
    debug_out[4] = (int)(barrier->sync_arr[barrier->mype]);

    // Step 3: Write to sync_arr[mype]
    barrier->sync_arr[barrier->mype] = 1;
    __threadfence_system();
    debug_out[5] = (int)(barrier->sync_arr[barrier->mype]);

    // Step 4: Check phase_handles[0] addresses
    debug_out[6] = (barrier->phase_handles[0].trigger_addr != nullptr) ? 1 : 0;
    debug_out[7] = (barrier->phase_handles[0].completion_addr != nullptr) ? 1 : 0;

    // Step 5: Trigger and wait for DWQ completion
    if (barrier->num_phases > 0) {
        gda_gpu_trigger(barrier->phase_handles[0]);
        debug_out[8] = 1;  // triggered

        // Step 6: Wait for completion (with timeout counter)
        int wait_count = 0;
        while (*barrier->phase_handles[0].completion_addr == 0) {
            wait_count++;
            if (wait_count > 100000000) {
                debug_out[9] = -1;  // timeout
                debug_out[10] = wait_count;
                return;
            }
        }
        __threadfence_system();
        debug_out[9] = (int)(*barrier->phase_handles[0].completion_addr);
        debug_out[10] = wait_count;

        // Step 7: Wait for remote value (what the barrier actually waits on)
        int src = barrier->phase_sources[0];
        debug_out[11] = src;  // source rank we're waiting on
        debug_out[12] = (int)(barrier->sync_arr[src]);  // Current value at source slot

        // Wait for source rank's value to appear
        int wait_count2 = 0;
        while (barrier->sync_arr[src] < 1) {
            wait_count2++;
            if (wait_count2 > 100000000) {
                debug_out[13] = -1;  // timeout waiting for remote
                debug_out[14] = wait_count2;
                return;
            }
        }
        __threadfence_system();
        debug_out[13] = (int)(barrier->sync_arr[src]);  // Final value
        debug_out[14] = wait_count2;
    }

    // Step 8: Done marker
    debug_out[15] = 999;
}

/**
 * Test kernel - run actual barrier iterations using gda_gpu_barrier_wait
 */
__global__ void barrier_test_kernel(gda_gpu_barrier_t* barrier, int num_iters, int* results)
{
    for (int i = 0; i < num_iters; i++) {
        gda_gpu_barrier_wait(barrier);
        results[i] = 1;  // Mark iteration complete
    }
}

int main(int argc, char** argv)
{
    // Initialize GDA
    if (gda_init() != 0) {
        std::cerr << "Failed to initialize OpenGDA" << std::endl;
        return 1;
    }

    int mype = gda_rank();
    int npes = gda_size();

    std::cout << "Rank " << mype << "/" << npes << " initialized" << std::endl;

    if (npes < 2) {
        std::cerr << "Need at least 2 ranks" << std::endl;
        gda_finalize();
        return 1;
    }

    // Allocate GPU barrier with 10 iterations
    const int max_barrier_iters = 10;
    std::cout << "Rank " << mype << ": Allocating GPU barrier..." << std::endl;
    gda_gpu_barrier_t* barrier = gda_gpu_barrier_alloc(max_barrier_iters);
    if (!barrier) {
        std::cerr << "Rank " << mype << ": Failed to allocate GPU barrier" << std::endl;
        gda_finalize();
        return 1;
    }
    std::cout << "Rank " << mype << ": GPU barrier allocated, num_phases=" << barrier->num_phases << std::endl;

    // Print barrier info
    std::cout << "Rank " << mype << ": barrier info:" << std::endl;
    std::cout << "  sync_arr=" << (void*)barrier->sync_arr << std::endl;
    std::cout << "  sync_counter=" << (void*)barrier->sync_counter << std::endl;
    for (int i = 0; i < barrier->num_phases; i++) {
        std::cout << "  phase " << i << ": target=" << barrier->phase_targets[i]
                  << ", source=" << barrier->phase_sources[i] << std::endl;
    }

    // Copy barrier to GPU
    gda_gpu_barrier_t* d_barrier;
    HIP_CHECK(hipMalloc(&d_barrier, sizeof(gda_gpu_barrier_t)));
    HIP_CHECK(hipMemcpy(d_barrier, barrier, sizeof(gda_gpu_barrier_t), hipMemcpyHostToDevice));

    // Allocate debug output array
    const int debug_size = 16;
    int* d_debug;
    HIP_CHECK(hipMalloc(&d_debug, debug_size * sizeof(int)));
    HIP_CHECK(hipMemset(d_debug, -1, debug_size * sizeof(int)));

    // Skip debug kernel - go directly to full barrier test
    HIP_CHECK(hipFree(d_debug));
    hipError_t err;

    // Now test full barrier iterations
    std::cout << "Rank " << mype << ": Testing full barrier..." << std::endl;

    // Reset barrier state
    gda_gpu_barrier_reset(barrier);

    // Recopy barrier to device (after reset)
    HIP_CHECK(hipMemcpy(d_barrier, barrier, sizeof(gda_gpu_barrier_t), hipMemcpyHostToDevice));

    // Allocate results array
    const int num_iters = 5;  // Test with 5 iterations
    int* d_results;
    HIP_CHECK(hipMalloc(&d_results, num_iters * sizeof(int)));
    HIP_CHECK(hipMemset(d_results, 0, num_iters * sizeof(int)));

    // Synchronize before test
    gda_barrier();

    // Launch barrier test kernel
    hipLaunchKernelGGL(barrier_test_kernel, dim3(1), dim3(1), 0, 0,
                       d_barrier, num_iters, d_results);

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Rank " << mype << ": Barrier test kernel failed: " << hipGetErrorString(err) << std::endl;
    }

    // Check results
    int h_results[num_iters];
    HIP_CHECK(hipMemcpy(h_results, d_results, num_iters * sizeof(int), hipMemcpyDeviceToHost));

    bool all_passed = true;
    for (int i = 0; i < num_iters; i++) {
        if (h_results[i] != 1) {
            std::cout << "Rank " << mype << ": Barrier iteration " << i << " FAILED (result=" << h_results[i] << ")" << std::endl;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "Rank " << mype << ": All " << num_iters << " barrier iterations PASSED!" << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipFree(d_results));
    HIP_CHECK(hipFree(d_barrier));
    gda_gpu_barrier_free(barrier);

    gda_barrier();
    gda_finalize();

    std::cout << "Rank " << mype << ": Done!" << std::endl;
    return 0;
}
