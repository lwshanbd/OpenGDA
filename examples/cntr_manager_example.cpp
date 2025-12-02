/**
 * cntr_manager_example.cpp - Demonstrates Counter Manager usage
 *
 * This example shows:
 * 1. Automatic counter initialization during gda_init()
 * 2. Allocating counter pairs for DWQ operations
 * 3. Accessing counter information (MMIO addresses, GPU pointers)
 * 4. GPU kernel writing to trigger counter (verifying GPU access)
 * 5. Querying statistics
 * 6. Proper cleanup
 *
 * Compile:
 *   hipcc -o cntr_example cntr_manager_example.cpp \
 *       -I../src -I/opt/cray/libfabric/2.1/include \
 *       -L../build/src -lopengda -Wl,-rpath,/p/lustre2/shan4/opengda/build/src \
 *       -DUSE_AMDGPU
 */

#include "gda.h"
#include "network/ofi.hpp"
#include <iostream>
#include <iomanip>
#include <sched.h>

#ifdef USE_AMDGPU
#include <hip/hip_runtime.h>

// GPU kernel to write to trigger counter
__global__ void write_trigger_counter(volatile uint64_t* counter_addr, uint64_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Write the value to trigger counter doorbell
        *counter_addr = value;
        __threadfence_system();  // Ensure write is visible to NIC
        __syncthreads();
    }
}
#endif

// Access to global OFI instance
extern std::unique_ptr<OFI> ofi;

int main() {
    
    std::cout << "=== OpenGDA Counter Manager Example ===\n\n";

    // ========================================================================
    // Step 1: Initialize OpenGDA
    // ========================================================================

    std::cout << "Initializing OpenGDA...\n";
    if (gda_init() != 0) {
        std::cerr << "Failed to initialize OpenGDA\n";
        return 1;
    }

    std::cout << "OpenGDA version: " << gda_get_version() << "\n\n";

    // Counters are automatically created during gda_init()!
    std::cout << "✓ Counters automatically initialized\n\n";

    // ========================================================================
    // Step 2: Check Initial Statistics
    // ========================================================================

    // std::cout << "--- Initial Counter Statistics ---\n";
    // ofi->print_cntr_stats();
    // std::cout << "\n";

    // ========================================================================
    // Step 3: Access Initialized Counter Pairs
    // ========================================================================

    std::cout << "--- Accessing Pre-initialized Counter Pairs ---\n";
    std::cout << "All 16 counter pairs were created during gda_init()\n";
    std::cout << "Let's examine the first 3 pairs:\n\n";

    CntrManager* mgr = ofi->get_cntr_manager();

    for (int i = 0; i < 3; i++) {
        CntrManager::CntrPair* pair = mgr->get_pair(i);
        if (!pair) {
            std::cerr << "Failed to get counter pair " << i << "\n";
            gda_finalize();
            return 1;
        }

        std::cout << "Counter pair " << i << ":\n";

        // Show trigger counter info
        CntrManager::CntrInfo* trigger = pair->trigger;
        std::cout << "  Trigger counter:\n";
        std::cout << "    - libfabric handle: " << trigger->cntr << "\n";
        std::cout << "    - MMIO address:     " << trigger->mmio_addr << "\n";
        std::cout << "    - MMIO length:      " << trigger->mmio_len << " bytes\n";

#ifdef USE_AMDGPU
        if (trigger->hip_registered) {
            std::cout << "    - GPU device addr:  " << (void*)trigger->dev_addr << "\n";
            std::cout << "    - HIP registered:   yes\n";
        } else {
            std::cout << "    - HIP registered:   no (warning)\n";
        }
#endif

        // Show completion counter info
        CntrManager::CntrInfo* completion = pair->completion;
        std::cout << "  Completion counter:\n";
        std::cout << "    - libfabric handle: " << completion->cntr << "\n";
        std::cout << "    - MMIO address:     " << completion->mmio_addr << "\n";
        std::cout << "    - MMIO length:      " << completion->mmio_len << " bytes\n";

#ifdef USE_AMDGPU
        if (completion->hip_registered) {
            std::cout << "    - GPU device addr:  " << (void*)completion->dev_addr << "\n";
            std::cout << "    - HIP registered:   yes\n";
        } else {
            std::cout << "    - HIP registered:   no (warning)\n";
        }
#endif

        std::cout << "\n";
    }

    // ========================================================================
    // Step 4: GPU Kernel Test - Write to Trigger Counter
    // ========================================================================

#ifdef USE_AMDGPU
    std::cout << "--- GPU Kernel Test: Writing to Trigger Counter ---\n";

    // Use the first counter pair (index 0) for testing
    CntrManager::CntrPair* test_pair = mgr->get_pair(0);
    CntrManager::CntrInfo* test_trigger = test_pair->trigger;

    if (!test_trigger->hip_registered) {
        std::cerr << "Warning: Trigger counter not registered with HIP, skipping GPU test\n\n";
    } else {
        // Create HIP stream
        hipStream_t stream;
        hipError_t hip_err = hipStreamCreate(&stream);
        if (hip_err != hipSuccess) {
            std::cerr << "hipStreamCreate failed: " << hipGetErrorString(hip_err) << "\n\n";
        } else {
            // Read initial counter value
            uint64_t initial_value = fi_cntr_read(test_trigger->cntr);
            std::cout << "Initial trigger counter value: " << initial_value << "\n";

            // IMPORTANT: Counter doorbell is ADD operation (accumulation), not SET!
            // The written value will be ADDED to the current counter value.
            uint64_t increment_value = 5;
            std::cout << "Launching GPU kernel to increment counter by: " << increment_value << "\n";
            std::cout << "GPU device pointer: " << (void*)test_trigger->dev_addr << "\n";

            // Launch GPU kernel to write to trigger counter
            hipLaunchKernelGGL(write_trigger_counter,
                               dim3(1),           // 1 block
                               dim3(1),           // 1 thread (only need one)
                               0,                 // no shared memory
                               stream,            // use stream
                               test_trigger->dev_addr,  // GPU device pointer
                               increment_value);        // value to ADD

            // Wait for kernel to complete
            hip_err = hipStreamSynchronize(stream);
            if (hip_err != hipSuccess) {
                std::cerr << "hipStreamSynchronize failed: " << hipGetErrorString(hip_err) << "\n\n";
            } else {
                std::cout << "✓ GPU kernel completed\n";

                // IMPORTANT: Counter update is NOT immediate!
                // Need to poll and wait for the update to propagate to NIC hardware
                std::cout << "Waiting for counter update...\n";

                uint64_t after_value = initial_value;
                uint64_t expected_value = initial_value + increment_value;
                int timeout = 1000;  // Max attempts
                int attempts = 0;

                while (attempts < timeout) {
                    after_value = fi_cntr_read(test_trigger->cntr);
                    if (after_value != initial_value) {
                        break;  // Counter updated!
                    }
                    attempts++;
                    sched_yield();  // Give up CPU to allow hardware to process
                }

                std::cout << "Counter updated after " << attempts << " attempts\n";
                std::cout << "  Initial value:  " << initial_value << "\n";
                std::cout << "  New value:      " << after_value << "\n";
                std::cout << "  Increment:      " << (after_value - initial_value) << "\n";
                std::cout << "  Expected incr:  " << increment_value << "\n";

                // Verify the write succeeded
                if (after_value == expected_value) {
                    std::cout << "✓ SUCCESS: GPU successfully wrote to trigger counter!\n";
                } else if (after_value != initial_value) {
                    std::cout << "⚠ PARTIAL SUCCESS: Counter updated but value unexpected\n";
                    std::cout << "  This might be due to other operations affecting the counter\n";
                } else {
                    std::cout << "✗ FAILED: Counter did not update\n";
                }
            }

            hipError_t destroy_err = hipStreamDestroy(stream);
            (void)destroy_err;  // Suppress unused warning
            std::cout << "\n";
        }
    }
#else
    std::cout << "--- GPU Test Skipped (USE_AMDGPU not defined) ---\n\n";
#endif

    // ========================================================================
    // Step 5: Counter Statistics
    // ========================================================================

    // std::cout << "--- Counter Statistics ---\n";
    // ofi->print_cntr_stats();
    // std::cout << "\n";

    CntrManager::CntrStats stats = ofi->get_cntr_stats();
    std::cout << "Total available pairs: " << stats.total_pairs << "\n\n";

    // ========================================================================
    // Step 6: Demonstrate Counter Access Patterns
    // ========================================================================

    std::cout << "--- Counter Access Patterns ---\n";

    // Access counter pair 0 in different ways
    std::cout << "Accessing counter pair 0:\n";

    // Pattern 1: Via CntrManager
    CntrManager::CntrPair* pair0 = mgr->get_pair(0);
    std::cout << "  Via mgr->get_pair(0):       " << pair0 << "\n";

    // Pattern 2: Via OFI
    CntrManager::CntrPair* pair0_via_ofi = ofi->get_cntr_pair(0);
    std::cout << "  Via ofi->get_cntr_pair(0):  " << pair0_via_ofi << "\n";
    std::cout << "  Same pointer? " << (pair0 == pair0_via_ofi ? "yes" : "no") << "\n\n";

    // Pattern 3: Direct counter access
    CntrManager::CntrInfo* trigger0 = mgr->get_trigger_cntr(0);
    CntrManager::CntrInfo* completion0 = mgr->get_completion_cntr(0);
    std::cout << "Direct counter access (pair 0):\n";
    std::cout << "  Trigger:    " << trigger0->cntr << "\n";
    std::cout << "  Completion: " << completion0->cntr << "\n\n";

    // ========================================================================
    // Step 7: Simulate DWQ Usage
    // ========================================================================

    std::cout << "--- Simulated DWQ Usage ---\n";

    // In real DWQ code, you would:
    // 1. Setup fi_deferred_work structure
    // 2. Assign trigger and completion counters
    // 3. Queue the work with fi_control(FI_QUEUE_WORK)
    // 4. GPU writes to trigger->dev_addr to trigger operation
    // 5. NIC updates completion counter when done

    CntrManager::CntrPair* dwq_pair = mgr->get_pair(0);
    std::cout << "Example DWQ setup using counter pair 0:\n";
    std::cout << "  struct fi_deferred_work work;\n";
    std::cout << "  work.triggering_cntr = " << dwq_pair->trigger->cntr << ";\n";
    std::cout << "  work.completion_cntr = " << dwq_pair->completion->cntr << ";\n";
    std::cout << "  work.threshold = 1;\n";
    std::cout << "  // ... configure operation ...\n";
    std::cout << "  fi_control(&domain->fid, FI_QUEUE_WORK, &work);\n\n";

#ifdef USE_AMDGPU
    std::cout << "GPU kernel would write to:\n";
    std::cout << "  trigger_addr = " << (void*)dwq_pair->trigger->dev_addr << "\n";
    std::cout << "  *trigger_addr = threshold;  // Trigger RDMA\n\n";
#endif

    // ========================================================================
    // Step 8: Cleanup
    // ========================================================================

    std::cout << "Finalizing OpenGDA...\n";
    if (gda_finalize() != 0) {
        std::cerr << "Failed to finalize OpenGDA\n";
        return 1;
    }

    // Counters are automatically cleaned up during gda_finalize()!
    std::cout << "✓ Counters automatically finalized\n\n";

    std::cout << "=== Example Completed Successfully ===\n";
    return 0;
}

// g++ -o cntr_example cntr_manager_example.cpp -I../src -L../build -lopengda -Wl,-rpath,../build -DUSE_AMDGPU=ON
// hipcc -o cntr_example cntr_manager_example.cpp -I../build/include -L../build/src -lopengda -Wl,-rpath,$(realpath ../build/src) -DUSE_AMDGPU=ON