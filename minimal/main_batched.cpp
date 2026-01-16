/*
 * main_batched.cpp - Batched DWQ benchmark with counter pool
 *
 * Demonstrates how to exceed hardware DWQ limits by processing
 * streams in batches and reusing counter pool slots.
 */

#include <cstdio>
#include <cstdlib>

#include "hip_device_context.hpp"
#include "pmi_session.hpp"
#include "fabric_dwq_context.hpp"
#include "benchmark_batched.hpp"

int main() {
    // Unset ROCR_VISIBLE_DEVICES before any initialization
    unset_rocr_visible_devices();

    // Initialize GPU
    constexpr int GPU_ID = 7;
    HipDeviceContext hip(GPU_ID);

    // Initialize PMI2
    PmiSession pmi;

    if (pmi.size != 2) {
        if (pmi.rank == 0) {
            fprintf(stderr, "This program requires exactly 2 processes\n");
        }
        return 1;
    }

    // Initialize Fabric/DWQ Context
    FabricDwqContext fabric(pmi.rank);

    // Run Batched Benchmark (16 streams in batches of 6)
    BatchedBenchmarkRunner runner(pmi, fabric);
    runner.run();

    return 0;
}
