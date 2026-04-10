/*
 * main.cpp - GPU-Direct Async (GDA) RDMA Benchmark using Deferred Work Queue
 *
 * Minimal encapsulation of the prototype implementation.
 *
 * Architecture:
 * - CPU: Queue fi_deferred_work via fi_control(FI_QUEUE_WORK)
 * - GPU: Write triggering counter MMIO → NIC detects threshold
 * - NIC: Execute queued RMA write autonomously
 * - NIC: Increment completion counter when done
 * - NIC: Execute chained atomic to signal GPU completion
 * - GPU: Poll atomic_result, exit when signaled
 */

#include <cstdio>
#include <cstdlib>

#include "hip_device_context.hpp"
#include "pmi_session.hpp"
#include "device_affinity.hpp"
#include "fabric_dwq_context.hpp"
#include "benchmark_runner.hpp"

int main() {
    // =========================================================================
    // CRITICAL: Unset ROCR_VISIBLE_DEVICES BEFORE ANY INITIALIZATION
    // Must happen before PMI2/HIP init - they may read env vars during init
    // Flux sets ROCR_VISIBLE_DEVICES which causes HSA virtualization issues
    // =========================================================================
    unset_rocr_visible_devices();

    // =========================================================================
    // Detect GPU-NIC affinity using hwloc
    // This selects the best GPU-NIC pair based on NUMA topology
    // =========================================================================
    DeviceAffinityDetector affinity;

    // =========================================================================
    // Initialize GPU with affinity-selected device
    // =========================================================================
    HipDeviceContext hip(affinity.selected_gpu_id);

    // =========================================================================
    // Initialize PMI2
    // =========================================================================
    PmiSession pmi;

    if (pmi.size != 2) {
        if (pmi.rank == 0) {
            fprintf(stderr, "This program requires exactly 2 processes\n");
        }
        return 1;
    }

    // =========================================================================
    // Initialize Fabric/DWQ Context with affinity-selected CXI
    // =========================================================================
    FabricDwqContext fabric(pmi.rank, &affinity);

    // =========================================================================
    // Run Concurrent Benchmark (N_STREAMS parallel DWQ operations)
    // =========================================================================
    BenchmarkRunner runner(pmi, fabric);
    runner.run();

    return 0;
}
