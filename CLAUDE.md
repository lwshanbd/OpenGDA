# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Development Focus

**The active development is in the `minimal/` directory.** The `src/` directory contains legacy code and should be ignored for new development.

## Project Overview

OpenGDA (Open Interface for GPU-Direct Async) is an HPC communication library enabling GPU-initiated asynchronous RDMA operations without CPU intervention. It leverages libfabric (OFI) and AMD ROCm/HIP for direct GPU-to-NIC communication via Deferred Work Queue (DWQ) operations on HPE Slingshot (CXI) networks.

## Minimal Implementation

The `minimal/` directory contains a clean, header-only implementation that demonstrates:
- 32 concurrent GPU-triggered RDMA transfers
- Sub-microsecond per-transfer latency (~0.96us for small messages)
- Shared trigger counter design to bypass hardware limits

### Build Commands

```bash
cd minimal
mkdir -p build && cd build
cmake ..
make -j

# Run benchmark (requires 2 nodes)
FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 ./gda_benchmark
```

### Directory Structure

```
minimal/
├── CMakeLists.txt           # Build configuration
├── main.cpp                 # Entry point
├── hip_device_context.hpp   # GPU initialization (RAII)
├── pmi_session.hpp          # PMI2 process management (RAII)
├── fabric_dwq_context.hpp   # Libfabric + DWQ context (RAII)
├── memory_region.hpp        # Memory region registration (RAII)
├── dwq_work_builder.hpp     # DWQ operation builder
└── benchmark_runner.hpp     # Benchmark with concurrent streams
```

## How DWQ (Deferred Work Queue) Works

DWQ enables GPU-initiated RDMA without CPU intervention. The key mechanism:

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GPU writes trigger_counter → NIC executes RMA → completion_counter++  │
│                                     ↓                                   │
│                              atomic_result++ (GPU polls this)           │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **CPU Setup Phase** (before kernel launch):
   - `DwqWorkBuilder::queue_rma_write()` - Queue RMA write, triggered when `trigger_cntr >= threshold`
   - `DwqWorkBuilder::queue_atomic_signal()` - Queue atomic op to signal GPU when RMA completes

2. **GPU Trigger Phase** (in kernel):
   - GPU writes to `trigger_cntr` MMIO address
   - NIC automatically executes queued RMA operation
   - On RMA completion, NIC increments `completion_cntr`
   - Atomic operation fires, incrementing `atomic_result` in GPU memory

3. **GPU Wait Phase** (in kernel):
   - GPU polls `atomic_result` until it reaches expected value
   - No CPU involvement during the entire transfer

### Key Insight: Shared Trigger Counter

The hardware limits per-stream MMIO mappings (~6). The minimal implementation uses a **shared trigger counter** design:

- All streams share ONE trigger counter (single MMIO mapping)
- Each stream uses a different threshold value (stream 0 → threshold 1, stream 1 → threshold 2, ...)
- GPU writes N_STREAMS to trigger all operations at once
- Each stream has its own `completion_cntr` and `atomic_result`

```cpp
// Queue all streams with different thresholds
for (int i = 0; i < N_STREAMS; i++) {
    dwq->queue_rma_write(...,
        fabric.trigger_cntr,    // SHARED trigger counter
        streams[i].completion_cntr,  // Per-stream completion
        i + 1);                 // Threshold = stream_id + 1
}

// GPU kernel: single write triggers all
*shared_trigger_addr = N_STREAMS;

// Each thread polls its own atomic_result
while (*atomic_results[stream_id] < 1) { }
```

## Core Components

### 1. HipDeviceContext (`hip_device_context.hpp`)

RAII wrapper for HIP GPU initialization.

```cpp
HipDeviceContext hip(GPU_ID);  // Sets device, gets properties
```

### 2. PmiSession (`pmi_session.hpp`)

RAII wrapper for PMI2 process management.

```cpp
PmiSession pmi;           // Init PMI2, get rank/size
pmi.barrier();            // Global barrier
pmi.kvs_put(key, value);  // Key-value store for bootstrap
pmi.kvs_get(key, buf, len);
```

### 3. FabricDwqContext (`fabric_dwq_context.hpp`)

RAII wrapper for libfabric initialization with DWQ support.

Key members:
- `domain`, `ep`, `av`, `cq` - Libfabric objects
- `trigger_cntr`, `completion_cntr` - DWQ counters
- `dev_trigger_cntr` - GPU-accessible MMIO address for trigger

Key operations:
- Initializes CXI provider with FI_MR_VIRT_ADDR mode
- Maps trigger counter MMIO to GPU via `hipHostRegister`
- Provides `local_addr` for address exchange

### 4. MemoryRegion (`memory_region.hpp`)

RAII wrapper for libfabric memory registration.

```cpp
MemoryRegion mr(domain, ep, info, buf, size, is_device_mem, device_id, rank);
// mr.desc - descriptor for RMA operations
// mr.key  - key for remote access
```

### 5. DwqWorkBuilder (`dwq_work_builder.hpp`)

Builds and queues DWQ operations. Holds persistent structures that must outlive the operation.

```cpp
DwqWorkBuilder dwq(rank);

// Queue RMA write (triggered when trigger_cntr >= threshold)
dwq.queue_rma_write(domain, ep,
    src_buf, desc, size,
    dest_addr, remote_addr, remote_key,
    trigger_cntr, completion_cntr, threshold);

// Queue atomic signal (fires when completion_cntr >= threshold)
dwq.queue_atomic_signal(domain, ep,
    operand_buf, operand_desc,
    result_buf, result_key, result_addr,
    target_addr,
    trigger_cntr, completion_cntr, threshold);
```

### 6. BenchmarkRunner (`benchmark_runner.hpp`)

Orchestrates the benchmark with N_STREAMS concurrent transfers.

Configuration:
```cpp
constexpr int N_STREAMS = 32;      // Concurrent transfers
constexpr int NUM_ITERATIONS = 20;
constexpr size_t MAX_SIZE = 16 * 1024 * 1024;  // 16MB per stream
```

## Performance Results

32 concurrent streams benchmark:

| Size | Total (us) | Per-transfer (us) |
|------|------------|-------------------|
| 1B-1KB | ~31 | ~0.96 |
| 64KB | 117 | 3.68 |
| 1MB | 1417 | 44.3 |
| 16MB | 22149 | 692 |

## Environment Variables

- `FI_MR_CACHE_MAX_COUNT=0` - **Required** to disable MR cache and free DWQ resources
- `FI_LOG_LEVEL` - LibFabric debug output
- `FI_PROVIDER=cxi` - Force CXI provider

## Dependencies

- libfabric (Cray at `/opt/cray/libfabric/2.1`)
- PMI2 (Cray PMI at `/opt/cray/pe/pmi/6.1.15`)
- ROCm/HIP (`/opt/rocm`)

## Legacy Code (Ignore for New Development)

The `src/` directory contains the original OpenGDA implementation with:
- Complex C API wrapper
- CPU proxy barrier for unlimited iterations
- Multiple network backends

This code is not actively maintained. Use `minimal/` for new development.
