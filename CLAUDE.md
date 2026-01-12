# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenGDA (Open Interface for GPU-Direct Async) is an HPC communication library enabling GPU-initiated asynchronous RDMA operations without CPU intervention. It leverages libfabric (OFI) and AMD ROCm/HIP for direct GPU-to-NIC communication via Deferred Work Queue (DWQ) operations on HPE Slingshot (CXI) networks.

## Build Commands

```bash
# Configure (from project root)
mkdir -p build && cd build
cmake -DUSE_AMDGPU=ON -DBOOTSTRAP_TYPE=PMI2 -DNETWORK_TYPE=OFI ..

# Build library
make -j

# Build specific example
make mm_hip_gda_proxy -j
```

### CMake Options

- `-DUSE_AMDGPU=ON` - AMD ROCm/HIP support (default)
- `-DUSE_NVGPU=ON` - NVIDIA CUDA support (future)
- `-DBOOTSTRAP_TYPE=PMI2|PMIX` - Process management (PMI2 default)
- `-DNETWORK_TYPE=OFI|OFI_PROXY|IBV` - Network backend (OFI default)
- `-DDEBUG_OFI=ON` - Enable debug/trace logging
- `-DDEFAULT_HOST_MR_SIZE=<bytes>` - Default host MR size (8GB)
- `-DDEFAULT_GPU_MR_SIZE=<bytes>` - Default GPU MR size (8GB)

## Running Tests

```bash
# Single-rank tests
./examples/print_limits
./examples/ofi-bigmr

# Multi-rank tests (requires Slurm)
srun -n 2 ./examples/concurrent_put_example
srun -n 2 ./examples/ofi-bigmr-remote

# Matrix multiplication with proxy barrier
srun -N 4 -n 4 --ntasks-per-node=1 ./examples/mm_hip_gda_proxy 512
srun -N 4 -n 16 --ntasks-per-node=4 ./examples/mm_hip_gda_proxy 512

# With debug logging
OPENGDA_LOG_LEVEL=debug srun -n 2 ./examples/concurrent_put_example
```

## Environment Variables

- `OPENGDA_HOST_MR_SIZE` / `OPENGDA_GPU_MR_SIZE` - Override default MR sizes
- `OPENGDA_LOG_LEVEL` - Logging level: error|warn|info|debug|trace
- `FI_LOG_LEVEL` - LibFabric debug output
- `FI_PROVIDER=cxi` - Force CXI provider

## Architecture

### Directory Structure

```
src/
├── gda.h / gda.cpp              # Public C API
├── gda_barrier_proxy.h/cpp      # CPU proxy barrier implementation
├── bootstrap/                   # Process management (PMI2, PMIx)
├── network/                     # Network backends
│   ├── ofi.hpp/cpp              # Main OFI implementation
│   └── ofi-proxy.hpp/cpp        # CPU proxy mode
└── common/log.hpp               # Logging infrastructure

examples/
├── mm_hip_gda_proxy.cpp         # Matrix multiplication with proxy barrier
└── ...
```

### Core Components (in `src/network/ofi.hpp`)

1. **OFI** - Main orchestrator managing libfabric domain, endpoints, address exchange
2. **CntrManager** - Counter pair management (256 pairs for DWQ triggers/completions)
3. **MRManager** - Memory region tracking and statistics
4. **CompletionQueue** - Sequence-based operation completion tracking
5. **DWQOperation** - GPU-triggered RDMA operation state machine
6. **CompletionSignalPool** - GPU-side completion signal allocation

### Data Flow

1. Bootstrap exchanges endpoint addresses and MR keys across ranks
2. Default host/GPU memory regions pre-registered at init
3. DWQOperations created with trigger/completion counter pairs
4. GPU kernel writes to trigger counter MMIO address
5. NIC executes deferred RDMA operation
6. Completion counter incremented, GPU polls for completion

### Public C API (`src/gda.h`)

```c
gda_init() / gda_finalize()
gda_register_memory(buf, size, is_device_mem) -> gda_mr_t*
gda_deregister_memory(mr)
gda_mr_get_key(mr) / gda_mr_get_desc(mr)
gda_put(src, size, target_rank, offset) -> gda_handle_t*
gda_get(dst, size, target_rank, offset) -> gda_handle_t*
gda_gpu_trigger_all(gpu_handle)  // GPU macro to trigger DWQ operation
```

## Key Implementation Details

- GPU MR limited to 8GB (CXI hardware limitation)
- 256 pre-allocated counter pairs for DWQ operations
- Default MRs pre-allocated to avoid per-operation registration overhead
- Signal pool allocates completion signals from default GPU MR
- GPU direct MMIO writes enable zero CPU overhead triggers

## CPU Proxy Barrier for Unlimited Iterations

The standard DWQ operations are one-shot and cannot be reset from GPU code. The **CPU proxy barrier** enables unlimited barrier iterations by having a background CPU thread continuously rearm completed DWQ operations.

### Algorithm: All-to-All Barrier

Each barrier epoch performs:
1. GPU waits for slot to be armed (slot = epoch % window_size)
2. GPU writes 1 to trigger counter (fires all atomics at once, threshold=1)
3. Each rank atomically adds +1 to ALL other ranks' `d_arrive` counter
4. GPU waits for `d_arrive >= (epoch+1) * num_peers`
5. GPU marks slot as NEED_QUEUE
6. CPU proxy thread rearms completed slots (queues new DWQ operations)

### DWQ Resource Management

DWQ operations consume queue slots that are shared per NIC:
- Each barrier slot needs `num_peers` DWQ atomic operations
- With 4 ranks per node sharing a NIC, total DWQ usage = 4 * window_size * num_peers
- Window size is automatically adjusted based on peer count to avoid overflow:
  - `max_window = 200 / num_peers` (budget ~200 DWQ ops per rank for barrier)
  - Example: 16 ranks (15 peers) → max window = 13 slots
  - Example: 4 ranks (3 peers) → max window = 66 slots (capped at 16)

### API

```c
// Allocate proxy barrier (window_size auto-adjusted based on npes)
gda_proxy_barrier_t* gda_proxy_barrier_alloc(int window_size);

// Get device context for GPU kernel
gda_proxy_barrier_dev_t* gda_proxy_barrier_get_dev(gda_proxy_barrier_t* barrier);

// Start proxy thread (call BEFORE launching GPU kernel)
int gda_proxy_start(gda_proxy_barrier_t* barrier);

// Stop proxy thread (call AFTER GPU kernel completes)
int gda_proxy_stop(gda_proxy_barrier_t* barrier);

// Free resources
void gda_proxy_barrier_free(gda_proxy_barrier_t* barrier);

// Get statistics
int gda_proxy_barrier_get_stats(gda_proxy_barrier_t* barrier, gda_proxy_stats_t* stats);

// GPU macro (use in kernel, master thread only)
gda_gpu_proxy_barrier_wait(dev_barrier);
```

### Two-Barrier Pattern for Put/Get Synchronization

When combining barrier with put/get operations, use **two barriers** per iteration to ensure data correctness:

```cpp
__global__ void kernel(gda_gpu_handle_t* put_handles, gda_proxy_barrier_dev_t* barrier, int npes) {
    bool is_master = (threadIdx.x == 0 && blockIdx.x == 0);

    for (int s = 0; s < npes; s++) {
        // Trigger async put to neighbor
        gda_gpu_trigger_all(put_handles[s]);
        __syncthreads();

        // Barrier 1: Ensures all ranks have triggered their puts
        // Atomics sent AFTER put trigger, so when barrier completes,
        // all remote puts have been initiated
        if (is_master) {
            gda_gpu_proxy_barrier_wait(barrier);
        }
        __syncthreads();

        // Barrier 2: Ensures all puts have completed
        // By waiting for all ranks to reach this point,
        // the network has had time to complete all transfers
        if (is_master) {
            gda_gpu_proxy_barrier_wait(barrier);
        }
        __syncthreads();

        // Now safe to use received data
        // ... computation ...
    }
}
```

**Why two barriers?**
- Barrier atomics and put RMA operations use different DWQ trigger counters
- No ordering guarantee between them
- First barrier: ensures all puts have been triggered
- Second barrier: ensures network has completed all transfers

### Usage Example (Matrix Multiplication)

```cpp
// Host code
int window_size = 16;  // Will be auto-adjusted if needed
gda_proxy_barrier_t* barrier = gda_proxy_barrier_alloc(window_size);
gda_proxy_barrier_dev_t* dev_host = gda_proxy_barrier_get_dev(barrier);

// Copy device context to GPU
gda_proxy_barrier_dev_t* d_barrier;
hipMalloc(&d_barrier, sizeof(gda_proxy_barrier_dev_t));
hipMemcpy(d_barrier, dev_host, sizeof(gda_proxy_barrier_dev_t), hipMemcpyHostToDevice);

// Pre-create put handles
std::vector<gda_handle_t*> put_handles(npes);
for (int s = 0; s < npes; s++) {
    put_handles[s] = gda_put(src, size, left_neighbor, offset);
}

// Copy GPU handles to device
gda_gpu_handle_t* d_put_handles;
hipMalloc(&d_put_handles, npes * sizeof(gda_gpu_handle_t));
// ... copy gpu handles ...

// Start proxy thread BEFORE launching kernel
gda_proxy_start(barrier);

// Launch kernel
mm_kernel<<<grid, block>>>(d_As, d_Bs, d_Cs, d_put_handles, d_barrier, npes);
hipDeviceSynchronize();

// Stop proxy thread AFTER kernel completes
gda_proxy_stop(barrier);

// Get statistics
gda_proxy_stats_t stats;
gda_proxy_barrier_get_stats(barrier, &stats);
printf("Rearms: %lu, Polls: %lu\n", stats.total_rearms, stats.queue_polls);

// Cleanup
gda_proxy_barrier_free(barrier);
for (auto h : put_handles) gda_free(h);
```

### Running Matrix Multiplication Example

```bash
# 4 nodes, 1 rank per node
srun -N 4 -n 4 --ntasks-per-node=1 ./examples/mm_hip_gda_proxy 512
# Expected: ~17ms

# 4 nodes, 4 ranks per node (16 total)
srun -N 4 -n 16 --ntasks-per-node=4 ./examples/mm_hip_gda_proxy 512
# Expected: ~11ms
```

### Statistics Interpretation

- `total_rearms` - Number of slots rearmed by proxy thread
- `queue_polls` - Number of proxy thread poll iterations
- `cq_events_drained` - CQ events processed to prevent overflow

If `total_rearms` is low compared to barrier iterations, the proxy is keeping up well.

## Dependencies

- libfabric (prefers Cray at `/opt/cray/libfabric/2.1`)
- PMI2 or PMIx (prefers Cray PMI)
- ROCm/HIP (`/opt/rocm`)
- hwloc
