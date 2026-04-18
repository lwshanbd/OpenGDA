# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GICC (GPU-Initiated Communication and Coordination) is an HPC communication library enabling GPU-initiated asynchronous RDMA operations without CPU intervention. It supports multiple backends:
- **MLX5/InfiniBand** (NVIDIA GPUs) — active development on MAPLE cluster
- **CXI/Slingshot** (AMD GPUs via ROCm/HIP) — legacy code in `minimal/`

## IMPORTANT: Machine-Specific Environment

### MAPLE Cluster (NVIDIA + InfiniBand)

This is the primary development machine. **Before building or running anything, load these modules:**

```bash
module use /shared/data1/Projects/CSE_HPC/apps/modules/aarch64/maple
module load gcc/11.4.1
module load cuda/12.4.131 nvhpc/24.5.cuda_12.4 openmpi-gcc/5.0.5/cuda.12.4
module load cmake/3.29.6
```

Run programs with `srun`:
```bash
srun -p maple --account=app -N 2 --ntasks-per-node=1 --gres=gpu:1 --mpi=pmix ./my_program
```

### Tioga Cluster (AMD + Slingshot)

For the `minimal/` directory only. Use the `/tioga-run` skill for distributed execution.

## Directory Structure

```
src/                          # GICC library source (shared library + headers)
  gicc/                       # Public API headers
    gicc.hpp                  # Top-level host API
    gicc_types.hpp            # Buffer, RemoteBufferInfo types
    gicc_device.cuh           # Device-side API (put, quiet, flush)
    bootstrap/                # MPI / PMI2 bootstrap
    coll.hpp                  # Collective ops
    platform/mlx5/            # MLX5 backend: Runtime, Fabric, DevxContext,
                              #   Am, device-side WQE/CQ code, all MLX5 bits
    platform/ofi/             # OFI/CXI backend: Runtime, Fabric, Am, Barrier
    util/                     # MemoryRegion, ibv_context, cuda_device_context
  CMakeLists.txt              # Builds libgicc.so
  gicc_lib.cu                 # Library compilation unit

examples/gicc/                # MLX5 examples (CUDA) linking against libgicc
  pingpong_bench.cu           # Latency/bandwidth benchmark
  mm.cu                       # Distributed matrix multiplication
  jacobi.cu                   # Jacobi solver with fused RDMA

examples/ofi/                 # OFI examples (HIP) — built when GICC_BACKEND=ofi
  benchmark.cpp               # 32-stream DWQ micro-benchmark
  mm_minimal.cpp              # Ring-topology distributed matmul
  barrier_test.cpp            # GPU-triggered dissemination barrier
  am_pingpong.cpp             # Active Message ping-pong latency
  test_gicc.cpp               # Minimal put/quiet/flush sanity check

legacy/, minimal/, prototype/ # Pre-GICC (OpenGDA-era) code, not built
```

## Build Commands (MAPLE)

```bash
./build.sh          # Configure + build
./build.sh clean    # Clean rebuild
```

Or manually:
```bash
module use /shared/data1/Projects/CSE_HPC/apps/modules/aarch64/maple
module load gcc/11.4.1 cuda/12.4.131 nvhpc/24.5.cuda_12.4 openmpi-gcc/5.0.5/cuda.12.4
module load cmake/3.29.6

mkdir -p build && cd build
cmake -DCMAKE_CUDA_COMPILER=$(which nvcc) ..
make -j
```

## GICC API Overview

### Host API (`gicc::Runtime`)
```cpp
#include <gicc/gicc.hpp>

gicc::Runtime rt;                                     // Init MPI, GPU, IB, QPs
auto buf = rt.register_buffer(d_ptr, size, true);     // Register GPU buffer
rt.exchange();                                        // Collective buffer info exchange
auto* ctx = rt.prepare(peer_rank, buf.index);         // Create device context
// ... launch kernel with ctx ...
rt.reset();                                           // Free device contexts
```

### Device API (`gicc::put`, `gicc::quiet`)
```cpp
#include <gicc/gicc_device.cuh>

__global__ void kernel(gicc::DeviceCtx* ctx, ...) {
    gicc::put(ctx, local_addr, local_lkey, remote_addr, remote_rkey, size);
    gicc::quiet(ctx);  // Wait for completion
}
```

## Namespace Convention

- `gicc::` — Public API (Runtime, Buffer, put/quiet/flush)
- `gicc::mlx5::` — MLX5 backend internals (DevxQp, WQE building, BlueFlame doorbell)

## Dependencies (MAPLE)

- CUDA 12.4 (via module)
- OpenMPI 5.0.5 (via module, CUDA-aware)
- libibverbs + libmlx5 (system)
- libpmix (via OpenMPI)
