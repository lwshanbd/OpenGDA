# GICC

**GICC** (GPU-Initiated Communication and Coordination) is a high-performance runtime that enables GPU-initiated asynchronous RDMA operations without CPU intervention in modern HPC systems.

GICC supports multiple network backends:

| Backend | Network | Transport |
|---------|---------|-----------|
| MLX5 | InfiniBand | DevX QP, BlueFlame doorbell |
| OFI/CXI | HPE Slingshot | Deferred Work Queue (DWQ) |

## Publication

> **GICC: A High-Performance Runtime for GPU-Initiated Communication and Coordination in Modern HPC Systems**
>
> The 35th International Symposium on High-Performance Parallel and Distributed Computing (HPDC '26), July 13--16, 2026, Cleveland, OH, USA
>
> DOI: [10.1145/3806645.3807576](https://doi.org/10.1145/3806645.3807576)
>
> Paper will be available soon.

## Building

GICC uses CMake. The backend is selected at configure time via `-DGICC_BACKEND=<mlx5|ofi>` (default: `mlx5`).

```bash
mkdir build && cd build
cmake -DGICC_BACKEND=<backend> [backend-specific options] ..
make -j
```

See `examples/` for sample programs demonstrating each backend.

## API Overview

### Host Side

```cpp
#include <gicc/gicc.hpp>

gicc::Runtime rt;                                   // Initialize runtime
auto buf = rt.register_buffer(d_ptr, size, true);   // Register GPU buffer
rt.exchange();                                      // Exchange buffer metadata
auto* ctx = rt.prepare(peer_rank, buf.index);       // Create device context

// Launch kernel, then cleanup
rt.reset();
```

### Device Side

```cpp
#include <gicc/gicc_device.cuh>

__global__ void kernel(gicc::DeviceCtx* ctx, ...) {
    // MLX5: GPU posts RDMA directly via doorbell
    gicc::put(ctx, local_addr, local_lkey, remote_addr, remote_rkey, size);
    gicc::quiet(ctx);  // Wait for completion

    // OFI/CXI: GPU triggers pre-queued DWQ operations
    gicc::flush(ctx);  // Write MMIO trigger counter
    gicc::quiet(ctx);  // Poll completion slots
}
```

### GPU-Triggered Barrier (OFI/CXI)

```cpp
#include <gicc/platform/ofi/internal/gicc_barrier.hpp>

gicc::Barrier barrier(comm);
barrier.init();

// Single barrier per kernel
barrier.setup();
my_kernel<<<1, 1>>>(args, barrier.device_ctx());
hipDeviceSynchronize();
barrier.reset();

// Or continuous: N barriers in one kernel
barrier.start_continuous(N);
multi_barrier_kernel<<<1, 1>>>(barrier.device_ctx(), N);
barrier.wait_continuous();

barrier.finalize();
```

## Directory Structure

```
src/gicc/                        # GICC library
  gicc.hpp                       # Top-level host API
  gicc_types.hpp                 # Buffer, RemoteBufferInfo types
  gicc_device.cuh                # Device-side API (put, quiet, flush)
  platform/
    mlx5/                        # MLX5 backend (InfiniBand)
    ofi/                         # OFI/CXI backend (Slingshot)
  mlx5/                          # Low-level MLX5 transport

examples/
  gicc/                          # MLX5 examples (pingpong, matrix multiply, Jacobi)
  ofi/                           # OFI examples (test, benchmark, barrier test)
```

## Dependencies

**MLX5 backend**: CUDA, MPI, libibverbs, libmlx5, libpmix

**OFI backend**: ROCm/HIP, libfabric (with CXI extensions), PMI2, hwloc, MPI

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

Copyright (c) 2026 Stony Brook University.
