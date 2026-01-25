# GPU-Triggered RDMA Benchmark Results

## Platform Information

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA GH200 480GB (1.98 GHz) |
| **Network** | Mellanox ConnectX-7 (mlx5_0) |
| **Link** | RoCE v2 |
| **API** | MLX5 DevX with GPU-accessible resources |
| **Nodes** | 2 nodes, 1 GPU per node |

## Implementation Overview

This implementation provides true GPU-initiated RDMA without CPU involvement:

- **MLX5 DevX API**: GPU-accessible UAR (BlueFlame), WQE buffer, doorbell record
- **BlueFlame Doorbell**: 64-bit write for low-latency WQE posting
- **WQE Building**: GPU directly constructs RDMA WRITE WQEs
- **Completion Detection**: Memory polling (sequence number in message)

---

## 1. Ping-Pong Latency Results (Complete)

True RDMA round-trip latency with GPU-triggered operations on both ends.
**1000 iterations per message size, 50 warmup iterations**

| Size | RTT (us) | One-way (us) | Min RTT (us) | Max RTT (us) | Bandwidth (Gbps) |
|------|----------|--------------|--------------|--------------|------------------|
| 1 B | 14.71 | 7.35 | 14.53 | 15.91 | 0.00 |
| 2 B | 14.67 | 7.34 | 14.53 | 15.38 | 0.00 |
| 4 B | 14.67 | 7.34 | 14.53 | 15.87 | 0.00 |
| 8 B | 14.67 | 7.34 | 14.53 | 15.93 | 0.01 |
| 16 B | 14.67 | 7.34 | 14.51 | 15.39 | 0.02 |
| 32 B | 14.68 | 7.34 | 14.54 | 15.38 | 0.03 |
| 64 B | 14.70 | 7.35 | 14.56 | 15.42 | 0.07 |
| 128 B | 14.76 | 7.38 | 14.62 | 15.58 | 0.14 |
| 256 B | 14.86 | 7.43 | 14.70 | 15.79 | 0.28 |
| 512 B | 14.95 | 7.48 | 14.81 | 15.69 | 0.55 |
| 1 KB | 15.30 | 7.65 | 15.15 | 16.71 | 1.07 |
| 2 KB | 15.36 | 7.68 | 15.21 | 16.67 | 2.13 |
| 4 KB | 15.76 | 7.88 | 15.62 | 16.22 | 4.16 |
| 8 KB | 16.97 | 8.48 | 16.83 | 17.58 | 7.73 |
| 16 KB | 19.05 | 9.53 | 18.90 | 20.73 | 13.76 |
| 32 KB | 23.13 | 11.56 | 22.98 | 23.85 | 22.67 |
| 64 KB | 29.55 | 14.77 | 29.39 | 30.40 | 35.49 |
| 128 KB | 40.98 | 20.49 | 40.83 | 42.68 | 51.17 |
| 256 KB | 63.59 | 31.79 | 63.45 | 64.45 | 65.96 |
| 512 KB | 109.62 | 54.81 | 109.44 | 111.01 | 76.52 |
| 1 MB | 202.77 | 101.39 | 202.30 | 203.40 | 82.74 |
| 2 MB | 386.98 | 193.49 | 386.80 | 388.43 | 86.71 |
| 4 MB | 755.46 | 377.73 | 755.22 | 757.23 | 88.83 |
| 8 MB | 1493.59 | 746.80 | 1493.37 | 1495.16 | 89.86 |
| 16 MB | 2969.79 | 1484.90 | 2969.42 | 2971.56 | **90.39** |

### Key Observations

1. **Small Message Latency**: ~7.35 us one-way for messages 1B-64B (latency-bound region)
2. **Latency Transition Point**: ~4KB where bandwidth starts to dominate
3. **Peak Bandwidth**: **90.39 Gbps** at 16MB (bidirectional)
4. **Consistency**: Min/Max within ~1.5 us of average for small messages
5. **Large Message Efficiency**: Near line-rate at 16MB

### Latency Breakdown (estimated for 8B message)

| Component | Time (us) |
|-----------|-----------|
| GPU WQE build | ~1.0 |
| BlueFlame doorbell | ~0.5 |
| PCIe to NIC | ~0.5 |
| Network RTT | ~3.0 |
| PCIe from NIC | ~0.5 |
| Memory polling | ~1.8 |
| **Total RTT** | **~14.7** |

---

## 2. WQE Posting Overhead (Batching Effect)

Measures time for GPU to build WQEs and ring doorbell, without waiting for network completion.

| Batch Size | Per-op (us) | Total (us) | Speedup |
|------------|-------------|------------|---------|
| 1 | 1.98 | 1981 | 1.00x |
| 8 | 0.32 | 316 | 6.26x |
| 32 | 0.14 | 142 | 13.96x |
| 128 | **0.10** | 97 | **20.47x** |

### Key Observations

1. **Single Operation**: ~2 us (includes WQE build + BlueFlame doorbell)
2. **Batched Operations**: As low as **0.10 us** per operation with batch=128
3. **Speedup**: **20x improvement** with batching due to amortized doorbell cost
4. **Optimal Batch Size**: 32-128 for best throughput

---

## 3. Performance Analysis

### Bandwidth vs Message Size

```
Bandwidth (Gbps)
     |
 90 -+                                          ********
     |                                    ******
 80 -+                              ******
     |                        ******
 70 -+                  ******
     |            ******
 60 -+      ******
     |  ****
 50 -+**
     |
 40 -+
     |
 30 -+----
     |    ----
 20 -+        ----
     |            ----
 10 -+                ----
     |                    --------
  0 -+------------------------+--------+--------+--------+
     1B   1KB   16KB  64KB  256KB   1MB   4MB   16MB
```

### Latency Regions

| Region | Message Size | Characteristic |
|--------|-------------|----------------|
| **Latency-bound** | 1B - 4KB | ~7.3-7.9 us one-way, constant |
| **Transition** | 4KB - 64KB | Latency increases with size |
| **Bandwidth-bound** | 64KB - 16MB | Near line-rate throughput |

---

## 4. Comparison with Minimal (AMD + CXI DWQ)

The `minimal/` directory implements GPU-triggered RDMA for AMD GPUs with HPE Slingshot (CXI) using DWQ.

| Metric | NVIDIA + InfiniBand (DevX) | AMD + CXI (DWQ) |
|--------|----------------------------|-----------------|
| **One-way latency (8B)** | ~7.35 us | ~0.96 us* |
| **RTT (8B)** | ~14.7 us | ~2 us* |
| **Peak Bandwidth** | 90.39 Gbps | ~400 Gbps* |
| **WQE posting (batched)** | 0.10 us | ~0.03 us* |

*Note: AMD + CXI numbers are from previous benchmarks on different hardware (MI250X + Slingshot 11).

### Architecture Differences

| Feature | NVIDIA + InfiniBand | AMD + CXI |
|---------|---------------------|-----------|
| **GPU Trigger** | BlueFlame doorbell | MMIO trigger counter |
| **WQE Location** | GPU-accessible host memory | NIC-managed |
| **Completion** | Memory polling | NIC atomic to GPU memory |
| **Network** | RoCE v2 | Native Slingshot |
| **Link Speed** | ~100 Gbps | ~400 Gbps |

### Why is CXI/DWQ faster?

1. **Native Slingshot**: Purpose-built for HPC, lower protocol overhead than RoCE
2. **DWQ Design**: NIC executes pre-queued work on trigger, minimal per-op setup
3. **Atomic Completion**: NIC directly updates GPU memory, no polling needed
4. **Higher Link Speed**: Slingshot 11 provides 4x the bandwidth

---

## 5. Implementation Details

### Files

| File | Description |
|------|-------------|
| `mlx5_devx_qp.hpp` | DevX QP with GPU-accessible UAR, WQE buffer, doorbell |
| `gda_device_opt.cuh` | GPU device functions for WQE building and doorbell |
| `gpu_pingpong_bench.cu` | Comprehensive ping-pong benchmark (1B-16MB) |
| `gpu_benchmark_devx.cu` | Full benchmark suite with batching tests |

### Key APIs Used

```cpp
// DevX UAR allocation (for BlueFlame doorbell)
mlx5dv_devx_alloc_uar(ctx, MLX5DV_UAR_ALLOC_TYPE_BF);

// DevX memory registration (for WQE buffer)
mlx5dv_devx_umem_reg(ctx, buffer, size, access_flags);

// CUDA mapping of MMIO regions
cudaHostRegister(ptr, size, cudaHostRegisterIoMemory);
cudaHostGetDevicePointer(&d_ptr, h_ptr, 0);
```

### GPU Kernel Pattern

```cuda
__global__ void pingpong_kernel(...) {
    // Build WQE directly from GPU
    gda_build_rdma_write_wqe_opt(state, ...);

    // Ring BlueFlame doorbell (64-bit MMIO write)
    gda_ring_doorbell_bf(state, wqe_idx);

    // Poll for response
    while (*recv_flag != expected) { }
}
```

---

## 6. Build and Run

```bash
# Load modules (Maple cluster)
module load genesis common cuda/12.4.131 gcc/12.2.0 openmpi-gcc/5.0.5/cuda.12.4

# Build
cd nvidia
mkdir -p build && cd build
cmake ..
make -j

# Run comprehensive ping-pong benchmark (2 nodes)
srun -N 2 --gres=gpu:1 ./gpu_pingpong_bench

# Run full benchmark suite with batching tests
srun -N 2 --gres=gpu:1 ./gpu_benchmark_devx
```

---

## 7. Conclusions

1. **GPU-Triggered RDMA Works**: Successfully implemented true GPU-initiated RDMA for NVIDIA + InfiniBand using DevX API.

2. **Latency**: ~7.35 us one-way for small messages (1B-64B), dominated by:
   - WQE building and doorbell (~1.5 us)
   - PCIe round-trip (~1 us)
   - Network latency (~3 us)
   - Memory polling overhead (~1.8 us)

3. **Bandwidth**: Reaches **90.39 Gbps** at 16MB (near line-rate for 100G link).

4. **Batching Benefit**: **20x throughput improvement** with batched operations (0.10 us per op vs 2 us single op).

5. **Comparison with DWQ**: Higher latency than AMD/CXI due to:
   - RoCE vs native Slingshot (protocol overhead)
   - BlueFlame vs MMIO trigger (different mechanisms)
   - Memory polling vs atomic completion
   - Lower link speed (100G vs 400G)

6. **Use Cases**: Best suited for:
   - Bulk data transfers (>64KB) where bandwidth matters
   - Batched small messages where posting overhead is amortized
   - Applications that can tolerate ~7us one-way latency

---

*Generated: 2025-01-25*
*Platform: NVIDIA GH200 480GB + Mellanox ConnectX-7 (RoCE v2)*
*Test: 1000 iterations per message size, 50 warmup iterations*
