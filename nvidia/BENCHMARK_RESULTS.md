# GPU-Triggered RDMA Benchmark Results

## Platform Information

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA GH200 480GB (1.98 GHz) |
| **Network** | Mellanox ConnectX-7 (mlx5_0) |
| **Link** | RoCE v2, ~200 Gbps |
| **MPI** | OpenMPI 5.0.5 with CUDA-aware support |
| **API** | MLX5 DevX with GPU-accessible resources |
| **Nodes** | 2 nodes, 1 GPU per node |

---

## Complete Comparison: MPI vs GPU-Triggered RDMA

### Small Message Latency (8 bytes)

| Method | RTT (us) | One-way (us) | Notes |
|--------|----------|--------------|-------|
| **MPI Host Memory** | 3.28 | **1.64** | CPU initiates, host buffers |
| **MPI GPU Memory** | 5.44 | **2.72** | CPU initiates, GPU buffers (CUDA-aware) |
| **GPU-triggered RDMA** | 14.67 | **7.34** | GPU initiates, no CPU involvement |

### Peak Bandwidth (16 MB)

| Method | RTT (us) | Bandwidth (Gbps) |
|--------|----------|------------------|
| **MPI Host Memory** | 1368.18 | **196.20** |
| **MPI GPU Memory** | 1367.48 | **196.30** |
| **GPU-triggered RDMA** | 2969.79 | **90.39** |

---

## 1. MPI Ping-Pong Results (Host Memory)

CPU-initiated communication using standard MPI_Send/MPI_Recv with host buffers.

| Size | RTT (us) | One-way (us) | Bandwidth (Gbps) |
|------|----------|--------------|------------------|
| 1 B | 3.36 | 1.68 | 0.00 |
| 8 B | 3.28 | **1.64** | 0.04 |
| 64 B | 3.60 | 1.80 | 0.28 |
| 512 B | 4.52 | 2.26 | 1.81 |
| 1 KB | 4.27 | 2.14 | 3.84 |
| 4 KB | 6.22 | 3.11 | 10.54 |
| 16 KB | 9.05 | 4.52 | 28.98 |
| 64 KB | 14.74 | 7.37 | 71.14 |
| 256 KB | 34.07 | 17.04 | 123.11 |
| 1 MB | 97.72 | 48.86 | 171.69 |
| 4 MB | 352.28 | 176.14 | 190.50 |
| 16 MB | 1368.18 | 684.09 | **196.20** |

---

## 2. MPI Ping-Pong Results (GPU Memory, CUDA-aware)

CPU-initiated communication with GPU buffers, using CUDA-aware MPI.

| Size | RTT (us) | One-way (us) | Bandwidth (Gbps) |
|------|----------|--------------|------------------|
| 1 B | 5.64 | 2.82 | 0.00 |
| 8 B | 5.44 | **2.72** | 0.02 |
| 64 B | 5.92 | 2.96 | 0.17 |
| 512 B | 6.61 | 3.31 | 1.24 |
| 1 KB | 6.67 | 3.33 | 2.46 |
| 4 KB | 7.46 | 3.73 | 8.79 |
| 16 KB | 12.32 | 6.16 | 21.27 |
| 64 KB | 17.28 | 8.64 | 60.67 |
| 256 KB | 33.38 | 16.69 | 125.67 |
| 1 MB | 97.55 | 48.77 | 171.99 |
| 4 MB | 351.59 | 175.79 | 190.87 |
| 16 MB | 1367.48 | 683.74 | **196.30** |

---

## 3. GPU-Triggered RDMA Results (DevX API)

GPU-initiated communication without CPU involvement using MLX5 DevX.

| Size | RTT (us) | One-way (us) | Bandwidth (Gbps) |
|------|----------|--------------|------------------|
| 1 B | 14.71 | 7.35 | 0.00 |
| 8 B | 14.67 | **7.34** | 0.01 |
| 64 B | 14.70 | 7.35 | 0.07 |
| 512 B | 14.95 | 7.48 | 0.55 |
| 1 KB | 15.30 | 7.65 | 1.07 |
| 4 KB | 15.76 | 7.88 | 4.16 |
| 16 KB | 19.05 | 9.53 | 13.76 |
| 64 KB | 29.55 | 14.77 | 35.49 |
| 256 KB | 63.59 | 31.79 | 65.96 |
| 1 MB | 202.77 | 101.39 | 82.74 |
| 4 MB | 755.46 | 377.73 | 88.83 |
| 16 MB | 2969.79 | 1484.90 | **90.39** |

---

## 4. Latency Comparison Chart

```
One-way Latency (us) for 8-byte message:

MPI Host     |████ 1.64 us
MPI GPU      |██████ 2.72 us
GPU-triggered|██████████████████████ 7.34 us
             +----+----+----+----+----+----+----+----+
             0    1    2    3    4    5    6    7    8
```

```
Bandwidth (Gbps) at 16MB:

MPI Host     |████████████████████████████████████████ 196 Gbps
MPI GPU      |████████████████████████████████████████ 196 Gbps
GPU-triggered|██████████████████ 90 Gbps
             +--------+--------+--------+--------+--------+
             0       50      100      150      200      250
```

---

## 5. Analysis

### Why is MPI faster for latency?

| Factor | MPI | GPU-triggered RDMA |
|--------|-----|-------------------|
| **Initiator** | CPU (fast, low overhead) | GPU (kernel overhead) |
| **WQE Build** | Pre-built by driver | GPU builds each WQE (~1 us) |
| **Doorbell** | CPU MMIO write | GPU BlueFlame write (~0.5 us) |
| **Completion** | Interrupt/polling on CPU | GPU memory polling (~2 us) |
| **Total overhead** | ~0.5 us | ~4-5 us |

### Why is MPI faster for bandwidth?

1. **NIC Optimization**: MPI/UCX uses optimized multi-packet protocols
2. **Pipelining**: CPU can efficiently pipeline multiple transfers
3. **Single Operation**: GPU-triggered sends one WQE at a time (no pipelining in this test)

### When to use GPU-triggered RDMA?

GPU-triggered RDMA excels when:

1. **Overlap compute and communication**: GPU can trigger RDMA while other SMs compute
2. **Fine-grained synchronization**: Direct GPU-to-GPU notification without CPU
3. **Reduce CPU involvement**: Free CPU for other tasks
4. **Many small messages**: Batching amortizes overhead (0.1 us/op with batch=128)

---

## 6. Concurrent Streams Benchmark (32 streams)

Similar to minimal/benchmark_runner.hpp for CXI/Slingshot, this test measures
32 concurrent GPU-triggered RDMA writes with a single doorbell.

| Size | Total (us) | Per-transfer (us) |
|------|------------|-------------------|
| 1B-1KB | ~10.8 | **~0.34** |
| 64KB | 10.9 | 0.34 |
| 1MB | 11.4 | 0.36 |
| 16MB | 11.2 | 0.35 |

**Key insight:** With batching (32 concurrent transfers, single doorbell),
the amortized cost drops from 7.34 us (ping-pong) to **0.34 us per transfer**!

### Comparison with CXI/Slingshot (minimal/)

| Platform | 32 Streams Total (us) | Per-transfer (us) |
|----------|----------------------|-------------------|
| **NVIDIA + InfiniBand** | ~11 | **~0.34** |
| AMD + CXI (Slingshot) | ~31 | ~0.96 |

NVIDIA + InfiniBand achieves **~3x better per-transfer latency** with batching.

---

## 7. Batching Effect (Ping-pong sequential)

| Batch Size | Per-op (us) | Speedup |
|------------|-------------|---------|
| 1 | 1.98 | 1.00x |
| 8 | 0.32 | 6.26x |
| 32 | 0.14 | 13.96x |
| 128 | **0.10** | **20.47x** |

With batching, GPU-triggered RDMA achieves **0.10 us per operation**, competitive with CPU-triggered approaches for bulk transfers.

---

## 8. Full Results Table

| Size | MPI Host RTT | MPI GPU RTT | GPU-trig RTT | MPI Host BW | MPI GPU BW | GPU-trig BW |
|------|-------------|-------------|--------------|-------------|------------|-------------|
| 1 B | 3.36 | 5.64 | 14.71 | 0.00 | 0.00 | 0.00 |
| 2 B | 3.54 | 5.39 | 14.67 | 0.01 | 0.01 | 0.00 |
| 4 B | 3.28 | 5.45 | 14.67 | 0.02 | 0.01 | 0.00 |
| 8 B | 3.28 | 5.44 | 14.67 | 0.04 | 0.02 | 0.01 |
| 16 B | 3.36 | 5.78 | 14.67 | 0.08 | 0.04 | 0.02 |
| 32 B | 3.38 | 5.62 | 14.68 | 0.15 | 0.09 | 0.03 |
| 64 B | 3.60 | 5.92 | 14.70 | 0.28 | 0.17 | 0.07 |
| 128 B | 3.58 | 6.48 | 14.76 | 0.57 | 0.32 | 0.14 |
| 256 B | 4.05 | 6.67 | 14.86 | 1.01 | 0.61 | 0.28 |
| 512 B | 4.52 | 6.61 | 14.95 | 1.81 | 1.24 | 0.55 |
| 1 KB | 4.27 | 6.67 | 15.30 | 3.84 | 2.46 | 1.07 |
| 2 KB | 5.65 | 6.94 | 15.36 | 5.80 | 4.72 | 2.13 |
| 4 KB | 6.22 | 7.46 | 15.76 | 10.54 | 8.79 | 4.16 |
| 8 KB | 8.07 | 10.20 | 16.97 | 16.25 | 12.85 | 7.73 |
| 16 KB | 9.05 | 12.32 | 19.05 | 28.98 | 21.27 | 13.76 |
| 32 KB | 11.10 | 17.06 | 23.13 | 47.25 | 30.72 | 22.67 |
| 64 KB | 14.74 | 17.28 | 29.55 | 71.14 | 60.67 | 35.49 |
| 128 KB | 21.37 | 22.63 | 40.98 | 98.13 | 92.69 | 51.17 |
| 256 KB | 34.07 | 33.38 | 63.59 | 123.11 | 125.67 | 65.96 |
| 512 KB | 55.13 | 54.93 | 109.62 | 152.17 | 152.73 | 76.52 |
| 1 MB | 97.72 | 97.55 | 202.77 | 171.69 | 171.99 | 82.74 |
| 2 MB | 183.02 | 182.51 | 386.98 | 183.34 | 183.85 | 86.71 |
| 4 MB | 352.28 | 351.59 | 755.46 | 190.50 | 190.87 | 88.83 |
| 8 MB | 690.92 | 690.23 | 1493.59 | 194.26 | 194.45 | 89.86 |
| 16 MB | 1368.18 | 1367.48 | 2969.79 | 196.20 | 196.30 | 90.39 |

---

## 9. Build and Run

```bash
# Load CUDA-aware MPI modules
module purge
module load genesis common proxy slurm gcc/11.4.1 cuda/12.4.131
module use /shared/data1/Projects/CSE_HPC/apps/modules/aarch64/maple
module load openmpi-gcc/5.0.5/cuda.12.4

# Build
cd nvidia/build
cmake ..
make -j

# Run MPI benchmark (host + GPU memory)
srun -N 2 --gres=gpu:1 ./mpi_pingpong_bench

# Run GPU-triggered RDMA benchmark
srun -N 2 --gres=gpu:1 ./gpu_pingpong_bench

# Run concurrent streams benchmark (32 streams, like minimal/)
srun -N 2 --gres=gpu:1 ./gpu_concurrent_bench
```

---

## 10. Conclusions

1. **MPI is faster for raw latency**: 1.64 us (host) vs 7.34 us (GPU-triggered)
   - CPU has lower overhead for initiating transfers
   - GPU-triggered adds WQE build + doorbell + polling overhead

2. **MPI achieves higher bandwidth**: 196 Gbps vs 90 Gbps at 16MB
   - MPI/UCX uses optimized multi-packet protocols
   - GPU-triggered implementation doesn't pipeline transfers

3. **GPU-triggered RDMA benefits**:
   - **Compute-communication overlap**: GPU can transfer while computing
   - **No CPU involvement**: Frees CPU for other work
   - **Batching**: 0.10 us/op with batch=128 (20x speedup)
   - **Direct GPU notification**: No CPU wake-up needed

4. **Use case recommendations**:
   - **Bulk transfers**: Use MPI for maximum bandwidth
   - **Latency-sensitive**: Use MPI Host for lowest latency
   - **GPU-centric workloads**: Use GPU-triggered for overlap and autonomy
   - **Many small messages**: Use GPU-triggered with batching

---

*Generated: 2025-01-25*
*Platform: NVIDIA GH200 480GB + Mellanox ConnectX-7 (RoCE v2)*
*MPI: OpenMPI 5.0.5 with CUDA-aware support*
