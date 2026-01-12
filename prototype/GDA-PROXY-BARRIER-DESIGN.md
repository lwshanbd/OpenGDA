# GDA Proxy Barrier Design Document

## Overview

`gda-proxy-barrier` implements a GPU-initiated barrier synchronization primitive that can execute unlimited iterations within a GPU kernel. It combines:
- **DWQ (Deferred Work Queue)** for GPU-triggered RDMA operations
- **CPU Proxy Thread** for continuously rearming completed barrier slots

## Problem Statement

### DWQ Limitation
DWQ operations are **one-shot**: once triggered, they cannot be reused. This limits GPU kernels to a fixed number of barrier operations equal to the pre-allocated counter pairs.

### Solution
Use a **windowed slot reuse** approach:
1. Pre-allocate a small window of barrier slots (e.g., 64)
2. GPU uses slots in round-robin: `slot = iteration % num_slots`
3. CPU proxy thread continuously rearms completed slots
4. GPU waits for slot to be armed before using it

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Rank 0                                      │
│  ┌─────────────────┐                     ┌─────────────────────┐    │
│  │   GPU Kernel    │                     │   CPU Proxy Thread  │    │
│  │                 │                     │                     │    │
│  │  for each iter: │                     │  while running:     │    │
│  │    slot = i%N   │                     │    scan slots       │    │
│  │    wait(ARMED)  │───slot_state───────>│    if NEED_QUEUE:   │    │
│  │    reset recv   │                     │      reset counters │    │
│  │    trigger send │───trigger_addr────> │      queue DWQ work │    │
│  │    wait(recv)   │                     │      set ARMED      │    │
│  │    set NEED_Q   │<──slot_state────────│    drain CQ         │    │
│  └─────────────────┘                     └─────────────────────┘    │
│           │                                        │                 │
│           │ RDMA Write                             │                 │
│           ▼                                        │                 │
├───────────────────────────────────────────────────────────────────── │
│     recv_flags[slot]  <──────────────────────────────────────────────│
│                                                                      │
│                              Network (CXI/Slingshot)                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ RDMA Write
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          Rank 1                                      │
│     recv_flags[slot]  <───────────────────────────────────────────── │
│           ▲                                        │                 │
│           │                                        │                 │
│  ┌─────────────────┐                     ┌─────────────────────┐    │
│  │   GPU Kernel    │                     │   CPU Proxy Thread  │    │
│  │   (same logic)  │                     │   (same logic)      │    │
│  └─────────────────┘                     └─────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Structures

### BarrierSlot
```cpp
struct BarrierSlot {
  // Trigger counter - GPU writes to this MMIO address to initiate RDMA
  struct fid_cntr *trigger_cntr;
  volatile uint64_t *dev_trigger_addr;  // GPU-accessible MMIO
  
  // Send completion counter - tracks when RDMA write completes
  struct fid_cntr *send_completion_cntr;
  
  bool initialized;
};
```

### ProxyContext
```cpp
struct ProxyContext {
  // Slot states (GPU-accessible pinned memory)
  volatile int *slot_state;      // SLOT_NEED_QUEUE or SLOT_ARMED
  
  // Barrier slots array
  BarrierSlot *slots;
  int num_slots;
  
  // Barrier flag buffers (GPU memory)
  volatile uint64_t *d_send_flags;  // Values we send to peer
  volatile uint64_t *d_recv_flags;  // Buffer where peer writes
  
  // Remote target info
  uint64_t peer_recv_addr;
  uint64_t peer_recv_key;
  
  // Control
  std::atomic<bool> stop_requested;
  std::atomic<uint64_t> total_rearms;
};
```

## Slot State Machine

```
         ┌────────────────────────────────────────┐
         │                                        │
         ▼                                        │
  ┌──────────────┐                        ┌──────────────┐
  │ SLOT_NEED_   │     CPU Proxy:         │ SLOT_ARMED   │
  │ QUEUE (0)    │ ───────────────────>   │     (1)      │
  │              │  reset counters,       │              │
  │              │  queue DWQ work        │              │
  └──────────────┘                        └──────────────┘
         ▲                                        │
         │                                        │
         │     GPU Kernel:                        │
         │     barrier complete                   │
         └────────────────────────────────────────┘
```

## GPU Kernel Flow

```cpp
__global__ void gpu_barrier_kernel(...) {
  for (int iter = 0; iter < num_iterations; iter++) {
    int slot = iter % num_slots;
    
    // 1. Wait for CPU proxy to arm this slot
    while (slot_state[slot] != SLOT_ARMED) { /* spin */ }
    
    // 2. Reset recv flag (prepare to receive peer's signal)
    recv_flags[slot] = 0;
    __threadfence_system();
    
    // 3. Trigger RDMA write to peer (signal our arrival)
    *trigger_addrs[slot] = 1;
    
    // 4. Wait for peer's signal
    while (recv_flags[slot] == 0) { /* spin */ }
    
    // 5. Signal CPU proxy to rearm this slot
    slot_state[slot] = SLOT_NEED_QUEUE;
    __threadfence_system();
  }
}
```

## CPU Proxy Thread Flow

```cpp
void *proxy_thread_func(void *arg) {
  while (!stop_requested) {
    for (int i = 0; i < num_slots; i++) {
      if (slot_state[i] == SLOT_NEED_QUEUE) {
        // 1. Reset counters
        fi_cntr_set(trigger_cntr, 0);
        fi_cntr_set(completion_cntr, 0);
        
        // 2. Queue DWQ work: RDMA write to peer's recv_flags[i]
        fi_control(&domain->fid, FI_QUEUE_WORK, &work);
        
        // 3. Mark slot as armed
        slot_state[i] = SLOT_ARMED;
      }
    }
    
    // Drain CQ to prevent overflow
    fi_cq_read(cq, entries, 32);
  }
}
```

## Barrier Semantics

For a 2-rank barrier:
1. **Rank 0** writes `ARRIVED` flag to **Rank 1's** `recv_flags[slot]`
2. **Rank 1** writes `ARRIVED` flag to **Rank 0's** `recv_flags[slot]`
3. Both ranks wait until their `recv_flags[slot]` becomes non-zero
4. Both ranks proceed past the barrier

## Performance Characteristics

| Metric | Description |
|--------|-------------|
| **Latency** | ~few μs per barrier (network RTT + trigger overhead) |
| **Throughput** | 100K+ barriers/sec possible with sufficient slots |
| **GPU Spin** | Time waiting for slot arm + peer signal |
| **Proxy Rearms** | Should match number of barrier iterations |

## Key Design Decisions

### 1. GPU Memory for Flags
- `recv_flags` must be in GPU memory for NIC to write directly
- `send_flags` must be in GPU memory for DWQ RDMA source
- GPU polls `recv_flags` without CPU involvement

### 2. Pinned Memory for Slot State
- `slot_state` is pinned host memory mapped to GPU
- Both CPU proxy and GPU kernel can access it
- Provides coordination without system calls

### 3. Window Size Tuning
- Too small: GPU spins waiting for proxy to rearm
- Too large: Wasted resources (counters, memory)
- Default 64 slots balances latency vs resource usage

## Usage

```bash
# Build
cd prototype
make gda-proxy-barrier

# Run with 10,000 barrier iterations using 64 slots
srun -n 2 ./gda-proxy-barrier 10000 64

# Run with default settings
srun -n 2 ./gda-proxy-barrier
```

## Extension to N Ranks

For N > 2 ranks, use dissemination barrier algorithm:
- `ceil(log2(N))` phases per barrier
- Each phase: exchange with partner at distance `2^phase`
- Requires N * num_phases * num_slots DWQ operations

## Comparison with gda-proxy.cpp

| Aspect | gda-proxy | gda-proxy-barrier |
|--------|-----------|-------------------|
| Purpose | Unidirectional RDMA | Bidirectional barrier |
| Operations | Write + atomic notification | Write to peer's recv flag |
| Completion | Wait on local atomic result | Wait on peer's RDMA write |
| Use case | One-way data transfer | Synchronization |

## Debugging Tips

1. **GPU hangs waiting for ARMED**
   - Check proxy thread is running
   - Check `fi_control` return values
   - Increase `usleep()` delay in proxy

2. **GPU hangs waiting for recv_flags**
   - Verify peer rank started
   - Check RDMA address exchange
   - Verify MR registration and keys

3. **Verification failed**
   - Check flag values (should be 1 after barrier)
   - Verify `__threadfence_system()` placement
