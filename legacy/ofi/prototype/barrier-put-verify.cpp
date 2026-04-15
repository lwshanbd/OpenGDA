/*
 * barrier-put-verify.cpp - Functional Verification of GPU Barrier + Put with CPU Proxy
 *
 * This test verifies that the GPU barrier with CPU proxy mechanism:
 * 1. Correctly synchronizes all ranks at each barrier
 * 2. Works correctly with interleaved put operations
 * 3. Ensures data consistency after barriers
 * 4. Supports unlimited iterations via slot reuse
 *
 * Test Algorithm (Ring Exchange with Verification):
 * For each iteration:
 *   1. Each rank prepares unique data: value = (rank << 24) | iteration
 *   2. Each rank performs RDMA put to next rank: rank -> (rank+1) % size
 *   3. Dissemination barrier to synchronize all ranks
 *   4. Each rank verifies received data matches expected value
 *   5. Second barrier to ensure all verifications complete before next iteration
 *
 * This tests both the barrier correctness and the interaction between
 * put operations and barriers, catching any synchronization bugs.
 *
 * Usage: srun -N 4 -n 4 --ntasks-per-node=1 ./barrier-put-verify [num_iters] [window_size]
 *
 * Author: Based on barrier-verify.cpp prototype
 */

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <pmi2.h>
#include <pthread.h>
#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_cxi_ext.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_trigger.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vector>

// =============================================================================
// Configuration
// =============================================================================

#define DEFAULT_WINDOW_SIZE 16
#define DEFAULT_ITERATIONS 100
#define MAX_WINDOW_SIZE 64
#define MAX_RANKS 16

// Slot states for CPU-GPU coordination
#define SLOT_NEED_QUEUE 0
#define SLOT_ARMED 1

// Operation types for DWQ
#define OP_TYPE_BARRIER 0  // Atomic add for barrier phase
#define OP_TYPE_PUT 1      // RDMA write for data transfer

// =============================================================================
// Error Checking Macros
// =============================================================================

#define CHECK(x, msg)                                                          \
  do {                                                                         \
    int ret = (x);                                                             \
    if (ret) {                                                                 \
      fprintf(stderr, "Rank %d: %s failed: %s (%d)\n", myrank, msg,            \
              fi_strerror(-ret), ret);                                         \
      PMI2_Finalize();                                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_HIP(x, msg)                                                      \
  do {                                                                         \
    hipError_t err = (x);                                                      \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "Rank %d: %s failed: %s (%d)\n", myrank, msg,            \
              hipGetErrorString(err), err);                                    \
      PMI2_Finalize();                                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int myrank = -1;

// =============================================================================
// Data Structures
// =============================================================================

// Counter pair for a single DWQ operation slot
struct OperationSlot {
  struct fid_cntr *trigger_cntr;
  struct fi_cxi_cntr_ops *trigger_ops;
  void *trigger_mmio_addr;
  size_t trigger_mmio_len;
  volatile uint64_t *dev_trigger_addr;  // GPU-accessible MMIO for trigger

  struct fid_cntr *completion_cntr;
  struct fi_cxi_cntr_ops *completion_ops;
  void *completion_mmio_addr;
  size_t completion_mmio_len;
  volatile uint64_t *dev_completion_addr;  // GPU-accessible MMIO for completion

  int op_type;     // OP_TYPE_BARRIER or OP_TYPE_PUT
  int target_rank; // Target rank for this operation
  int phase;       // For barrier: which phase (0..num_phases-1)

  bool initialized;
};

// Remote memory region info for barrier phase counters
struct RemotePhaseInfo {
  uint64_t addr;
  uint64_t key;
};

// Remote memory region info for put buffers
struct RemotePutInfo {
  uint64_t addr;
  uint64_t key;
};

// Proxy thread context
struct ProxyContext {
  // Fabric objects
  struct fid_domain *domain;
  struct fid_ep *ep;
  struct fid_cq *cq;
  struct fi_info *info;

  // Slot configuration
  int window_size;
  int num_phases;
  int barrier_slots_per_window;  // num_phases slots for barrier per window
  int put_slots_per_window;      // 1 slot for put per window
  int total_slots;               // window_size * (num_phases + 1)

  // Slot array
  OperationSlot *slots;
  volatile int *slot_state;  // Host-pinned, GPU-visible

  // Rank info
  int size;
  int rank;
  fi_addr_t *peer_fi_addrs;

  // Barrier: per-phase remote info
  std::vector<RemotePhaseInfo> *peer_phase_info;  // [phase][rank]

  // Put: remote buffer info
  RemotePutInfo *peer_put_info;  // [rank]

  // Local buffers
  uint64_t *d_atomic_operand;       // Always 1 for atomic add
  void *desc_atomic_operand;
  struct fid_mr *mr_atomic_operand;

  uint64_t *d_send_buf;             // Buffer to send via put
  void *desc_send_buf;
  struct fid_mr *mr_send_buf;

  // DWQ structures for barrier (atomic ops)
  std::vector<struct fi_deferred_work> *barrier_works;
  std::vector<struct fi_op_atomic> *barrier_atomics;
  std::vector<struct fi_msg_atomic> *barrier_msgs;
  std::vector<struct fi_ioc> *barrier_iovs;
  std::vector<struct fi_rma_ioc> *barrier_rma_iovs;

  // DWQ structures for put (RMA write ops)
  std::vector<struct fi_deferred_work> *put_works;
  std::vector<struct fi_op_rma> *put_ops;
  std::vector<struct fi_msg_rma> *put_msgs;
  std::vector<struct iovec> *put_iovs;
  std::vector<struct fi_rma_iov> *put_rma_iovs;

  // Control
  std::atomic<bool> stop_flag;
  std::atomic<uint64_t> total_rearms;
  std::atomic<uint64_t> barrier_rearms;
  std::atomic<uint64_t> put_rearms;
};

// =============================================================================
// Utility Functions
// =============================================================================

static double get_time_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void bytes_to_hex(const uint8_t *in, size_t len, char *out) {
  static const char *h = "0123456789abcdef";
  for (size_t i = 0; i < len; i++) {
    out[2 * i] = h[(in[i] >> 4) & 0xF];
    out[2 * i + 1] = h[in[i] & 0xF];
  }
  out[2 * len] = '\0';
}

static int hexval(char c) {
  if ('0' <= c && c <= '9')
    return c - '0';
  if ('a' <= c && c <= 'f')
    return c - 'a' + 10;
  if ('A' <= c && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

static int hex_to_bytes(const char *in, uint8_t *out, size_t outlen) {
  size_t n = strlen(in);
  if (n % 2 != 0 || outlen < n / 2)
    return -1;
  for (size_t i = 0; i < n; i += 2) {
    int hi = hexval(in[i]);
    int lo = hexval(in[i + 1]);
    if (hi < 0 || lo < 0)
      return -1;
    out[i / 2] = (uint8_t)((hi << 4) | lo);
  }
  return (int)(n / 2);
}

static int exchange_addrs(int rank, int size, const char *my_hex,
                          size_t hex_len, char *all_hex) {
  char key[PMI2_MAX_KEYLEN];
  snprintf(key, sizeof(key), "addr-%d", rank);
  int rc = PMI2_KVS_Put(key, my_hex);
  if (rc != PMI2_SUCCESS)
    return rc;

  rc = PMI2_KVS_Fence();
  if (rc != PMI2_SUCCESS)
    return rc;

  for (int i = 0; i < size; i++) {
    snprintf(key, sizeof(key), "addr-%d", i);
    char val[PMI2_MAX_VALLEN];
    int vallen;
    rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key, val, sizeof(val), &vallen);
    if (rc != PMI2_SUCCESS)
      return rc;

    char *slot = all_hex + i * (hex_len + 1);
    strncpy(slot, val, hex_len);
    slot[hex_len] = '\0';
  }
  return PMI2_SUCCESS;
}

static int register_memory_region(struct fid_domain *domain, struct fid_ep *ep,
                                  struct fi_info *info, void *buf, size_t size,
                                  struct fid_mr **mr_out, bool is_device_mem) {
  struct fi_mr_attr mr_attr = {};
  struct iovec iov;
  iov.iov_base = buf;
  iov.iov_len = size;

  mr_attr.mr_iov = &iov;
  mr_attr.iov_count = 1;
  mr_attr.access =
      FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;

  if (is_device_mem) {
    mr_attr.iface = FI_HMEM_ROCR;
    mr_attr.device.reserved = 0;
  } else {
    mr_attr.iface = FI_HMEM_SYSTEM;
  }

  struct fid_mr *mr = NULL;
  int ret = fi_mr_regattr(domain, &mr_attr, 0, &mr);
  if (ret)
    return ret;

  if (info->domain_attr->mr_mode & FI_MR_ENDPOINT) {
    ret = fi_mr_bind(mr, &ep->fid, 0);
    if (ret) {
      fi_close(&mr->fid);
      return ret;
    }
    ret = fi_mr_enable(mr);
    if (ret) {
      fi_close(&mr->fid);
      return ret;
    }
  }

  *mr_out = mr;
  return 0;
}

// =============================================================================
// DWQ Work Queueing Functions
// =============================================================================

// Queue barrier (atomic add) work for a specific phase slot
static int queue_barrier_work(ProxyContext *ctx, int slot_idx, int phase,
                              int target_rank) {
  OperationSlot &os = ctx->slots[slot_idx];
  int barrier_idx = slot_idx;

  // Reset counters
  fi_cntr_set(os.trigger_cntr, 0);
  fi_cntr_set(os.completion_cntr, 0);

  // Get remote info for this phase and target rank
  RemotePhaseInfo &rinfo = ctx->peer_phase_info[phase][target_rank];

  // Setup atomic IOV (source operand = 1)
  ctx->barrier_iovs->at(barrier_idx).addr = ctx->d_atomic_operand;
  ctx->barrier_iovs->at(barrier_idx).count = 1;

  // Setup remote RMA IOV (target's phase counter)
  ctx->barrier_rma_iovs->at(barrier_idx).addr = rinfo.addr;
  ctx->barrier_rma_iovs->at(barrier_idx).count = 1;
  ctx->barrier_rma_iovs->at(barrier_idx).key = rinfo.key;

  // Setup atomic message
  ctx->barrier_msgs->at(barrier_idx).msg_iov = &ctx->barrier_iovs->at(barrier_idx);
  ctx->barrier_msgs->at(barrier_idx).desc = &ctx->desc_atomic_operand;
  ctx->barrier_msgs->at(barrier_idx).iov_count = 1;
  ctx->barrier_msgs->at(barrier_idx).addr = ctx->peer_fi_addrs[target_rank];
  ctx->barrier_msgs->at(barrier_idx).rma_iov = &ctx->barrier_rma_iovs->at(barrier_idx);
  ctx->barrier_msgs->at(barrier_idx).rma_iov_count = 1;
  ctx->barrier_msgs->at(barrier_idx).datatype = FI_UINT64;
  ctx->barrier_msgs->at(barrier_idx).op = FI_SUM;

  // Setup atomic op
  ctx->barrier_atomics->at(barrier_idx).ep = ctx->ep;
  ctx->barrier_atomics->at(barrier_idx).msg = ctx->barrier_msgs->at(barrier_idx);
  ctx->barrier_atomics->at(barrier_idx).flags = 0;

  // Setup deferred work
  ctx->barrier_works->at(barrier_idx).triggering_cntr = os.trigger_cntr;
  ctx->barrier_works->at(barrier_idx).completion_cntr = os.completion_cntr;
  ctx->barrier_works->at(barrier_idx).threshold = 1;
  ctx->barrier_works->at(barrier_idx).op_type = FI_OP_ATOMIC;
  ctx->barrier_works->at(barrier_idx).op.atomic = &ctx->barrier_atomics->at(barrier_idx);

  int ret = fi_control(&ctx->domain->fid, FI_QUEUE_WORK,
                       &ctx->barrier_works->at(barrier_idx));
  if (ret) {
    fprintf(stderr, "Rank %d: fi_control(barrier slot %d) failed: %s\n",
            ctx->rank, slot_idx, fi_strerror(-ret));
    return ret;
  }

  return 0;
}

// Queue put (RDMA write) work for a specific put slot
static int queue_put_work(ProxyContext *ctx, int slot_idx, int target_rank) {
  OperationSlot &os = ctx->slots[slot_idx];
  int put_idx = slot_idx;

  // Reset counters
  fi_cntr_set(os.trigger_cntr, 0);
  fi_cntr_set(os.completion_cntr, 0);

  // Get remote put buffer info
  RemotePutInfo &rinfo = ctx->peer_put_info[target_rank];

  // Setup local IOV (source buffer)
  ctx->put_iovs->at(put_idx).iov_base = ctx->d_send_buf;
  ctx->put_iovs->at(put_idx).iov_len = sizeof(uint64_t);

  // Setup remote RMA IOV (target's recv buffer)
  ctx->put_rma_iovs->at(put_idx).addr = rinfo.addr;
  ctx->put_rma_iovs->at(put_idx).len = sizeof(uint64_t);
  ctx->put_rma_iovs->at(put_idx).key = rinfo.key;

  // Setup RMA message
  ctx->put_msgs->at(put_idx).msg_iov = &ctx->put_iovs->at(put_idx);
  ctx->put_msgs->at(put_idx).desc = &ctx->desc_send_buf;
  ctx->put_msgs->at(put_idx).iov_count = 1;
  ctx->put_msgs->at(put_idx).addr = ctx->peer_fi_addrs[target_rank];
  ctx->put_msgs->at(put_idx).rma_iov = &ctx->put_rma_iovs->at(put_idx);
  ctx->put_msgs->at(put_idx).rma_iov_count = 1;

  // Setup RMA op
  ctx->put_ops->at(put_idx).ep = ctx->ep;
  ctx->put_ops->at(put_idx).msg = ctx->put_msgs->at(put_idx);
  ctx->put_ops->at(put_idx).flags = 0;

  // Setup deferred work
  ctx->put_works->at(put_idx).triggering_cntr = os.trigger_cntr;
  ctx->put_works->at(put_idx).completion_cntr = os.completion_cntr;
  ctx->put_works->at(put_idx).threshold = 1;
  ctx->put_works->at(put_idx).op_type = FI_OP_WRITE;
  ctx->put_works->at(put_idx).op.rma = &ctx->put_ops->at(put_idx);

  int ret = fi_control(&ctx->domain->fid, FI_QUEUE_WORK,
                       &ctx->put_works->at(put_idx));
  if (ret) {
    fprintf(stderr, "Rank %d: fi_control(put slot %d) failed: %s\n",
            ctx->rank, slot_idx, fi_strerror(-ret));
    return ret;
  }

  return 0;
}

// =============================================================================
// CPU Proxy Thread
// =============================================================================

static void *proxy_thread_func(void *arg) {
  ProxyContext *ctx = (ProxyContext *)arg;

  while (!ctx->stop_flag.load(std::memory_order_relaxed)) {
    bool did_work = false;

    // Scan all slots for those needing re-queue
    for (int s = 0; s < ctx->total_slots; s++) {
      int state = ((volatile int *)ctx->slot_state)[s];

      if (state == SLOT_NEED_QUEUE) {
        OperationSlot &os = ctx->slots[s];
        int ret = 0;

        if (os.op_type == OP_TYPE_BARRIER) {
          ret = queue_barrier_work(ctx, s, os.phase, os.target_rank);
          if (ret == 0) {
            ctx->barrier_rearms.fetch_add(1, std::memory_order_relaxed);
          }
        } else if (os.op_type == OP_TYPE_PUT) {
          ret = queue_put_work(ctx, s, os.target_rank);
          if (ret == 0) {
            ctx->put_rearms.fetch_add(1, std::memory_order_relaxed);
          }
        }

        if (ret == 0) {
          __atomic_store_n(&ctx->slot_state[s], SLOT_ARMED, __ATOMIC_RELEASE);
          ctx->total_rearms.fetch_add(1, std::memory_order_relaxed);
          did_work = true;
        }
      }
    }

    // Drain CQ to prevent overflow
    struct fi_cq_entry cq_entries[32];
    int ret;
    while ((ret = fi_cq_read(ctx->cq, cq_entries, 32)) > 0) {
      // Just drain, completion tracking is done via counters
    }

    if (!did_work) {
      usleep(10);
    }
  }

  return NULL;
}

// =============================================================================
// GPU Kernel
// =============================================================================

/*
 * GPU kernel for barrier + put verification
 *
 * Slot layout per window:
 *   slots[0..num_phases-1]: barrier phase slots
 *   slots[num_phases]:      put slot
 *
 * Each iteration does:
 *   1. Prepare send data (write to d_send_buf)
 *   2. Trigger put to next rank
 *   3. First barrier (sync all ranks)
 *   4. Verify received data
 *   5. Second barrier (ensure all verifications complete)
 */
__global__ void gpu_barrier_put_kernel(
    volatile int *slot_state,              // Slot states [total_slots]
    volatile uint64_t **dev_trigger_addrs, // Trigger addresses [total_slots]
    volatile uint64_t *d_phase_counters,   // Phase counters [num_phases]
    volatile uint64_t *d_send_buf,         // Buffer to send (1 uint64_t)
    volatile uint64_t *d_recv_buf,         // Buffer for received data
    int num_iters,
    int window_size,
    int num_phases,
    int size,
    int rank,
    uint64_t *out_errors,
    uint64_t *out_put_errors,
    uint64_t *out_barrier_errors,
    uint64_t *out_cycles) {

  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  uint64_t start = clock64();
  uint64_t errors = 0;
  uint64_t put_errors = 0;
  uint64_t barrier_errors = 0;

  // Slots per window: num_phases (barrier) + 1 (put)
  int slots_per_window = num_phases + 1;

  for (int epoch = 0; epoch < num_iters; epoch++) {
    int window_slot = epoch % window_size;
    int base_slot = window_slot * slots_per_window;
    int put_slot = base_slot + num_phases;  // Last slot in group is put

    // =========================================================================
    // Step 1: First Barrier - Synchronize before puts
    // =========================================================================
    // This ensures all ranks are ready before any rank starts its put
    for (int phase = 0; phase < num_phases; phase++) {
      int barrier_slot = base_slot + phase;

      while (__atomic_load_n(&slot_state[barrier_slot], __ATOMIC_ACQUIRE) != SLOT_ARMED) {
        // Spin wait
      }

      // Expected arrivals for this phase (3 barriers per epoch)
      uint64_t expected = (uint64_t)(epoch * 3 + 1);

      __threadfence_system();
      *dev_trigger_addrs[barrier_slot] = 1;
      __threadfence_system();

      while (__atomic_load_n((unsigned long long *)&d_phase_counters[phase],
                             __ATOMIC_ACQUIRE) < expected) {
        // Spin wait
      }

      __atomic_store_n(&slot_state[barrier_slot], SLOT_NEED_QUEUE, __ATOMIC_RELEASE);
    }

    // =========================================================================
    // Step 2: Prepare and send data
    // =========================================================================
    // Unique value: high bits = rank, low bits = iteration
    uint64_t send_value = ((uint64_t)rank << 24) | (uint64_t)epoch;
    *d_send_buf = send_value;
    __threadfence_system();

    // Wait for put slot to be armed
    while (__atomic_load_n(&slot_state[put_slot], __ATOMIC_ACQUIRE) != SLOT_ARMED) {
      // Spin wait
    }

    // Trigger the put
    __threadfence_system();
    *dev_trigger_addrs[put_slot] = 1;
    __threadfence_system();

    // Mark slot for requeue
    __atomic_store_n(&slot_state[put_slot], SLOT_NEED_QUEUE, __ATOMIC_RELEASE);

    // =========================================================================
    // Step 3: Second Barrier - Wait for all puts to complete
    // =========================================================================
    // By the time this barrier completes, all ranks have triggered their puts
    // and the network has had time to deliver them
    for (int phase = 0; phase < num_phases; phase++) {
      int barrier_slot = base_slot + phase;

      while (__atomic_load_n(&slot_state[barrier_slot], __ATOMIC_ACQUIRE) != SLOT_ARMED) {
        // Spin wait
      }

      uint64_t expected = (uint64_t)(epoch * 3 + 2);

      __threadfence_system();
      *dev_trigger_addrs[barrier_slot] = 1;
      __threadfence_system();

      while (__atomic_load_n((unsigned long long *)&d_phase_counters[phase],
                             __ATOMIC_ACQUIRE) < expected) {
        // Spin wait
      }

      __atomic_store_n(&slot_state[barrier_slot], SLOT_NEED_QUEUE, __ATOMIC_RELEASE);
    }

    // =========================================================================
    // Step 4: Verify received data
    // =========================================================================
    // Expected sender is previous rank: (rank - 1 + size) % size
    int sender = (rank - 1 + size) % size;
    uint64_t expected_value = ((uint64_t)sender << 24) | (uint64_t)epoch;

    __threadfence_system();
    uint64_t received = *d_recv_buf;

    if (received != expected_value) {
      put_errors++;
      errors++;
    }

    // =========================================================================
    // Step 5: Third Barrier - Ensure all verifications complete
    // =========================================================================
    for (int phase = 0; phase < num_phases; phase++) {
      int barrier_slot = base_slot + phase;

      while (__atomic_load_n(&slot_state[barrier_slot], __ATOMIC_ACQUIRE) != SLOT_ARMED) {
        // Spin wait
      }

      uint64_t expected = (uint64_t)(epoch * 3 + 3);

      __threadfence_system();
      *dev_trigger_addrs[barrier_slot] = 1;
      __threadfence_system();

      while (__atomic_load_n((unsigned long long *)&d_phase_counters[phase],
                             __ATOMIC_ACQUIRE) < expected) {
        // Spin wait
      }

      __atomic_store_n(&slot_state[barrier_slot], SLOT_NEED_QUEUE, __ATOMIC_RELEASE);
    }
  }

  uint64_t end = clock64();
  *out_errors = errors;
  *out_put_errors = put_errors;
  *out_barrier_errors = barrier_errors;
  *out_cycles = end - start;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
  int window_size = DEFAULT_WINDOW_SIZE;
  int num_iters = DEFAULT_ITERATIONS;

  if (argc > 1) {
    num_iters = atoi(argv[1]);
    if (num_iters < 1)
      num_iters = 1;
  }
  if (argc > 2) {
    window_size = atoi(argv[2]);
    if (window_size < 1)
      window_size = 1;
    if (window_size > MAX_WINDOW_SIZE)
      window_size = MAX_WINDOW_SIZE;
  }

  // Environment setup for Tioga
  unsetenv("ROCR_VISIBLE_DEVICES");
  setenv("FI_CXI_DEVICE_NAME", "cxi3", 1);
  int device_id = 7;
  CHECK_HIP(hipSetDevice(device_id), "hipSetDevice");

  // PMI2 Initialization
  int spawned, size, appnum;
  PMI2_Init(&spawned, &size, &myrank, &appnum);

  if (size < 2) {
    if (myrank == 0) {
      fprintf(stderr, "This program requires at least 2 processes\n");
    }
    PMI2_Finalize();
    return 1;
  }

  if (size > MAX_RANKS) {
    if (myrank == 0) {
      fprintf(stderr, "Maximum %d ranks supported\n", MAX_RANKS);
    }
    PMI2_Finalize();
    return 1;
  }

  // Calculate number of phases for dissemination barrier
  int num_phases = (int)ceil(log2((double)size));
  int slots_per_window = num_phases + 1;  // barrier phases + 1 put slot
  int total_slots = window_size * slots_per_window;

  printf("Rank %d/%d: Barrier+Put verify - %d iters, window=%d, phases=%d, slots=%d\n",
         myrank, size, num_iters, window_size, num_phases, total_slots);

  // HIP setup
  hipDeviceProp_t prop;
  CHECK_HIP(hipGetDeviceProperties(&prop, device_id), "hipGetDeviceProperties");
  double gpu_clock_mhz = prop.clockRate / 1000.0;
  printf("Rank %d: GPU %d: %s (%.0f MHz)\n", myrank, device_id, prop.name,
         gpu_clock_mhz);

  // Libfabric Initialization
  struct fi_info *hints = fi_allocinfo();
  hints->caps = FI_RMA | FI_MSG | FI_HMEM | FI_ATOMIC;
  hints->mode = FI_CONTEXT2;
  hints->ep_attr->type = FI_EP_RDM;
  hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                FI_MR_PROV_KEY | FI_MR_LOCAL | FI_MR_ENDPOINT;
  hints->domain_attr->threading = FI_THREAD_SAFE;
  hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
  hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

  struct fi_info *info = NULL;
  int ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL,
                       NULL, 0, hints, &info);
  fi_freeinfo(hints);

  if (ret) {
    fprintf(stderr, "Rank %d: fi_getinfo failed: %s\n", myrank,
            fi_strerror(-ret));
    PMI2_Finalize();
    exit(1);
  }

  // Find CXI provider
  struct fi_info *cxi_info = NULL;
  for (struct fi_info *cur = info; cur; cur = cur->next) {
    if (cur->fabric_attr && cur->fabric_attr->prov_name &&
        strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
      cxi_info = cur;
      break;
    }
  }
  if (!cxi_info) {
    fprintf(stderr, "Rank %d: CXI provider not found!\n", myrank);
    fi_freeinfo(info);
    PMI2_Finalize();
    exit(1);
  }

  // Create Fabric, Domain, AV, CQ, EP
  struct fid_fabric *fabric = NULL;
  CHECK(fi_fabric(cxi_info->fabric_attr, &fabric, NULL), "fi_fabric");

  struct fid_domain *domain = NULL;
  CHECK(fi_domain(fabric, cxi_info, &domain, NULL), "fi_domain");

  struct fid_av *av = NULL;
  struct fi_av_attr av_attr = {};
  av_attr.type = FI_AV_TABLE;
  CHECK(fi_av_open(domain, &av_attr, &av, NULL), "fi_av_open");

  struct fid_cq *cq = NULL;
  struct fi_cq_attr cq_attr = {};
  cq_attr.size = 8192;
  cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  CHECK(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");

  struct fid_ep *ep = NULL;
  CHECK(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");
  CHECK(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");
  CHECK(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
  CHECK(fi_enable(ep), "fi_enable");

  // Allocate operation slots
  std::vector<OperationSlot> slots(total_slots);
  printf("Rank %d: Allocating %d operation slots...\n", myrank, total_slots);

  for (int i = 0; i < total_slots; i++) {
    OperationSlot &os = slots[i];
    os.initialized = false;

    // Determine operation type and parameters
    int slot_in_window = i % slots_per_window;
    (void)slots_per_window;  // Used in loop logic above

    if (slot_in_window < num_phases) {
      // Barrier slot
      os.op_type = OP_TYPE_BARRIER;
      os.phase = slot_in_window;
      // Target for dissemination barrier phase p: (rank + 2^p) % size
      int step = 1 << slot_in_window;
      os.target_rank = (myrank + step) % size;
    } else {
      // Put slot
      os.op_type = OP_TYPE_PUT;
      os.phase = -1;
      // Put target: next rank in ring
      os.target_rank = (myrank + 1) % size;
    }

    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;

    CHECK(fi_cntr_open(domain, &cntr_attr, &os.trigger_cntr, NULL),
          "fi_cntr_open(trigger)");
    CHECK(fi_cntr_open(domain, &cntr_attr, &os.completion_cntr, NULL),
          "fi_cntr_open(completion)");

    CHECK(fi_open_ops(&os.trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                      (void **)&os.trigger_ops, NULL),
          "fi_open_ops(trigger)");

    CHECK(os.trigger_ops->get_mmio_addr(&os.trigger_cntr->fid,
                                        &os.trigger_mmio_addr,
                                        &os.trigger_mmio_len),
          "get_mmio_addr(trigger)");

    CHECK_HIP(hipHostRegister(os.trigger_mmio_addr, os.trigger_mmio_len,
                              hipHostRegisterMapped),
              "hipHostRegister(trigger)");
    CHECK_HIP(hipHostGetDevicePointer((void **)&os.dev_trigger_addr,
                                      os.trigger_mmio_addr, 0),
              "hipHostGetDevicePointer(trigger)");

    // Setup completion counter MMIO for GPU polling (especially for put slots)
    CHECK(fi_open_ops(&os.completion_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                      (void **)&os.completion_ops, NULL),
          "fi_open_ops(completion)");

    CHECK(os.completion_ops->get_mmio_addr(&os.completion_cntr->fid,
                                           &os.completion_mmio_addr,
                                           &os.completion_mmio_len),
          "get_mmio_addr(completion)");

    CHECK_HIP(hipHostRegister(os.completion_mmio_addr, os.completion_mmio_len,
                              hipHostRegisterMapped),
              "hipHostRegister(completion)");
    CHECK_HIP(hipHostGetDevicePointer((void **)&os.dev_completion_addr,
                                      os.completion_mmio_addr, 0),
              "hipHostGetDevicePointer(completion)");

    os.initialized = true;
  }

  printf("Rank %d: Created %d operation slots (barrier=%d, put=%d per window)\n",
         myrank, total_slots, num_phases, 1);

  // Exchange endpoint addresses
  size_t addrlen = 0;
  fi_getname(&ep->fid, NULL, &addrlen);
  void *local_addr = malloc(addrlen);
  CHECK(fi_getname(&ep->fid, local_addr, &addrlen), "fi_getname");

  char *my_hex = (char *)malloc(2 * addrlen + 1);
  bytes_to_hex((uint8_t *)local_addr, addrlen, my_hex);

  char *all_hex = (char *)malloc(size * (2 * addrlen + 1));
  CHECK(exchange_addrs(myrank, size, my_hex, 2 * addrlen, all_hex),
        "exchange_addrs");

  uint8_t *all_bin = (uint8_t *)malloc(size * addrlen);
  for (int i = 0; i < size; i++) {
    const char *hex = all_hex + i * (2 * addrlen + 1);
    CHECK(hex_to_bytes(hex, all_bin + i * addrlen, addrlen) > 0 ? 0 : -1,
          "hex_to_bytes");
  }

  std::vector<fi_addr_t> peer_fi_addrs(size);
  for (int i = 0; i < size; i++) {
    void *addr = all_bin + i * addrlen;
    int inserted = fi_av_insert(av, addr, 1, &peer_fi_addrs[i], 0, NULL);
    if (inserted != 1 || peer_fi_addrs[i] == FI_ADDR_NOTAVAIL) {
      fprintf(stderr, "Rank %d: fi_av_insert failed for rank %d\n", myrank, i);
      PMI2_Finalize();
      exit(1);
    }
  }

  // Allocate GPU buffers
  // Phase counters: one per phase for barrier sync
  uint64_t *d_phase_counters = NULL;
  CHECK_HIP(hipMalloc(&d_phase_counters, sizeof(uint64_t) * num_phases),
            "hipMalloc(phase_counters)");
  CHECK_HIP(hipMemset(d_phase_counters, 0, sizeof(uint64_t) * num_phases),
            "hipMemset(phase_counters)");

  // Atomic operand (always 1)
  uint64_t *d_atomic_operand = NULL;
  CHECK_HIP(hipMalloc(&d_atomic_operand, sizeof(uint64_t)),
            "hipMalloc(atomic_operand)");
  uint64_t one = 1;
  CHECK_HIP(hipMemcpy(d_atomic_operand, &one, sizeof(uint64_t),
                      hipMemcpyHostToDevice),
            "hipMemcpy(atomic_operand)");

  // Send buffer for put
  uint64_t *d_send_buf = NULL;
  CHECK_HIP(hipMalloc(&d_send_buf, sizeof(uint64_t)), "hipMalloc(send_buf)");
  CHECK_HIP(hipMemset(d_send_buf, 0, sizeof(uint64_t)), "hipMemset(send_buf)");

  // Receive buffer for put (will be written by remote rank)
  uint64_t *d_recv_buf = NULL;
  CHECK_HIP(hipMalloc(&d_recv_buf, sizeof(uint64_t)), "hipMalloc(recv_buf)");
  CHECK_HIP(hipMemset(d_recv_buf, 0xFF, sizeof(uint64_t)), "hipMemset(recv_buf)");

  // Output buffers
  uint64_t *d_errors = NULL;
  uint64_t *d_put_errors = NULL;
  uint64_t *d_barrier_errors = NULL;
  uint64_t *d_cycles = NULL;
  CHECK_HIP(hipMalloc(&d_errors, sizeof(uint64_t)), "hipMalloc(errors)");
  CHECK_HIP(hipMalloc(&d_put_errors, sizeof(uint64_t)), "hipMalloc(put_errors)");
  CHECK_HIP(hipMalloc(&d_barrier_errors, sizeof(uint64_t)), "hipMalloc(barrier_errors)");
  CHECK_HIP(hipMalloc(&d_cycles, sizeof(uint64_t)), "hipMalloc(cycles)");

  // Register memory regions
  struct fid_mr *mr_phase_counters = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_phase_counters,
                               sizeof(uint64_t) * num_phases,
                               &mr_phase_counters, true),
        "register_memory_region(phase_counters)");

  struct fid_mr *mr_atomic_operand = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_atomic_operand,
                               sizeof(uint64_t), &mr_atomic_operand, true),
        "register_memory_region(atomic_operand)");

  struct fid_mr *mr_send_buf = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_send_buf,
                               sizeof(uint64_t), &mr_send_buf, true),
        "register_memory_region(send_buf)");

  struct fid_mr *mr_recv_buf = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_recv_buf,
                               sizeof(uint64_t), &mr_recv_buf, true),
        "register_memory_region(recv_buf)");

  // Exchange phase counter MR info
  std::vector<std::vector<RemotePhaseInfo>> peer_phase_info(num_phases);
  for (int p = 0; p < num_phases; p++) {
    peer_phase_info[p].resize(size);
  }

  for (int p = 0; p < num_phases; p++) {
    RemotePhaseInfo my_info;
    if (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
      my_info.addr = (uint64_t)&d_phase_counters[p];
    } else {
      my_info.addr = p * sizeof(uint64_t);
    }
    my_info.key = fi_mr_key(mr_phase_counters);

    char info_hex[128];
    bytes_to_hex((uint8_t *)&my_info, sizeof(my_info), info_hex);

    char key_str[PMI2_MAX_KEYLEN];
    snprintf(key_str, sizeof(key_str), "phase-%d-rank-%d", p, myrank);
    CHECK(PMI2_KVS_Put(key_str, info_hex), "PMI2_KVS_Put(phase)");
  }

  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(phase)");

  for (int p = 0; p < num_phases; p++) {
    for (int r = 0; r < size; r++) {
      char key_str[PMI2_MAX_KEYLEN];
      snprintf(key_str, sizeof(key_str), "phase-%d-rank-%d", p, r);
      char peer_hex[PMI2_MAX_VALLEN];
      int vallen;
      CHECK(PMI2_KVS_Get(NULL, PMI2_ID_NULL, key_str, peer_hex, sizeof(peer_hex),
                         &vallen),
            "PMI2_KVS_Get(phase)");
      CHECK(hex_to_bytes(peer_hex, (uint8_t *)&peer_phase_info[p][r],
                         sizeof(RemotePhaseInfo)) > 0
                ? 0
                : -1,
            "hex_to_bytes(phase)");
    }
  }

  // Exchange recv buffer MR info for put operations
  std::vector<RemotePutInfo> peer_put_info(size);

  RemotePutInfo my_put_info;
  if (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
    my_put_info.addr = (uint64_t)d_recv_buf;
  } else {
    my_put_info.addr = 0;
  }
  my_put_info.key = fi_mr_key(mr_recv_buf);

  char put_info_hex[128];
  bytes_to_hex((uint8_t *)&my_put_info, sizeof(my_put_info), put_info_hex);

  char put_key_str[PMI2_MAX_KEYLEN];
  snprintf(put_key_str, sizeof(put_key_str), "put-rank-%d", myrank);
  CHECK(PMI2_KVS_Put(put_key_str, put_info_hex), "PMI2_KVS_Put(put)");

  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(put)");

  for (int r = 0; r < size; r++) {
    snprintf(put_key_str, sizeof(put_key_str), "put-rank-%d", r);
    char peer_hex[PMI2_MAX_VALLEN];
    int vallen;
    CHECK(PMI2_KVS_Get(NULL, PMI2_ID_NULL, put_key_str, peer_hex,
                       sizeof(peer_hex), &vallen),
          "PMI2_KVS_Get(put)");
    CHECK(hex_to_bytes(peer_hex, (uint8_t *)&peer_put_info[r],
                       sizeof(RemotePutInfo)) > 0
              ? 0
              : -1,
          "hex_to_bytes(put)");
  }

  printf("Rank %d: Exchanged phase and put MR info\n", myrank);

  // Allocate slot_state in host-pinned memory
  int *h_slot_state = NULL;
  CHECK_HIP(hipHostMalloc(&h_slot_state, sizeof(int) * total_slots,
                          hipHostMallocMapped),
            "hipHostMalloc(slot_state)");

  for (int i = 0; i < total_slots; i++) {
    h_slot_state[i] = SLOT_NEED_QUEUE;
  }

  volatile int *d_slot_state = NULL;
  CHECK_HIP(hipHostGetDevicePointer((void **)&d_slot_state, h_slot_state, 0),
            "hipHostGetDevicePointer(slot_state)");

  // Allocate trigger address array for GPU
  volatile uint64_t **h_trigger_addrs = NULL;
  CHECK_HIP(hipHostMalloc(&h_trigger_addrs,
                          sizeof(volatile uint64_t *) * total_slots,
                          hipHostMallocMapped),
            "hipHostMalloc(trigger_addrs)");

  for (int i = 0; i < total_slots; i++) {
    h_trigger_addrs[i] = slots[i].dev_trigger_addr;
  }

  volatile uint64_t **d_trigger_addrs = NULL;
  CHECK_HIP(hipHostGetDevicePointer((void **)&d_trigger_addrs, h_trigger_addrs, 0),
            "hipHostGetDevicePointer(trigger_addrs)");

  // Allocate completion address array for GPU (to wait for put completion)
  volatile uint64_t **h_completion_addrs = NULL;
  CHECK_HIP(hipHostMalloc(&h_completion_addrs,
                          sizeof(volatile uint64_t *) * total_slots,
                          hipHostMallocMapped),
            "hipHostMalloc(completion_addrs)");

  for (int i = 0; i < total_slots; i++) {
    h_completion_addrs[i] = slots[i].dev_completion_addr;
  }

  volatile uint64_t **d_completion_addrs = NULL;
  CHECK_HIP(hipHostGetDevicePointer((void **)&d_completion_addrs, h_completion_addrs, 0),
            "hipHostGetDevicePointer(completion_addrs)");

  // Allocate DWQ structures for barrier
  std::vector<struct fi_deferred_work> barrier_works(total_slots);
  std::vector<struct fi_op_atomic> barrier_atomics(total_slots);
  std::vector<struct fi_msg_atomic> barrier_msgs(total_slots);
  std::vector<struct fi_ioc> barrier_iovs(total_slots);
  std::vector<struct fi_rma_ioc> barrier_rma_iovs(total_slots);

  // Allocate DWQ structures for put
  std::vector<struct fi_deferred_work> put_works(total_slots);
  std::vector<struct fi_op_rma> put_ops(total_slots);
  std::vector<struct fi_msg_rma> put_msgs(total_slots);
  std::vector<struct iovec> put_iovs(total_slots);
  std::vector<struct fi_rma_iov> put_rma_iovs(total_slots);

  // Setup proxy context
  ProxyContext proxy_ctx;
  proxy_ctx.domain = domain;
  proxy_ctx.ep = ep;
  proxy_ctx.cq = cq;
  proxy_ctx.info = cxi_info;
  proxy_ctx.window_size = window_size;
  proxy_ctx.num_phases = num_phases;
  proxy_ctx.barrier_slots_per_window = num_phases;
  proxy_ctx.put_slots_per_window = 1;
  proxy_ctx.total_slots = total_slots;
  proxy_ctx.slots = slots.data();
  proxy_ctx.slot_state = h_slot_state;
  proxy_ctx.size = size;
  proxy_ctx.rank = myrank;
  proxy_ctx.peer_fi_addrs = peer_fi_addrs.data();
  proxy_ctx.peer_phase_info = peer_phase_info.data();
  proxy_ctx.peer_put_info = peer_put_info.data();
  proxy_ctx.d_atomic_operand = d_atomic_operand;
  proxy_ctx.desc_atomic_operand = fi_mr_desc(mr_atomic_operand);
  proxy_ctx.mr_atomic_operand = mr_atomic_operand;
  proxy_ctx.d_send_buf = d_send_buf;
  proxy_ctx.desc_send_buf = fi_mr_desc(mr_send_buf);
  proxy_ctx.mr_send_buf = mr_send_buf;
  proxy_ctx.barrier_works = &barrier_works;
  proxy_ctx.barrier_atomics = &barrier_atomics;
  proxy_ctx.barrier_msgs = &barrier_msgs;
  proxy_ctx.barrier_iovs = &barrier_iovs;
  proxy_ctx.barrier_rma_iovs = &barrier_rma_iovs;
  proxy_ctx.put_works = &put_works;
  proxy_ctx.put_ops = &put_ops;
  proxy_ctx.put_msgs = &put_msgs;
  proxy_ctx.put_iovs = &put_iovs;
  proxy_ctx.put_rma_iovs = &put_rma_iovs;
  proxy_ctx.stop_flag.store(false);
  proxy_ctx.total_rearms.store(0);
  proxy_ctx.barrier_rearms.store(0);
  proxy_ctx.put_rearms.store(0);

  // Sync before starting
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(pre-test)");
  usleep(100000);

  printf("Rank %d: Starting proxy thread and GPU kernel\n", myrank);

  // Start proxy thread
  pthread_t proxy_thread;
  ret = pthread_create(&proxy_thread, NULL, proxy_thread_func, &proxy_ctx);
  if (ret != 0) {
    fprintf(stderr, "Rank %d: pthread_create failed: %d\n", myrank, ret);
    PMI2_Finalize();
    exit(1);
  }

  // Wait for all slots to be armed
  printf("Rank %d: Waiting for slots to be armed...\n", myrank);
  bool all_armed = false;
  while (!all_armed) {
    all_armed = true;
    for (int i = 0; i < total_slots; i++) {
      if (__atomic_load_n(&h_slot_state[i], __ATOMIC_ACQUIRE) != SLOT_ARMED) {
        all_armed = false;
        break;
      }
    }
    if (!all_armed) {
      usleep(100);
    }
  }
  printf("Rank %d: All %d slots armed, launching kernel\n", myrank, total_slots);

  // Final sync
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(pre-kernel)");

  double start_time = get_time_us();

  // Launch GPU kernel
  hipLaunchKernelGGL(gpu_barrier_put_kernel, dim3(1), dim3(1), 0, 0,
                     d_slot_state, d_trigger_addrs,
                     d_phase_counters, d_send_buf, d_recv_buf,
                     num_iters, window_size, num_phases, size, myrank,
                     d_errors, d_put_errors, d_barrier_errors, d_cycles);

  CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");

  double end_time = get_time_us();

  // Stop proxy thread
  proxy_ctx.stop_flag.store(true, std::memory_order_release);
  pthread_join(proxy_thread, NULL);

  printf("Rank %d: Kernel completed\n", myrank);

  // Get results
  uint64_t h_errors, h_put_errors, h_barrier_errors, h_cycles;
  CHECK_HIP(hipMemcpy(&h_errors, d_errors, sizeof(uint64_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy(errors)");
  CHECK_HIP(hipMemcpy(&h_put_errors, d_put_errors, sizeof(uint64_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy(put_errors)");
  CHECK_HIP(hipMemcpy(&h_barrier_errors, d_barrier_errors, sizeof(uint64_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy(barrier_errors)");
  CHECK_HIP(hipMemcpy(&h_cycles, d_cycles, sizeof(uint64_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy(cycles)");

  // Sync and print results
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(post-test)");

  double wall_time_ms = (end_time - start_time) / 1000.0;
  double avg_iter_us = (end_time - start_time) / num_iters;
  double gpu_time_us = (double)h_cycles / gpu_clock_mhz;

  printf("\n");
  printf("======================================\n");
  printf("Rank %d Results:\n", myrank);
  printf("======================================\n");
  printf("  Configuration:\n");
  printf("    Iterations: %d\n", num_iters);
  printf("    Barriers per iter: 3\n");
  printf("    Puts per iter: 1\n");
  printf("    Ranks: %d\n", size);
  printf("    Phases per barrier: %d\n", num_phases);
  printf("    Window size: %d\n", window_size);
  printf("    Total slots: %d\n", total_slots);
  printf("  Verification:\n");
  printf("    Total errors: %lu\n", h_errors);
  printf("    Put errors: %lu\n", h_put_errors);
  printf("    Barrier errors: %lu\n", h_barrier_errors);
  printf("  Performance:\n");
  printf("    Wall time: %.2f ms\n", wall_time_ms);
  printf("    GPU time: %.2f ms\n", gpu_time_us / 1000.0);
  printf("    Avg iter latency: %.2f us\n", avg_iter_us);
  printf("  Proxy Stats:\n");
  printf("    Total rearms: %lu\n",
         proxy_ctx.total_rearms.load(std::memory_order_relaxed));
  printf("    Barrier rearms: %lu\n",
         proxy_ctx.barrier_rearms.load(std::memory_order_relaxed));
  printf("    Put rearms: %lu\n",
         proxy_ctx.put_rearms.load(std::memory_order_relaxed));
  printf("======================================\n");

  if (h_errors == 0) {
    printf("  Status: PASS - All verifications successful!\n");
  } else {
    printf("  Status: FAIL - %lu errors detected!\n", h_errors);
  }
  printf("======================================\n");

  // Cleanup
  for (int i = 0; i < total_slots; i++) {
    if (slots[i].initialized) {
      hipHostUnregister(slots[i].trigger_mmio_addr);
      hipHostUnregister(slots[i].completion_mmio_addr);
      fi_close(&slots[i].trigger_cntr->fid);
      fi_close(&slots[i].completion_cntr->fid);
    }
  }

  hipHostFree(h_slot_state);
  hipHostFree((void *)h_trigger_addrs);
  hipHostFree((void *)h_completion_addrs);
  hipFree(d_phase_counters);
  hipFree(d_atomic_operand);
  hipFree(d_send_buf);
  hipFree(d_recv_buf);
  hipFree(d_errors);
  hipFree(d_put_errors);
  hipFree(d_barrier_errors);
  hipFree(d_cycles);

  fi_close(&mr_phase_counters->fid);
  fi_close(&mr_atomic_operand->fid);
  fi_close(&mr_send_buf->fid);
  fi_close(&mr_recv_buf->fid);
  fi_close(&ep->fid);
  fi_close(&cq->fid);
  fi_close(&av->fid);
  fi_close(&domain->fid);
  fi_close(&fabric->fid);
  fi_freeinfo(info);

  free(local_addr);
  free(my_hex);
  free(all_hex);
  free(all_bin);

  PMI2_Finalize();

  printf("Rank %d: Done!\n", myrank);
  return 0;
}
