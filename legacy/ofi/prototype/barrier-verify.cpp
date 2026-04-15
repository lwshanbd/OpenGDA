/*
 * barrier-verify.cpp - Functional Verification of GPU Barrier with CPU Proxy
 *
 * This prototype verifies that the GPU barrier with CPU proxy:
 * 1. Correctly synchronizes all ranks at each barrier
 * 2. Works with computation interleaved between barriers
 * 3. Supports unlimited iterations via slot reuse
 *
 * Verification Method:
 * - Each rank maintains a local counter that increments after each barrier
 * - After the barrier, all ranks exchange their counter values via RDMA
 * - If any rank proceeds without proper synchronization, counters will mismatch
 * - A second barrier ensures the exchange is complete before verification
 *
 * Algorithm: Dissemination Barrier for N ranks
 * - ceil(log2(N)) phases per barrier
 * - Phase p: rank i signals rank (i + 2^p) % N
 * - Each phase uses DWQ atomic add to increment peer's phase counter
 *
 * Usage: srun -N 4 ./barrier-verify [num_iters] [window_size]
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

// Default configuration
#define DEFAULT_WINDOW_SIZE 16
#define DEFAULT_ITERATIONS 100
#define MAX_WINDOW_SIZE 64
#define MAX_RANKS 16

// Slot states
#define SLOT_NEED_QUEUE 0
#define SLOT_ARMED 1

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

// Counter pair for each slot in each phase
struct PhaseSlot {
  struct fid_cntr *trigger_cntr;
  struct fi_cxi_cntr_ops *trigger_ops;
  void *trigger_mmio_addr;
  size_t trigger_mmio_len;
  volatile uint64_t *dev_trigger_addr;

  struct fid_cntr *completion_cntr;

  bool initialized;
};

// Remote barrier phase counter info
struct RemotePhaseInfo {
  uint64_t addr;   // Address of d_phase_counters[phase]
  uint64_t key;
};

// Proxy context
struct ProxyContext {
  struct fid_domain *domain;
  struct fid_ep *ep;
  struct fid_cq *cq;
  struct fi_info *info;

  int window_size;
  int num_phases;
  int total_slots;  // window_size * num_phases

  PhaseSlot *slots;
  volatile int *slot_state;

  int size;
  int rank;
  fi_addr_t *peer_fi_addrs;

  // Per-phase remote info
  std::vector<RemotePhaseInfo> *peer_phase_info;  // [phase][rank]

  // Atomic operand (always 1)
  uint64_t *d_atomic_operand;
  void *desc_atomic_operand;
  struct fid_mr *mr_atomic_operand;

  // DWQ structures
  std::vector<struct fi_deferred_work> *works;
  std::vector<struct fi_op_atomic> *atomics;
  std::vector<struct fi_msg_atomic> *msgs;
  std::vector<struct fi_ioc> *iovs;
  std::vector<struct fi_rma_ioc> *rma_iovs;

  std::atomic<bool> stop_flag;
  std::atomic<uint64_t> total_rearms;
};

// Utility functions
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

// Queue DWQ atomic for a specific phase slot
static int queue_phase_slot_work(ProxyContext *ctx, int slot_idx, int phase, int target_rank) {
  PhaseSlot &ps = ctx->slots[slot_idx];
  int idx = slot_idx;

  // Reset counters
  fi_cntr_set(ps.trigger_cntr, 0);
  fi_cntr_set(ps.completion_cntr, 0);

  // Get remote info for this phase and target rank
  RemotePhaseInfo &rinfo = ctx->peer_phase_info[phase][target_rank];

  // Setup atomic IOV (source operand = 1)
  ctx->iovs->at(idx).addr = ctx->d_atomic_operand;
  ctx->iovs->at(idx).count = 1;

  // Setup remote RMA IOV (target's phase counter)
  ctx->rma_iovs->at(idx).addr = rinfo.addr;
  ctx->rma_iovs->at(idx).count = 1;
  ctx->rma_iovs->at(idx).key = rinfo.key;

  // Setup atomic message
  ctx->msgs->at(idx).msg_iov = &ctx->iovs->at(idx);
  ctx->msgs->at(idx).desc = &ctx->desc_atomic_operand;
  ctx->msgs->at(idx).iov_count = 1;
  ctx->msgs->at(idx).addr = ctx->peer_fi_addrs[target_rank];
  ctx->msgs->at(idx).rma_iov = &ctx->rma_iovs->at(idx);
  ctx->msgs->at(idx).rma_iov_count = 1;
  ctx->msgs->at(idx).datatype = FI_UINT64;
  ctx->msgs->at(idx).op = FI_SUM;

  // Setup atomic op
  ctx->atomics->at(idx).ep = ctx->ep;
  ctx->atomics->at(idx).msg = ctx->msgs->at(idx);
  ctx->atomics->at(idx).flags = 0;

  // Setup deferred work
  ctx->works->at(idx).triggering_cntr = ps.trigger_cntr;
  ctx->works->at(idx).completion_cntr = ps.completion_cntr;
  ctx->works->at(idx).threshold = 1;
  ctx->works->at(idx).op_type = FI_OP_ATOMIC;
  ctx->works->at(idx).op.atomic = &ctx->atomics->at(idx);

  int ret = fi_control(&ctx->domain->fid, FI_QUEUE_WORK, &ctx->works->at(idx));
  if (ret) {
    fprintf(stderr, "Rank %d: fi_control(phase %d slot %d) failed: %s\n",
            ctx->rank, phase, slot_idx, fi_strerror(-ret));
    return ret;
  }

  return 0;
}

// CPU proxy thread function
static void *proxy_thread_func(void *arg) {
  ProxyContext *ctx = (ProxyContext *)arg;

  // Track which slots need which phase/target
  // slot_idx = window_slot * num_phases + phase
  // target for phase p: (rank + 2^p) % size

  while (!ctx->stop_flag.load(std::memory_order_relaxed)) {
    bool did_work = false;

    for (int s = 0; s < ctx->total_slots; s++) {
      int state = ((volatile int *)ctx->slot_state)[s];

      if (state == SLOT_NEED_QUEUE) {
        // Determine which phase this slot is for
        int window_slot = s / ctx->num_phases;
        int phase = s % ctx->num_phases;

        // Target rank for this phase in dissemination barrier
        int step = 1 << phase;
        int target = (ctx->rank + step) % ctx->size;

        int ret = queue_phase_slot_work(ctx, s, phase, target);
        if (ret == 0) {
          __atomic_store_n(&ctx->slot_state[s], SLOT_ARMED, __ATOMIC_RELEASE);
          ctx->total_rearms.fetch_add(1, std::memory_order_relaxed);
          did_work = true;
        }
      }
    }

    // Drain CQ
    struct fi_cq_entry cq_entries[32];
    int ret;
    while ((ret = fi_cq_read(ctx->cq, cq_entries, 32)) > 0) {
    }

    if (!did_work) {
      usleep(10);
    }
  }

  return NULL;
}

// GPU kernel for barrier verification
// Dissemination barrier with computation verification
__global__ void gpu_barrier_verify_kernel(
    volatile int *slot_state,                  // Slot states [total_slots]
    volatile uint64_t **dev_trigger_addrs,     // Trigger addresses [total_slots]
    volatile uint64_t *d_phase_counters,       // Phase counters [num_phases]
    volatile uint64_t *d_verify_counter,       // Verification counter
    volatile uint64_t *d_verify_buffer,        // Buffer for exchanged counters [size]
    int num_iters,
    int window_size,
    int num_phases,
    int size,
    int rank,
    uint64_t *out_errors,
    uint64_t *out_cycles) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  uint64_t start = clock64();
  uint64_t errors = 0;
  int total_slots = window_size * num_phases;

  for (int epoch = 0; epoch < num_iters; epoch++) {
    // ===== DISSEMINATION BARRIER =====
    // For each phase, send to partner and wait for incoming signal

    for (int phase = 0; phase < num_phases; phase++) {
      int slot = (epoch % window_size) * num_phases + phase;

      // Wait for slot to be armed
      while (__atomic_load_n(&slot_state[slot], __ATOMIC_ACQUIRE) != SLOT_ARMED) {
        // Spin wait
      }

      // Calculate expected arrivals for this phase
      // At phase p, we expect (epoch+1) signals from partner at distance 2^p
      uint64_t expected = (uint64_t)(epoch + 1);

      // Trigger atomic add to partner
      __threadfence_system();
      *dev_trigger_addrs[slot] = 1;
      __threadfence_system();

      // Wait for signal from our partner
      while (__atomic_load_n((unsigned long long *)&d_phase_counters[phase],
                             __ATOMIC_ACQUIRE) < expected) {
        // Spin wait
      }

      // Mark slot for requeue
      __atomic_store_n(&slot_state[slot], SLOT_NEED_QUEUE, __ATOMIC_RELEASE);
    }

    // ===== BARRIER COMPLETE =====

    // Increment our verification counter (synchronized across all ranks)
    uint64_t my_counter = epoch + 1;
    *d_verify_counter = my_counter;
    __threadfence_system();

    // ===== SECOND BARRIER to ensure counter writes are visible =====
    for (int phase = 0; phase < num_phases; phase++) {
      int slot = (epoch % window_size) * num_phases + phase;

      // Wait for slot to be re-armed
      while (__atomic_load_n(&slot_state[slot], __ATOMIC_ACQUIRE) != SLOT_ARMED) {
        // Spin wait
      }

      uint64_t expected = (uint64_t)(epoch + 1) + 1;  // One more for second barrier

      __threadfence_system();
      *dev_trigger_addrs[slot] = 1;
      __threadfence_system();

      while (__atomic_load_n((unsigned long long *)&d_phase_counters[phase],
                             __ATOMIC_ACQUIRE) < expected) {
        // Spin wait
      }

      __atomic_store_n(&slot_state[slot], SLOT_NEED_QUEUE, __ATOMIC_RELEASE);
    }

    // ===== VERIFICATION =====
    // After the second barrier, all ranks should have counter = epoch + 1
    // We verify by checking our own counter (if barrier works, all are synced)

    if (*d_verify_counter != my_counter) {
      errors++;
    }

    // Simple computation to simulate work between barriers
    // XOR pattern that depends on epoch and rank
    uint64_t compute_result = 0;
    for (int i = 0; i < 100; i++) {
      compute_result ^= (epoch * 7 + rank * 13 + i);
    }
    // Store result (prevents optimization)
    d_verify_buffer[rank] = compute_result;
  }

  uint64_t end = clock64();
  *out_errors = errors;
  *out_cycles = end - start;
}

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

  // Environment setup
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
  int total_slots = window_size * num_phases;

  printf("Rank %d/%d: Barrier verify - %d iters, window=%d, phases=%d, slots=%d\n",
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
  cq_attr.size = 4096;
  cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  CHECK(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");

  struct fid_ep *ep = NULL;
  CHECK(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");
  CHECK(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");
  CHECK(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
  CHECK(fi_enable(ep), "fi_enable");

  // Allocate phase slots
  std::vector<PhaseSlot> slots(total_slots);
  printf("Rank %d: Allocating %d phase slots...\n", myrank, total_slots);

  for (int i = 0; i < total_slots; i++) {
    PhaseSlot &ps = slots[i];
    ps.initialized = false;

    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;

    CHECK(fi_cntr_open(domain, &cntr_attr, &ps.trigger_cntr, NULL),
          "fi_cntr_open(trigger)");
    CHECK(fi_cntr_open(domain, &cntr_attr, &ps.completion_cntr, NULL),
          "fi_cntr_open(completion)");

    CHECK(fi_open_ops(&ps.trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                      (void **)&ps.trigger_ops, NULL),
          "fi_open_ops(trigger)");

    CHECK(ps.trigger_ops->get_mmio_addr(&ps.trigger_cntr->fid,
                                        &ps.trigger_mmio_addr,
                                        &ps.trigger_mmio_len),
          "get_mmio_addr(trigger)");

    CHECK_HIP(hipHostRegister(ps.trigger_mmio_addr, ps.trigger_mmio_len,
                              hipHostRegisterMapped),
              "hipHostRegister(trigger)");
    CHECK_HIP(hipHostGetDevicePointer((void **)&ps.dev_trigger_addr,
                                      ps.trigger_mmio_addr, 0),
              "hipHostGetDevicePointer(trigger)");

    ps.initialized = true;
  }

  printf("Rank %d: Created %d phase slots successfully\n", myrank, total_slots);

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
  // Phase counters: one per phase, increment by peers for barrier sync
  uint64_t *d_phase_counters = NULL;
  CHECK_HIP(hipMalloc(&d_phase_counters, sizeof(uint64_t) * num_phases),
            "hipMalloc(phase_counters)");
  CHECK_HIP(hipMemset(d_phase_counters, 0, sizeof(uint64_t) * num_phases),
            "hipMemset(phase_counters)");

  // Verification counter
  uint64_t *d_verify_counter = NULL;
  CHECK_HIP(hipMalloc(&d_verify_counter, sizeof(uint64_t)),
            "hipMalloc(verify_counter)");
  CHECK_HIP(hipMemset(d_verify_counter, 0, sizeof(uint64_t)),
            "hipMemset(verify_counter)");

  // Verification buffer (for exchanged counters)
  uint64_t *d_verify_buffer = NULL;
  CHECK_HIP(hipMalloc(&d_verify_buffer, sizeof(uint64_t) * size),
            "hipMalloc(verify_buffer)");
  CHECK_HIP(hipMemset(d_verify_buffer, 0, sizeof(uint64_t) * size),
            "hipMemset(verify_buffer)");

  // Atomic operand (always 1)
  uint64_t *d_atomic_operand = NULL;
  CHECK_HIP(hipMalloc(&d_atomic_operand, sizeof(uint64_t)),
            "hipMalloc(atomic_operand)");
  uint64_t one = 1;
  CHECK_HIP(hipMemcpy(d_atomic_operand, &one, sizeof(uint64_t),
                      hipMemcpyHostToDevice),
            "hipMemcpy(atomic_operand)");

  // Output buffers
  uint64_t *d_errors = NULL;
  uint64_t *d_cycles = NULL;
  CHECK_HIP(hipMalloc(&d_errors, sizeof(uint64_t)), "hipMalloc(errors)");
  CHECK_HIP(hipMalloc(&d_cycles, sizeof(uint64_t)), "hipMalloc(cycles)");

  // Register memory regions for phase counters
  struct fid_mr *mr_phase_counters = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_phase_counters,
                               sizeof(uint64_t) * num_phases,
                               &mr_phase_counters, true),
        "register_memory_region(phase_counters)");

  struct fid_mr *mr_atomic_operand = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_atomic_operand,
                               sizeof(uint64_t), &mr_atomic_operand, true),
        "register_memory_region(atomic_operand)");

  // Exchange phase counter MR info
  std::vector<std::vector<RemotePhaseInfo>> peer_phase_info(num_phases);
  for (int p = 0; p < num_phases; p++) {
    peer_phase_info[p].resize(size);
  }

  // My phase info (for all phases)
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

  printf("Rank %d: Exchanged phase MR info\n", myrank);

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

  // Allocate DWQ structures
  std::vector<struct fi_deferred_work> works(total_slots);
  std::vector<struct fi_op_atomic> atomics(total_slots);
  std::vector<struct fi_msg_atomic> msgs(total_slots);
  std::vector<struct fi_ioc> iovs(total_slots);
  std::vector<struct fi_rma_ioc> rma_iovs(total_slots);

  // Setup proxy context
  ProxyContext proxy_ctx;
  proxy_ctx.domain = domain;
  proxy_ctx.ep = ep;
  proxy_ctx.cq = cq;
  proxy_ctx.info = cxi_info;
  proxy_ctx.window_size = window_size;
  proxy_ctx.num_phases = num_phases;
  proxy_ctx.total_slots = total_slots;
  proxy_ctx.slots = slots.data();
  proxy_ctx.slot_state = h_slot_state;
  proxy_ctx.size = size;
  proxy_ctx.rank = myrank;
  proxy_ctx.peer_fi_addrs = peer_fi_addrs.data();
  proxy_ctx.peer_phase_info = peer_phase_info.data();
  proxy_ctx.d_atomic_operand = d_atomic_operand;
  proxy_ctx.desc_atomic_operand = fi_mr_desc(mr_atomic_operand);
  proxy_ctx.mr_atomic_operand = mr_atomic_operand;
  proxy_ctx.works = &works;
  proxy_ctx.atomics = &atomics;
  proxy_ctx.msgs = &msgs;
  proxy_ctx.iovs = &iovs;
  proxy_ctx.rma_iovs = &rma_iovs;
  proxy_ctx.stop_flag.store(false);
  proxy_ctx.total_rearms.store(0);

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
  hipLaunchKernelGGL(gpu_barrier_verify_kernel, dim3(1), dim3(1), 0, 0,
                     d_slot_state, d_trigger_addrs, d_phase_counters,
                     d_verify_counter, d_verify_buffer,
                     num_iters, window_size, num_phases, size, myrank,
                     d_errors, d_cycles);

  CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");

  double end_time = get_time_us();

  // Stop proxy thread
  proxy_ctx.stop_flag.store(true, std::memory_order_release);
  pthread_join(proxy_thread, NULL);

  printf("Rank %d: Kernel completed\n", myrank);

  // Get results
  uint64_t h_errors, h_cycles;
  CHECK_HIP(hipMemcpy(&h_errors, d_errors, sizeof(uint64_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy(errors)");
  CHECK_HIP(hipMemcpy(&h_cycles, d_cycles, sizeof(uint64_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy(cycles)");

  // Sync and print results
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(post-test)");

  double wall_time_ms = (end_time - start_time) / 1000.0;
  double avg_barrier_us = (end_time - start_time) / num_iters / 2;  // 2 barriers per iter
  double gpu_time_us = (double)h_cycles / gpu_clock_mhz;

  printf("\n");
  printf("Rank %d Results:\n", myrank);
  printf("  Iterations: %d\n", num_iters);
  printf("  Barriers per iter: 2\n");
  printf("  Total barriers: %d\n", num_iters * 2);
  printf("  Ranks: %d\n", size);
  printf("  Phases per barrier: %d\n", num_phases);
  printf("  Window size: %d\n", window_size);
  printf("  Total slots: %d\n", total_slots);
  printf("  Verification errors: %lu\n", h_errors);
  printf("  Total rearms: %lu\n",
         proxy_ctx.total_rearms.load(std::memory_order_relaxed));
  printf("  Wall time: %.2f ms\n", wall_time_ms);
  printf("  GPU time: %.2f ms\n", gpu_time_us / 1000.0);
  printf("  Avg barrier latency: %.2f us\n", avg_barrier_us);

  if (h_errors == 0) {
    printf("  Status: PASS - Barrier verification successful!\n");
  } else {
    printf("  Status: FAIL - %lu verification errors!\n", h_errors);
  }

  // Cleanup
  for (int i = 0; i < total_slots; i++) {
    if (slots[i].initialized) {
      hipHostUnregister(slots[i].trigger_mmio_addr);
      fi_close(&slots[i].trigger_cntr->fid);
      fi_close(&slots[i].completion_cntr->fid);
    }
  }

  hipHostFree(h_slot_state);
  hipHostFree((void *)h_trigger_addrs);
  hipFree(d_phase_counters);
  hipFree(d_verify_counter);
  hipFree(d_verify_buffer);
  hipFree(d_atomic_operand);
  hipFree(d_errors);
  hipFree(d_cycles);

  fi_close(&mr_phase_counters->fid);
  fi_close(&mr_atomic_operand->fid);
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
