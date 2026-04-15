/*
 * gda-proxy.cpp - GPU-Direct Async with CPU proxy for unlimited iterations
 *
 * Core design:
 * - Small window of counter pair slots (NUM_SLOTS) that get reused
 * - GPU runs a persistent kernel for many iterations (e.g., 10,000+)
 * - CPU proxy thread continuously rearms slots as they complete
 *
 * Slot state machine (in GPU-accessible pinned memory):
 *   0 = NEED_QUEUE: GPU finished using this slot, needs re-arming
 *   1 = ARMED: CPU proxy has queued DWQ work, GPU can use it
 *
 * Flow:
 *   GPU: wait(slot_state[i]==ARMED) -> trigger -> wait(completion) -> slot_state[i]=NEED_QUEUE
 *   CPU: scan(slot_state[i]==NEED_QUEUE) -> reset counters -> queue work -> slot_state[i]=ARMED
 */

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <pmi2.h>
#include <pthread.h>
#include <rdma/fabric.h>
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

// Number of slots (counter pairs) for window-based reuse
#define NUM_SLOTS 64

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

// Counter slot structure
struct CounterSlot {
  // Trigger counter - GPU writes to this to trigger operations
  struct fid_cntr *trigger_cntr;
  struct fi_cxi_cntr_ops *trigger_ops;
  void *trigger_mmio_addr;
  size_t trigger_mmio_len;
  volatile uint64_t *dev_trigger_addr;

  // Completion counter - NIC updates this when RDMA completes
  struct fid_cntr *completion_cntr;
  struct fi_cxi_cntr_ops *completion_ops;
  void *completion_mmio_addr;
  size_t completion_mmio_len;
  volatile uint64_t *dev_completion_addr;

  // Atomic completion counter - tracks when atomic operation completes
  struct fid_cntr *atomic_completion_cntr;

  bool initialized;
};

// Proxy context - shared between main thread and proxy thread
struct ProxyContext {
  // Slot state array (GPU-accessible pinned memory)
  volatile int *slot_state;       // Host pointer
  volatile int *dev_slot_state;   // Device pointer to same memory

  // Counter slots
  CounterSlot *slots;
  int num_slots;

  // Libfabric resources
  struct fid_domain *domain;
  struct fid_ep *ep;
  struct fid_cq *cq;
  struct fi_info *info;
  fi_addr_t peer_fi_addr;
  fi_addr_t local_fi_addr;

  // Memory regions and buffers
  void *d_local_buf;
  void *d_atomic_results;
  void *d_atomic_operands;
  void *desc_local;
  void *desc_atomic_operands;
  void *desc_atomic_results;
  struct fid_mr *mr_atomic_results;
  uint64_t remote_addr_for_rma;
  uint64_t peer_remote_key;
  size_t transfer_size;

  // DWQ structures (pre-allocated per slot)
  struct fi_deferred_work *works;
  struct fi_op_rma *op_rmas;
  struct fi_msg_rma *msg_rmas;
  struct iovec *iovs;
  struct fi_rma_iov *rma_iovs;

  struct fi_deferred_work *atomic_works;
  struct fi_op_atomic *atomics;
  struct fi_msg_atomic *atomic_msgs;
  struct fi_ioc *atomic_iovs;
  struct fi_rma_ioc *atomic_rma_iovs;

  // Control flags
  std::atomic<bool> proxy_running;
  std::atomic<bool> stop_requested;
  std::atomic<uint64_t> total_rearms;
  std::atomic<uint64_t> cq_events_drained;

  // Thread handle
  pthread_t proxy_thread;
};

// GPU kernel data structure (passed to persistent kernel)
struct GpuKernelData {
  volatile int *slot_state;           // Slot state array
  volatile uint64_t **trigger_addrs;  // Array of trigger MMIO addresses
  volatile uint64_t *atomic_results;  // Array of atomic results
  int num_slots;
  int num_iterations;
};

// Persistent GPU kernel - runs many iterations using windowed slots
__global__ void gpu_persistent_kernel(
    volatile int *slot_state,
    volatile uint64_t **trigger_addrs,
    volatile uint64_t *atomic_results,
    int num_slots,
    int num_iterations,
    uint64_t *total_cycles,
    uint64_t *gpu_spin_cycles) {

  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  uint64_t total_start = clock64();
  uint64_t spin_cycles = 0;

  for (int iter = 0; iter < num_iterations; iter++) {
    int slot = iter % num_slots;

    // Wait for slot to be armed by CPU proxy
    uint64_t spin_start = clock64();
    while (slot_state[slot] != SLOT_ARMED) {
      // Spin wait - CPU proxy will arm this slot
    }
    spin_cycles += (clock64() - spin_start);

    // Reset atomic result for this slot
    atomic_results[slot] = 0;
    __threadfence_system();

    // Trigger DWQ operation by writing to MMIO doorbell
    *trigger_addrs[slot] = 1;

    // Wait for atomic completion (NIC writes to atomic_results)
    while (atomic_results[slot] < 1) {
      // Spin wait for NIC completion
    }

    // Signal CPU proxy that slot needs re-arming
    slot_state[slot] = SLOT_NEED_QUEUE;
    __threadfence_system();
  }

  uint64_t total_end = clock64();
  *total_cycles = total_end - total_start;
  *gpu_spin_cycles = spin_cycles;
}

// CPU proxy thread function
void *proxy_thread_func(void *arg) {
  ProxyContext *ctx = (ProxyContext *)arg;
  int ret;

  printf("Rank %d: Proxy thread started\n", myrank);
  fflush(stdout);

  while (!ctx->stop_requested.load()) {
    bool did_work = false;

    // Scan all slots for NEED_QUEUE state
    for (int i = 0; i < ctx->num_slots; i++) {
      if (ctx->slot_state[i] == SLOT_NEED_QUEUE) {
        CounterSlot &cs = ctx->slots[i];

        // Step 1: Reset counters
        fi_cntr_set(cs.trigger_cntr, 0);
        fi_cntr_set(cs.completion_cntr, 0);
        fi_cntr_set(cs.atomic_completion_cntr, 0);

        // Step 2: Setup and queue RDMA write work
        size_t offset = i * ctx->transfer_size;

        ctx->iovs[i].iov_base = (char *)ctx->d_local_buf + offset;
        ctx->iovs[i].iov_len = ctx->transfer_size;

        ctx->rma_iovs[i].addr = ctx->remote_addr_for_rma + offset;
        ctx->rma_iovs[i].len = ctx->transfer_size;
        ctx->rma_iovs[i].key = ctx->peer_remote_key;

        memset(&ctx->msg_rmas[i], 0, sizeof(ctx->msg_rmas[i]));
        ctx->msg_rmas[i].msg_iov = &ctx->iovs[i];
        ctx->msg_rmas[i].desc = &ctx->desc_local;
        ctx->msg_rmas[i].iov_count = 1;
        ctx->msg_rmas[i].addr = ctx->peer_fi_addr;
        ctx->msg_rmas[i].rma_iov = &ctx->rma_iovs[i];
        ctx->msg_rmas[i].rma_iov_count = 1;

        memset(&ctx->op_rmas[i], 0, sizeof(ctx->op_rmas[i]));
        ctx->op_rmas[i].ep = ctx->ep;
        ctx->op_rmas[i].msg = ctx->msg_rmas[i];
        ctx->op_rmas[i].flags = FI_COMPLETION;

        ctx->works[i].triggering_cntr = cs.trigger_cntr;
        ctx->works[i].completion_cntr = cs.completion_cntr;
        ctx->works[i].threshold = 1;
        ctx->works[i].op_type = FI_OP_WRITE;
        ctx->works[i].op.rma = &ctx->op_rmas[i];

        ret = fi_control(&ctx->domain->fid, FI_QUEUE_WORK, &ctx->works[i]);
        if (ret) {
          fprintf(stderr, "Rank %d: Proxy fi_control(work %d) failed: %s\n",
                  myrank, i, fi_strerror(-ret));
          continue;
        }

        // Step 3: Setup and queue atomic operation
        ctx->atomic_iovs[i].addr = &((uint64_t *)ctx->d_atomic_operands)[i];
        ctx->atomic_iovs[i].count = 1;

        uint64_t atomic_result_addr = (uint64_t)&((uint64_t *)ctx->d_atomic_results)[i];
        if (ctx->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
          ctx->atomic_rma_iovs[i].addr = atomic_result_addr;
        } else {
          ctx->atomic_rma_iovs[i].addr = i * sizeof(uint64_t);
        }
        ctx->atomic_rma_iovs[i].count = 1;
        ctx->atomic_rma_iovs[i].key = fi_mr_key(ctx->mr_atomic_results);

        ctx->atomic_msgs[i].msg_iov = &ctx->atomic_iovs[i];
        ctx->atomic_msgs[i].desc = &ctx->desc_atomic_operands;
        ctx->atomic_msgs[i].iov_count = 1;
        ctx->atomic_msgs[i].addr = ctx->local_fi_addr;
        ctx->atomic_msgs[i].rma_iov = &ctx->atomic_rma_iovs[i];
        ctx->atomic_msgs[i].rma_iov_count = 1;
        ctx->atomic_msgs[i].datatype = FI_UINT64;
        ctx->atomic_msgs[i].op = FI_SUM;

        ctx->atomics[i].ep = ctx->ep;
        ctx->atomics[i].msg = ctx->atomic_msgs[i];
        ctx->atomics[i].flags = FI_COMPLETION;

        ctx->atomic_works[i].op_type = FI_OP_ATOMIC;
        ctx->atomic_works[i].op.atomic = &ctx->atomics[i];
        ctx->atomic_works[i].triggering_cntr = cs.completion_cntr;
        ctx->atomic_works[i].completion_cntr = cs.atomic_completion_cntr;
        ctx->atomic_works[i].threshold = 1;

        ret = fi_control(&ctx->domain->fid, FI_QUEUE_WORK, &ctx->atomic_works[i]);
        if (ret) {
          fprintf(stderr, "Rank %d: Proxy fi_control(atomic %d) failed: %s\n",
                  myrank, i, fi_strerror(-ret));
          continue;
        }

        // Step 4: Mark slot as armed
        ctx->slot_state[i] = SLOT_ARMED;
        __sync_synchronize();  // Memory barrier

        ctx->total_rearms.fetch_add(1);
        did_work = true;
      }
    }

    // Drain CQ to prevent overflow
    struct fi_cq_entry cq_entries[32];
    while (true) {
      ret = fi_cq_read(ctx->cq, cq_entries, 32);
      if (ret > 0) {
        ctx->cq_events_drained.fetch_add(ret);
      } else if (ret == -FI_EAGAIN) {
        break;
      } else if (ret < 0) {
        fprintf(stderr, "Rank %d: Proxy fi_cq_read error: %s\n",
                myrank, fi_strerror(-ret));
        break;
      }
    }

    // Small sleep if no work done to reduce CPU spinning
    if (!did_work) {
      usleep(1);
    }
  }

  printf("Rank %d: Proxy thread exiting (total rearms: %lu, CQ events: %lu)\n",
         myrank, ctx->total_rearms.load(), ctx->cq_events_drained.load());
  fflush(stdout);

  return NULL;
}

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
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return c - 'a' + 10;
  if ('A' <= c && c <= 'F') return c - 'A' + 10;
  return -1;
}

static int hex_to_bytes(const char *in, uint8_t *out, size_t outlen) {
  size_t n = strlen(in);
  if (n % 2 != 0 || outlen < n / 2) return -1;
  for (size_t i = 0; i < n; i += 2) {
    int hi = hexval(in[i]);
    int lo = hexval(in[i + 1]);
    if (hi < 0 || lo < 0) return -1;
    out[i / 2] = (uint8_t)((hi << 4) | lo);
  }
  return (int)(n / 2);
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
  mr_attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE |
                   FI_REMOTE_READ | FI_REMOTE_WRITE;

  if (is_device_mem) {
    mr_attr.iface = FI_HMEM_ROCR;
    mr_attr.device.reserved = 0;
  } else {
    mr_attr.iface = FI_HMEM_SYSTEM;
  }

  struct fid_mr *mr = NULL;
  int ret = fi_mr_regattr(domain, &mr_attr, 0, &mr);
  if (ret) return ret;

  if (info->domain_attr->mr_mode & FI_MR_ENDPOINT) {
    ret = fi_mr_bind(mr, &ep->fid, 0);
    if (ret) { fi_close(&mr->fid); return ret; }
    ret = fi_mr_enable(mr);
    if (ret) { fi_close(&mr->fid); return ret; }
  }

  *mr_out = mr;
  return 0;
}

static int exchange_addrs(int rank, int size, const char *my_hex,
                          size_t hex_len, char *all_hex) {
  char key[PMI2_MAX_KEYLEN];
  snprintf(key, sizeof(key), "addr-%d", rank);
  int rc = PMI2_KVS_Put(key, my_hex);
  if (rc != PMI2_SUCCESS) return rc;

  rc = PMI2_KVS_Fence();
  if (rc != PMI2_SUCCESS) return rc;

  for (int i = 0; i < size; i++) {
    snprintf(key, sizeof(key), "addr-%d", i);
    char val[PMI2_MAX_VALLEN];
    int vallen;
    rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key, val, sizeof(val), &vallen);
    if (rc != PMI2_SUCCESS) return rc;

    char *slot = all_hex + i * (hex_len + 1);
    strncpy(slot, val, hex_len);
    slot[hex_len] = '\0';
  }
  return PMI2_SUCCESS;
}

int main(int argc, char **argv) {
  int num_iterations = 10000;
  int num_slots = NUM_SLOTS;

  if (argc > 1) {
    num_iterations = atoi(argv[1]);
    if (num_iterations < 1) num_iterations = 1;
  }
  if (argc > 2) {
    num_slots = atoi(argv[2]);
    if (num_slots < 1) num_slots = 1;
    if (num_slots > NUM_SLOTS) num_slots = NUM_SLOTS;
  }

  // Environment setup
  unsetenv("ROCR_VISIBLE_DEVICES");
  setenv("FI_CXI_DEVICE_NAME", "cxi3", 1);
  int device_id = 7;
  CHECK_HIP(hipSetDevice(device_id), "hipSetDevice");

  // PMI2 Initialization
  int spawned, size, appnum;
  PMI2_Init(&spawned, &size, &myrank, &appnum);

  if (size != 2) {
    if (myrank == 0) {
      fprintf(stderr, "This program requires exactly 2 processes\n");
    }
    PMI2_Finalize();
    return 1;
  }

  printf("Rank %d: CPU Proxy mode - %d iterations using %d slots\n",
         myrank, num_iterations, num_slots);

  // HIP Initialization
  hipDeviceProp_t prop;
  CHECK_HIP(hipGetDeviceProperties(&prop, device_id), "hipGetDeviceProperties");
  double gpu_clock_mhz = prop.clockRate / 1000.0;
  printf("Rank %d: Using GPU %d: %s (%.0f MHz)\n",
         myrank, device_id, prop.name, gpu_clock_mhz);

  // Libfabric Initialization
  struct fi_info *hints = fi_allocinfo();
  hints->caps = FI_RMA | FI_MSG | FI_HMEM;
  hints->mode = FI_CONTEXT2;
  hints->ep_attr->type = FI_EP_RDM;
  hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                FI_MR_PROV_KEY | FI_MR_LOCAL | FI_MR_ENDPOINT;
  hints->domain_attr->threading = FI_THREAD_SAFE;
  hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
  hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

  struct fi_info *info = NULL;
  int ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                       NULL, NULL, 0, hints, &info);
  fi_freeinfo(hints);
  if (ret) {
    fprintf(stderr, "Rank %d: fi_getinfo failed: %s\n", myrank, fi_strerror(-ret));
    PMI2_Finalize();
    return 1;
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
    return 1;
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
  cq_attr.size = 1024;  // Larger CQ for many operations
  cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  CHECK(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");

  struct fid_ep *ep = NULL;
  CHECK(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");
  CHECK(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");
  CHECK(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
  CHECK(fi_enable(ep), "fi_enable");

  // Allocate slot state array (GPU-accessible pinned memory)
  volatile int *slot_state = NULL;
  CHECK_HIP(hipHostMalloc((void **)&slot_state, sizeof(int) * num_slots,
                          hipHostMallocMapped), "hipHostMalloc(slot_state)");
  volatile int *dev_slot_state = NULL;
  CHECK_HIP(hipHostGetDevicePointer((void **)&dev_slot_state,
                                    (void *)slot_state, 0),
            "hipHostGetDevicePointer(slot_state)");

  // Initialize all slots to NEED_QUEUE (will be armed by proxy)
  for (int i = 0; i < num_slots; i++) {
    slot_state[i] = SLOT_NEED_QUEUE;
  }

  // Allocate counter slots
  std::vector<CounterSlot> counter_slots(num_slots);
  printf("Rank %d: Allocating %d counter slots...\n", myrank, num_slots);

  for (int i = 0; i < num_slots; i++) {
    CounterSlot &cs = counter_slots[i];
    cs.initialized = false;

    // Create trigger counter
    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;
    CHECK(fi_cntr_open(domain, &cntr_attr, &cs.trigger_cntr, NULL),
          "fi_cntr_open(trigger)");

    // Create completion counter
    struct fi_cntr_attr comp_cntr_attr = {};
    comp_cntr_attr.events = FI_CNTR_EVENTS_COMP;
    comp_cntr_attr.wait_obj = FI_WAIT_UNSPEC;
    CHECK(fi_cntr_open(domain, &comp_cntr_attr, &cs.completion_cntr, NULL),
          "fi_cntr_open(completion)");

    // Create atomic completion counter
    CHECK(fi_cntr_open(domain, &cntr_attr, &cs.atomic_completion_cntr, NULL),
          "fi_cntr_open(atomic_completion)");

    // Get counter ops for MMIO access
    CHECK(fi_open_ops(&cs.trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                      (void **)&cs.trigger_ops, NULL),
          "fi_open_ops(trigger)");

    CHECK(fi_open_ops(&cs.completion_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                      (void **)&cs.completion_ops, NULL),
          "fi_open_ops(completion)");

    // Get MMIO addresses
    CHECK(cs.trigger_ops->get_mmio_addr(&cs.trigger_cntr->fid,
                                        &cs.trigger_mmio_addr,
                                        &cs.trigger_mmio_len),
          "get_mmio_addr(trigger)");

    CHECK(cs.completion_ops->get_mmio_addr(&cs.completion_cntr->fid,
                                           &cs.completion_mmio_addr,
                                           &cs.completion_mmio_len),
          "get_mmio_addr(completion)");

    // Map MMIO to GPU
    CHECK_HIP(hipHostRegister(cs.trigger_mmio_addr, cs.trigger_mmio_len,
                              hipHostRegisterMapped), "hipHostRegister(trigger)");
    CHECK_HIP(hipHostRegister(cs.completion_mmio_addr, cs.completion_mmio_len,
                              hipHostRegisterMapped), "hipHostRegister(completion)");

    CHECK_HIP(hipHostGetDevicePointer((void **)&cs.dev_trigger_addr,
                                      cs.trigger_mmio_addr, 0),
              "hipHostGetDevicePointer(trigger)");
    CHECK_HIP(hipHostGetDevicePointer((void **)&cs.dev_completion_addr,
                                      cs.completion_mmio_addr, 0),
              "hipHostGetDevicePointer(completion)");

    cs.initialized = true;
  }
  printf("Rank %d: Created %d counter slots\n", myrank, num_slots);

  // Allocate trigger address array for GPU (array of pointers)
  volatile uint64_t **h_trigger_addrs = NULL;
  CHECK_HIP(hipHostMalloc((void **)&h_trigger_addrs,
                          sizeof(uint64_t *) * num_slots,
                          hipHostMallocMapped), "hipHostMalloc(trigger_addrs)");
  volatile uint64_t **dev_trigger_addrs = NULL;
  CHECK_HIP(hipHostGetDevicePointer((void **)&dev_trigger_addrs,
                                    (void *)h_trigger_addrs, 0),
            "hipHostGetDevicePointer(trigger_addrs)");

  for (int i = 0; i < num_slots; i++) {
    h_trigger_addrs[i] = counter_slots[i].dev_trigger_addr;
  }

  // Exchange Address Information
  size_t addrlen = 0;
  fi_getname(&ep->fid, NULL, &addrlen);
  void *local_addr = malloc(addrlen);
  CHECK(fi_getname(&ep->fid, local_addr, &addrlen), "fi_getname");

  char *my_hex = (char *)malloc(2 * addrlen + 1);
  bytes_to_hex((uint8_t *)local_addr, addrlen, my_hex);

  char *all_hex = (char *)malloc(size * (2 * addrlen + 1));
  CHECK(exchange_addrs(myrank, size, my_hex, 2 * addrlen, all_hex), "exchange_addrs");

  uint8_t *all_bin = (uint8_t *)malloc(size * addrlen);
  for (int i = 0; i < size; i++) {
    const char *hex = all_hex + i * (2 * addrlen + 1);
    CHECK(hex_to_bytes(hex, all_bin + i * addrlen, addrlen) > 0 ? 0 : -1, "hex_to_bytes");
  }

  int peer = (myrank == 0) ? 1 : 0;
  void *peer_addr = all_bin + peer * addrlen;

  fi_addr_t peer_fi_addr = FI_ADDR_NOTAVAIL;
  int inserted = fi_av_insert(av, peer_addr, 1, &peer_fi_addr, 0, NULL);
  if (inserted != 1 || peer_fi_addr == FI_ADDR_NOTAVAIL) {
    fprintf(stderr, "Rank %d: fi_av_insert(peer) failed\n", myrank);
    PMI2_Finalize();
    return 1;
  }

  fi_addr_t local_fi_addr = FI_ADDR_NOTAVAIL;
  inserted = fi_av_insert(av, local_addr, 1, &local_fi_addr, 0, NULL);
  if (inserted != 1 || local_fi_addr == FI_ADDR_NOTAVAIL) {
    fprintf(stderr, "Rank %d: fi_av_insert(local) failed\n", myrank);
    PMI2_Finalize();
    return 1;
  }

  // Allocate GPU buffers
  size_t transfer_size = 256 * 1024;  // 256KB per transfer
  size_t total_size = transfer_size * num_slots;

  void *d_local_buf = NULL;
  CHECK_HIP(hipMalloc(&d_local_buf, total_size), "hipMalloc(local)");

  void *d_remote_buf = NULL;
  CHECK_HIP(hipMalloc(&d_remote_buf, total_size), "hipMalloc(remote)");

  // Atomic results/operands (GPU memory)
  uint64_t *d_atomic_results = NULL;
  CHECK_HIP(hipMalloc(&d_atomic_results, sizeof(uint64_t) * num_slots),
            "hipMalloc(atomic_results)");
  CHECK_HIP(hipMemset(d_atomic_results, 0, sizeof(uint64_t) * num_slots),
            "hipMemset(atomic_results)");

  uint64_t *d_atomic_operands = NULL;
  CHECK_HIP(hipMalloc(&d_atomic_operands, sizeof(uint64_t) * num_slots),
            "hipMalloc(atomic_operands)");
  std::vector<uint64_t> h_operands(num_slots, 1);
  CHECK_HIP(hipMemcpy(d_atomic_operands, h_operands.data(),
                      sizeof(uint64_t) * num_slots, hipMemcpyHostToDevice),
            "hipMemcpy(operands)");

  // Timing/stats buffers
  uint64_t *d_total_cycles = NULL;
  uint64_t *d_spin_cycles = NULL;
  CHECK_HIP(hipMalloc(&d_total_cycles, sizeof(uint64_t)), "hipMalloc(total_cycles)");
  CHECK_HIP(hipMalloc(&d_spin_cycles, sizeof(uint64_t)), "hipMalloc(spin_cycles)");

  // Host verification buffer
  uint8_t *h_verify_buf = (uint8_t *)malloc(total_size);

  // Register memory regions
  struct fid_mr *mr_local = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_local_buf, total_size,
                               &mr_local, true), "register_memory_region(local)");

  struct fid_mr *mr_remote = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_remote_buf, total_size,
                               &mr_remote, true), "register_memory_region(remote)");

  struct fid_mr *mr_atomic_results = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_atomic_results,
                               sizeof(uint64_t) * num_slots,
                               &mr_atomic_results, true), "register_memory_region(atomic_results)");

  struct fid_mr *mr_atomic_operands = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_atomic_operands,
                               sizeof(uint64_t) * num_slots,
                               &mr_atomic_operands, true), "register_memory_region(atomic_operands)");

  uint64_t my_remote_key = fi_mr_key(mr_remote);
  uint64_t my_remote_addr = (uint64_t)d_remote_buf;
  void *desc_local = fi_mr_desc(mr_local);
  void *desc_atomic_operands = fi_mr_desc(mr_atomic_operands);
  void *desc_atomic_results = fi_mr_desc(mr_atomic_results);

  // Exchange RMA info
  struct { uint64_t addr; uint64_t key; } my_rma_info, peer_rma_info;
  my_rma_info.addr = my_remote_addr;
  my_rma_info.key = my_remote_key;

  char rma_hex[128];
  bytes_to_hex((uint8_t *)&my_rma_info, sizeof(my_rma_info), rma_hex);

  char key_str[PMI2_MAX_KEYLEN];
  snprintf(key_str, sizeof(key_str), "rma-%d", myrank);
  CHECK(PMI2_KVS_Put(key_str, rma_hex), "PMI2_KVS_Put(rma)");
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(rma)");

  snprintf(key_str, sizeof(key_str), "rma-%d", peer);
  char peer_rma_hex[PMI2_MAX_VALLEN];
  int vallen;
  CHECK(PMI2_KVS_Get(NULL, PMI2_ID_NULL, key_str, peer_rma_hex,
                     sizeof(peer_rma_hex), &vallen), "PMI2_KVS_Get(rma)");
  CHECK(hex_to_bytes(peer_rma_hex, (uint8_t *)&peer_rma_info,
                     sizeof(peer_rma_info)) > 0 ? 0 : -1, "hex_to_bytes(rma)");

  uint64_t peer_remote_addr = peer_rma_info.addr;
  uint64_t peer_remote_key = peer_rma_info.key;

  uint64_t remote_addr_for_rma;
  if (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
    remote_addr_for_rma = peer_remote_addr;
  } else {
    remote_addr_for_rma = 0;
  }

  // Sync before test
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(pre-test)");
  usleep(100000);

  // Allocate DWQ structures for proxy
  std::vector<struct fi_deferred_work> works(num_slots);
  std::vector<struct fi_op_rma> op_rmas(num_slots);
  std::vector<struct fi_msg_rma> msg_rmas(num_slots);
  std::vector<struct iovec> iovs(num_slots);
  std::vector<struct fi_rma_iov> rma_iovs(num_slots);

  std::vector<struct fi_deferred_work> atomic_works(num_slots);
  std::vector<struct fi_op_atomic> atomics(num_slots);
  std::vector<struct fi_msg_atomic> atomic_msgs(num_slots);
  std::vector<struct fi_ioc> atomic_iovs(num_slots);
  std::vector<struct fi_rma_ioc> atomic_rma_iovs(num_slots);

  printf("\n");
  if (myrank == 0) {
    printf("========================================\n");
    printf("CPU Proxy Test: %d iterations, %d slots\n", num_iterations, num_slots);
    printf("Transfer size: %zu bytes per operation\n", transfer_size);
    printf("========================================\n\n");
  }

  // Initialize buffers
  if (myrank == 0) {
    // Fill source buffer with pattern (each slot region gets unique byte)
    for (int i = 0; i < num_slots; i++) {
      uint8_t pattern = (uint8_t)(i & 0xFF);
      CHECK_HIP(hipMemset((char *)d_local_buf + i * transfer_size,
                          pattern, transfer_size), "hipMemset(pattern)");
    }
  }
  if (myrank == 1) {
    CHECK_HIP(hipMemset(d_remote_buf, 0xFF, total_size), "hipMemset(remote)");
  }
  CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence");

  double total_start = get_time_us();

  if (myrank == 0) {
    // Setup proxy context
    ProxyContext ctx;
    ctx.slot_state = slot_state;
    ctx.dev_slot_state = dev_slot_state;
    ctx.slots = counter_slots.data();
    ctx.num_slots = num_slots;
    ctx.domain = domain;
    ctx.ep = ep;
    ctx.cq = cq;
    ctx.info = cxi_info;
    ctx.peer_fi_addr = peer_fi_addr;
    ctx.local_fi_addr = local_fi_addr;
    ctx.d_local_buf = d_local_buf;
    ctx.d_atomic_results = d_atomic_results;
    ctx.d_atomic_operands = d_atomic_operands;
    ctx.desc_local = desc_local;
    ctx.desc_atomic_operands = desc_atomic_operands;
    ctx.desc_atomic_results = desc_atomic_results;
    ctx.mr_atomic_results = mr_atomic_results;
    ctx.remote_addr_for_rma = remote_addr_for_rma;
    ctx.peer_remote_key = peer_remote_key;
    ctx.transfer_size = transfer_size;
    ctx.works = works.data();
    ctx.op_rmas = op_rmas.data();
    ctx.msg_rmas = msg_rmas.data();
    ctx.iovs = iovs.data();
    ctx.rma_iovs = rma_iovs.data();
    ctx.atomic_works = atomic_works.data();
    ctx.atomics = atomics.data();
    ctx.atomic_msgs = atomic_msgs.data();
    ctx.atomic_iovs = atomic_iovs.data();
    ctx.atomic_rma_iovs = atomic_rma_iovs.data();
    ctx.proxy_running.store(false);
    ctx.stop_requested.store(false);
    ctx.total_rearms.store(0);
    ctx.cq_events_drained.store(0);

    // Start proxy thread
    printf("Rank %d: Starting proxy thread...\n", myrank);
    ret = pthread_create(&ctx.proxy_thread, NULL, proxy_thread_func, &ctx);
    if (ret != 0) {
      fprintf(stderr, "Rank %d: pthread_create failed: %d\n", myrank, ret);
      PMI2_Finalize();
      return 1;
    }
    ctx.proxy_running.store(true);

    // Wait for initial slots to be armed
    printf("Rank %d: Waiting for initial slots to be armed...\n", myrank);
    bool all_armed = false;
    while (!all_armed) {
      all_armed = true;
      for (int i = 0; i < num_slots; i++) {
        if (slot_state[i] != SLOT_ARMED) {
          all_armed = false;
          break;
        }
      }
      usleep(100);
    }
    printf("Rank %d: All %d slots armed, launching persistent GPU kernel...\n",
           myrank, num_slots);

    // Launch persistent GPU kernel
    double kernel_start = get_time_us();

    hipLaunchKernelGGL(gpu_persistent_kernel, dim3(1), dim3(1), 0, 0,
                       dev_slot_state,
                       dev_trigger_addrs,
                       d_atomic_results,
                       num_slots,
                       num_iterations,
                       d_total_cycles,
                       d_spin_cycles);

    CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize(kernel)");

    double kernel_end = get_time_us();

    // Stop proxy thread
    ctx.stop_requested.store(true);
    pthread_join(ctx.proxy_thread, NULL);

    // Read timing stats from GPU
    uint64_t h_total_cycles, h_spin_cycles;
    CHECK_HIP(hipMemcpy(&h_total_cycles, d_total_cycles, sizeof(uint64_t),
                        hipMemcpyDeviceToHost), "hipMemcpy(total_cycles)");
    CHECK_HIP(hipMemcpy(&h_spin_cycles, d_spin_cycles, sizeof(uint64_t),
                        hipMemcpyDeviceToHost), "hipMemcpy(spin_cycles)");

    double total_end = get_time_us();
    double wall_time_us = total_end - total_start;
    double kernel_time_us = kernel_end - kernel_start;
    double gpu_total_us = h_total_cycles / gpu_clock_mhz;
    double gpu_spin_us = h_spin_cycles / gpu_clock_mhz;

    printf("\n========================================\n");
    printf("Results:\n");
    printf("  Iterations: %d\n", num_iterations);
    printf("  Slots (window size): %d\n", num_slots);
    printf("  Total wall time: %.2f ms\n", wall_time_us / 1000.0);
    printf("  Kernel time: %.2f ms\n", kernel_time_us / 1000.0);
    printf("  GPU total cycles: %.2f ms\n", gpu_total_us / 1000.0);
    printf("  GPU spin waiting: %.2f ms (%.1f%%)\n",
           gpu_spin_us / 1000.0, 100.0 * gpu_spin_us / gpu_total_us);
    printf("  Proxy thread rearms: %lu\n", ctx.total_rearms.load());
    printf("  CQ events drained: %lu\n", ctx.cq_events_drained.load());
    printf("  Per-iteration latency: %.2f us\n", kernel_time_us / num_iterations);
    printf("  Operations per second: %.0f\n",
           num_iterations / (kernel_time_us / 1e6));
    printf("  Total data transferred: %.2f MB\n",
           (double)(transfer_size * num_iterations) / (1024 * 1024));
    printf("  Effective bandwidth: %.2f GB/s\n",
           (double)(transfer_size * num_iterations) / kernel_time_us / 1000.0);
    printf("========================================\n");
  }

  // Sync and verify
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence");

  // Rank 1: Verify received data
  if (myrank == 1) {
    usleep(100000);  // Wait for writes to complete
    CHECK_HIP(hipMemcpy(h_verify_buf, d_remote_buf, total_size,
                        hipMemcpyDeviceToHost), "hipMemcpy(verify)");

    int total_errors = 0;
    int regions_verified = 0;

    for (int region = 0; region < num_slots; region++) {
      // Each region should have pattern = region & 0xFF
      uint8_t expected = (uint8_t)(region & 0xFF);
      size_t region_offset = region * transfer_size;
      int region_errors = 0;

      for (size_t j = 0; j < transfer_size; j++) {
        if (h_verify_buf[region_offset + j] != expected) {
          region_errors++;
        }
      }

      if (region_errors > 0) {
        if (total_errors < 10) {
          fprintf(stderr,
                  "Rank %d: Region %d FAILED - expected 0x%02X, "
                  "got 0x%02X at first byte (%d/%zu bytes wrong)\n",
                  myrank, region, expected,
                  h_verify_buf[region_offset], region_errors, transfer_size);
        }
        total_errors += region_errors;
      } else {
        regions_verified++;
      }
    }

    if (total_errors > 0) {
      fprintf(stderr, "Rank %d: VERIFICATION FAILED - %d/%d regions correct\n",
              myrank, regions_verified, num_slots);
    } else {
      printf("Rank %d: VERIFICATION PASSED - all %d regions correct\n",
             myrank, num_slots);
      printf("Rank %d: Verified %zu bytes from %d RDMA operations\n",
             myrank, total_size, num_iterations);
    }
  }

  // Cleanup
  for (int i = 0; i < num_slots; i++) {
    if (counter_slots[i].initialized) {
      hipHostUnregister(counter_slots[i].trigger_mmio_addr);
      hipHostUnregister(counter_slots[i].completion_mmio_addr);
      fi_close(&counter_slots[i].trigger_cntr->fid);
      fi_close(&counter_slots[i].completion_cntr->fid);
      fi_close(&counter_slots[i].atomic_completion_cntr->fid);
    }
  }

  hipHostFree((void *)slot_state);
  hipHostFree((void *)h_trigger_addrs);
  hipFree(d_local_buf);
  hipFree(d_remote_buf);
  hipFree(d_atomic_results);
  hipFree(d_atomic_operands);
  hipFree(d_total_cycles);
  hipFree(d_spin_cycles);
  free(h_verify_buf);

  fi_close(&mr_local->fid);
  fi_close(&mr_remote->fid);
  fi_close(&mr_atomic_results->fid);
  fi_close(&mr_atomic_operands->fid);
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
