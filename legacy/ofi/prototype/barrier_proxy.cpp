/*
 * barrier_proxy.cpp - CPU Proxy + GPU-Driven Barrier Demo
 *
 * Implements unlimited barrier iterations using:
 * - W slots (windowed) with reusable DWQ counter pairs
 * - GPU persistent kernel driving the barrier (writes trigger doorbell)
 * - CPU proxy thread continuously re-queuing DWQ work for consumed slots
 * - All-to-all arrival barrier using atomic add operations
 *
 * Algorithm:
 * - Each rank has d_arrive counter (GPU memory), init 0
 * - Each barrier epoch: each rank atomically adds +1 to all other ranks' d_arrive
 * - GPU waits for d_arrive >= (epoch+1) * (size-1)
 * - slot_state[W] coordinates GPU/CPU: 0=NEED_QUEUE, 1=ARMED
 */

#include <atomic>
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
#define DEFAULT_ITERATIONS 10000
#define MAX_WINDOW_SIZE 64

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

// Counter pair structure for each slot
struct SlotCounters {
  struct fid_cntr *trigger_cntr;
  struct fi_cxi_cntr_ops *trigger_ops;
  void *trigger_mmio_addr;
  size_t trigger_mmio_len;
  volatile uint64_t *dev_trigger_addr;

  struct fid_cntr *completion_cntr;

  bool initialized;
};

// RMA info for remote arrive counters
struct RemoteArriveInfo {
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

  // Slot management
  int window_size;
  SlotCounters *slots;
  volatile int *slot_state;  // Host-pinned, GPU-visible

  // Atomic operation parameters
  int size;  // Number of ranks
  int rank;
  fi_addr_t *peer_fi_addrs;
  RemoteArriveInfo *peer_arrive_info;

  // Local atomic operand (always 1)
  uint64_t *d_atomic_operand;
  void *desc_atomic_operand;
  struct fid_mr *mr_atomic_operand;

  // DWQ structures (one set per slot per peer)
  std::vector<struct fi_deferred_work> *works;
  std::vector<struct fi_op_atomic> *atomics;
  std::vector<struct fi_msg_atomic> *msgs;
  std::vector<struct fi_ioc> *iovs;
  std::vector<struct fi_rma_ioc> *rma_iovs;

  // Control
  std::atomic<bool> stop_flag;
  std::atomic<uint64_t> total_rearms;

  // Iteration tracking for each slot
  int *slot_epoch;
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

// Queue DWQ work for a slot (atomics to all peers)
static int queue_slot_work(ProxyContext *ctx, int slot) {
  SlotCounters &sc = ctx->slots[slot];
  int num_peers = ctx->size - 1;
  int idx_base = slot * num_peers;

  // Reset counters
  fi_cntr_set(sc.trigger_cntr, 0);
  fi_cntr_set(sc.completion_cntr, 0);

  // Queue atomic add to each peer's d_arrive
  int peer_idx = 0;
  for (int r = 0; r < ctx->size; r++) {
    if (r == ctx->rank)
      continue;

    int idx = idx_base + peer_idx;

    // Setup atomic IOV (source operand)
    ctx->iovs->at(idx).addr = ctx->d_atomic_operand;
    ctx->iovs->at(idx).count = 1;

    // Setup remote RMA IOV (target: peer's d_arrive)
    ctx->rma_iovs->at(idx).addr = ctx->peer_arrive_info[r].addr;
    ctx->rma_iovs->at(idx).count = 1;
    ctx->rma_iovs->at(idx).key = ctx->peer_arrive_info[r].key;

    // Setup atomic message
    ctx->msgs->at(idx).msg_iov = &ctx->iovs->at(idx);
    ctx->msgs->at(idx).desc = &ctx->desc_atomic_operand;
    ctx->msgs->at(idx).iov_count = 1;
    ctx->msgs->at(idx).addr = ctx->peer_fi_addrs[r];
    ctx->msgs->at(idx).rma_iov = &ctx->rma_iovs->at(idx);
    ctx->msgs->at(idx).rma_iov_count = 1;
    ctx->msgs->at(idx).datatype = FI_UINT64;
    ctx->msgs->at(idx).op = FI_SUM;

    // Setup atomic op
    ctx->atomics->at(idx).ep = ctx->ep;
    ctx->atomics->at(idx).msg = ctx->msgs->at(idx);
    ctx->atomics->at(idx).flags = 0;  // No completion event needed

    // Setup deferred work
    ctx->works->at(idx).triggering_cntr = sc.trigger_cntr;
    ctx->works->at(idx).completion_cntr = sc.completion_cntr;
    ctx->works->at(idx).threshold = 1;
    ctx->works->at(idx).op_type = FI_OP_ATOMIC;
    ctx->works->at(idx).op.atomic = &ctx->atomics->at(idx);

    int ret = fi_control(&ctx->domain->fid, FI_QUEUE_WORK, &ctx->works->at(idx));
    if (ret) {
      fprintf(stderr, "Rank %d: fi_control(atomic slot %d peer %d) failed: %s\n",
              ctx->rank, slot, r, fi_strerror(-ret));
      return ret;
    }

    peer_idx++;
  }

  return 0;
}

// CPU proxy thread function
static void *proxy_thread_func(void *arg) {
  ProxyContext *ctx = (ProxyContext *)arg;

  while (!ctx->stop_flag.load(std::memory_order_relaxed)) {
    bool did_work = false;

    // Scan all slots for those needing re-queue
    for (int s = 0; s < ctx->window_size; s++) {
      // Use volatile read to see GPU's writes
      int state = ((volatile int *)ctx->slot_state)[s];

      if (state == SLOT_NEED_QUEUE) {
        // Re-queue work for this slot
        int ret = queue_slot_work(ctx, s);
        if (ret == 0) {
          // Mark as armed (use memory barrier for GPU visibility)
          __atomic_store_n(&ctx->slot_state[s], SLOT_ARMED, __ATOMIC_RELEASE);
          ctx->total_rearms.fetch_add(1, std::memory_order_relaxed);
          did_work = true;
        }
      }
    }

    // Drain CQ periodically to avoid overflow
    struct fi_cq_entry cq_entries[32];
    int ret;
    while ((ret = fi_cq_read(ctx->cq, cq_entries, 32)) > 0) {
      // Just drain, we don't need to process these
    }

    if (!did_work) {
      // Short sleep to avoid spinning too hard
      usleep(10);
    }
  }

  return NULL;
}

// GPU persistent kernel for barrier
__global__ void gpu_barrier_kernel(
    volatile int *slot_state,           // GPU-visible slot states
    volatile uint64_t **dev_trigger_addrs,  // Trigger doorbell addresses per slot
    volatile uint64_t *d_arrive,        // Local arrive counter
    int num_iters,
    int window_size,
    int num_peers,                      // size - 1
    uint64_t *out_final_arrive,         // Output: final arrive value
    uint64_t *out_cycles                // Output: total cycles
) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  uint64_t start = clock64();

  for (int epoch = 0; epoch < num_iters; epoch++) {
    int slot = epoch % window_size;

    // 1. Wait for slot to be armed
    while (__atomic_load_n(&slot_state[slot], __ATOMIC_ACQUIRE) != SLOT_ARMED) {
      // Spin wait
    }

    // 2. Trigger the queued atomics by writing to doorbell
    __threadfence_system();
    *dev_trigger_addrs[slot] = 1;
    __threadfence_system();

    // 3. Wait for arrivals from all peers
    // Expected: (epoch + 1) * num_peers
    uint64_t expected = (uint64_t)(epoch + 1) * num_peers;
    while (__atomic_load_n((unsigned long long *)d_arrive, __ATOMIC_ACQUIRE) < expected) {
      // Spin wait
    }

    // 4. Mark slot as needing re-queue for future use
    __atomic_store_n(&slot_state[slot], SLOT_NEED_QUEUE, __ATOMIC_RELEASE);
  }

  uint64_t end = clock64();

  *out_final_arrive = *d_arrive;
  *out_cycles = end - start;
}

int main(int argc, char **argv) {
  int window_size = DEFAULT_WINDOW_SIZE;
  int num_iters = DEFAULT_ITERATIONS;

  // Parse arguments
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

  printf("Rank %d/%d: Barrier proxy test - %d iterations, window=%d\n",
         myrank, size, num_iters, window_size);

  // HIP device setup
  hipDeviceProp_t prop;
  CHECK_HIP(hipGetDeviceProperties(&prop, device_id), "hipGetDeviceProperties");
  double gpu_clock_mhz = prop.clockRate / 1000.0;
  printf("Rank %d: Using GPU %d: %s (%.0f MHz)\n", myrank, device_id,
         prop.name, gpu_clock_mhz);

  // Libfabric Initialization with FI_ATOMIC
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
    fprintf(stderr, "Rank %d: fi_getinfo failed: %s\n", myrank, fi_strerror(-ret));
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
  cq_attr.size = 1024;
  cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  CHECK(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");

  struct fid_ep *ep = NULL;
  CHECK(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");
  CHECK(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");
  CHECK(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
  CHECK(fi_enable(ep), "fi_enable");

  // Allocate slot counters
  std::vector<SlotCounters> slots(window_size);
  printf("Rank %d: Allocating %d slots...\n", myrank, window_size);

  for (int i = 0; i < window_size; i++) {
    SlotCounters &sc = slots[i];
    sc.initialized = false;

    // Create trigger counter
    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;

    CHECK(fi_cntr_open(domain, &cntr_attr, &sc.trigger_cntr, NULL),
          "fi_cntr_open(trigger)");

    // Create completion counter
    CHECK(fi_cntr_open(domain, &cntr_attr, &sc.completion_cntr, NULL),
          "fi_cntr_open(completion)");

    // Get CXI ops for MMIO access
    CHECK(fi_open_ops(&sc.trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                      (void **)&sc.trigger_ops, NULL),
          "fi_open_ops(trigger)");

    CHECK(sc.trigger_ops->get_mmio_addr(&sc.trigger_cntr->fid,
                                        &sc.trigger_mmio_addr,
                                        &sc.trigger_mmio_len),
          "get_mmio_addr(trigger)");

    // Map MMIO to GPU
    CHECK_HIP(hipHostRegister(sc.trigger_mmio_addr, sc.trigger_mmio_len,
                              hipHostRegisterMapped),
              "hipHostRegister(trigger)");
    CHECK_HIP(hipHostGetDevicePointer((void **)&sc.dev_trigger_addr,
                                      sc.trigger_mmio_addr, 0),
              "hipHostGetDevicePointer(trigger)");

    sc.initialized = true;
  }

  printf("Rank %d: Created %d slots successfully\n", myrank, window_size);

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

  // Insert all addresses into AV
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
  // d_arrive: each rank's arrival counter (1 uint64_t)
  uint64_t *d_arrive = NULL;
  CHECK_HIP(hipMalloc(&d_arrive, sizeof(uint64_t)), "hipMalloc(d_arrive)");
  CHECK_HIP(hipMemset(d_arrive, 0, sizeof(uint64_t)), "hipMemset(d_arrive)");

  // Atomic operand (always 1)
  uint64_t *d_atomic_operand = NULL;
  CHECK_HIP(hipMalloc(&d_atomic_operand, sizeof(uint64_t)),
            "hipMalloc(d_atomic_operand)");
  uint64_t one = 1;
  CHECK_HIP(hipMemcpy(d_atomic_operand, &one, sizeof(uint64_t),
                      hipMemcpyHostToDevice),
            "hipMemcpy(atomic_operand)");

  // Output buffers
  uint64_t *d_final_arrive = NULL;
  uint64_t *d_cycles = NULL;
  CHECK_HIP(hipMalloc(&d_final_arrive, sizeof(uint64_t)), "hipMalloc(final)");
  CHECK_HIP(hipMalloc(&d_cycles, sizeof(uint64_t)), "hipMalloc(cycles)");

  // Register d_arrive as MR for remote atomics
  struct fid_mr *mr_arrive = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_arrive, sizeof(uint64_t),
                               &mr_arrive, true),
        "register_memory_region(arrive)");

  struct fid_mr *mr_atomic_operand = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_atomic_operand,
                               sizeof(uint64_t), &mr_atomic_operand, true),
        "register_memory_region(operand)");

  // Exchange d_arrive MR info
  std::vector<RemoteArriveInfo> peer_arrive_info(size);

  RemoteArriveInfo my_arrive_info;
  if (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
    my_arrive_info.addr = (uint64_t)d_arrive;
  } else {
    my_arrive_info.addr = 0;
  }
  my_arrive_info.key = fi_mr_key(mr_arrive);

  char arrive_hex[128];
  bytes_to_hex((uint8_t *)&my_arrive_info, sizeof(my_arrive_info), arrive_hex);

  char key_str[PMI2_MAX_KEYLEN];
  snprintf(key_str, sizeof(key_str), "arrive-%d", myrank);
  CHECK(PMI2_KVS_Put(key_str, arrive_hex), "PMI2_KVS_Put(arrive)");
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(arrive)");

  for (int i = 0; i < size; i++) {
    snprintf(key_str, sizeof(key_str), "arrive-%d", i);
    char peer_hex[PMI2_MAX_VALLEN];
    int vallen;
    CHECK(PMI2_KVS_Get(NULL, PMI2_ID_NULL, key_str, peer_hex, sizeof(peer_hex),
                       &vallen),
          "PMI2_KVS_Get(arrive)");
    CHECK(hex_to_bytes(peer_hex, (uint8_t *)&peer_arrive_info[i],
                       sizeof(RemoteArriveInfo)) > 0
              ? 0
              : -1,
          "hex_to_bytes(arrive)");
  }

  printf("Rank %d: Exchanged arrive MR info\n", myrank);

  // Allocate slot_state in host-pinned memory (GPU-visible)
  int *h_slot_state = NULL;
  CHECK_HIP(hipHostMalloc(&h_slot_state, sizeof(int) * window_size,
                          hipHostMallocMapped),
            "hipHostMalloc(slot_state)");

  // Initialize all slots as NEED_QUEUE (proxy will arm them)
  for (int i = 0; i < window_size; i++) {
    h_slot_state[i] = SLOT_NEED_QUEUE;
  }

  // Get device pointer for slot_state
  volatile int *d_slot_state = NULL;
  CHECK_HIP(hipHostGetDevicePointer((void **)&d_slot_state, h_slot_state, 0),
            "hipHostGetDevicePointer(slot_state)");

  // Allocate trigger address array for GPU
  volatile uint64_t **h_trigger_addrs = NULL;
  CHECK_HIP(hipHostMalloc(&h_trigger_addrs,
                          sizeof(volatile uint64_t *) * window_size,
                          hipHostMallocMapped),
            "hipHostMalloc(trigger_addrs)");

  for (int i = 0; i < window_size; i++) {
    h_trigger_addrs[i] = slots[i].dev_trigger_addr;
  }

  volatile uint64_t **d_trigger_addrs = NULL;
  CHECK_HIP(hipHostGetDevicePointer((void **)&d_trigger_addrs, h_trigger_addrs, 0),
            "hipHostGetDevicePointer(trigger_addrs)");

  // Allocate DWQ structures for proxy
  int num_peers = size - 1;
  int total_ops = window_size * num_peers;

  std::vector<struct fi_deferred_work> works(total_ops);
  std::vector<struct fi_op_atomic> atomics(total_ops);
  std::vector<struct fi_msg_atomic> msgs(total_ops);
  std::vector<struct fi_ioc> iovs(total_ops);
  std::vector<struct fi_rma_ioc> rma_iovs(total_ops);

  // Setup proxy context
  ProxyContext proxy_ctx;
  proxy_ctx.domain = domain;
  proxy_ctx.ep = ep;
  proxy_ctx.cq = cq;
  proxy_ctx.info = cxi_info;
  proxy_ctx.window_size = window_size;
  proxy_ctx.slots = slots.data();
  proxy_ctx.slot_state = h_slot_state;
  proxy_ctx.size = size;
  proxy_ctx.rank = myrank;
  proxy_ctx.peer_fi_addrs = peer_fi_addrs.data();
  proxy_ctx.peer_arrive_info = peer_arrive_info.data();
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
  proxy_ctx.slot_epoch = (int *)calloc(window_size, sizeof(int));

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

  // Wait for all slots to be armed initially
  printf("Rank %d: Waiting for slots to be armed...\n", myrank);
  bool all_armed = false;
  while (!all_armed) {
    all_armed = true;
    for (int i = 0; i < window_size; i++) {
      if (__atomic_load_n(&h_slot_state[i], __ATOMIC_ACQUIRE) != SLOT_ARMED) {
        all_armed = false;
        break;
      }
    }
    if (!all_armed) {
      usleep(100);
    }
  }
  printf("Rank %d: All %d slots armed, launching kernel\n", myrank, window_size);

  // Final sync before launching kernels
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(pre-kernel)");

  double start_time = get_time_us();

  // Launch persistent GPU kernel
  hipLaunchKernelGGL(gpu_barrier_kernel, dim3(1), dim3(1), 0, 0,
                     d_slot_state, d_trigger_addrs, d_arrive,
                     num_iters, window_size, num_peers,
                     d_final_arrive, d_cycles);

  CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");

  double end_time = get_time_us();

  // Stop proxy thread
  proxy_ctx.stop_flag.store(true, std::memory_order_release);
  pthread_join(proxy_thread, NULL);

  printf("Rank %d: Kernel completed\n", myrank);

  // Verify results
  uint64_t h_final_arrive, h_cycles;
  CHECK_HIP(hipMemcpy(&h_final_arrive, d_final_arrive, sizeof(uint64_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy(final_arrive)");
  CHECK_HIP(hipMemcpy(&h_cycles, d_cycles, sizeof(uint64_t),
                      hipMemcpyDeviceToHost),
            "hipMemcpy(cycles)");

  uint64_t expected_arrive = (uint64_t)num_iters * num_peers;

  // Sync and print results
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(post-test)");

  double wall_time_ms = (end_time - start_time) / 1000.0;
  double avg_barrier_us = (end_time - start_time) / num_iters;
  double gpu_time_us = (double)h_cycles / gpu_clock_mhz;

  printf("\n");
  printf("Rank %d Results:\n", myrank);
  printf("  Iterations: %d\n", num_iters);
  printf("  Window size: %d\n", window_size);
  printf("  Peers: %d\n", num_peers);
  printf("  Final d_arrive: %lu (expected: %lu)\n", h_final_arrive, expected_arrive);
  printf("  Total rearms by proxy: %lu\n",
         proxy_ctx.total_rearms.load(std::memory_order_relaxed));
  printf("  Wall time: %.2f ms\n", wall_time_ms);
  printf("  GPU time: %.2f ms (%.2f us per barrier)\n",
         gpu_time_us / 1000.0, gpu_time_us / num_iters);
  printf("  Avg barrier latency: %.2f us\n", avg_barrier_us);

  if (h_final_arrive == expected_arrive) {
    printf("  Status: PASS\n");
  } else {
    printf("  Status: FAIL (arrive mismatch)\n");
  }

  // Cleanup
  free(proxy_ctx.slot_epoch);

  for (int i = 0; i < window_size; i++) {
    if (slots[i].initialized) {
      hipHostUnregister(slots[i].trigger_mmio_addr);
      fi_close(&slots[i].trigger_cntr->fid);
      fi_close(&slots[i].completion_cntr->fid);
    }
  }

  hipHostFree(h_slot_state);
  hipHostFree((void *)h_trigger_addrs);
  hipFree(d_arrive);
  hipFree(d_atomic_operand);
  hipFree(d_final_arrive);
  hipFree(d_cycles);

  fi_close(&mr_arrive->fid);
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
