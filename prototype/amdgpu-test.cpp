/*
 * rank-amdgpu-gda.cpp - GPU-Driven RDMA using Deferred Work Queue
 *
 * This program implements true GPU-Driven RDMA using libfabric's
 * Deferred Work Queue (DWQ) interface where:
 * 1. CPU submits deferred work request to domain work queue
 * 2. GPU signals by writing to triggering counter doorbell (MMIO)
 * 3. NIC automatically executes RDMA when threshold is met
 * 4. Completion counter is incremented when done
 *
 * Key difference from previous version:
 * - Uses fi_control(FI_QUEUE_WORK) instead of FI_TRIGGER flag
 * - Uses fi_deferred_work structure with triggering/completion counters
 * - NIC directly executes queued work (no CPU intervention)
 *
 * Architecture:
 * - CPU: Queue fi_deferred_work via fi_control(FI_QUEUE_WORK)
 * - GPU: Write triggering counter MMIO → NIC detects threshold
 * - NIC: Execute queued RMA write autonomously
 * - NIC: Increment completion counter when done
 *
 * Based on libfabric Deferred Work Queue specification
 */

#include <cstdio>
#include <hip/hip_runtime.h>
#include <pmi2.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_cxi_ext.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_trigger.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

// =============================================================================
// Global Variables
// =============================================================================
int myrank = -1;

// Test parameters
#define NUM_ITERATIONS 10
const size_t test_sizes[] = {
    16 * 1024,   // 16KB
    32 * 1024,   // 32KB
    64 * 1024,   // 64KB
    128 * 1024,  // 128KB
    256 * 1024,  // 256KB
    512 * 1024,  // 512KB
    1024 * 1024, // 1MB
                 // 16 * 1024 * 1024, // 16MB
                 // 128 * 1024 * 1024 // 128MB
};
const int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

// =============================================================================
// GPU Doorbell Kernel
// =============================================================================

// GPU kernel to write counter doorbell (trigger NIC operation)
__global__ void gpu_write_counter_doorbell(volatile uint64_t *counter_addr,
                                           uint64_t value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *counter_addr = value;
    __threadfence_system();
    __syncthreads();
  }
}

// GPU kernel to poll completion counter
__global__ void gpu_poll_completion_counter(volatile uint64_t *counter_addr,
                                            uint64_t target_value,
                                            volatile int *done_flag) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    while (1) {
      uint64_t val = *counter_addr;
      // Extract success counter (lower 48 bits)
      uint64_t success_count = val & 0xFFFFFFFFFFFFULL;
      if (success_count >= target_value) {
        *done_flag = 1;
        __threadfence_system();
        break;
      }
    }
  }
}

// =============================================================================
// Utility Functions
// =============================================================================

static double get_time_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static const char *format_size(size_t size, char *buf) {
  if (size < 1024) {
    snprintf(buf, 32, "%zuB", size);
  } else if (size < 1024 * 1024) {
    snprintf(buf, 32, "%zuKB", size / 1024);
  } else {
    snprintf(buf, 32, "%zuMB", size / (1024 * 1024));
  }
  return buf;
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

// =============================================================================
// Memory Registration
// =============================================================================

static int register_memory_region(struct fid_domain *domain, struct fid_ep *ep,
                                  struct fi_info *info, void *buf, size_t size,
                                  struct fid_mr **mr_out, bool is_device_mem,
                                  int device_id) {
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
// PMI2 Helper Functions
// =============================================================================

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
static inline int smart_cntr_wait(struct fid_cntr *cntr, uint64_t threshold,
                                  size_t transfer_size) {
  uint64_t val;

  if (transfer_size <= 64 * 1024) { // <= 64KB: 紧密轮询
    for (int i = 0; i < 10000; i++) {
      val = fi_cntr_read(cntr);
      if ((val & 0xFFFFFFFFFFFFULL) >= threshold)
        return 0;
      __asm__ __volatile__("pause" ::: "memory");
    }
  }

  // 中等大小: 轮询 + yield
  for (int i = 0; i < 1000; i++) {
    val = fi_cntr_read(cntr);
    if ((val & 0xFFFFFFFFFFFFULL) >= threshold)
      return 0;
    sched_yield();
  }

  // 大数据: 轮询 + 短睡眠
  while (1) {
    val = fi_cntr_read(cntr);
    if ((val & 0xFFFFFFFFFFFFULL) >= threshold)
      return 0;
    usleep(10); // 10us
  }

  return 0;
}
// =============================================================================
// Main Function
// =============================================================================

int main(void) {
  // -------------------------------------------------------------------------
  // CRITICAL FIX: Unset ROCR_VISIBLE_DEVICES BEFORE ANY INITIALIZATION
  // -------------------------------------------------------------------------
  // Must unset BEFORE PMI2/HIP init - they may read env vars during init
  // Flux sets ROCR_VISIBLE_DEVICES which causes HSA virtualization
  unsetenv("ROCR_VISIBLE_DEVICES");

  // -------------------------------------------------------------------------
  // STEP 1: PMI2 Initialization
  // -------------------------------------------------------------------------
  int spawned, size, appnum;
  PMI2_Init(&spawned, &size, &myrank, &appnum);

  if (size != 2) {
    if (myrank == 0) {
      fprintf(stderr, "This program requires exactly 2 processes\n");
    }
    PMI2_Finalize();
    return 1;
  }

  // -------------------------------------------------------------------------
  // STEP 2: HIP Initialization
  // -------------------------------------------------------------------------
  int device_count;
  CHECK_HIP(hipGetDeviceCount(&device_count), "hipGetDeviceCount");

  // IMPORTANT: In multi-node environment, use rank % device_count
  int gpu_id = myrank % device_count;
  CHECK_HIP(hipSetDevice(gpu_id), "hipSetDevice");

  hipDeviceProp_t prop;
  CHECK_HIP(hipGetDeviceProperties(&prop, gpu_id), "hipGetDeviceProperties");
  printf("Rank %d: Using GPU %d: %s\n", myrank, gpu_id, prop.name);
  fflush(stderr);

  // -------------------------------------------------------------------------
  // STEP 3: Libfabric Initialization - Get Provider Info
  // -------------------------------------------------------------------------
  struct fi_info *hints = fi_allocinfo();
  hints->caps = FI_RMA | FI_MSG | FI_HMEM;
  hints->mode = FI_CONTEXT2; // DWQ requires FI_CONTEXT2
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
    fprintf(stderr, "Rank %d: fi_getinfo failed: %s (%d)\n", myrank,
            fi_strerror(-ret), ret);
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

  // -------------------------------------------------------------------------
  // STEP 4: Create Fabric, Domain, and Endpoint
  // -------------------------------------------------------------------------
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
  cq_attr.size = 128;
  cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  CHECK(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");

  struct fid_ep *ep = NULL;
  CHECK(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");
  CHECK(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");
  CHECK(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
  CHECK(fi_enable(ep), "fi_enable");

  // -------------------------------------------------------------------------
  // STEP 5: Create Counters for DWQ (Trigger + Completion)
  // -------------------------------------------------------------------------
  printf("Rank %d: Creating DWQ counters...\n", myrank);

  // Triggering counter - GPU will write to this to trigger operations
  struct fid_cntr *trigger_cntr = NULL;
  struct fi_cntr_attr cntr_attr = {};
  cntr_attr.events = FI_CNTR_EVENTS_COMP;
  CHECK(fi_cntr_open(domain, &cntr_attr, &trigger_cntr, NULL),
        "fi_cntr_open(trigger)");

  // Completion counter - NIC will update this when operations complete
  struct fi_cntr_attr completion_cntr_attr = {};
  completion_cntr_attr.events = FI_CNTR_EVENTS_COMP;
//   completion_cntr_attr.wait_obj = FI_WAIT_UNSPEC;
  struct fid_cntr *completion_cntr = NULL;
  CHECK(fi_cntr_open(domain, &completion_cntr_attr, &completion_cntr, NULL),
        "fi_cntr_open(completion)");

  // Get counter ops for MMIO access
  struct fi_cxi_cntr_ops *trigger_cntr_ops = NULL;
  CHECK(fi_open_ops(&trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                    (void **)&trigger_cntr_ops, NULL),
        "fi_open_ops(trigger)");

  struct fi_cxi_cntr_ops *completion_cntr_ops = NULL;
  CHECK(fi_open_ops(&completion_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                    (void **)&completion_cntr_ops, NULL),
        "fi_open_ops(completion)");

  // Get MMIO addresses
  void *trigger_mmio_addr = NULL;
  size_t trigger_mmio_len = 0;
  CHECK(trigger_cntr_ops->get_mmio_addr(&trigger_cntr->fid, &trigger_mmio_addr,
                                        &trigger_mmio_len),
        "get_mmio_addr(trigger)");
  printf("Rank %d: Trigger MMIO at %p, len=%zu\n", myrank, trigger_mmio_addr,
         trigger_mmio_len);

  void *completion_mmio_addr = NULL;
  size_t completion_mmio_len = 0;
  CHECK(completion_cntr_ops->get_mmio_addr(
            &completion_cntr->fid, &completion_mmio_addr, &completion_mmio_len),
        "get_mmio_addr(completion)");
  printf("Rank %d: Completion MMIO at %p, len=%zu\n", myrank,
         completion_mmio_addr, completion_mmio_len);

  // Map MMIO to GPU
  CHECK_HIP(hipHostRegister(trigger_mmio_addr, trigger_mmio_len,
                            hipHostRegisterMapped),
            "hipHostRegister(trigger)");
  CHECK_HIP(hipHostRegister(completion_mmio_addr, completion_mmio_len,
                            hipHostRegisterMapped),
            "hipHostRegister(completion)");

  volatile uint64_t *dev_trigger_cntr = NULL;
  volatile uint64_t *dev_completion_cntr = NULL;
  CHECK_HIP(
      hipHostGetDevicePointer((void **)&dev_trigger_cntr, trigger_mmio_addr, 0),
      "hipHostGetDevicePointer(trigger)");
  CHECK_HIP(hipHostGetDevicePointer((void **)&dev_completion_cntr,
                                    completion_mmio_addr, 0),
            "hipHostGetDevicePointer(completion)");

  // -------------------------------------------------------------------------
  // STEP 6: Exchange Address Information
  // -------------------------------------------------------------------------
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

  int peer = (myrank == 0) ? 1 : 0;
  void *peer_addr = all_bin + peer * addrlen;

  fi_addr_t peer_fi_addr = FI_ADDR_NOTAVAIL;
  int inserted = fi_av_insert(av, peer_addr, 1, &peer_fi_addr, 0, NULL);
  if (inserted != 1 || peer_fi_addr == FI_ADDR_NOTAVAIL) {
    fprintf(stderr, "Rank %d: fi_av_insert failed\n", myrank);
    PMI2_Finalize();
    exit(1);
  }

  printf("Rank %d: Peer address inserted\n", myrank);

  // -------------------------------------------------------------------------
  // STEP 7: Allocate and Register GPU Buffers
  // -------------------------------------------------------------------------
  size_t max_size = 128 * 1024 * 1024; // 1MB

  void *d_local_buf = NULL;
  CHECK_HIP(hipMalloc(&d_local_buf, max_size), "hipMalloc(local)");
  CHECK_HIP(hipMemset(d_local_buf, 0xAA, max_size), "hipMemset(local)");

  void *d_remote_buf = NULL;
  CHECK_HIP(hipMalloc(&d_remote_buf, max_size), "hipMalloc(remote)");
  if (myrank == 0) {
    CHECK_HIP(hipMemset(d_remote_buf, 0xCC, max_size), "hipMemset(remote)");
  } else {
    CHECK_HIP(hipMemset(d_remote_buf, 0xDD, max_size), "hipMemset(remote)");
  }
  CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize after memset");

  struct fid_mr *mr_local = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_local_buf, max_size,
                               &mr_local, true, 0),
        "register_memory_region(local)");

  struct fid_mr *mr_remote = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_remote_buf, max_size,
                               &mr_remote, true, 0),
        "register_memory_region(remote)");

  uint64_t my_remote_key = fi_mr_key(mr_remote);
  printf("Rank %d: my_remote_key = %lu\n", myrank, my_remote_key);
  uint64_t my_remote_addr = (uint64_t)d_remote_buf;
  void *desc_local = fi_mr_desc(mr_local);

  // Exchange RMA info
  struct {
    uint64_t addr;
    uint64_t key;
  } my_rma_info, peer_rma_info;

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
                     sizeof(peer_rma_hex), &vallen),
        "PMI2_KVS_Get(rma)");
  CHECK(hex_to_bytes(peer_rma_hex, (uint8_t *)&peer_rma_info,
                     sizeof(peer_rma_info)) > 0
            ? 0
            : -1,
        "hex_to_bytes(rma)");

  uint64_t peer_remote_addr = peer_rma_info.addr;
  uint64_t peer_remote_key = peer_rma_info.key;

  // Determine remote address mode
  uint64_t remote_addr_for_rma;
  if (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
    remote_addr_for_rma = peer_remote_addr;
  } else {
    remote_addr_for_rma = 0; // Use offset from MR base (CXI case)
  }

  printf("Rank %d: Peer remote_key=%lu, remote_addr=0x%lx\n", myrank,
         peer_remote_key, remote_addr_for_rma);

  // -------------------------------------------------------------------------
  // STEP 8: Sync before benchmark
  // -------------------------------------------------------------------------
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(pre-benchmark)");
  //   usleep(100000); // 100ms for readiness

  // -------------------------------------------------------------------------
  // STEP 9: DWQ Benchmark Loop
  // -------------------------------------------------------------------------
  if (myrank == 0) {
    printf("\n");
    printf("========================================\n");
    printf("GPU-to-GPU RDMA Latency Benchmark (DWQ)\n");
    printf("========================================\n");
    printf("%-10s %10s %10s\n", "Size", "Iterations", "Avg Latency");
    printf("%-10s %10s %10s\n", "", "", "(us)");
    printf("----------------------------------------\n");
  }

  // Allocate persistent structures for DWQ (must remain valid until completion)
  struct fi_deferred_work work;
  struct fi_op_rma *op_rma =
      (struct fi_op_rma *)malloc(sizeof(struct fi_op_rma));
  struct fi_msg_rma *msg_rma =
      (struct fi_msg_rma *)malloc(sizeof(struct fi_msg_rma));
  struct iovec *iov = (struct iovec *)malloc(sizeof(struct iovec));
  struct fi_rma_iov *rma_iov =
      (struct fi_rma_iov *)malloc(sizeof(struct fi_rma_iov));

  for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
    size_t current_size = test_sizes[size_idx];
    double total_time = 0.0;
    int successful_iterations = 0;

    // Warmup iteration (not counted)
    fi_cntr_set(trigger_cntr, 0);
    fi_cntr_set(completion_cntr, 0);

    // if (myrank == 0) {
    //   // Setup IOV for local buffer
    //   iov->iov_base = d_local_buf;
    //   iov->iov_len = current_size;

    //   // Setup RMA IOV for remote target
    //   rma_iov->addr = remote_addr_for_rma;
    //   rma_iov->len = current_size;
    //   rma_iov->key = peer_remote_key;

    //   // Fill fi_msg_rma
    //   memset(msg_rma, 0, sizeof(*msg_rma));
    //   msg_rma->msg_iov = iov;
    //   msg_rma->desc = &desc_local;
    //   msg_rma->iov_count = 1;
    //   msg_rma->addr = peer_fi_addr;
    //   msg_rma->rma_iov = rma_iov;
    //   msg_rma->rma_iov_count = 1;
    //   msg_rma->context = NULL;
    //   msg_rma->data = 0;

    //   // Fill fi_op_rma
    //   memset(op_rma, 0, sizeof(*op_rma));
    //   op_rma->ep = ep;
    //   op_rma->msg = *msg_rma;
    //   op_rma->flags = FI_COMPLETION | FI_CXI_CNTR_WB;

    //   // Setup deferred work
    //   work.triggering_cntr = trigger_cntr;
    //   work.completion_cntr = completion_cntr;
    //   work.threshold = 1;
    //   work.op_type = FI_OP_WRITE;
    //   work.op.rma = op_rma;

    //   // Queue deferred work
    //   int ret = fi_control(&domain->fid, FI_QUEUE_WORK, &work);
    //   if (ret) {
    //     fprintf(stderr, "Rank %d: Warmup FI_QUEUE_WORK failed: %s (%d)\n",
    //             myrank, fi_strerror(-ret), ret);
    //     continue;
    //   }

    //   // Trigger and wait for warmup
    //   hipLaunchKernelGGL(gpu_write_counter_doorbell, dim3(1), dim3(1), 0, 0,
    //                      dev_trigger_cntr, work.threshold);
    //   CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");
    //   fi_cntr_wait(completion_cntr, 1, -1);
    // }

    CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence"); // sync after warmup

    // Timed iterations
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
      fi_cntr_set(trigger_cntr, 0);
      fi_cntr_set(completion_cntr, 0);

      if (myrank == 0) {
        // ---------- Rank 0: Sender using DWQ ----------

        // Reset d_local_buf with iteration-dependent pattern
        uint8_t pattern = 0x10 + (iter % 240);
        CHECK_HIP(hipMemset(d_local_buf, pattern, current_size),
                  "hipMemset(d_local_buf)");
        CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize after memset");

        // 1. Setup IOV for local buffer
        iov->iov_base = d_local_buf;
        iov->iov_len = current_size;

        // 2. Setup RMA IOV for remote target
        rma_iov->addr = remote_addr_for_rma;
        rma_iov->len = current_size;
        rma_iov->key = peer_remote_key;

        // 3. Fill fi_msg_rma
        memset(msg_rma, 0, sizeof(*msg_rma));
        msg_rma->msg_iov = iov;
        msg_rma->desc = &desc_local; // Include descriptor for GPU memory
        msg_rma->iov_count = 1;
        msg_rma->addr = peer_fi_addr;
        msg_rma->rma_iov = rma_iov;
        msg_rma->rma_iov_count = 1;
        msg_rma->context = NULL; // Not needed for DWQ
        msg_rma->data = 0;

        // 4. Fill fi_op_rma
        memset(op_rma, 0, sizeof(*op_rma));
        op_rma->ep = ep;
        op_rma->msg = *msg_rma;
        // Use FI_CXI_CNTR_WB to ensure counter writeback
        op_rma->flags = FI_COMPLETION | FI_CXI_CNTR_WB;

        // 5. Setup deferred work
        work.triggering_cntr = trigger_cntr;    // Counter GPU will write to
        work.completion_cntr = completion_cntr; // Counter NIC will increment
        work.threshold = 1;                     // Trigger when counter >= 1
        work.op_type = FI_OP_WRITE;             // RMA write operation
        work.op.rma = op_rma;

        // 6. Queue deferred work to domain
        int ret = fi_control(&domain->fid, FI_QUEUE_WORK, &work);

        // 7. Start timing and fire GPU doorbell to trigger
        double start_time = get_time_us();
        hipLaunchKernelGGL(gpu_write_counter_doorbell, dim3(1), dim3(1), 0, 0,
                           dev_trigger_cntr, 1);
        // // CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");

        // 8. Wait for completion counter to increment
        // fi_cntr_wait(completion_cntr, 1, -1);
        smart_cntr_wait(completion_cntr, 1, current_size);
        // uint64_t completion_val = 0;
        // completion_val = fi_cntr_read(completion_cntr);
        // do {
        //   completion_val = fi_cntr_read(completion_cntr);
        //   // usleep(10);
        // } while (completion_val < 1);
        double end_time = get_time_us();

        total_time += (end_time - start_time);
        successful_iterations++;
        // CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence");
      } else if (myrank == 1) {
        // ---------- Rank 1: Receiver - Verify d_remote_buf ----------
        // CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence");
        // Wait for data to arrive
        // usleep(1000);

        // Allocate host buffer for verification
        // uint8_t *h_verify_buf = (uint8_t *)malloc(current_size);
        // if (!h_verify_buf) {
        //   fprintf(stderr, "Rank %d: Failed to allocate verify buffer\n",
        //           myrank);
        //   continue;
        // }
        // CHECK_HIP(hipDeviceSynchronize(), "pre-verify sync on receiver");
        // // Copy d_remote_buf to host for verification
        // CHECK_HIP(hipMemcpy(h_verify_buf, d_remote_buf, current_size,
        //                     hipMemcpyDeviceToHost),
        //           "hipMemcpy D2H for verification");

        // // Expected pattern from rank 0
        // uint8_t expected_pattern = 0x10 + (iter % 240);

        // // Verify the data
        // bool verification_passed = true;
        // size_t errors = 0;
        // size_t max_errors_to_show = 5;

        // for (size_t i = 0; i < current_size; i++) {
        //   if (h_verify_buf[i] != expected_pattern) {
        //     verification_passed = false;
        //     if (errors < max_errors_to_show) {
        //       fprintf(stderr,
        //               "Rank %d: Verification failed at byte %zu: expected "
        //               "0x%02x, got 0x%02x\n",
        //               myrank, i, expected_pattern, h_verify_buf[i]);
        //     }
        //     errors++;
        //   }
        // }

        // if (verification_passed) {
        //   printf("Rank %d: Iteration %d - Verification PASSED
        //   (pattern=0x%02x, "
        //          "size=%zu)\n",
        //          myrank, iter, expected_pattern, current_size);
        // } else {
        //   fprintf(stderr,
        //           "Rank %d: Iteration %d - Verification FAILED (%zu errors
        //           out " "of %zu bytes)\n", myrank, iter, errors,
        //           current_size);
        // }

        // free(h_verify_buf);
      }

      CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence");
    }

    // Print results for this size
    if (myrank == 0 && successful_iterations > 0) {
      double avg_latency = total_time / successful_iterations;
      char size_str[32];
      format_size(current_size, size_str);
      printf("%-10s %10d %10.2f\n", size_str, successful_iterations,
             avg_latency);
    }
  }

  if (myrank == 0) {
    printf("========================================\n\n");
  }

  // -------------------------------------------------------------------------
  // STEP 10: Final Synchronization
  // -------------------------------------------------------------------------
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(final)");

  // Free persistent DWQ structures
  free(rma_iov);
  free(iov);
  free(msg_rma);
  free(op_rma);

  // -------------------------------------------------------------------------
  // STEP 11: Cleanup
  // -------------------------------------------------------------------------

  hipHostUnregister(trigger_mmio_addr);
  hipHostUnregister(completion_mmio_addr);
  hipFree(d_local_buf);
  hipFree(d_remote_buf);

  fi_close(&mr_local->fid);
  fi_close(&mr_remote->fid);
  fi_close(&trigger_cntr->fid);
  fi_close(&completion_cntr->fid);
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
  printf("Rank %d: Done\n", myrank);
  return 0;
}
