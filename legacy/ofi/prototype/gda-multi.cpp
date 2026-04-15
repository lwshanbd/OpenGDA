/*
 * gda-multi.cpp - GPU-Direct Async with multiple counter pairs
 *
 * Combines gda-comp.cpp and gda-large.cpp:
 * - Allocates 75 counter pairs (trigger + completion + atomic_completion)
 * - Executes 75 kernels, each using a different counter pair
 * - Tests concurrent DWQ operations with multiple counters
*/

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <pmi2.h>
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

// Number of counter pairs to allocate
#define NUM_COUNTER_PAIRS 100

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

// Counter pair structure
struct CounterPair {
  // Trigger counter - GPU writes to this to trigger operations
  struct fid_cntr *trigger_cntr;
  struct fi_cxi_cntr_ops *trigger_ops;
  void *trigger_mmio_addr;
  size_t trigger_mmio_len;
  volatile uint64_t *dev_trigger_addr;

  // Completion counter - NIC updates this when operations complete
  struct fid_cntr *completion_cntr;
  struct fi_cxi_cntr_ops *completion_ops;
  void *completion_mmio_addr;
  size_t completion_mmio_len;
  volatile uint64_t *dev_completion_addr;

  // Atomic completion counter - tracks when atomic operation completes
  struct fid_cntr *atomic_completion_cntr;

  bool initialized;
};

// GPU kernel to write counter doorbell and wait for completion
__global__ void gpu_trigger_and_wait(volatile uint64_t *trigger_addr,
                                     volatile uint64_t *atomic_result,
                                     uint64_t trigger_value,
                                     uint64_t *latency_cycles) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint64_t start = clock64();

    // Write to trigger counter to initiate RDMA
    *trigger_addr = trigger_value;

    // Poll atomic_result until NIC completes
    while (*atomic_result < 1) {
      // Spin wait
    }

    uint64_t end = clock64();

    if (latency_cycles) {
      *latency_cycles = end - start;
    }
  }
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

int main(int argc, char **argv) {
  int num_pairs = NUM_COUNTER_PAIRS;
  if (argc > 1) {
    num_pairs = atoi(argv[1]);
    if (num_pairs < 1)
      num_pairs = 1;
    if (num_pairs > 100)
      num_pairs = 100;
  }

  // Unset ROCR_VISIBLE_DEVICES before any initialization
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

  printf("Rank %d: Testing %d counter pairs with %d kernels\n", myrank,
         num_pairs, num_pairs);

  // HIP Initialization
  int device_count;
  CHECK_HIP(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
  int gpu_id = device_id;
  CHECK_HIP(hipSetDevice(gpu_id), "hipSetDevice");

  hipDeviceProp_t prop;
  CHECK_HIP(hipGetDeviceProperties(&prop, gpu_id), "hipGetDeviceProperties");
  double gpu_clock_mhz = prop.clockRate / 1000.0;
  printf("Rank %d: Using GPU %d: %s (%.0f MHz)\n", myrank, gpu_id, prop.name,
         gpu_clock_mhz);

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
  cq_attr.size = 128;
  cq_attr.format = FI_CQ_FORMAT_CONTEXT;
  CHECK(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");

  struct fid_ep *ep = NULL;
  CHECK(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");
  CHECK(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");
  CHECK(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
  CHECK(fi_enable(ep), "fi_enable");

  // Allocate counter pairs
  std::vector<CounterPair> counter_pairs(num_pairs);
  printf("Rank %d: Allocating %d counter pairs...\n", myrank, num_pairs);

  int created_pairs = 0;
  for (int i = 0; i < num_pairs; i++) {
    CounterPair &cp = counter_pairs[i];
    cp.initialized = false;

    // Create trigger counter
    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;

    ret = fi_cntr_open(domain, &cntr_attr, &cp.trigger_cntr, NULL);
    if (ret) {
      fprintf(stderr,
              "Rank %d: fi_cntr_open(trigger) failed for "
              "pair %d: %s\n",
              myrank, i, fi_strerror(-ret));
      break;
    }

    // Create completion counter
    struct fi_cntr_attr comp_cntr_attr = {};
    comp_cntr_attr.events = FI_CNTR_EVENTS_COMP;
    comp_cntr_attr.wait_obj = FI_WAIT_UNSPEC;

    ret = fi_cntr_open(domain, &comp_cntr_attr, &cp.completion_cntr, NULL);
    if (ret) {
      fprintf(stderr,
              "Rank %d: fi_cntr_open(completion) failed for "
              "pair %d: %s\n",
              myrank, i, fi_strerror(-ret));
      fi_close(&cp.trigger_cntr->fid);
      break;
    }

    // Create atomic completion counter
    ret = fi_cntr_open(domain, &cntr_attr, &cp.atomic_completion_cntr, NULL);
    if (ret) {
      fprintf(stderr,
              "Rank %d: fi_cntr_open(atomic_completion) "
              "failed for pair %d: "
              "%s\n",
              myrank, i, fi_strerror(-ret));
      fi_close(&cp.trigger_cntr->fid);
      fi_close(&cp.completion_cntr->fid);
      break;
    }

    // Get counter ops for MMIO access (trigger)
    ret = fi_open_ops(&cp.trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                      (void **)&cp.trigger_ops, NULL);
    if (ret) {
      fprintf(stderr,
              "Rank %d: fi_open_ops(trigger) failed for pair "
              "%d: %s\n",
              myrank, i, fi_strerror(-ret));
      fi_close(&cp.trigger_cntr->fid);
      fi_close(&cp.completion_cntr->fid);
      fi_close(&cp.atomic_completion_cntr->fid);
      break;
    }

    // Get counter ops for MMIO access (completion)
    ret = fi_open_ops(&cp.completion_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                      (void **)&cp.completion_ops, NULL);
    if (ret) {
      fprintf(stderr,
              "Rank %d: fi_open_ops(completion) failed for "
              "pair %d: %s\n",
              myrank, i, fi_strerror(-ret));
      fi_close(&cp.trigger_cntr->fid);
      fi_close(&cp.completion_cntr->fid);
      fi_close(&cp.atomic_completion_cntr->fid);
      break;
    }

    // Get MMIO addresses
    ret = cp.trigger_ops->get_mmio_addr(
        &cp.trigger_cntr->fid, &cp.trigger_mmio_addr, &cp.trigger_mmio_len);
    if (ret) {
      fprintf(stderr,
              "Rank %d: get_mmio_addr(trigger) failed for "
              "pair %d: %s\n",
              myrank, i, fi_strerror(-ret));
      fi_close(&cp.trigger_cntr->fid);
      fi_close(&cp.completion_cntr->fid);
      fi_close(&cp.atomic_completion_cntr->fid);
      break;
    }

    ret = cp.completion_ops->get_mmio_addr(&cp.completion_cntr->fid,
                                           &cp.completion_mmio_addr,
                                           &cp.completion_mmio_len);
    if (ret) {
      fprintf(stderr,
              "Rank %d: get_mmio_addr(completion) failed for "
              "pair %d: %s\n",
              myrank, i, fi_strerror(-ret));
      fi_close(&cp.trigger_cntr->fid);
      fi_close(&cp.completion_cntr->fid);
      fi_close(&cp.atomic_completion_cntr->fid);
      break;
    }

    // Map MMIO to GPU
    CHECK_HIP(hipHostRegister(cp.trigger_mmio_addr, cp.trigger_mmio_len,
                              hipHostRegisterMapped),
              "hipHostRegister(trigger)");
    CHECK_HIP(hipHostRegister(cp.completion_mmio_addr, cp.completion_mmio_len,
                              hipHostRegisterMapped),
              "hipHostRegister(completion)");

    CHECK_HIP(hipHostGetDevicePointer((void **)&cp.dev_trigger_addr,
                                      cp.trigger_mmio_addr, 0),
              "hipHostGetDevicePointer(trigger)");
    CHECK_HIP(hipHostGetDevicePointer((void **)&cp.dev_completion_addr,
                                      cp.completion_mmio_addr, 0),
              "hipHostGetDevicePointer(completion)");

    cp.initialized = true;
    created_pairs++;

    if ((i + 1) % 10 == 0) {
      printf("Rank %d: Created %d/%d counter pairs\n", myrank, created_pairs,
             num_pairs);
    }
  }

  printf("Rank %d: Successfully created %d counter pairs\n", myrank,
         created_pairs);

  // Exchange Address Information
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

  // Insert local address into AV for atomic operations to self
  fi_addr_t local_fi_addr = FI_ADDR_NOTAVAIL;
  inserted = fi_av_insert(av, local_addr, 1, &local_fi_addr, 0, NULL);
  if (inserted != 1 || local_fi_addr == FI_ADDR_NOTAVAIL) {
    fprintf(stderr, "Rank %d: fi_av_insert(local) failed\n", myrank);
    PMI2_Finalize();
    exit(1);
  }

  // Allocate and Register GPU Buffers
  size_t transfer_size = 256 * 1024 * 1; // 64KB per transfer
  size_t total_size = transfer_size * created_pairs;

  void *d_local_buf = NULL;
  CHECK_HIP(hipMalloc(&d_local_buf, total_size), "hipMalloc(local)");

  void *d_remote_buf = NULL;
  CHECK_HIP(hipMalloc(&d_remote_buf, total_size), "hipMalloc(remote)");

  // Allocate atomic result and operand arrays (one per counter pair)
  uint64_t *d_atomic_results = NULL;
  CHECK_HIP(hipMalloc(&d_atomic_results, sizeof(uint64_t) * created_pairs),
            "hipMalloc(atomic_results)");
  CHECK_HIP(hipMemset(d_atomic_results, 0, sizeof(uint64_t) * created_pairs),
            "hipMemset(atomic_results)");

  uint64_t *d_atomic_operands = NULL;
  CHECK_HIP(hipMalloc(&d_atomic_operands, sizeof(uint64_t) * created_pairs),
            "hipMalloc(atomic_operands)");

  // Initialize atomic operands to 1
  std::vector<uint64_t> h_operands(created_pairs, 1);
  CHECK_HIP(hipMemcpy(d_atomic_operands, h_operands.data(),
                      sizeof(uint64_t) * created_pairs, hipMemcpyHostToDevice),
            "hipMemcpy(atomic_operands)");

  // Allocate latency buffer
  uint64_t *d_latencies = NULL;
  CHECK_HIP(hipMalloc(&d_latencies, sizeof(uint64_t) * created_pairs),
            "hipMalloc(latencies)");

  // Host verification buffer
  uint8_t *h_verify_buf = (uint8_t *)malloc(total_size);

  struct fid_mr *mr_local = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_local_buf, total_size,
                               &mr_local, true, device_id),
        "register_memory_region(local)");

  struct fid_mr *mr_remote = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_remote_buf, total_size,
                               &mr_remote, true, device_id),
        "register_memory_region(remote)");

  struct fid_mr *mr_atomic_results = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_atomic_results,
                               sizeof(uint64_t) * created_pairs,
                               &mr_atomic_results, true, device_id),
        "register_memory_region(atomic_results)");

  struct fid_mr *mr_atomic_operands = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_atomic_operands,
                               sizeof(uint64_t) * created_pairs,
                               &mr_atomic_operands, true, device_id),
        "register_memory_region(atomic_operands)");

  uint64_t my_remote_key = fi_mr_key(mr_remote);
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
    remote_addr_for_rma = 0;
  }

  // Sync before test
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(pre-test)");
  usleep(100000);

  // Allocate DWQ structures for all pairs
  std::vector<struct fi_deferred_work> works(created_pairs);
  std::vector<struct fi_op_rma> op_rmas(created_pairs);
  std::vector<struct fi_msg_rma> msg_rmas(created_pairs);
  std::vector<struct iovec> iovs(created_pairs);
  std::vector<struct fi_rma_iov> rma_iovs(created_pairs);

  // Atomic structures
  std::vector<struct fi_deferred_work> atomic_works(created_pairs);
  std::vector<struct fi_op_atomic> atomics(created_pairs);
  std::vector<struct fi_msg_atomic> atomic_msgs(created_pairs);
  std::vector<struct fi_ioc> atomic_iovs(created_pairs);
  std::vector<struct fi_rma_ioc> atomic_rma_iovs(created_pairs);

  printf("\n");
  if (myrank == 0) {
    printf("========================================\n");
    printf("Running %d kernels with different counters\n", created_pairs);
    printf("Transfer size: %zu bytes per operation\n", transfer_size);
    printf("========================================\n\n");
  }
  std::vector<uint64_t> h_latencies(created_pairs);
  // Initialize remote buffer to 0xFF (will be overwritten by RDMA)
  // Source buffer is filled dynamically per-iteration with unique patterns
  if (myrank == 1) {
    CHECK_HIP(hipMemset(d_remote_buf, 0xFF, total_size), "hipMemset(remote)");
  }
  CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence");

  double total_start = get_time_us();

  if (myrank == 0) {
    void *desc_atomic_operands = fi_mr_desc(mr_atomic_operands);
    void *desc_atomic_results = fi_mr_desc(mr_atomic_results);

    // Reset all counters and atomic results
    for (int i = 0; i < created_pairs; i++) {
      fi_cntr_set(counter_pairs[i].trigger_cntr, 0);
      fi_cntr_set(counter_pairs[i].completion_cntr, 0);
      fi_cntr_set(counter_pairs[i].atomic_completion_cntr, 0);
    }
    CHECK_HIP(hipMemset(d_atomic_results, 0, sizeof(uint64_t) * created_pairs),
              "hipMemset reset atomic_results");
    CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize reset");

    // Queue all DWQ operations first
    printf("Rank %d: Starting 1000 iterations...\n", myrank);
    fflush(stdout);

    for (int t = 0; t < 1000; t++) {
      if (t == 0) {
        printf("Rank %d: Starting iteration 0\n", myrank);
        fflush(stdout);
      }
      int i = t % created_pairs;
      CounterPair &cp = counter_pairs[i];
      size_t offset = i * transfer_size;

      // Reset counters and atomic result when reusing (after first round)
      if (t >= created_pairs) {
        if (t == created_pairs) {
          printf("Rank %d: Starting second round, resetting counters\n", myrank);
          fflush(stdout);
        }
        fi_cntr_set(cp.trigger_cntr, 0);
        fi_cntr_set(cp.completion_cntr, 0);
        fi_cntr_set(cp.atomic_completion_cntr, 0);
        CHECK_HIP(hipMemset(&d_atomic_results[i], 0, sizeof(uint64_t)),
                  "hipMemset reset atomic_result");
        CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize after reset");
      }
      
      if (t == 0) {
        printf("Rank %d: About to queue RDMA work\n", myrank);
        fflush(stdout);
      }

      // Fill source region with iteration-specific pattern
      // Each iteration uses a unique byte value: (t & 0xFF)
      // This allows us to verify which iteration's data arrived
      uint8_t pattern = (uint8_t)(t & 0xFF);
      CHECK_HIP(hipMemset((char *)d_local_buf + offset, pattern, transfer_size),
                "hipMemset iteration pattern");
      CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize after memset");

      // Setup RDMA write
      iovs[i].iov_base = (char *)d_local_buf + offset;
      iovs[i].iov_len = transfer_size;

      rma_iovs[i].addr = remote_addr_for_rma + offset;
      rma_iovs[i].len = transfer_size;
      rma_iovs[i].key = peer_remote_key;

      memset(&msg_rmas[i], 0, sizeof(msg_rmas[i]));
      msg_rmas[i].msg_iov = &iovs[i];
      msg_rmas[i].desc = &desc_local;
      msg_rmas[i].iov_count = 1;
      msg_rmas[i].addr = peer_fi_addr;
      msg_rmas[i].rma_iov = &rma_iovs[i];
      msg_rmas[i].rma_iov_count = 1;

      memset(&op_rmas[i], 0, sizeof(op_rmas[i]));
      op_rmas[i].ep = ep;
      op_rmas[i].msg = msg_rmas[i];
      op_rmas[i].flags = FI_COMPLETION;  // Removed FI_CXI_CNTR_WB to avoid TLE accumulation

      works[i].triggering_cntr = cp.trigger_cntr;
      works[i].completion_cntr = cp.completion_cntr;
      works[i].threshold = 1;
      works[i].op_type = FI_OP_WRITE;
      works[i].op.rma = &op_rmas[i];

      ret = fi_control(&domain->fid, FI_QUEUE_WORK, &works[i]);

      if (ret) {
        fprintf(stderr,
                "Rank %d: fi_control(work %d) failed: "
                "%s (%d) with idx = %d\n",
                myrank, i, fi_strerror(-ret), ret, t);
        PMI2_Finalize();
        exit(1);
      }
      // Setup atomic operation for GPU completion signaling
      atomic_iovs[i].addr = &d_atomic_operands[i];
      atomic_iovs[i].count = 1;

      uint64_t atomic_result_addr = (uint64_t)&d_atomic_results[i];
      if (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
        atomic_rma_iovs[i].addr = atomic_result_addr;
      } else {
        atomic_rma_iovs[i].addr = i * sizeof(uint64_t);
      }
      atomic_rma_iovs[i].count = 1;
      atomic_rma_iovs[i].key = fi_mr_key(mr_atomic_results);

      atomic_msgs[i].msg_iov = &atomic_iovs[i];
      atomic_msgs[i].desc = &desc_atomic_operands;
      atomic_msgs[i].iov_count = 1;
      atomic_msgs[i].addr = local_fi_addr;
      atomic_msgs[i].rma_iov = &atomic_rma_iovs[i];
      atomic_msgs[i].rma_iov_count = 1;
      atomic_msgs[i].datatype = FI_UINT64;
      atomic_msgs[i].op = FI_SUM;

      atomics[i].ep = ep;
      atomics[i].msg = atomic_msgs[i];
      atomics[i].flags = FI_COMPLETION;

      atomic_works[i].op_type = FI_OP_ATOMIC;
      atomic_works[i].op.atomic = &atomics[i];
      atomic_works[i].triggering_cntr = cp.completion_cntr;
      atomic_works[i].completion_cntr = cp.atomic_completion_cntr;
      atomic_works[i].threshold = 1;

      ret = fi_control(&domain->fid, FI_QUEUE_WORK, &atomic_works[i]);
      if (ret) {
        fprintf(stderr,
                "Rank %d: fi_control(atomic %d) "
                "failed: %s (%d)\n",
                myrank, i, fi_strerror(-ret), ret);
        PMI2_Finalize();
        exit(1);
      }
      //  }
      //  return 0;

      //  // printf("Rank %d: Launching %d kernels...\n",
      //  myrank, created_pairs);

      //  // Launch kernels sequentially, each with different
      //  counter

      //  for (int i = 0; i < created_pairs; i++) {
      //    CounterPair &cp = counter_pairs[i];

      // Launch kernel to trigger this counter pair
      if (t == 0) {
        printf("Rank %d: Launching kernel for iteration 0\n", myrank);
        fflush(stdout);
      }
      
      hipLaunchKernelGGL(gpu_trigger_and_wait, dim3(1), dim3(1), 0, 0,
                         cp.dev_trigger_addr, &d_atomic_results[i], 1,
                         &d_latencies[i]);

      if (t == 0) {
        printf("Rank %d: Kernel launched, waiting for synchronize...\n", myrank);
        fflush(stdout);
      }

      CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");

      if (t == 0) {
        printf("Rank %d: Iteration 0 completed!\n", myrank);
        fflush(stdout);
      }

      // Progress CQ to drain ALL completion events
      struct fi_cq_entry cq_entries[16];
      int total_processed = 0;
      while (true) {
        ret = fi_cq_read(cq, cq_entries, 16);
        if (ret > 0) {
          total_processed += ret;
        } else if (ret == -FI_EAGAIN) {
          break;
        } else {
          fprintf(stderr, "Rank %d: fi_cq_read failed: %s (%d)\n",
                  myrank, fi_strerror(-ret), ret);
          break;
        }
      }
      
      if (t == 0 && total_processed > 0) {
        printf("Rank %d: Processed %d CQ events after iteration 0\n", myrank, total_processed);
        fflush(stdout);
      }

      // Progress logging (no flush needed - TLEs auto-release when triggers fire)
      if ((t + 1) % 100 == 0) {
        printf("Rank %d: Completed %d/1000 iterations (processed %d CQ events)\n",
               myrank, t + 1, total_processed);
        fflush(stdout);
      }
    }

    // Read latencies
    CHECK_HIP(hipMemcpy(h_latencies.data(), d_latencies,
                        sizeof(uint64_t) * created_pairs,
                        hipMemcpyDeviceToHost),
              "hipMemcpy read latencies");

    double total_end = get_time_us();

    // Calculate statistics
    double total_latency_us = 0;
    double min_latency_us = 1e9;
    double max_latency_us = 0;

    for (int i = 0; i < created_pairs; i++) {
      double lat_us = h_latencies[i] / gpu_clock_mhz;
      total_latency_us += lat_us;
      if (lat_us < min_latency_us)
        min_latency_us = lat_us;
      if (lat_us > max_latency_us)
        max_latency_us = lat_us;
    }

    double avg_latency_us = total_latency_us / created_pairs;
    double wall_time_us = total_end - total_start;

    printf("\n========================================\n");
    printf("Results:\n");
    printf("  Counter pairs used: %d\n", created_pairs);
    printf("  Total wall time: %.2f ms\n", wall_time_us / 1000.0);
    printf("  Per-operation latency (GPU cycles):\n");
    printf("    Average: %.2f us\n", avg_latency_us);
    printf("    Min: %.2f us\n", min_latency_us);
    printf("    Max: %.2f us\n", max_latency_us);
    printf("  Total data transferred: %.2f MB\n",
           (double)(transfer_size * created_pairs) / (1024 * 1024));
    printf("  Effective bandwidth: %.2f GB/s\n",
           (double)(transfer_size * created_pairs) / wall_time_us / 1000.0);
    printf("========================================\n");
  }

  // Sync and verify
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence");

  // Rank 1: Verify received data
  // Each region i should contain pattern from its last iteration: (i + 900) & 0xFF
  // (1000 iterations, 100 pairs: pair i is written by iterations i, i+100, ..., i+900)
  if (myrank == 1) {
    usleep(10000); // Small delay to ensure all writes complete
    CHECK_HIP(hipMemcpy(h_verify_buf, d_remote_buf, total_size,
                        hipMemcpyDeviceToHost),
              "hipMemcpy D2H");

    int total_errors = 0;
    int regions_failed = 0;
    const int num_iterations = 1000;

    for (int region = 0; region < created_pairs; region++) {
      // Calculate expected pattern: last iteration that wrote to this region
      int last_iter = region + (num_iterations - created_pairs);  // region + 900
      uint8_t expected = (uint8_t)(last_iter & 0xFF);

      size_t region_offset = region * transfer_size;
      int region_errors = 0;
      uint8_t first_bad_got = 0;
      size_t first_bad_pos = 0;

      for (size_t j = 0; j < transfer_size; j++) {
        uint8_t got = h_verify_buf[region_offset + j];
        if (got != expected) {
          if (region_errors == 0) {
            first_bad_got = got;
            first_bad_pos = j;
          }
          region_errors++;
        }
      }

      if (region_errors > 0) {
        regions_failed++;
        total_errors += region_errors;
        if (regions_failed <= 10) {
          fprintf(stderr,
                  "Rank %d: Region %d FAILED - expected 0x%02X (iter %d), "
                  "got 0x%02X at offset %zu (%d/%zu bytes wrong)\n",
                  myrank, region, expected, last_iter,
                  first_bad_got, first_bad_pos,
                  region_errors, transfer_size);
        }
      }
    }

    if (total_errors > 0) {
      fprintf(stderr,
              "Rank %d: VERIFICATION FAILED - %d/%d regions incorrect, "
              "%d/%zu bytes wrong\n",
              myrank, regions_failed, created_pairs, total_errors, total_size);
    } else {
      printf("Rank %d: VERIFICATION PASSED - all %d regions correct, "
             "each with unique iteration pattern\n",
             myrank, created_pairs);
      printf("Rank %d: Verified %zu bytes across 1000 RDMA operations\n",
             myrank, total_size);
    }
  }

  // Cleanup (no flush needed - TLEs auto-release when triggers fire)
  for (int i = 0; i < created_pairs; i++) {
    if (counter_pairs[i].initialized) {
      hipHostUnregister(counter_pairs[i].trigger_mmio_addr);
      hipHostUnregister(counter_pairs[i].completion_mmio_addr);
      fi_close(&counter_pairs[i].trigger_cntr->fid);
      fi_close(&counter_pairs[i].completion_cntr->fid);
      fi_close(&counter_pairs[i].atomic_completion_cntr->fid);
    }
  }

  hipFree(d_local_buf);
  hipFree(d_remote_buf);
  hipFree(d_atomic_results);
  hipFree(d_atomic_operands);
  hipFree(d_latencies);
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
