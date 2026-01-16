/*
 * gda_put_bw.cpp - GPU-Direct Async Put Bandwidth Benchmark
 *
 * Uses libfabric DWQ with window-based batching for bandwidth measurement.
 * GPU kernel triggers multiple operations in parallel.
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

// Benchmark parameters (OSU-style)
#define WINDOW_SIZE 64
#define SKIP_LARGE 10
#define SKIP_SMALL 100
#define ITERATIONS_LARGE 100
#define ITERATIONS_SMALL 1000
#define LARGE_MESSAGE_SIZE 8192

// OSU standard message sizes
const size_t test_sizes[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304
};
const int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

// =============================================================================
// GPU Kernel - Trigger multiple operations and wait for all completions
// =============================================================================

// Each thread triggers one operation and waits for its completion
__global__ void gpu_trigger_wait_window(
    volatile uint64_t **trigger_addrs,    // Array of trigger counter addresses
    volatile uint64_t **completion_addrs, // Array of completion signal addresses
    uint64_t *thresholds,                 // Threshold values for each op
    int num_ops) {
  int tid = threadIdx.x;
  if (tid < num_ops) {
    // Trigger this thread's operation
    *trigger_addrs[tid] = thresholds[tid];
    __threadfence_system();
  }

  __syncthreads();

  if (tid < num_ops) {
    // Wait for completion
    while (*completion_addrs[tid] < 1) {
      // spin
    }
    __threadfence_system();
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
  mr_attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;

  if (is_device_mem) {
    mr_attr.iface = FI_HMEM_ROCR;
    mr_attr.device.reserved = device_id;
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

// =============================================================================
// PMI2 Helper Functions
// =============================================================================

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

// =============================================================================
// Main Function
// =============================================================================

int main(void) {
  unsetenv("ROCR_VISIBLE_DEVICES");
  CHECK_HIP(hipSetDevice(7), "hipSetDevice(7)");

  int spawned, size, appnum;
  PMI2_Init(&spawned, &size, &myrank, &appnum);

  if (size != 2) {
    if (myrank == 0) fprintf(stderr, "Requires exactly 2 processes\n");
    PMI2_Finalize();
    return 1;
  }

  int device_count;
  CHECK_HIP(hipGetDeviceCount(&device_count), "hipGetDeviceCount");

  // -------------------------------------------------------------------------
  // Libfabric Initialization
  // -------------------------------------------------------------------------
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
  int ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL, NULL, 0, hints, &info);
  fi_freeinfo(hints);
  if (ret) {
    fprintf(stderr, "Rank %d: fi_getinfo failed\n", myrank);
    PMI2_Finalize();
    exit(1);
  }

  struct fi_info *cxi_info = NULL;
  for (struct fi_info *cur = info; cur; cur = cur->next) {
    if (cur->fabric_attr && cur->fabric_attr->prov_name &&
        strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
      cxi_info = cur;
      break;
    }
  }
  if (!cxi_info) {
    fprintf(stderr, "Rank %d: CXI provider not found\n", myrank);
    PMI2_Finalize();
    exit(1);
  }

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
  // Create WINDOW_SIZE counter pairs for DWQ
  // -------------------------------------------------------------------------
  struct fid_cntr *trigger_cntrs[WINDOW_SIZE];
  struct fid_cntr *completion_cntrs[WINDOW_SIZE];
  volatile uint64_t *dev_trigger_ptrs[WINDOW_SIZE];
  volatile uint64_t *dev_completion_ptrs[WINDOW_SIZE];

  for (int i = 0; i < WINDOW_SIZE; i++) {
    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;
    CHECK(fi_cntr_open(domain, &cntr_attr, &trigger_cntrs[i], NULL), "fi_cntr_open(trigger)");
    CHECK(fi_cntr_open(domain, &cntr_attr, &completion_cntrs[i], NULL), "fi_cntr_open(completion)");

    struct fi_cxi_cntr_ops *trigger_ops = NULL;
    struct fi_cxi_cntr_ops *completion_ops = NULL;
    CHECK(fi_open_ops(&trigger_cntrs[i]->fid, FI_CXI_COUNTER_OPS, 0, (void **)&trigger_ops, NULL), "fi_open_ops(trigger)");
    CHECK(fi_open_ops(&completion_cntrs[i]->fid, FI_CXI_COUNTER_OPS, 0, (void **)&completion_ops, NULL), "fi_open_ops(completion)");

    void *trigger_mmio = NULL, *completion_mmio = NULL;
    size_t len;
    CHECK(trigger_ops->get_mmio_addr(&trigger_cntrs[i]->fid, &trigger_mmio, &len), "get_mmio_addr(trigger)");
    CHECK(completion_ops->get_mmio_addr(&completion_cntrs[i]->fid, &completion_mmio, &len), "get_mmio_addr(completion)");

    CHECK_HIP(hipHostRegister(trigger_mmio, len, hipHostRegisterMapped), "hipHostRegister(trigger)");
    CHECK_HIP(hipHostRegister(completion_mmio, len, hipHostRegisterMapped), "hipHostRegister(completion)");
    CHECK_HIP(hipHostGetDevicePointer((void **)&dev_trigger_ptrs[i], trigger_mmio, 0), "hipHostGetDevicePointer(trigger)");
    CHECK_HIP(hipHostGetDevicePointer((void **)&dev_completion_ptrs[i], completion_mmio, 0), "hipHostGetDevicePointer(completion)");
  }

  // -------------------------------------------------------------------------
  // Exchange Addresses
  // -------------------------------------------------------------------------
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
    hex_to_bytes(hex, all_bin + i * addrlen, addrlen);
  }

  int peer = (myrank == 0) ? 1 : 0;
  fi_addr_t peer_fi_addr = FI_ADDR_NOTAVAIL;
  fi_av_insert(av, all_bin + peer * addrlen, 1, &peer_fi_addr, 0, NULL);

  // -------------------------------------------------------------------------
  // Allocate GPU Buffers
  // -------------------------------------------------------------------------
  size_t max_size = 128 * 1024 * 1024;
  void *d_local_buf = NULL, *d_remote_buf = NULL;
  CHECK_HIP(hipMalloc(&d_local_buf, max_size), "hipMalloc(local)");
  CHECK_HIP(hipMalloc(&d_remote_buf, max_size), "hipMalloc(remote)");

  struct fid_mr *mr_local = NULL, *mr_remote = NULL;
  CHECK(register_memory_region(domain, ep, cxi_info, d_local_buf, max_size, &mr_local, true, 0), "register(local)");
  CHECK(register_memory_region(domain, ep, cxi_info, d_remote_buf, max_size, &mr_remote, true, 0), "register(remote)");

  void *desc_local = fi_mr_desc(mr_local);
  uint64_t my_remote_key = fi_mr_key(mr_remote);
  uint64_t my_remote_addr = (uint64_t)d_remote_buf;

  // Exchange RMA info
  struct { uint64_t addr; uint64_t key; } my_rma_info, peer_rma_info;
  my_rma_info.addr = my_remote_addr;
  my_rma_info.key = my_remote_key;

  char rma_hex[128];
  bytes_to_hex((uint8_t *)&my_rma_info, sizeof(my_rma_info), rma_hex);
  char key_str[PMI2_MAX_KEYLEN];
  snprintf(key_str, sizeof(key_str), "rma-%d", myrank);
  PMI2_KVS_Put(key_str, rma_hex);
  PMI2_KVS_Fence();

  snprintf(key_str, sizeof(key_str), "rma-%d", peer);
  char peer_rma_hex[PMI2_MAX_VALLEN];
  int vallen;
  PMI2_KVS_Get(NULL, PMI2_ID_NULL, key_str, peer_rma_hex, sizeof(peer_rma_hex), &vallen);
  hex_to_bytes(peer_rma_hex, (uint8_t *)&peer_rma_info, sizeof(peer_rma_info));

  uint64_t peer_remote_key = peer_rma_info.key;
  uint64_t remote_addr_for_rma = (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) ? peer_rma_info.addr : 0;

  // -------------------------------------------------------------------------
  // Allocate GPU arrays for kernel
  // -------------------------------------------------------------------------
  volatile uint64_t **d_trigger_ptrs = NULL;
  volatile uint64_t **d_completion_ptrs = NULL;
  uint64_t *d_thresholds = NULL;
  CHECK_HIP(hipMalloc(&d_trigger_ptrs, WINDOW_SIZE * sizeof(void*)), "hipMalloc(d_trigger_ptrs)");
  CHECK_HIP(hipMalloc(&d_completion_ptrs, WINDOW_SIZE * sizeof(void*)), "hipMalloc(d_completion_ptrs)");
  CHECK_HIP(hipMalloc(&d_thresholds, WINDOW_SIZE * sizeof(uint64_t)), "hipMalloc(d_thresholds)");

  CHECK_HIP(hipMemcpy(d_trigger_ptrs, dev_trigger_ptrs, WINDOW_SIZE * sizeof(void*), hipMemcpyHostToDevice), "hipMemcpy(trigger_ptrs)");
  CHECK_HIP(hipMemcpy(d_completion_ptrs, dev_completion_ptrs, WINDOW_SIZE * sizeof(void*), hipMemcpyHostToDevice), "hipMemcpy(completion_ptrs)");

  // -------------------------------------------------------------------------
  // Pre-allocate DWQ structures
  // -------------------------------------------------------------------------
  struct fi_deferred_work works[WINDOW_SIZE];
  struct fi_op_rma op_rmas[WINDOW_SIZE];
  struct fi_msg_rma msg_rmas[WINDOW_SIZE];
  struct iovec iovs[WINDOW_SIZE];
  struct fi_rma_iov rma_iovs[WINDOW_SIZE];

  PMI2_KVS_Fence();
  usleep(100000);

  // -------------------------------------------------------------------------
  // Benchmark
  // -------------------------------------------------------------------------
  if (myrank == 0) {
    printf("# GDA Put Bandwidth Benchmark (Window=%d)\n", WINDOW_SIZE);
    printf("%-12s %15s\n", "Size", "BW (MB/s)");
  }

  for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
    size_t msg_size = test_sizes[size_idx];

    if (msg_size * WINDOW_SIZE > max_size) {
      if (myrank == 0) printf("%-12zu %15s\n", msg_size, "SKIP");
      continue;
    }

    int iterations = (msg_size >= LARGE_MESSAGE_SIZE) ? ITERATIONS_LARGE : ITERATIONS_SMALL;
    int skip = (msg_size >= LARGE_MESSAGE_SIZE) ? SKIP_LARGE : SKIP_SMALL;
    double total_time = 0.0;

    PMI2_KVS_Fence();

    for (int iter = 0; iter < iterations + skip; iter++) {
      if (myrank == 0) {
        // Reset counters
        for (int i = 0; i < WINDOW_SIZE; i++) {
          fi_cntr_set(trigger_cntrs[i], 0);
          fi_cntr_set(completion_cntrs[i], 0);
        }

        // Setup DWQ work items
        uint64_t h_thresholds[WINDOW_SIZE];
        for (int i = 0; i < WINDOW_SIZE; i++) {
          iovs[i].iov_base = (uint8_t*)d_local_buf + i * msg_size;
          iovs[i].iov_len = msg_size;

          rma_iovs[i].addr = remote_addr_for_rma + i * msg_size;
          rma_iovs[i].len = msg_size;
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
          op_rmas[i].flags = FI_COMPLETION | FI_CXI_CNTR_WB;

          works[i].triggering_cntr = trigger_cntrs[i];
          works[i].completion_cntr = completion_cntrs[i];
          works[i].threshold = 1;
          works[i].op_type = FI_OP_WRITE;
          works[i].op.rma = &op_rmas[i];

          h_thresholds[i] = 1;

          fi_control(&domain->fid, FI_QUEUE_WORK, &works[i]);
        }

        hipMemcpy(d_thresholds, h_thresholds, WINDOW_SIZE * sizeof(uint64_t), hipMemcpyHostToDevice);
        hipDeviceSynchronize();

        // Start timing
        double t_start = get_time_us();

        // Launch kernel
        hipLaunchKernelGGL(gpu_trigger_wait_window, dim3(1), dim3(WINDOW_SIZE),
                           0, 0, d_trigger_ptrs, d_completion_ptrs, d_thresholds, WINDOW_SIZE);
        hipDeviceSynchronize();

        double t_end = get_time_us();

        if (iter >= skip) {
          total_time += (t_end - t_start);
        }

        fi_control(&domain->fid, FI_FLUSH_WORK, NULL);
      }

      PMI2_KVS_Fence();
    }

    if (myrank == 0) {
      double avg_time_us = total_time / iterations;
      double bw = (double)(msg_size * WINDOW_SIZE) / avg_time_us;  // MB/s
      printf("%-12zu %15.2f\n", msg_size, bw);
      fflush(stdout);
    }

    PMI2_KVS_Fence();
  }

  // -------------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------------
  hipFree(d_trigger_ptrs);
  hipFree(d_completion_ptrs);
  hipFree(d_thresholds);
  hipFree(d_local_buf);
  hipFree(d_remote_buf);

  for (int i = 0; i < WINDOW_SIZE; i++) {
    fi_close(&trigger_cntrs[i]->fid);
    fi_close(&completion_cntrs[i]->fid);
  }

  fi_close(&mr_local->fid);
  fi_close(&mr_remote->fid);
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
  return 0;
}
