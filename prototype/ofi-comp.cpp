/*
 * rank-amdgpu.cpp - GPU-to-GPU RDMA Latency Benchmark using Libfabric
 *
 * This program measures RDMA write latency between two processes using AMD GPU
 * memory with the CXI provider. It tests various message sizes and reports
 * average latency for each.
 *
 * Key Features:
 * - Uses AMD GPU memory (HIP/ROCm) instead of CPU memory
 * - Tests multiple message sizes: 1B to 128MB (14 different sizes)
 * - Each size tested 10 times to compute average latency
 * - Uses PMI2 for process coordination and information exchange
 * - Handles CXI provider's lack of FI_MR_VIRT_ADDR support
 * - Uses fi_writemsg with FI_DELIVERY_COMPLETE for reliable transfers
 * - Each process uses GPU 0
 *
 * Based on rank.cpp and LCI OFI backend implementation
 */

#include <cstdio>
#include <hip/hip_runtime.h>
#include <pmi2.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_cxi_ext.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
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
int myrank = -1; // Global for error handling

static uint64_t g_next_rdma_key = 0;

// GPU computation kernel - SAXPY operation (y = a*x + y)
// Used to add computation workload before and after communication
__global__ void gpu_compute_saxpy(float *x, float *y, float a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Perform SAXPY: y[i] = a * x[i] + y[i]
    // Loop multiple times to increase computation time
    #pragma unroll 1
    for (int iter = 0; iter < 100; iter++) {
      y[idx] = a * x[idx] + y[idx];
    }
  }
}

// Test parameters
#define NUM_ITERATIONS 20  // Run 20 iterations, pick best 10
const size_t test_sizes[] = {
    // 1 * 1024, // 1KB
    // 2 * 1024, // 2KB
    // 4 * 1024, // 4KB
    // 8 * 1024, // 8KB
    16 * 1024,        // 16KB
    32 * 1024,        // 32KB
    64 * 1024,        // 64KB
    128 * 1024,       // 128KB
    256 * 1024,       // 256KB
    512 * 1024,       // 512KB
    1024 * 1024,      // 1MB
    2 * 1024 * 1024,  // 2MB
    4 * 1024 * 1024,  // 4MB
    8 * 1024 * 1024,  // 8MB
    16 * 1024 * 1024, // 16MB
};
const int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

// =============================================================================
// Utility Functions
// =============================================================================

// Get current time in microseconds
static double get_time_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// Format size for display
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
// PMI2 Address Exchange
// =============================================================================

int exchange_addrs(int rank, int size, const char *my_hex, size_t hex_len,
                   char *all_hex) {
  int rc;
  char jobid[PMI2_MAX_VALLEN];
  rc = PMI2_Job_GetId(jobid, sizeof(jobid));
  if (rc != PMI2_SUCCESS) {
    fprintf(stderr, "Rank %d: PMI2_Job_GetId failed (%d)\n", rank, rc);
    return rc;
  }

  char key[PMI2_MAX_KEYLEN];
  snprintf(key, sizeof(key), "addr-%d", rank);
  rc = PMI2_KVS_Put(key, my_hex);
  if (rc != PMI2_SUCCESS) {
    fprintf(stderr, "Rank %d: PMI2_KVS_Put failed (%d)\n", rank, rc);
    return rc;
  }

  rc = PMI2_KVS_Fence();
  if (rc != PMI2_SUCCESS) {
    fprintf(stderr, "Rank %d: PMI2_KVS_Fence failed (%d)\n", rank, rc);
    return rc;
  }

  for (int i = 0; i < size; i++) {
    int vallen;
    snprintf(key, sizeof(key), "addr-%d", i);
    char val[PMI2_MAX_VALLEN];
    rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key, val, sizeof(val), &vallen);
    if (rc != PMI2_SUCCESS) {
      fprintf(stderr, "Rank %d: PMI2_KVS_Get(%d) failed (%d)\n", rank, i, rc);
      return rc;
    }
    char *slot = all_hex + i * (hex_len + 1);
    strncpy(slot, val, hex_len);
    slot[hex_len] = '\0';
  }
  return PMI2_SUCCESS;
}

// =============================================================================
// Memory Region Registration
// =============================================================================

int register_memory_region(struct fid_domain *domain, struct fid_ep *ep,
                           struct fi_info *cxi_info, void *buffer,
                           size_t buffer_size, struct fid_mr **mr_out,
                           bool is_gpu_mem, int gpu_device) {
  struct fid_mr *mr;
  struct fi_mr_attr mr_attr;
  struct iovec iov;

  iov.iov_base = buffer;
  iov.iov_len = buffer_size;

  memset(&mr_attr, 0, sizeof(mr_attr));
  mr_attr.mr_iov = &iov;
  mr_attr.iov_count = 1;
  mr_attr.access = FI_RMA | FI_SEND | FI_RECV | FI_READ | FI_WRITE |
                   FI_REMOTE_WRITE | FI_REMOTE_READ;

  uint64_t rdma_key = 0;
  if (cxi_info->domain_attr->mr_mode & FI_MR_PROV_KEY) {
    rdma_key = 0;
  } else {
    rdma_key = g_next_rdma_key++;
  }
  mr_attr.requested_key = rdma_key;

  // Set memory interface type
  if (is_gpu_mem) {
    mr_attr.iface = FI_HMEM_ROCR; // AMD GPU memory
    // Note: libfabric 2.1 doesn't have device.rocr, use reserved or cuda field
    mr_attr.device.reserved = gpu_device; // Set GPU device number
  } else {
    mr_attr.iface = FI_HMEM_SYSTEM; // CPU memory
  }

  int ret = fi_mr_regattr(domain, &mr_attr, 0, &mr);
  if (ret) {
    return ret;
  }

  if (cxi_info->domain_attr->mr_mode & FI_MR_ENDPOINT) {
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
// Main Program
// =============================================================================

int main(void) {
  // Libfabric objects
  struct fi_info *info = NULL;
  struct fi_info *cxi_info = NULL;
  struct fid_fabric *fabric = NULL;
  struct fid_domain *domain = NULL;
  struct fid_ep *ep = NULL;
  struct fid_av *av = NULL;
  struct fid_cq *cq = NULL;

  // =============================================================================
  // STEP 0: GPU Initialization
  // =============================================================================
  // Set to use GPU 0
  CHECK_HIP(hipSetDevice(0), "hipSetDevice");

  // Verify GPU is accessible
  int device;
  CHECK_HIP(hipGetDevice(&device), "hipGetDevice");

  hipDeviceProp_t prop;
  CHECK_HIP(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");

  // Note: Don't print yet because myrank is not set

  // =============================================================================
  // STEP 1: PMI2 Initialization
  // =============================================================================
  int spawned, size, appnum = 0;
  if (!PMI2_Initialized()) {
    int rc = PMI2_Init(&spawned, &size, &myrank, &appnum);
    if (rc != PMI2_SUCCESS) {
      fprintf(stderr, "PMI2_Init failed (%d)\n", rc);
      exit(1);
    }
  }

  printf("Rank %d: Using GPU %d: %s\n", myrank, device, prop.name);

  // =============================================================================
  // STEP 2: Libfabric Provider Setup
  // =============================================================================
  struct fi_info *hints = fi_allocinfo();
  hints->caps = FI_RMA | FI_MSG | FI_HMEM; // Add FI_HMEM for GPU memory support
  hints->mode = FI_CONTEXT;
  hints->ep_attr->type = FI_EP_RDM;
  hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                FI_MR_PROV_KEY | FI_MR_LOCAL | FI_MR_ENDPOINT;
  hints->domain_attr->threading = FI_THREAD_SAFE;
  hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
  hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

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

  if (myrank == 0) {
    printf("Rank 0: Using CXI provider %s\n", cxi_info->domain_attr->name);
    printf("  - mr_mode: 0x%lx\n",
           (unsigned long)cxi_info->domain_attr->mr_mode);
    printf("  - inject_size: %zu bytes\n", cxi_info->tx_attr->inject_size);
    printf("  - max_msg_size: %zu bytes\n", cxi_info->ep_attr->max_msg_size);
  }

  // =============================================================================
  // STEP 3: Create Fabric and Domain
  // =============================================================================
  CHECK(fi_fabric(cxi_info->fabric_attr, &fabric, NULL), "fi_fabric");
  CHECK(fi_domain(fabric, cxi_info, &domain, NULL), "fi_domain");

  // =============================================================================
  // STEP 4: Create Endpoint
  // =============================================================================
  CHECK(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");

  // =============================================================================
  // STEP 5: Create and Bind Completion Queue (CQ)
  // =============================================================================
  struct fi_cq_attr cq_attr;
  memset(&cq_attr, 0, sizeof(cq_attr));
  cq_attr.format = FI_CQ_FORMAT_DATA;
  cq_attr.wait_obj = FI_WAIT_NONE;
  cq_attr.size = 128;
  CHECK(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");
  CHECK(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");

  // =============================================================================
  // STEP 6: Create and Bind Address Vector (AV)
  // =============================================================================
  struct fi_av_attr av_attr{};
  av_attr.type = FI_AV_MAP;
  CHECK(fi_av_open(domain, &av_attr, &av, NULL), "fi_av_open");
  CHECK(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");

  // Enable endpoint
  CHECK(fi_enable(ep), "fi_enable");

  // =============================================================================
  // STEP 7: Exchange Endpoint Addresses (via PMI2)
  // =============================================================================
  size_t addrlen = 0;
  fi_getname(&ep->fid, NULL, &addrlen);
  void *local_addr = malloc(addrlen);
  CHECK(fi_getname(&ep->fid, local_addr, &addrlen), "fi_getname");

  // Get local endpoint address
  char *my_hex = (char *)malloc(2 * addrlen + 1);
  bytes_to_hex((uint8_t *)local_addr, addrlen, my_hex);

  // Exchange addresses with all ranks
  char *all_hex = (char *)malloc(size * (2 * addrlen + 1));
  CHECK(exchange_addrs(myrank, size, my_hex, 2 * addrlen, all_hex),
        "exchange_addrs");

  // Convert from hex to binary
  uint8_t *all_bin = (uint8_t *)malloc(size * addrlen);
  for (int i = 0; i < size; i++) {
    const char *hex = all_hex + i * (2 * addrlen + 1);
    CHECK(hex_to_bytes(hex, all_bin + i * addrlen, addrlen) > 0 ? 0 : -1,
          "hex_to_bytes");
  }

  // Insert peer address into AV
  int peer = (myrank == 0) ? 1 : 0;
  void *peer_addr = all_bin + peer * addrlen;

  fi_addr_t peer_fi_addr = FI_ADDR_NOTAVAIL;
  int inserted = fi_av_insert(av, peer_addr, 1, &peer_fi_addr, 0, NULL);
  if (inserted != 1) {
    fprintf(stderr, "Rank %d: fi_av_insert failed, inserted=%d\n", myrank,
            inserted);
    PMI2_Finalize();
    exit(1);
  }
  if (peer_fi_addr == FI_ADDR_NOTAVAIL) {
    fprintf(stderr, "Rank %d: peer_fi_addr is FI_ADDR_NOTAVAIL!\n", myrank);
    PMI2_Finalize();
    exit(1);
  }

  // =============================================================================
  // STEP 8: Allocate and Register GPU Memory
  // =============================================================================
  // Use maximum test size for buffer allocation
  size_t max_buf_size = test_sizes[num_test_sizes - 1];

  // Allocate GPU memory buffers
  char *d_local_buf = NULL;  // GPU buffer for sending
  char *d_remote_buf = NULL; // GPU buffer for receiving

  CHECK_HIP(hipMalloc(&d_local_buf, max_buf_size), "hipMalloc(local)");
  CHECK_HIP(hipMalloc(&d_remote_buf, max_buf_size), "hipMalloc(remote)");

  // Initialize GPU buffers using CPU staging
  char *h_temp = (char *)malloc(max_buf_size);
  memset(h_temp, 0xAB, max_buf_size); // Fill with test pattern

  // Copy to GPU
  CHECK_HIP(hipMemcpy(d_local_buf, h_temp, max_buf_size, hipMemcpyHostToDevice),
            "hipMemcpy H2D local");

  memset(h_temp, 0, max_buf_size);
  CHECK_HIP(
      hipMemcpy(d_remote_buf, h_temp, max_buf_size, hipMemcpyHostToDevice),
      "hipMemcpy H2D remote");

  CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");

  // Allocate computation buffers for SAXPY workload
  // Size is proportional to max communication size
  // compute_n will be adjusted per iteration based on current_size
  const size_t max_compute_n = max_buf_size / sizeof(float);
  const size_t max_compute_bytes = max_compute_n * sizeof(float);
  float *d_compute_x = NULL;
  float *d_compute_y = NULL;
  CHECK_HIP(hipMalloc(&d_compute_x, max_compute_bytes), "hipMalloc(compute_x)");
  CHECK_HIP(hipMalloc(&d_compute_y, max_compute_bytes), "hipMalloc(compute_y)");

  // Initialize computation buffers
  CHECK_HIP(hipMemset(d_compute_x, 0, max_compute_bytes), "hipMemset(compute_x)");
  CHECK_HIP(hipMemset(d_compute_y, 0, max_compute_bytes), "hipMemset(compute_y)");
  CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");

  const int threads_per_block = 256;
  const float compute_a = 2.5f;

  if (myrank == 0) {
    printf("Rank 0: GPU buffers allocated (%zu bytes) - local=%p, remote=%p\n",
           max_buf_size, d_local_buf, d_remote_buf);
    printf("Rank 0: Computation buffers allocated (%zu bytes, max %zu elements)\n",
           max_compute_bytes, max_compute_n);
  }

  // Register GPU buffers with libfabric
  struct fid_mr *mr_local;
  CHECK(register_memory_region(domain, ep, cxi_info, d_local_buf, max_buf_size,
                               &mr_local, true, device),
        "register_memory_region(local GPU)");

  struct fid_mr *mr_remote;
  CHECK(register_memory_region(domain, ep, cxi_info, d_remote_buf, max_buf_size,
                               &mr_remote, true, device),
        "register_memory_region(remote GPU)");

  // Get MR keys and descriptors
  uint64_t remote_key = fi_mr_key(mr_remote);
  void *desc_local = fi_mr_desc(mr_local);

  // =============================================================================
  // STEP 9: Exchange RMA Information (address + key)
  // =============================================================================
  struct {
    uint64_t addr;
    uint64_t key;
  } my_rma_info, peer_rma_info;

  // Prepare my RMA info
  my_rma_info.addr = (uint64_t)d_remote_buf;
  my_rma_info.key = remote_key;

  // Publish my RMA info to PMI2
  char rma_hex[128];
  bytes_to_hex((uint8_t *)&my_rma_info, sizeof(my_rma_info), rma_hex);

  char key[PMI2_MAX_KEYLEN];
  snprintf(key, sizeof(key), "rma-%d", myrank);
  CHECK(PMI2_KVS_Put(key, rma_hex), "PMI2_KVS_Put(rma)");
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(rma)");

  // Retrieve peer's RMA info
  snprintf(key, sizeof(key), "rma-%d", peer);
  char peer_rma_hex[PMI2_MAX_VALLEN];
  int vallen;
  CHECK(PMI2_KVS_Get(NULL, PMI2_ID_NULL, key, peer_rma_hex,
                     sizeof(peer_rma_hex), &vallen),
        "PMI2_KVS_Get(rma)");
  CHECK(hex_to_bytes(peer_rma_hex, (uint8_t *)&peer_rma_info,
                     sizeof(peer_rma_info)) > 0
            ? 0
            : -1,
        "hex_to_bytes(rma)");

  uint64_t peer_remote_addr = peer_rma_info.addr;
  uint64_t peer_remote_key = peer_rma_info.key;

  // =============================================================================
  // STEP 10: Synchronize Before RDMA
  // =============================================================================
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(pre-write)");
  usleep(100000); // 100ms delay for CXI provider readiness

  // =============================================================================
  // STEP 11: Latency Benchmark - Test Multiple Sizes
  // =============================================================================
  if (myrank == 0) {
    printf("\n");
    printf("========================================================================\n");
    printf("CPU-Driven RDMA Latency Benchmark (with GPU computation)\n");
    printf("========================================================================\n");
    printf("%-10s  %12s  %s\n", "Size", "Latency(us)", "Statistics");
    printf("==========  ============  ===============================================\n");
    printf("Note: Latency is average of best 10/%d iterations\n", NUM_ITERATIONS);
    printf("      Includes 2x SAXPY computation (proportional to transfer size)\n");
    printf("      Compute size = transfer_size / sizeof(float), 100 iters per SAXPY\n\n");
  }

  // Setup remote address mode (check FI_MR_VIRT_ADDR support)
  uint64_t remote_addr_for_rma;
  if (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
    remote_addr_for_rma = peer_remote_addr;
  } else {
    remote_addr_for_rma = 0; // Use offset from MR base (CXI case)
  }

  // Loop through different message sizes
  for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
    size_t current_size = test_sizes[size_idx];

    // Store all iteration times for statistical analysis
    double iteration_times[NUM_ITERATIONS];
    int successful_iterations = 0;

    // Warmup iteration (not counted)
    if (myrank == 0) {
      ret = fi_write(ep, d_local_buf, current_size, desc_local, peer_fi_addr,
                     remote_addr_for_rma, peer_remote_key, NULL);

      struct fi_cq_data_entry cqe;
      while (fi_cq_read(cq, &cqe, 1) == -FI_EAGAIN)
        ;
    }

    // Timed iterations - with computation before and after communication
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
      if (myrank == 0) {
        // Compute size is proportional to communication size
        const int compute_n = current_size / sizeof(float);
        const int num_blocks = (compute_n + threads_per_block - 1) / threads_per_block;

        double start_time = get_time_us();

        // === Computation Phase 1: Pre-communication ===
        hipLaunchKernelGGL(gpu_compute_saxpy,
                           dim3(num_blocks), dim3(threads_per_block), 0, 0,
                           d_compute_x, d_compute_y, compute_a, compute_n);
        CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize compute1");

        // === Communication Phase ===
        // Choose strategy based on message size
        const size_t SMALL_MSG_THRESHOLD = 8192 * 4; // 8KB
        bool use_fast_path = (current_size <= SMALL_MSG_THRESHOLD);

        if (use_fast_path) {
          // Fast path for small messages: use fi_write
          ret =
              fi_write(ep, d_local_buf, current_size, desc_local, peer_fi_addr,
                       remote_addr_for_rma, peer_remote_key, NULL);
        } else {
          // Standard path for large messages: use fi_writemsg
          struct iovec iov;
          iov.iov_base = d_local_buf;
          iov.iov_len = current_size;

          struct fi_rma_iov rma_iov;
          rma_iov.addr = remote_addr_for_rma;
          rma_iov.len = current_size;
          rma_iov.key = peer_remote_key;

          struct fi_msg_rma msg;
          memset(&msg, 0, sizeof(msg));
          msg.msg_iov = &iov;
          msg.desc = &desc_local;
          msg.iov_count = 1;
          msg.addr = peer_fi_addr;
          msg.rma_iov = &rma_iov;
          msg.rma_iov_count = 1;
          msg.context = NULL;
          msg.data = 0;

          ret = fi_writemsg(ep, &msg, FI_COMPLETION | FI_DELIVERY_COMPLETE);
        }

        if (ret) {
          fprintf(stderr, "Rank 0: RDMA write failed: %s (%d)\n",
                  fi_strerror(-ret), ret);
          continue;
        }

        // Poll for completion - aggressive polling
        struct fi_cq_data_entry cqe;
        ssize_t rc;
        do {
          rc = fi_cq_read(cq, &cqe, 1);
        } while (rc == -FI_EAGAIN);

        // === Computation Phase 2: Post-communication ===
        hipLaunchKernelGGL(gpu_compute_saxpy,
                           dim3(num_blocks), dim3(threads_per_block), 0, 0,
                           d_compute_x, d_compute_y, compute_a, compute_n);
        CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize compute2");

        double end_time = get_time_us();

        // Check for errors
        if (rc < 0) {
          if (rc == -FI_EAVAIL) {
            struct fi_cq_err_entry err_entry;
            int err_ret = fi_cq_readerr(cq, &err_entry, 0);
            if (err_ret > 0) {
              fprintf(stderr, "Rank 0: CQ error: err=%d (%s), prov_errno=%d\n",
                      err_entry.err, fi_strerror(err_entry.err),
                      err_entry.prov_errno);
            }
          }
          fprintf(stderr, "Rank 0: Iteration %d failed\n", iter);
        } else {
          // Store this iteration's time
          iteration_times[successful_iterations] = (end_time - start_time);
          successful_iterations++;
        }
      }
    }

    // Print results for this size
    if (myrank == 0 && successful_iterations > 0) {
      // Sort iteration times in ascending order
      for (int i = 0; i < successful_iterations - 1; i++) {
        for (int j = i + 1; j < successful_iterations; j++) {
          if (iteration_times[j] < iteration_times[i]) {
            double temp = iteration_times[i];
            iteration_times[i] = iteration_times[j];
            iteration_times[j] = temp;
          }
        }
      }

      // Select top 10 fastest iterations (or all if less than 10)
      int samples_to_average = (successful_iterations < 10) ? successful_iterations : 10;
      double sum_best = 0.0;
      for (int i = 0; i < samples_to_average; i++) {
        sum_best += iteration_times[i];
      }
      double avg_best_latency = sum_best / samples_to_average;

      char size_str[32];
      format_size(current_size, size_str);
      printf("%-10s  %12.2f  (best %d/%d: min=%.2f max=%.2f)\n",
             size_str,
             avg_best_latency,
             samples_to_average,
             successful_iterations,
             iteration_times[0],
             iteration_times[samples_to_average - 1]);
    }
  }

  if (myrank == 0) {
    printf("========================================================================\n\n");
  }

  // =============================================================================
  // STEP 12: Final Synchronization
  // =============================================================================
  CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(final)");

  // =============================================================================
  // STEP 13: Cleanup Resources
  // =============================================================================

  fi_close(&mr_local->fid);
  fi_close(&mr_remote->fid);

  CHECK_HIP(hipFree(d_local_buf), "hipFree(local)");
  CHECK_HIP(hipFree(d_remote_buf), "hipFree(remote)");
  CHECK_HIP(hipFree(d_compute_x), "hipFree(compute_x)");
  CHECK_HIP(hipFree(d_compute_y), "hipFree(compute_y)");
  free(h_temp);

  free(local_addr);
  free(my_hex);
  free(all_hex);
  free(all_bin);

  fi_close(&ep->fid);
  fi_close(&av->fid);
  fi_close(&cq->fid);
  fi_close(&domain->fid);
  fi_close(&fabric->fid);
  fi_freeinfo(info);

  PMI2_Finalize();
  return 0;
}
