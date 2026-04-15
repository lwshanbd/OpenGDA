/*
 * rank-amdgpu-gda.cpp - GPU-Direct Async (GDA) RDMA using Deferred Work Queue
 *
 * This program implements GPU-Direct Async (GDA) using libfabric's
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
 
 // Test parameters
 #define NUM_ITERATIONS 20 // Single iteration for data verification test
 const size_t test_sizes[] = {
   // 1 * 1024, // 1KB
   // 2 * 1024, // 2KB
   // 4 * 1024, // 4KB
   // 8 * 1024, // 8KB
   16 * 1024, // 16KB
   32 * 1024, // 32KB
   64 * 1024, // 64KB
   128 * 1024, // 128KB
   256 * 1024, // 256KB
   512 * 1024, // 512KB
   1024 * 1024, // 1MB
   2 * 1024 * 1024, // 2MB
   4 * 1024 * 1024, // 4MB
   8 * 1024 * 1024, // 8MB
   16 * 1024 * 1024, // 16MB
 };
 const int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
 
 // =============================================================================
 // GPU Doorbell Kernel
 // =============================================================================

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

 // GPU kernel to write counter doorbell (trigger NIC operation)
 // Polls atomic_result until NIC completes the atomic write
 // Records timing inside GPU for accurate latency measurement
 // Now includes computation before and after communication
 __global__ void gpu_write_counter_doorbell(volatile uint64_t *counter_addr,
                                            volatile uint64_t *atomic_result,
                                            uint64_t value,
                                            uint64_t *start_clock,
                                            uint64_t *end_clock,
                                            float *compute_x,
                                            float *compute_y,
                                            float compute_a,
                                            int compute_n) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   // Record start time (GPU clock cycles) - thread 0 only
   if (threadIdx.x == 0 && blockIdx.x == 0) {
     uint64_t t_start = clock64();
     if (start_clock) *start_clock = t_start;
   }
   __syncthreads();

   // === Computation Phase 1: Pre-communication ===
   if (idx < compute_n) {
     #pragma unroll 1
     for (int iter = 0; iter < 100; iter++) {
       compute_y[idx] = compute_a * compute_x[idx] + compute_y[idx];
     }
   }
   __syncthreads();

   // === Communication Phase: Only thread 0 handles this ===
   if (threadIdx.x == 0 && blockIdx.x == 0) {
     // Step 1: Write to trigger counter to initiate RDMA write
     *counter_addr = value;

     // Step 2: Poll atomic_result until NIC writes to it
     do {
       // __threadfence_system();
     } while (*atomic_result < 1);
   }
   __syncthreads();

   // === Computation Phase 2: Post-communication ===
   if (idx < compute_n) {
     #pragma unroll 1
     for (int iter = 0; iter < 100; iter++) {
       compute_y[idx] = compute_a * compute_x[idx] + compute_y[idx];
     }
   }
   __syncthreads();

   // Record end time (GPU clock cycles) - thread 0 only
   if (threadIdx.x == 0 && blockIdx.x == 0) {
     __threadfence_system();
     uint64_t t_end = clock64();
     if (end_clock) *end_clock = t_end;
   }
   __syncthreads();
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
   setenv("FI_CXI_DEVICE_NAME", "cxi3", 1);
   int device_id = 7;
   CHECK_HIP(hipSetDevice(device_id), "hipSetDevice");
 
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
   int gpu_id = 7;
   CHECK_HIP(hipSetDevice(gpu_id), "hipSetDevice");
 
   hipDeviceProp_t prop;
   CHECK_HIP(hipGetDeviceProperties(&prop, gpu_id), "hipGetDeviceProperties");
 
   // Get GPU clock rate for time conversion (clock64() returns cycles)
   double gpu_clock_mhz = prop.clockRate / 1000.0; // Convert kHz to MHz
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
 
   // Triggering counter - GPU will write to this to trigger operations
   struct fid_cntr *trigger_cntr = NULL;
   struct fi_cntr_attr cntr_attr = {};
   cntr_attr.events = FI_CNTR_EVENTS_COMP;
   CHECK(fi_cntr_open(domain, &cntr_attr, &trigger_cntr, NULL),
         "fi_cntr_open(trigger)");
 
   // Completion counter - NIC will update this when operations complete
   struct fid_cntr *completion_cntr = NULL;
   struct fi_cntr_attr completion_cntr_attr = {};
   completion_cntr_attr.events = FI_CNTR_EVENTS_COMP;
   completion_cntr_attr.wait_obj = FI_WAIT_UNSPEC;
   CHECK(fi_cntr_open(domain, &completion_cntr_attr, &completion_cntr, NULL),
         "fi_cntr_open(completion)");
 
   // Atomic completion counter - tracks when atomic operation completes
   struct fid_cntr *atomic_completion_cntr = NULL;
   CHECK(fi_cntr_open(domain, &cntr_attr, &atomic_completion_cntr, NULL),
         "fi_cntr_open(atomic_completion)");
 
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
   void *completion_mmio_addr = NULL;
   size_t completion_mmio_len = 0;
   CHECK(completion_cntr_ops->get_mmio_addr(
             &completion_cntr->fid, &completion_mmio_addr, &completion_mmio_len),
         "get_mmio_addr(completion)");
 
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
 
   // Insert local address into AV for atomic operations to self
   fi_addr_t local_fi_addr = FI_ADDR_NOTAVAIL;
   inserted = fi_av_insert(av, local_addr, 1, &local_fi_addr, 0, NULL);
   if (inserted != 1 || local_fi_addr == FI_ADDR_NOTAVAIL) {
     fprintf(stderr, "Rank %d: fi_av_insert(local) failed\n", myrank);
     PMI2_Finalize();
     exit(1);
   }
 
   // -------------------------------------------------------------------------
   // STEP 7: Allocate and Register GPU Buffers
   // -------------------------------------------------------------------------
   size_t max_size = 128 * 1024 * 1024; // 128MB
 
   void *d_local_buf = NULL;
   CHECK_HIP(hipMalloc(&d_local_buf, max_size), "hipMalloc(local)");
 
   void *d_remote_buf = NULL;
   CHECK_HIP(hipMalloc(&d_remote_buf, max_size), "hipMalloc(remote)");
 
   // Allocate host verification buffer
   uint8_t *h_verify_buf = (uint8_t *)malloc(max_size);
   if (!h_verify_buf) {
     fprintf(stderr, "Rank %d: malloc(h_verify_buf) failed\n", myrank);
     PMI2_Finalize();
     exit(1);
   }
 
   // allocate memory for atomic operation result and source operand
   uint64_t *atomic_result = NULL;
   CHECK_HIP(hipMalloc(&atomic_result, sizeof(uint64_t)), "hipMalloc(atomic)");
   CHECK_HIP(hipMemset(atomic_result, 0, sizeof(uint64_t)), "hipMemset(atomic)");
 
   // Allocate source operand for atomic operation (value to add)
   uint64_t *atomic_operand = NULL;
   CHECK_HIP(hipMalloc(&atomic_operand, sizeof(uint64_t)), "hipMalloc(atomic_operand)");
   uint64_t operand_value = 1; // Value to add
   CHECK_HIP(hipMemcpy(atomic_operand, &operand_value, sizeof(uint64_t), hipMemcpyHostToDevice),
             "hipMemcpy(atomic_operand)");
 
   // Allocate timestamp buffers for GPU timing
   uint64_t *d_start_clock = NULL;
   uint64_t *d_end_clock = NULL;
   CHECK_HIP(hipMalloc(&d_start_clock, sizeof(uint64_t)), "hipMalloc(start_clock)");
   CHECK_HIP(hipMalloc(&d_end_clock, sizeof(uint64_t)), "hipMalloc(end_clock)");
 
   CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize after memset");
 

  // Allocate computation buffers (proportional to max transfer size)
  const size_t max_compute_n = max_size / sizeof(float);
  const size_t max_compute_bytes = max_compute_n * sizeof(float);
  float *d_compute_x = NULL;
  float *d_compute_y = NULL;
  CHECK_HIP(hipMalloc(&d_compute_x, max_compute_bytes), "hipMalloc(compute_x)");
  CHECK_HIP(hipMalloc(&d_compute_y, max_compute_bytes), "hipMalloc(compute_y)");

  // Initialize computation buffers with some data
  CHECK_HIP(hipMemset(d_compute_x, 1, max_compute_bytes), "hipMemset(compute_x)");
  CHECK_HIP(hipMemset(d_compute_y, 0, max_compute_bytes), "hipMemset(compute_y)");

   struct fid_mr *mr_local = NULL;
   CHECK(register_memory_region(domain, ep, cxi_info, d_local_buf, max_size,
                                &mr_local, true, device_id),
         "register_memory_region(local)");
 
   struct fid_mr *mr_remote = NULL;
   CHECK(register_memory_region(domain, ep, cxi_info, d_remote_buf, max_size,
                                &mr_remote, true, device_id),
         "register_memory_region(remote)");
 
   struct fid_mr *mr_atomic = NULL;
   CHECK(register_memory_region(domain, ep, cxi_info, atomic_result, sizeof(uint64_t),
                                &mr_atomic, true, device_id),
         "register_memory_region(atomic)");
 
   struct fid_mr *mr_atomic_operand = NULL;
   CHECK(register_memory_region(domain, ep, cxi_info, atomic_operand, sizeof(uint64_t),
                                &mr_atomic_operand, true, device_id),
         "register_memory_region(atomic_operand)");
 
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
     remote_addr_for_rma = 0; // Use offset from MR base (CXI case)
   }
 
   // -------------------------------------------------------------------------
   // STEP 8: Sync before benchmark
   // -------------------------------------------------------------------------
   CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(pre-benchmark)");
   usleep(100000); // 100ms for readiness
 
   // -------------------------------------------------------------------------
   // STEP 9: DWQ Benchmark Loop
   // -------------------------------------------------------------------------
 
   // Note: Skip DWQ support check - will detect on first fi_control call
 
   if (myrank == 0) {
     printf("%-8s  %12s  %s\n", "Size", "Latency(us)", "Statistics");
     printf("========  ============  ===============================================\n");
     printf("Note: Latency is average of best 10/%d iterations\n\n", NUM_ITERATIONS);
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
 
   // DWQ Work, NIC to GPU completion signaling
     struct fi_op_atomic atomic;
     struct fi_msg_atomic atomic_msg;
     struct fi_ioc atomic_iov;
     struct fi_rma_ioc atomic_rma_iov;
   struct fi_deferred_work atomic_work;
 
   int total_verifications = 0;
   int total_verification_failures = 0;
 
   for (int size_idx = 0; size_idx < num_test_sizes; size_idx++) {
     size_t current_size = test_sizes[size_idx];
     CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence"); // sync before each size
 
     // Store all iteration times for statistical analysis
     double iteration_times[NUM_ITERATIONS];
     int successful_iterations = 0;
 
     for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
       fi_cntr_set(trigger_cntr, 0);
       fi_cntr_set(completion_cntr, 0);
       fi_cntr_set(atomic_completion_cntr, 0);
 
       // CRITICAL: Reset atomic_result to 0 before each iteration
       // GPU kernel polls this value, so it must start at 0 each time
       if (myrank == 0) {
         uint64_t zero = 0;
         CHECK_HIP(hipMemcpy(atomic_result, &zero, sizeof(uint64_t),
                             hipMemcpyHostToDevice),
                   "hipMemcpy reset atomic_result");
         CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize reset");
       }
 
       // Initialize buffers with iteration-specific patterns for verification
       if (myrank == 0) {
         // Rank 0: Fill send buffer with pattern (iter + 0xA0)
         uint8_t pattern = (iter + 0xA0) & 0xFF;
         CHECK_HIP(hipMemset(d_local_buf, pattern, current_size),
                   "hipMemset(local)");
         CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");
       } else {
         // Rank 1: Clear receive buffer with different pattern (0xFF)
         CHECK_HIP(hipMemset(d_remote_buf, 0xFF, current_size),
                   "hipMemset(remote)");
         CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");
       }
       CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence");
 
       if (myrank == 0) {
         
         // WORK 1: RDMA write to peer
         iov->iov_base = d_local_buf;
         iov->iov_len = current_size;
 
         rma_iov->addr = remote_addr_for_rma;
         rma_iov->len = current_size;
         rma_iov->key = peer_remote_key;
 
         memset(msg_rma, 0, sizeof(*msg_rma));
         msg_rma->msg_iov = iov;
         msg_rma->desc = &desc_local; // Include descriptor for GPU memory
         msg_rma->iov_count = 1;
         msg_rma->addr = peer_fi_addr;
         msg_rma->rma_iov = rma_iov;
         msg_rma->rma_iov_count = 1;
         msg_rma->context = NULL; // Not needed for DWQ
         msg_rma->data = 0;
 
         memset(op_rma, 0, sizeof(*op_rma));
         op_rma->ep = ep;
         op_rma->msg = *msg_rma;
         // Use FI_CXI_CNTR_WB to ensure counter writeback
         op_rma->flags = FI_COMPLETION | FI_CXI_CNTR_WB;
 
         work.triggering_cntr = trigger_cntr;    // Counter GPU will write to
         work.completion_cntr = completion_cntr; // Counter NIC will increment
         work.threshold = 1;                     // Trigger when counter >= 1
         work.op_type = FI_OP_WRITE;             // RMA write operation
         work.op.rma = op_rma;
 
         int ret = fi_control(&domain->fid, FI_QUEUE_WORK, &work);
 
         // WORK 2: NIC atomic to signal GPU completion
         // This atomic operation will be triggered when the RMA write completes
         // and increments the completion counter. It writes to atomic_result
         // which the GPU kernel is polling.
 
         void *desc_atomic_operand = fi_mr_desc(mr_atomic_operand);
         void *desc_atomic_result = fi_mr_desc(mr_atomic);
 
         // Setup source operand (value to add)
         atomic_iov.addr = atomic_operand;
         atomic_iov.count = 1; // Number of elements, not bytes
 
         // Setup destination (atomic_result on GPU)
         uint64_t atomic_result_addr = (uint64_t)atomic_result;
         if (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
           atomic_rma_iov.addr = atomic_result_addr;
         } else {
           atomic_rma_iov.addr = 0; // Offset from MR base
         }
         atomic_rma_iov.count = 1; // Number of elements, not bytes
         atomic_rma_iov.key = fi_mr_key(mr_atomic);
 
         // Setup atomic message
         atomic_msg.msg_iov = &atomic_iov;
         atomic_msg.desc = &desc_atomic_operand;
         atomic_msg.iov_count = 1;
         atomic_msg.addr = local_fi_addr; // Target is local EP
         atomic_msg.rma_iov = &atomic_rma_iov;
         atomic_msg.rma_iov_count = 1;
         atomic_msg.datatype = FI_UINT64;
         atomic_msg.op = FI_SUM; // Atomic add operation
         atomic_msg.context = NULL;
         atomic_msg.data = 0; // Not used for atomic operations
 
         // Setup atomic work structure
         atomic.ep = ep;
         atomic.msg = atomic_msg;
         atomic.flags = FI_COMPLETION; // Need this to increment completion counter!
 
         atomic_work.op_type = FI_OP_ATOMIC;
         atomic_work.op.atomic = &atomic;
         // Triggered when RMA completion counter reaches threshold
         atomic_work.triggering_cntr = completion_cntr;
         atomic_work.completion_cntr = atomic_completion_cntr; // Track atomic completion
         atomic_work.threshold = 1;
 
         ret = fi_control(&domain->fid, FI_QUEUE_WORK, &atomic_work);
 
         if (ret) {
           fprintf(stderr, "Rank %d: fi_control(atomic) failed: %s (%d)\n",
                   myrank, fi_strerror(-ret), ret);
           PMI2_Finalize();
           exit(1);
         }
 
        // Calculate computation parameters (proportional to transfer size)
        const int compute_n = current_size / sizeof(float);
        const int threads_per_block = 256;
        const int num_blocks = (compute_n + threads_per_block - 1) / threads_per_block;
        const float compute_a = 2.5f;

         // GPU writes trigger counter and waits for atomic_result
         // Timing is done inside GPU kernel using clock64()
        hipLaunchKernelGGL(gpu_write_counter_doorbell,
                           dim3(num_blocks), dim3(threads_per_block), 0, 0,
                           dev_trigger_cntr, atomic_result, work.threshold,
                           d_start_clock, d_end_clock,
                           d_compute_x, d_compute_y, compute_a, compute_n);
         CHECK_HIP(hipDeviceSynchronize(), "hipDeviceSynchronize");
 
         // Read GPU timestamps
         uint64_t start_clock, end_clock;
         CHECK_HIP(hipMemcpy(&start_clock, d_start_clock, sizeof(uint64_t),
                             hipMemcpyDeviceToHost),
                   "hipMemcpy read start_clock");
         CHECK_HIP(hipMemcpy(&end_clock, d_end_clock, sizeof(uint64_t),
                             hipMemcpyDeviceToHost),
                   "hipMemcpy read end_clock");
 
         // Convert GPU cycles to microseconds
         double elapsed_us = (end_clock - start_clock) / gpu_clock_mhz;
 
         // Store this iteration's time
         iteration_times[successful_iterations] = elapsed_us;
         successful_iterations++;
 
       } else {
         // ---------- Rank 1: Receiver ----------
         // Wait for data to arrive (just barrier sync for now)
       }
 
       // Synchronize both ranks before verification
       CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence");
 
       // Rank 1: Verify received data
       if (myrank == 1) {
         // usleep(10000);
         // Copy data from GPU to host for verification
         CHECK_HIP(hipMemcpy(h_verify_buf, d_remote_buf, current_size,
                             hipMemcpyDeviceToHost),
                   "hipMemcpy D2H");
 
         // Verify the data matches the expected pattern
         uint8_t expected_pattern = (iter + 0xA0) & 0xFF;
         int errors = 0;
         int first_error_idx = -1;
         uint8_t first_error_val = 0;
 
         for (size_t i = 0; i < current_size; i++) {
           if (h_verify_buf[i] != expected_pattern) {
             if (first_error_idx == -1) {
               first_error_idx = i;
               first_error_val = h_verify_buf[i];
             }
             errors++;
             if (errors >= 10)
               break; // Limit error counting
           }
         }
 
         total_verifications++;
 
         if (errors > 0) {
           total_verification_failures++;
           fprintf(stderr,
                   "Rank %d: VERIFICATION FAILED - iter=%d, size=%zu, "
                   "expected=0x%02X, errors=%d\n",
                   myrank, iter, current_size, expected_pattern, errors);
         }
       }
     }
 
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
 
       char size_buf[32];
       printf("%-8s  %12.2f  (best %d/%d: min=%.2f max=%.2f)\n",
              format_size(current_size, size_buf),
              avg_best_latency,
              samples_to_average,
              successful_iterations,
              iteration_times[0],
              iteration_times[samples_to_average - 1]);
     }
     fi_deferred_work flush_work = {};
     ret = fi_control(&domain->fid, FI_FLUSH_WORK, NULL);
     if (ret) {
       fprintf(stderr, "Rank %d: FI_FLUSH_WORK failed: %s (%d)\n",
               myrank, fi_strerror(-ret), ret);
       PMI2_Finalize();
       exit(1);
     }
   }
 
   // Free persistent DWQ structures
   free(rma_iov);
   free(iov);
   free(msg_rma);
   free(op_rma);
 
   // -------------------------------------------------------------------------
   // Final Verification Report
   // -------------------------------------------------------------------------
   if (myrank == 1 && total_verification_failures > 0) {
     fprintf(stderr, "\nVerification: %d/%d failed\n",
             total_verification_failures, total_verifications);
   }
 
   // -------------------------------------------------------------------------
   // STEP 10: Cleanup
   // -------------------------------------------------------------------------
 
   hipHostUnregister(trigger_mmio_addr);
   hipHostUnregister(completion_mmio_addr);
   hipFree(d_local_buf);
   hipFree(d_remote_buf);
   hipFree(atomic_result);
   hipFree(atomic_operand);
   hipFree(d_start_clock);
   hipFree(d_end_clock);
  hipFree(d_compute_x);
  hipFree(d_compute_y);
   free(h_verify_buf);
 
   fi_close(&mr_local->fid);
   fi_close(&mr_remote->fid);
   fi_close(&mr_atomic->fid);
   fi_close(&mr_atomic_operand->fid);
   fi_close(&trigger_cntr->fid);
   fi_close(&completion_cntr->fid);
   fi_close(&atomic_completion_cntr->fid);
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
 