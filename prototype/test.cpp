#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_trigger.h>
#include <rdma/fi_cxi_ext.h>
#include <rdma/fi_cm.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/uio.h>

#define CHECK(x, msg) do { \
    int ret = (x); \
    if (ret) { \
        fprintf(stderr, "%s failed: %s (%d)\n", msg, fi_strerror(-ret), ret); \
        exit(1); \
    } \
} while(0)

/* ========== GPU Kernel Functions ========== */

/**
 * GPU kernel: Write to counter doorbell
 *
 * This kernel demonstrates how GPU directly writes to CXI counter MMIO address
 * to trigger network operations, implementing GPU-initiated communication.
 *
 * Important: Counter doorbell performs ADD operation (accumulation), not SET
 * The written value is added to the current counter value.
 *
 * @param counter_addr: GPU-accessible counter MMIO address
 * @param value: Increment value to write (will be accumulated to counter)
 */
__global__ void gpu_write_counter_doorbell(volatile uint64_t* counter_addr, uint64_t value) {
    // Only use the first thread to execute the write
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Method 1: Direct volatile write (recommended for MMIO)
        *counter_addr = value;

        // Ensure write operation completes and is visible to system
        __threadfence_system();

        // Additional memory barrier to ensure write reaches NIC
        __syncthreads();
    }
}

/**
 * GPU kernel: Execute computation
 *
 * Pure computation kernel, processes data transformation
 *
 * @param data: Data buffer
 * @param size: Data size
 */
__global__ void gpu_compute(uint64_t* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Parallel computation: each thread processes a portion of data
    if (idx < size) {
        data[idx] = data[idx] * 2 + 1;  // Example computation
    }
}

int main() {
    hipSetDevice(7);
    struct fi_info *info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_ep *ep;
    struct fid_av *av;
    struct fid_cq *cq;
    struct fid_cntr *cntr;
    struct fi_cxi_cntr_ops *cntr_ops;
    struct fi_cntr_attr cntr_attr;
    void *mmio_addr = NULL;
    size_t mmio_len = 0;
    void *dev_cntr = NULL;
    void *local_buf = NULL;
    hipStream_t stream;

    memset(&cntr_attr, 0, sizeof(cntr_attr));

    /* ---------- 1. Get provider information ---------- */
    // Use NULL hints to get all available providers
    int ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL, NULL, 0, NULL, &info);
    if (ret) {
        fprintf(stderr, "fi_getinfo failed: %s (%d)\n", fi_strerror(-ret), ret);
        exit(1);
    }

    // Iterate to find CXI provider
    struct fi_info *cxi_info = NULL;
    for (struct fi_info *cur = info; cur; cur = cur->next) {
        const char *prov_name = cur->fabric_attr ? cur->fabric_attr->prov_name : NULL;
        if (prov_name && strcmp(prov_name, "cxi") == 0 && !cxi_info) {
            cxi_info = cur;
            break;
        }
    }

    if (!cxi_info) {
        fprintf(stderr, "CXI provider not found!\n");
        fi_freeinfo(info);
        exit(1);
    }

    printf("Using CXI provider: %s\n", cxi_info->domain_attr->name);

    CHECK(fi_fabric(cxi_info->fabric_attr, &fabric, NULL), "fi_fabric");
    CHECK(fi_domain(fabric, cxi_info, &domain, NULL), "fi_domain");
    CHECK(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");

    // Create address vector
    struct fi_av_attr av_attr;
    memset(&av_attr, 0, sizeof(av_attr));
    av_attr.type = FI_AV_MAP;
    CHECK(fi_av_open(domain, &av_attr, &av, NULL), "fi_av_open");
    CHECK(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");

    // Create completion queue
    struct fi_cq_attr cq_attr;
    memset(&cq_attr, 0, sizeof(cq_attr));
    cq_attr.format = FI_CQ_FORMAT_MSG;
    cq_attr.size = 128;
    CHECK(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");
    CHECK(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");

    /* ---------- 2. Create and map counter ---------- */
    CHECK(fi_cntr_open(domain, &cntr_attr, &cntr, NULL), "fi_cntr_open");
    CHECK(fi_ep_bind(ep, &cntr->fid, FI_SEND | FI_RECV), "fi_ep_bind(counter)");
    CHECK(fi_enable(ep), "fi_enable");

    CHECK(fi_open_ops(&cntr->fid, FI_CXI_COUNTER_OPS, 0, (void**)&cntr_ops, NULL), "fi_open_ops");
    if (!cntr_ops) {
        fprintf(stderr, "cntr_ops is NULL!\n");
        exit(1);
    }

    CHECK(cntr_ops->get_mmio_addr(&cntr->fid, &mmio_addr, &mmio_len), "get_mmio_addr");
    printf("MMIO counter addr = %p, len = %zu\n", mmio_addr, mmio_len);

    CHECK(hipHostRegister(mmio_addr, mmio_len, hipHostRegisterMapped), "hipHostRegister");
    CHECK(hipHostGetDevicePointer(&dev_cntr, mmio_addr, 0), "hipHostGetDevicePointer");
    CHECK(hipStreamCreate(&stream), "hipStreamCreate");

    /* ---------- 3. Prepare GPU-Initiated communication ---------- */

    // Allocate local buffer
    local_buf = malloc(8);
    if (!local_buf) {
        fprintf(stderr, "Failed to allocate local buffer\n");
        exit(1);
    }
    uint64_t test_data = 0xDEADBEEFCAFEBABE;
    memcpy(local_buf, &test_data, 8);
    printf("Local buffer allocated: %p, data: 0x%lx\n", local_buf, test_data);

    // For demonstration, we will simulate the entire flow
    // In actual use, you need:
    // 1. Remote node address and memory registration key
    // 2. Add remote address via fi_av_insert
    //
    // Here we mainly demonstrate GPU writing to counter doorbell mechanism

    printf("Counter MMIO address: %p\n", mmio_addr);
    printf("GPU accessible address: %p\n", dev_cntr);

    /* ---------- 4. GPU Kernel write doorbell ---------- */

    // Read initial counter value
    uint64_t initial_count = fi_cntr_read(cntr);
    printf("Initial counter value: %lu\n", initial_count);

    // Method 1: Simple GPU kernel write
    uint64_t trigger_val = 5;
    printf("\n[Method 1] Launching GPU kernel to write counter doorbell (value=%lu)...\n", trigger_val);

    // Launch GPU kernel: 1 block, 1 thread
    hipLaunchKernelGGL(gpu_write_counter_doorbell,
                       dim3(1),           // 1 block
                       dim3(1),           // 1 thread
                       0,                 // shared memory
                       stream,            // stream
                       (volatile uint64_t*)dev_cntr,  // counter address
                       trigger_val);      // value to write

    // Synchronize and wait for GPU kernel to complete
    CHECK(hipStreamSynchronize(stream), "hipStreamSynchronize");

    /* ---------- 5. Verify Counter update ---------- */

    // Wait for counter update
    uint64_t new_count;
    int timeout = 1000;  // 1000 attempts
    int attempts = 0;

    while (attempts < timeout) {
        new_count = fi_cntr_read(cntr);
        if (new_count != initial_count) {
            break;
        }
        attempts++;
        sched_yield();
    }

    printf("Counter updated after %d attempts\n", attempts);
    printf("  Initial value: %lu\n", initial_count);
    printf("  New value: %lu\n", new_count);
    printf("  Increment: %lu\n", new_count - initial_count);

    if (new_count == initial_count) {
        fprintf(stderr, "WARNING: Counter did not update!\n");
    } else {
        printf("SUCCESS: GPU kernel successfully triggered counter update!\n");
    }

    /* ---------- 5.5 Demonstrate GPU computation + communication ---------- */

    // Allocate GPU memory for computation
    const int data_size = 1024;
    uint64_t *d_data = NULL;
    CHECK(hipMalloc(&d_data, data_size * sizeof(uint64_t)), "hipMalloc");

    // Initialize data
    uint64_t *h_data = (uint64_t*)malloc(data_size * sizeof(uint64_t));
    for (int i = 0; i < data_size; i++) {
        h_data[i] = i;
    }
    CHECK(hipMemcpy(d_data, h_data, data_size * sizeof(uint64_t), hipMemcpyHostToDevice), "hipMemcpy H2D");

    uint64_t compute_trigger_val = trigger_val + 10;  // Increment value

    // Launch computation kernel
    int threads_per_block = 256;
    int num_blocks = (data_size + threads_per_block - 1) / threads_per_block;
    hipLaunchKernelGGL(gpu_compute,
                       dim3(num_blocks),
                       dim3(threads_per_block),
                       0,
                       stream,
                       d_data,
                       data_size);

    // Wait for computation to complete
    CHECK(hipStreamSynchronize(stream), "hipStreamSynchronize after compute");

    // Read current counter value
    uint64_t before_method2 = fi_cntr_read(cntr);

    // Launch counter doorbell write kernel
    hipLaunchKernelGGL(gpu_write_counter_doorbell,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       (volatile uint64_t*)dev_cntr,
                       compute_trigger_val);

    CHECK(hipStreamSynchronize(stream), "hipStreamSynchronize after counter write");

    // Verify computation results
    CHECK(hipMemcpy(h_data, d_data, data_size * sizeof(uint64_t), hipMemcpyDeviceToHost), "hipMemcpy D2H");
    bool compute_correct = true;
    for (int i = 0; i < 10; i++) {  // Only check first 10
        uint64_t expected = i * 2 + 1;
        if (h_data[i] != expected) {
            compute_correct = false;
            break;
        }
    }
    printf("  Computation: %s\n", compute_correct ? "CORRECT" : "FAILED");

    // Verify counter update (doorbell is ADD operation, not SET)
    uint64_t expected_final = before_method2 + compute_trigger_val;
    uint64_t after_method2 = fi_cntr_read(cntr);

    // Wait for counter update
    int wait_attempts = 0;
    while (wait_attempts < 1000 && after_method2 < expected_final) {
        after_method2 = fi_cntr_read(cntr);
        wait_attempts++;
        sched_yield();
    }

    if (after_method2 == expected_final && compute_correct) {
        printf("SUCCESS: GPU compute+trigger pattern works correctly!\n");
    } else {
        printf("WARNING: Test failed (counter: %lu/%lu, compute: %s)\n",
               after_method2, expected_final, compute_correct ? "OK" : "FAILED");
    }

    // Cleanup
    hipFree(d_data);
    free(h_data);

    /* ---------- 6. Cleanup resources ---------- */

    CHECK(hipHostUnregister(mmio_addr), "hipHostUnregister");
    CHECK(hipStreamDestroy(stream), "hipStreamDestroy");
    fi_close(&cntr->fid);
    fi_close(&cq->fid);
    fi_close(&av->fid);
    fi_close(&ep->fid);
    fi_close(&domain->fid);
    fi_close(&fabric->fid);
    fi_freeinfo(info);
    free(local_buf);

    return 0;
}