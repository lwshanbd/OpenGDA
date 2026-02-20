/*
 * limits_test.cpp - Test Counter and DWQ limits on CXI/libfabric
 *
 * This benchmark tests:
 * 1. Maximum number of counters that can be created
 * 2. Maximum number of counters with MMIO mapping (GPU-accessible)
 * 3. Maximum number of DWQ (Deferred Work Queue) operations that can be queued
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hip/hip_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_cxi_ext.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_trigger.h>

#include "hip_device_context.hpp"
#include "pmi_session.hpp"
#include "device_affinity.hpp"

// =============================================================================
// Counter Info Structure
// =============================================================================

struct CounterInfo {
    struct fid_cntr* cntr;
    struct fi_cxi_cntr_ops* ops;
    void* mmio_addr;
    size_t mmio_len;
    volatile uint64_t* dev_ptr;  // GPU device pointer
    bool has_mmio;
    bool has_gpu_mapping;
};

// =============================================================================
// DWQ Work Structure (for limit testing)
// =============================================================================

struct DwqWorkInfo {
    struct fi_deferred_work work;
    struct fi_op_rma op_rma;
    struct fi_msg_rma msg_rma;
    struct iovec iov;
    struct fi_rma_iov rma_iov;
    void* stored_desc;
};

// =============================================================================
// Test Functions
// =============================================================================

void test_counter_limits(struct fid_domain* domain, int rank) {
    printf("\n========================================\n");
    printf("Testing Counter Creation Limits\n");
    printf("========================================\n");

    std::vector<CounterInfo> counters;
    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;

    int max_counters = 3000;  // Try up to 3000 counters (limit is ~2047)
    int created = 0;

    for (int i = 0; i < max_counters; i++) {
        CounterInfo info = {};

        int ret = fi_cntr_open(domain, &cntr_attr, &info.cntr, NULL);
        if (ret != 0) {
            printf("Counter %d: fi_cntr_open failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            break;
        }

        counters.push_back(info);
        created++;

        // Print progress every 1000 counters
        if (created % 1000 == 0) {
            printf("  ... %d counters created so far\n", created);
        }
    }

    printf("Result: Created %d counters successfully\n", created);
    printf(">>> Counter creation limit: %d <<<\n", created);

    // Cleanup
    printf("Cleaning up %d counters...\n", created);
    for (auto& info : counters) {
        if (info.cntr) fi_close(&info.cntr->fid);
    }
    counters.clear();
    printf("Cleanup done.\n");
}

void test_counter_mmio_limits(struct fid_domain* domain, int rank) {
    printf("\n========================================\n");
    printf("Testing Counter MMIO Mapping Limits\n");
    printf("========================================\n");

    std::vector<CounterInfo> counters;
    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;

    int max_counters = 10000;  // Try up to 10000 MMIO mappings
    int created = 0;
    int mmio_success = 0;

    for (int i = 0; i < max_counters; i++) {
        CounterInfo info = {};

        // Create counter
        int ret = fi_cntr_open(domain, &cntr_attr, &info.cntr, NULL);
        if (ret != 0) {
            printf("Counter %d: fi_cntr_open failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            break;
        }
        created++;

        // Get counter ops
        ret = fi_open_ops(&info.cntr->fid, FI_CXI_COUNTER_OPS, 0,
                          (void**)&info.ops, NULL);
        if (ret != 0) {
            printf("Counter %d: fi_open_ops failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            counters.push_back(info);
            continue;
        }

        // Get MMIO address
        ret = info.ops->get_mmio_addr(&info.cntr->fid, &info.mmio_addr, &info.mmio_len);
        if (ret != 0) {
            printf("Counter %d: get_mmio_addr failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            counters.push_back(info);
            continue;
        }

        info.has_mmio = true;
        mmio_success++;
        counters.push_back(info);
    }

    printf("Result: Created %d counters, %d with MMIO mapping\n", created, mmio_success);

    // Cleanup
    for (auto& info : counters) {
        if (info.cntr) fi_close(&info.cntr->fid);
    }
}

void test_counter_gpu_mapping_limits(struct fid_domain* domain, int rank) {
    printf("\n========================================\n");
    printf("Testing Counter GPU Mapping Limits\n");
    printf("(This is the critical limit for DWQ)\n");
    printf("========================================\n");

    std::vector<CounterInfo> counters;
    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;

    int max_counters = 100;  // Test up to 100 GPU mappings (save resources for multi-rank tests)
    int created = 0;
    int mmio_success = 0;
    int gpu_success = 0;

    for (int i = 0; i < max_counters; i++) {
        CounterInfo info = {};

        // Create counter
        int ret = fi_cntr_open(domain, &cntr_attr, &info.cntr, NULL);
        if (ret != 0) {
            printf("Counter %d: fi_cntr_open failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            break;
        }
        created++;

        // Get counter ops
        ret = fi_open_ops(&info.cntr->fid, FI_CXI_COUNTER_OPS, 0,
                          (void**)&info.ops, NULL);
        if (ret != 0) {
            printf("Counter %d: fi_open_ops failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            counters.push_back(info);
            continue;
        }

        // Get MMIO address
        ret = info.ops->get_mmio_addr(&info.cntr->fid, &info.mmio_addr, &info.mmio_len);
        if (ret != 0) {
            printf("Counter %d: get_mmio_addr failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            counters.push_back(info);
            continue;
        }
        info.has_mmio = true;
        mmio_success++;

        // Map to GPU
        hipError_t err = hipHostRegister(info.mmio_addr, info.mmio_len, hipHostRegisterMapped);
        if (err != hipSuccess) {
            printf("Counter %d: hipHostRegister failed: %s (%d)\n",
                   i, hipGetErrorString(err), err);
            counters.push_back(info);
            continue;
        }

        // Get device pointer
        err = hipHostGetDevicePointer((void**)&info.dev_ptr, info.mmio_addr, 0);
        if (err != hipSuccess) {
            printf("Counter %d: hipHostGetDevicePointer failed: %s (%d)\n",
                   i, hipGetErrorString(err), err);
            (void)hipHostUnregister(info.mmio_addr);
            counters.push_back(info);
            continue;
        }

        info.has_gpu_mapping = true;
        gpu_success++;
        counters.push_back(info);

        // Print progress every 100 successful mappings
        if (gpu_success % 100 == 0) {
            printf("  ... %d GPU mappings successful so far\n", gpu_success);
        }
    }

    printf("\nResult Summary:\n");
    printf("  Counters created:      %d\n", created);
    printf("  MMIO mappings:         %d\n", mmio_success);
    printf("  GPU mappings:          %d\n", gpu_success);
    printf("\n  >>> GPU-accessible counter limit: %d <<<\n", gpu_success);

    // Cleanup (reverse order)
    for (auto it = counters.rbegin(); it != counters.rend(); ++it) {
        if (it->has_gpu_mapping && it->mmio_addr) {
            (void)hipHostUnregister(it->mmio_addr);
        }
        if (it->cntr) fi_close(&it->cntr->fid);
    }
}

void test_dwq_queue_limits(struct fid_domain* domain, struct fid_ep* ep,
                           struct fid_cntr* trigger_cntr, struct fid_cntr* completion_cntr,
                           void* src_buf, void* src_desc, size_t buf_size,
                           fi_addr_t dest_addr, uint64_t remote_addr, uint64_t remote_key,
                           int rank) {
    printf("\n========================================\n");
    printf("Testing DWQ Queue Limits\n");
    printf("========================================\n");

    std::vector<DwqWorkInfo*> works;

    int max_ops = 1000;  // Try up to 1000 DWQ operations
    int queued = 0;

    for (int i = 0; i < max_ops; i++) {
        DwqWorkInfo* info = new DwqWorkInfo();
        memset(info, 0, sizeof(DwqWorkInfo));

        // Setup iovec
        info->iov.iov_base = src_buf;
        info->iov.iov_len = buf_size;

        // Setup remote address
        info->rma_iov.addr = remote_addr;
        info->rma_iov.len = buf_size;
        info->rma_iov.key = remote_key;

        // Setup message
        info->stored_desc = src_desc;
        info->msg_rma.msg_iov = &info->iov;
        info->msg_rma.desc = &info->stored_desc;
        info->msg_rma.iov_count = 1;
        info->msg_rma.addr = dest_addr;
        info->msg_rma.rma_iov = &info->rma_iov;
        info->msg_rma.rma_iov_count = 1;
        info->msg_rma.context = NULL;
        info->msg_rma.data = 0;

        // Setup op_rma
        info->op_rma.ep = ep;
        info->op_rma.msg = info->msg_rma;
        info->op_rma.flags = FI_COMPLETION;

        // Setup deferred work
        info->work.triggering_cntr = trigger_cntr;
        info->work.completion_cntr = completion_cntr;
        info->work.threshold = i + 1;  // Each with different threshold
        info->work.op_type = FI_OP_WRITE;
        info->work.op.rma = &info->op_rma;

        int ret = fi_control(&domain->fid, FI_QUEUE_WORK, &info->work);
        if (ret != 0) {
            printf("DWQ op %d: fi_control(FI_QUEUE_WORK) failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            delete info;
            break;
        }

        works.push_back(info);
        queued++;

        // Print progress
        if (queued % 100 == 0) {
            printf("  ... %d DWQ operations queued so far\n", queued);
        }
    }

    printf("\nResult: Queued %d DWQ operations successfully\n", queued);
    printf(">>> DWQ queue limit: %d operations <<<\n", queued);

    // Flush to clean up
    printf("Flushing DWQ...\n");
    int ret = fi_control(&domain->fid, FI_FLUSH_WORK, NULL);
    if (ret != 0) {
        printf("Warning: FI_FLUSH_WORK failed: %s (%d)\n", fi_strerror(-ret), ret);
    }

    // Cleanup
    for (auto* info : works) {
        delete info;
    }
}

void test_dwq_per_counter_limits(struct fid_domain* domain, struct fid_ep* ep,
                                  void* src_buf, void* src_desc, size_t buf_size,
                                  fi_addr_t dest_addr, uint64_t remote_addr, uint64_t remote_key,
                                  int rank) {
    printf("\n========================================\n");
    printf("Testing DWQ per-Counter Limits\n");
    printf("(Multiple counters, one DWQ per counter)\n");
    printf("========================================\n");

    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;

    std::vector<struct fid_cntr*> trigger_cntrs;
    std::vector<struct fid_cntr*> completion_cntrs;
    std::vector<DwqWorkInfo*> works;

    int max_ops = 100;
    int queued = 0;

    for (int i = 0; i < max_ops; i++) {
        // Create unique counter pair for each DWQ
        struct fid_cntr* trigger = nullptr;
        struct fid_cntr* completion = nullptr;

        int ret = fi_cntr_open(domain, &cntr_attr, &trigger, NULL);
        if (ret != 0) {
            printf("Counter pair %d: trigger fi_cntr_open failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            break;
        }

        ret = fi_cntr_open(domain, &cntr_attr, &completion, NULL);
        if (ret != 0) {
            printf("Counter pair %d: completion fi_cntr_open failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            fi_close(&trigger->fid);
            break;
        }

        trigger_cntrs.push_back(trigger);
        completion_cntrs.push_back(completion);

        // Queue DWQ
        DwqWorkInfo* info = new DwqWorkInfo();
        memset(info, 0, sizeof(DwqWorkInfo));

        info->iov.iov_base = src_buf;
        info->iov.iov_len = buf_size;

        info->rma_iov.addr = remote_addr;
        info->rma_iov.len = buf_size;
        info->rma_iov.key = remote_key;

        info->stored_desc = src_desc;
        info->msg_rma.msg_iov = &info->iov;
        info->msg_rma.desc = &info->stored_desc;
        info->msg_rma.iov_count = 1;
        info->msg_rma.addr = dest_addr;
        info->msg_rma.rma_iov = &info->rma_iov;
        info->msg_rma.rma_iov_count = 1;

        info->op_rma.ep = ep;
        info->op_rma.msg = info->msg_rma;
        info->op_rma.flags = FI_COMPLETION;

        info->work.triggering_cntr = trigger;
        info->work.completion_cntr = completion;
        info->work.threshold = 1;
        info->work.op_type = FI_OP_WRITE;
        info->work.op.rma = &info->op_rma;

        ret = fi_control(&domain->fid, FI_QUEUE_WORK, &info->work);
        if (ret != 0) {
            printf("DWQ op %d: fi_control(FI_QUEUE_WORK) failed: %s (%d)\n",
                   i, fi_strerror(-ret), ret);
            delete info;
            break;
        }

        works.push_back(info);
        queued++;

        if (queued % 20 == 0) {
            printf("  ... %d DWQ operations (with unique counters) queued\n", queued);
        }
    }

    printf("\nResult: Queued %d DWQ ops with unique counter pairs\n", queued);

    // Flush
    fi_control(&domain->fid, FI_FLUSH_WORK, NULL);

    // Cleanup
    for (auto* info : works) delete info;
    for (auto* cntr : trigger_cntrs) fi_close(&cntr->fid);
    for (auto* cntr : completion_cntrs) fi_close(&cntr->fid);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    // Unset ROCR_VISIBLE_DEVICES
    unset_rocr_visible_devices();

    // Initialize affinity detector
    DeviceAffinityDetector affinity;

    // Initialize HIP
    HipDeviceContext hip(affinity.selected_gpu_id);

    // Initialize PMI
    PmiSession pmi;

    printf("========================================\n");
    printf("Counter and DWQ Limits Test\n");
    printf("========================================\n");
    printf("Rank: %d / %d\n", pmi.rank, pmi.size);
    printf("GPU: %d (%s)\n", hip.gpu_id, hip.prop.name);

    // All ranks run the test to see shared resource limits
    pmi.barrier();  // Sync before starting

    // =========================================================================
    // Initialize Fabric
    // =========================================================================

    struct fi_info* hints = fi_allocinfo();
    hints->caps = FI_RMA | FI_MSG | FI_HMEM;
    hints->mode = FI_CONTEXT2;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                  FI_MR_PROV_KEY | FI_MR_LOCAL | FI_MR_ENDPOINT;
    hints->domain_attr->threading = FI_THREAD_SAFE;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

    struct fi_info* info = nullptr;
    int ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                         NULL, NULL, 0, hints, &info);
    fi_freeinfo(hints);

    if (ret) {
        fprintf(stderr, "fi_getinfo failed: %s\n", fi_strerror(-ret));
        return 1;
    }

    // Find CXI provider
    struct fi_info* cxi_info = affinity.select_cxi_provider(info);
    if (!cxi_info) {
        fprintf(stderr, "CXI provider not found\n");
        return 1;
    }

    printf("CXI provider: %s\n", cxi_info->domain_attr->name);

    // Create fabric and domain
    struct fid_fabric* fabric = nullptr;
    struct fid_domain* domain = nullptr;
    struct fid_av* av = nullptr;
    struct fid_cq* cq = nullptr;
    struct fid_ep* ep = nullptr;

    ret = fi_fabric(cxi_info->fabric_attr, &fabric, NULL);
    if (ret) {
        fprintf(stderr, "fi_fabric failed: %s\n", fi_strerror(-ret));
        return 1;
    }

    ret = fi_domain(fabric, cxi_info, &domain, NULL);
    if (ret) {
        fprintf(stderr, "fi_domain failed: %s\n", fi_strerror(-ret));
        return 1;
    }

    // Create AV
    struct fi_av_attr av_attr = {};
    av_attr.type = FI_AV_TABLE;
    ret = fi_av_open(domain, &av_attr, &av, NULL);
    if (ret) {
        fprintf(stderr, "fi_av_open failed: %s\n", fi_strerror(-ret));
        return 1;
    }

    // Create CQ
    struct fi_cq_attr cq_attr = {};
    cq_attr.size = 128;
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    ret = fi_cq_open(domain, &cq_attr, &cq, NULL);
    if (ret) {
        fprintf(stderr, "fi_cq_open failed: %s\n", fi_strerror(-ret));
        return 1;
    }

    // Create and enable endpoint
    ret = fi_endpoint(domain, cxi_info, &ep, NULL);
    if (ret) {
        fprintf(stderr, "fi_endpoint failed: %s\n", fi_strerror(-ret));
        return 1;
    }
    fi_ep_bind(ep, &av->fid, 0);
    fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
    fi_enable(ep);

    // =========================================================================
    // Test 1: Counter GPU mapping limits (critical for DWQ) - test first!
    // =========================================================================
    test_counter_gpu_mapping_limits(domain, pmi.rank);

    // =========================================================================
    // Setup for DWQ tests
    // =========================================================================

    // Create counters for DWQ tests
    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;

    struct fid_cntr* trigger_cntr = nullptr;
    struct fid_cntr* completion_cntr = nullptr;
    fi_cntr_open(domain, &cntr_attr, &trigger_cntr, NULL);
    fi_cntr_open(domain, &cntr_attr, &completion_cntr, NULL);

    // Allocate GPU buffer for DWQ tests
    void* d_buf = nullptr;
    size_t buf_size = 4096;
    hipError_t hip_err = hipMalloc(&d_buf, buf_size);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    // Register memory
    struct fi_mr_attr mr_attr = {};
    struct iovec iov;
    iov.iov_base = d_buf;
    iov.iov_len = buf_size;
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE |
                     FI_REMOTE_READ | FI_REMOTE_WRITE;
    mr_attr.iface = FI_HMEM_ROCR;
    mr_attr.device.reserved = hip.gpu_id;

    struct fid_mr* mr = nullptr;
    ret = fi_mr_regattr(domain, &mr_attr, 0, &mr);
    if (ret) {
        fprintf(stderr, "fi_mr_regattr failed: %s\n", fi_strerror(-ret));
        return 1;
    }

    if (cxi_info->domain_attr->mr_mode & FI_MR_ENDPOINT) {
        fi_mr_bind(mr, &ep->fid, 0);
        fi_mr_enable(mr);
    }

    void* mr_desc = fi_mr_desc(mr);
    uint64_t mr_key = fi_mr_key(mr);

    // Insert self into AV for loopback
    size_t addrlen = 0;
    fi_getname(&ep->fid, NULL, &addrlen);
    void* local_addr = malloc(addrlen);
    fi_getname(&ep->fid, local_addr, &addrlen);

    fi_addr_t self_addr;
    fi_av_insert(av, local_addr, 1, &self_addr, 0, NULL);
    free(local_addr);

    uint64_t remote_addr = (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
        ? (uint64_t)d_buf : 0;

    // =========================================================================
    // Test 2: DWQ queue limits (single counter, many operations)
    // =========================================================================
    test_dwq_queue_limits(domain, ep, trigger_cntr, completion_cntr,
                          d_buf, mr_desc, buf_size,
                          self_addr, remote_addr, mr_key, pmi.rank);

    // =========================================================================
    // Test 3: DWQ per-counter limits
    // =========================================================================
    test_dwq_per_counter_limits(domain, ep,
                                 d_buf, mr_desc, buf_size,
                                 self_addr, remote_addr, mr_key, pmi.rank);

    // =========================================================================
    // Test 4: Counter MMIO limits (run after GPU mapping test)
    // =========================================================================
    test_counter_mmio_limits(domain, pmi.rank);

    // =========================================================================
    // Test 5: Counter creation limits (run last - exhausts resources)
    // =========================================================================
    test_counter_limits(domain, pmi.rank);

    // =========================================================================
    // Cleanup
    // =========================================================================
    printf("\nCleaning up...\n");

    if (trigger_cntr) fi_close(&trigger_cntr->fid);
    if (completion_cntr) fi_close(&completion_cntr->fid);
    if (mr) fi_close(&mr->fid);
    if (d_buf) (void)hipFree(d_buf);
    if (ep) fi_close(&ep->fid);
    if (cq) fi_close(&cq->fid);
    if (av) fi_close(&av->fid);
    if (domain) fi_close(&domain->fid);
    if (fabric) fi_close(&fabric->fid);
    if (info) fi_freeinfo(info);

    pmi.barrier();

    printf("\n========================================\n");
    printf("Rank %d: All tests completed!\n", pmi.rank);
    printf("========================================\n");

    pmi.barrier();

    return 0;
}
