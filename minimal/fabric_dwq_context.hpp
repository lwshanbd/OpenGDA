/*
 * fabric_dwq_context.hpp - Libfabric DWQ context with RAII
 */
#pragma once

#include <hip/hip_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_cxi_ext.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_trigger.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

class FabricDwqContext {
public:
    // Libfabric objects (order matters for cleanup)
    struct fi_info* info;
    struct fi_info* cxi_info;  // Points into info list
    struct fid_fabric* fabric;
    struct fid_domain* domain;
    struct fid_av* av;
    struct fid_cq* cq;
    struct fid_ep* ep;

    // Counters for DWQ
    struct fid_cntr* trigger_cntr;
    struct fid_cntr* completion_cntr;
    struct fid_cntr* atomic_completion_cntr;

    // Counter ops for MMIO access
    struct fi_cxi_cntr_ops* trigger_cntr_ops;
    struct fi_cxi_cntr_ops* completion_cntr_ops;

    // MMIO addresses (host)
    void* trigger_mmio_addr;
    size_t trigger_mmio_len;
    void* completion_mmio_addr;
    size_t completion_mmio_len;

    // GPU device pointers for counter doorbells
    volatile uint64_t* dev_trigger_cntr;
    volatile uint64_t* dev_completion_cntr;

    // Address info
    void* local_addr;
    size_t addrlen;
    fi_addr_t peer_addr;
    fi_addr_t local_addr_in_av;

    // For error messages
    int rank;

    FabricDwqContext(int rank_) :
        info(nullptr), cxi_info(nullptr), fabric(nullptr), domain(nullptr),
        av(nullptr), cq(nullptr), ep(nullptr),
        trigger_cntr(nullptr), completion_cntr(nullptr), atomic_completion_cntr(nullptr),
        trigger_cntr_ops(nullptr), completion_cntr_ops(nullptr),
        trigger_mmio_addr(nullptr), trigger_mmio_len(0),
        completion_mmio_addr(nullptr), completion_mmio_len(0),
        dev_trigger_cntr(nullptr), dev_completion_cntr(nullptr),
        local_addr(nullptr), addrlen(0),
        peer_addr(FI_ADDR_NOTAVAIL), local_addr_in_av(FI_ADDR_NOTAVAIL),
        rank(rank_)
    {
        init_fabric();
        init_counters();
        init_mmio_mapping();
        get_local_address();
    }

    ~FabricDwqContext() {
        // Unregister MMIO from GPU first (ignore errors in cleanup)
        if (trigger_mmio_addr) (void)hipHostUnregister(trigger_mmio_addr);
        if (completion_mmio_addr) (void)hipHostUnregister(completion_mmio_addr);

        // Close libfabric objects in reverse order
        if (trigger_cntr) fi_close(&trigger_cntr->fid);
        if (completion_cntr) fi_close(&completion_cntr->fid);
        if (atomic_completion_cntr) fi_close(&atomic_completion_cntr->fid);
        if (ep) fi_close(&ep->fid);
        if (cq) fi_close(&cq->fid);
        if (av) fi_close(&av->fid);
        if (domain) fi_close(&domain->fid);
        if (fabric) fi_close(&fabric->fid);
        if (info) fi_freeinfo(info);
        if (local_addr) free(local_addr);
    }

    // No copy/move
    FabricDwqContext(const FabricDwqContext&) = delete;
    FabricDwqContext& operator=(const FabricDwqContext&) = delete;

    void flush_dwq() {
        int ret = fi_control(&domain->fid, FI_FLUSH_WORK, NULL);
        if (ret) {
            fprintf(stderr, "Rank %d: FI_FLUSH_WORK failed: %s (%d)\n",
                    rank, fi_strerror(-ret), ret);
            exit(1);
        }
    }

    bool is_virt_addr_mode() const {
        return (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0;
    }

private:
    void check(int ret, const char* msg) {
        if (ret) {
            fprintf(stderr, "Rank %d: %s failed: %s (%d)\n",
                    rank, msg, fi_strerror(-ret), ret);
            exit(1);
        }
    }

    void check_hip(hipError_t err, const char* msg) {
        if (err != hipSuccess) {
            fprintf(stderr, "Rank %d: %s failed: %s (%d)\n",
                    rank, msg, hipGetErrorString(err), err);
            exit(1);
        }
    }

    void init_fabric() {
        // Setup hints
        struct fi_info* hints = fi_allocinfo();
        hints->caps = FI_RMA | FI_MSG | FI_HMEM;
        hints->mode = FI_CONTEXT2;  // DWQ requires FI_CONTEXT2
        hints->ep_attr->type = FI_EP_RDM;
        hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                      FI_MR_PROV_KEY | FI_MR_LOCAL | FI_MR_ENDPOINT;
        hints->domain_attr->threading = FI_THREAD_SAFE;
        hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
        hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

        int ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                             NULL, NULL, 0, hints, &info);
        fi_freeinfo(hints);

        if (ret) {
            fprintf(stderr, "Rank %d: fi_getinfo failed: %s (%d)\n",
                    rank, fi_strerror(-ret), ret);
            exit(1);
        }

        // Find CXI provider
        for (struct fi_info* cur = info; cur; cur = cur->next) {
            if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
                cxi_info = cur;
                printf("Rank %d: Using CXI provider %s\n", rank, cxi_info->domain_attr->name);
            }
        }

        if (!cxi_info) {
            fprintf(stderr, "Rank %d: CXI provider not found!\n", rank);
            exit(1);
        }

        // Create fabric and domain
        check(fi_fabric(cxi_info->fabric_attr, &fabric, NULL), "fi_fabric");
        check(fi_domain(fabric, cxi_info, &domain, NULL), "fi_domain");

        // Create AV
        struct fi_av_attr av_attr = {};
        av_attr.type = FI_AV_TABLE;
        check(fi_av_open(domain, &av_attr, &av, NULL), "fi_av_open");

        // Create CQ
        struct fi_cq_attr cq_attr = {};
        cq_attr.size = 128;
        cq_attr.format = FI_CQ_FORMAT_CONTEXT;
        check(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");

        // Create and enable endpoint
        check(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");
        check(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");
        check(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
        check(fi_enable(ep), "fi_enable");
    }

    void init_counters() {
        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;

        // Trigger counter
        check(fi_cntr_open(domain, &cntr_attr, &trigger_cntr, NULL),
              "fi_cntr_open(trigger)");

        // Completion counter
        struct fi_cntr_attr completion_cntr_attr = {};
        completion_cntr_attr.events = FI_CNTR_EVENTS_COMP;
        completion_cntr_attr.wait_obj = FI_WAIT_UNSPEC;
        check(fi_cntr_open(domain, &completion_cntr_attr, &completion_cntr, NULL),
              "fi_cntr_open(completion)");

        // Atomic completion counter
        check(fi_cntr_open(domain, &cntr_attr, &atomic_completion_cntr, NULL),
              "fi_cntr_open(atomic_completion)");

        // Get counter ops
        check(fi_open_ops(&trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                          (void**)&trigger_cntr_ops, NULL),
              "fi_open_ops(trigger)");
        check(fi_open_ops(&completion_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                          (void**)&completion_cntr_ops, NULL),
              "fi_open_ops(completion)");
    }

    void init_mmio_mapping() {
        // Get MMIO addresses
        check(trigger_cntr_ops->get_mmio_addr(&trigger_cntr->fid,
                                               &trigger_mmio_addr, &trigger_mmio_len),
              "get_mmio_addr(trigger)");
        check(completion_cntr_ops->get_mmio_addr(&completion_cntr->fid,
                                                  &completion_mmio_addr, &completion_mmio_len),
              "get_mmio_addr(completion)");

        // Map MMIO to GPU
        check_hip(hipHostRegister(trigger_mmio_addr, trigger_mmio_len,
                                  hipHostRegisterMapped),
                  "hipHostRegister(trigger)");
        check_hip(hipHostRegister(completion_mmio_addr, completion_mmio_len,
                                  hipHostRegisterMapped),
                  "hipHostRegister(completion)");

        // Get device pointers
        check_hip(hipHostGetDevicePointer((void**)&dev_trigger_cntr,
                                          trigger_mmio_addr, 0),
                  "hipHostGetDevicePointer(trigger)");
        check_hip(hipHostGetDevicePointer((void**)&dev_completion_cntr,
                                          completion_mmio_addr, 0),
                  "hipHostGetDevicePointer(completion)");
    }

    void get_local_address() {
        addrlen = 0;
        fi_getname(&ep->fid, NULL, &addrlen);
        local_addr = malloc(addrlen);
        check(fi_getname(&ep->fid, local_addr, &addrlen), "fi_getname");
    }

public:
    // Helper structure for additional counter pairs (used by concurrent benchmark)
    struct CounterPair {
        struct fid_cntr* trigger_cntr;
        struct fid_cntr* completion_cntr;
        struct fid_cntr* atomic_completion_cntr;
        struct fi_cxi_cntr_ops* trigger_ops;
        void* trigger_mmio_addr;
        size_t trigger_mmio_len;
        volatile uint64_t* dev_trigger_cntr;

        CounterPair() : trigger_cntr(nullptr), completion_cntr(nullptr),
                        atomic_completion_cntr(nullptr), trigger_ops(nullptr),
                        trigger_mmio_addr(nullptr), trigger_mmio_len(0),
                        dev_trigger_cntr(nullptr) {}
    };

    // Create additional counter pair with MMIO mapping for concurrent operations
    CounterPair create_counter_pair() {
        CounterPair cp;
        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;

        // Trigger counter
        check(fi_cntr_open(domain, &cntr_attr, &cp.trigger_cntr, NULL),
              "fi_cntr_open(trigger)");

        // Completion counter
        struct fi_cntr_attr completion_attr = {};
        completion_attr.events = FI_CNTR_EVENTS_COMP;
        completion_attr.wait_obj = FI_WAIT_UNSPEC;
        check(fi_cntr_open(domain, &completion_attr, &cp.completion_cntr, NULL),
              "fi_cntr_open(completion)");

        // Atomic completion counter
        check(fi_cntr_open(domain, &cntr_attr, &cp.atomic_completion_cntr, NULL),
              "fi_cntr_open(atomic_completion)");

        // Get counter ops and MMIO
        check(fi_open_ops(&cp.trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                          (void**)&cp.trigger_ops, NULL),
              "fi_open_ops(trigger)");
        check(cp.trigger_ops->get_mmio_addr(&cp.trigger_cntr->fid,
                                             &cp.trigger_mmio_addr, &cp.trigger_mmio_len),
              "get_mmio_addr(trigger)");

        // Map to GPU
        check_hip(hipHostRegister(cp.trigger_mmio_addr, cp.trigger_mmio_len,
                                  hipHostRegisterMapped),
                  "hipHostRegister(trigger)");
        check_hip(hipHostGetDevicePointer((void**)&cp.dev_trigger_cntr,
                                          cp.trigger_mmio_addr, 0),
                  "hipHostGetDevicePointer(trigger)");

        return cp;
    }

    // Cleanup counter pair (ignore errors in cleanup)
    void destroy_counter_pair(CounterPair& cp) {
        if (cp.trigger_mmio_addr) (void)hipHostUnregister(cp.trigger_mmio_addr);
        if (cp.trigger_cntr) fi_close(&cp.trigger_cntr->fid);
        if (cp.completion_cntr) fi_close(&cp.completion_cntr->fid);
        if (cp.atomic_completion_cntr) fi_close(&cp.atomic_completion_cntr->fid);
    }
};
