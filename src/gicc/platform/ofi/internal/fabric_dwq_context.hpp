/*
 * fabric_dwq_context.hpp - Libfabric DWQ context with RAII
 */
#pragma once

#include "gpu_device_context.hpp"
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
#include <unistd.h>
#include <thread>
#include <atomic>

#include "device_affinity.hpp"

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

    // Affinity detector for CXI selection (optional, can be nullptr)
    DeviceAffinityDetector* affinity_detector_;

    // Background CQ progress thread
    std::thread cq_progress_thread_;
    std::atomic<bool> cq_progress_stop_{false};

    FabricDwqContext(int rank_, DeviceAffinityDetector* affinity_detector = nullptr) :
        info(nullptr), cxi_info(nullptr), fabric(nullptr), domain(nullptr),
        av(nullptr), cq(nullptr), ep(nullptr),
        trigger_cntr(nullptr), completion_cntr(nullptr), atomic_completion_cntr(nullptr),
        trigger_cntr_ops(nullptr), completion_cntr_ops(nullptr),
        trigger_mmio_addr(nullptr), trigger_mmio_len(0),
        completion_mmio_addr(nullptr), completion_mmio_len(0),
        dev_trigger_cntr(nullptr), dev_completion_cntr(nullptr),
        local_addr(nullptr), addrlen(0),
        peer_addr(FI_ADDR_NOTAVAIL), local_addr_in_av(FI_ADDR_NOTAVAIL),
        rank(rank_),
        affinity_detector_(affinity_detector)
    {
        init_fabric();
        init_counters();
        // GICC_SKIP_DWQ_INIT=1 lets CPU-proxy-only callers bypass the
        // CXI MMIO -> GPU mapping entirely. The proxy path does not use
        // dev_trigger_cntr; DWQ-trigger callers must NOT set this.
        //
        // This used to be the only way to run on Grace Hopper, where the
        // mapping failed. That was a wrong registration flag, not a
        // hardware limit: the trigger BAR needs the I/O-memory flag (see
        // gpuHostRegisterMmio). Verified working on GH200 + Slingshot,
        // so the DWQ-trigger path is available there.
        if (std::getenv("GICC_SKIP_DWQ_INIT") == nullptr) {
            init_mmio_mapping();
        } else if (rank == 0) {
            fprintf(stderr,
                    "[gicc] GICC_SKIP_DWQ_INIT=1: skipping CXI trigger MMIO "
                    "mapping (DWQ-trigger path will not work on this "
                    "Runtime; CPU-proxy path is unaffected)\n");
        }
        get_local_address();
        // The background CQ progress thread drains the CQ so host-side
        // completion counters tick after a kernel fires the trigger. It is
        // only useful when nothing else polls: the domain hint requests
        // FI_THREAD_SAFE (init_fabric below), so its tight fi_cq_read loop
        // holds the per-domain mutex against everyone else. For pure
        // CPU-proxy mode that contends with the proxy thread's fi_write.
        //
        // It contends just as badly with a caller that drives progress
        // itself — Runtime::reset() spins fi_cq_read on this same CQ, so in
        // host-wait mode the two threads form a lock convoy whose severity
        // is pure scheduling luck. Measured on GH200 + Slingshot with a
        // 2-rank Jacobi halo exchange, identical binaries varied 68x run to
        // run (0.20 vs 13.6 ms/iter), with the stall moving between reset()
        // and the barrier depending on which rank lost the race. Set
        // GICC_DWQ_CQ_THREAD=0 to leave progress to the caller.
        const char* cq_thread_env = std::getenv("GICC_DWQ_CQ_THREAD");
        const bool want_cq_thread =
            (cq_thread_env == nullptr || std::atoi(cq_thread_env) != 0);
        if (std::getenv("GICC_SKIP_DWQ_INIT") == nullptr && want_cq_thread) {
            start_cq_progress_thread();
        } else if (rank == 0) {
            fprintf(stderr,
                    "[gicc] skipping background CQ progress thread (%s); "
                    "the caller must drain the CQ itself\n",
                    want_cq_thread ? "GICC_SKIP_DWQ_INIT=1"
                                   : "GICC_DWQ_CQ_THREAD=0");
        }
    }

    ~FabricDwqContext() {
        // Stop background CQ progress thread first
        stop_cq_progress_thread();

        // Unregister MMIO from GPU first (ignore errors in cleanup)
        if (trigger_mmio_addr) (void)gpuHostUnregister(trigger_mmio_addr);
        if (completion_mmio_addr) (void)gpuHostUnregister(completion_mmio_addr);

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

    // Start background CQ progress thread
    void start_cq_progress_thread() {
        cq_progress_stop_.store(false, std::memory_order_relaxed);
        cq_progress_thread_ = std::thread([this]() {
            while (!cq_progress_stop_.load(std::memory_order_relaxed)) {
                fi_cq_read(cq, NULL, 0);
            }
        });
    }

    // Stop background CQ progress thread
    void stop_cq_progress_thread() {
        cq_progress_stop_.store(true, std::memory_order_relaxed);
        if (cq_progress_thread_.joinable()) {
            cq_progress_thread_.join();
        }
    }

    // fast_flush is now a no-op since background thread handles CQ progress
    void fast_flush(uint64_t /*expected_completions*/) {
        // No-op: background thread continuously progresses CQ
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

    void check_gpu(GpuError err, const char* msg) {
        if (err != GPU_SUCCESS) {
            fprintf(stderr, "Rank %d: %s failed: %s (%d)\n",
                    rank, msg, gpuGetErrorString(err), err);
            exit(1);
        }
    }

    void init_fabric() {
        // Setup hints
        struct fi_info* hints = fi_allocinfo();
        hints->caps = FI_RMA | FI_MSG | FI_HMEM | FI_ATOMIC;
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

        // Find CXI provider - use affinity detector if available
        if (affinity_detector_) {
            cxi_info = affinity_detector_->select_cxi_provider(info);
        } else {
            // Fallback: use first CXI provider
            for (struct fi_info* cur = info; cur; cur = cur->next) {
                if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                    strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
                    cxi_info = cur;
                    break;
                }
            }
        }

        // if (cxi_info) {
        //     printf("Rank %d: Using CXI provider %s\n", rank, cxi_info->domain_attr->name);
        // } else {
        //     fprintf(stderr, "Rank %d: CXI provider not found!\n", rank);
        //     exit(1);
        // }

        // Create fabric and domain
        check(fi_fabric(cxi_info->fabric_attr, &fabric, NULL), "fi_fabric");
        check(fi_domain(fabric, cxi_info, &domain, NULL), "fi_domain");

        // Create AV
        struct fi_av_attr av_attr = {};
        av_attr.type = FI_AV_TABLE;
        check(fi_av_open(domain, &av_attr, &av, NULL), "fi_av_open");

        // Create CQ. Size from env GICC_CQ_SIZE (default 128, range [16, 16384]).
        struct fi_cq_attr cq_attr = {};
        cq_attr.size = 128;
        if (const char* env = std::getenv("GICC_CQ_SIZE")) {
            int n = std::atoi(env);
            if (n >= 16 && n <= 16384) {
                cq_attr.size = n;
            } else {
                fprintf(stderr, "GICC: GICC_CQ_SIZE=%s out of range [16,16384], using default 128\n", env);
            }
        }
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

        // Map MMIO to GPU. These are NIC BAR pages, not ordinary host memory,
        // so they need the I/O-memory registration flag (see
        // gpuHostRegisterMmio in gpu_device_context.hpp).
        check_gpu(gpuHostRegister(trigger_mmio_addr, trigger_mmio_len,
                                  gpuHostRegisterMmio),
                  "gpuHostRegister(trigger)");
        check_gpu(gpuHostRegister(completion_mmio_addr, completion_mmio_len,
                                  gpuHostRegisterMmio),
                  "gpuHostRegister(completion)");

        // Get device pointers
        check_gpu(gpuHostGetDevicePointer((void**)&dev_trigger_cntr,
                                          trigger_mmio_addr, 0),
                  "gpuHostGetDevicePointer(trigger)");
        check_gpu(gpuHostGetDevicePointer((void**)&dev_completion_cntr,
                                          completion_mmio_addr, 0),
                  "gpuHostGetDevicePointer(completion)");
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
        check_gpu(gpuHostRegister(cp.trigger_mmio_addr, cp.trigger_mmio_len,
                                  gpuHostRegisterMmio),
                  "gpuHostRegister(trigger)");
        check_gpu(gpuHostGetDevicePointer((void**)&cp.dev_trigger_cntr,
                                          cp.trigger_mmio_addr, 0),
                  "gpuHostGetDevicePointer(trigger)");

        return cp;
    }

    // Cleanup counter pair (ignore errors in cleanup)
    void destroy_counter_pair(CounterPair& cp) {
        if (cp.trigger_mmio_addr) (void)gpuHostUnregister(cp.trigger_mmio_addr);
        if (cp.trigger_cntr) fi_close(&cp.trigger_cntr->fid);
        if (cp.completion_cntr) fi_close(&cp.completion_cntr->fid);
        if (cp.atomic_completion_cntr) fi_close(&cp.atomic_completion_cntr->fid);
    }
};
