/*
 * gda-large.cpp - Test allocating many counters to find MMIO limits
 *
 * Based on gda-comp.cpp, but allocates NUM_COUNTERS counters and registers
 * their MMIO addresses with HIP to test CXI MMIO mapping limits.
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
#include <unistd.h>
#include <vector>

// Number of counters to allocate - test MMIO limits
#define NUM_COUNTERS 75

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

struct CounterInfo {
    struct fid_cntr *cntr;
    struct fi_cxi_cntr_ops *ops;
    void *mmio_addr;
    size_t mmio_len;
    volatile uint64_t *dev_addr;
    bool hip_registered;
};

int main(int argc, char **argv) {
    // Parse command line for number of counters
    int num_counters = NUM_COUNTERS;
    if (argc > 1) {
        num_counters = atoi(argv[1]);
        if (num_counters < 1) num_counters = 1;
        if (num_counters > 500) num_counters = 500;
    }

    // Unset ROCR_VISIBLE_DEVICES before any initialization
    unsetenv("ROCR_VISIBLE_DEVICES");

    // PMI2 Initialization
    int spawned, size, appnum;
    PMI2_Init(&spawned, &size, &myrank, &appnum);

    printf("Rank %d/%d: Testing allocation of %d counters\n", myrank, size, num_counters);

    // HIP Initialization
    int device_count;
    CHECK_HIP(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    int gpu_id = myrank % device_count;
    CHECK_HIP(hipSetDevice(gpu_id), "hipSetDevice");

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, gpu_id), "hipGetDeviceProperties");
    printf("Rank %d: Using GPU %d: %s\n", myrank, gpu_id, prop.name);

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

    // Create Fabric and Domain
    struct fid_fabric *fabric = NULL;
    CHECK(fi_fabric(cxi_info->fabric_attr, &fabric, NULL), "fi_fabric");

    struct fid_domain *domain = NULL;
    CHECK(fi_domain(fabric, cxi_info, &domain, NULL), "fi_domain");

    // Allocate counter info array
    std::vector<CounterInfo> counters(num_counters);

    // Initialize all counters
    for (int i = 0; i < num_counters; i++) {
        counters[i].cntr = nullptr;
        counters[i].ops = nullptr;
        counters[i].mmio_addr = nullptr;
        counters[i].mmio_len = 0;
        counters[i].dev_addr = nullptr;
        counters[i].hip_registered = false;
    }

    printf("Rank %d: Creating %d counters...\n", myrank, num_counters);

    // Create all counters
    int created_count = 0;
    for (int i = 0; i < num_counters; i++) {
        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;
        cntr_attr.wait_obj = FI_WAIT_UNSPEC;

        ret = fi_cntr_open(domain, &cntr_attr, &counters[i].cntr, NULL);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_cntr_open failed for counter %d: %s (%d)\n",
                    myrank, i, fi_strerror(-ret), ret);
            break;
        }

        ret = fi_open_ops(&counters[i].cntr->fid, FI_CXI_COUNTER_OPS, 0,
                          (void **)&counters[i].ops, NULL);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_open_ops failed for counter %d: %s (%d)\n",
                    myrank, i, fi_strerror(-ret), ret);
            fi_close(&counters[i].cntr->fid);
            counters[i].cntr = nullptr;
            break;
        }

        ret = counters[i].ops->get_mmio_addr(&counters[i].cntr->fid,
                                              &counters[i].mmio_addr,
                                              &counters[i].mmio_len);
        if (ret) {
            fprintf(stderr, "Rank %d: get_mmio_addr failed for counter %d: %s (%d)\n",
                    myrank, i, fi_strerror(-ret), ret);
            fi_close(&counters[i].cntr->fid);
            counters[i].cntr = nullptr;
            break;
        }

        created_count++;
    }

    printf("Rank %d: Created %d counters successfully\n", myrank, created_count);

    // Now register MMIO with HIP
    printf("Rank %d: Registering MMIO addresses with HIP...\n", myrank);

    int registered_count = 0;
    for (int i = 0; i < created_count; i++) {
        hipError_t hip_err = hipHostRegister(counters[i].mmio_addr,
                                              counters[i].mmio_len,
                                              hipHostRegisterMapped);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "Rank %d: hipHostRegister failed for counter %d: %s (%d)\n",
                    myrank, i, hipGetErrorString(hip_err), hip_err);
            // Don't break - continue to see how many we can register
            continue;
        }
        counters[i].hip_registered = true;

        hip_err = hipHostGetDevicePointer((void **)&counters[i].dev_addr,
                                           counters[i].mmio_addr, 0);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "Rank %d: hipHostGetDevicePointer failed for counter %d: %s (%d)\n",
                    myrank, i, hipGetErrorString(hip_err), hip_err);
            hipHostUnregister(counters[i].mmio_addr);
            counters[i].hip_registered = false;
            continue;
        }

        registered_count++;

        // Print progress every 10 counters
        if ((i + 1) % 10 == 0) {
            printf("Rank %d: Registered %d/%d MMIO addresses\n", myrank, registered_count, i + 1);
        }
    }

    printf("Rank %d: Successfully registered %d/%d MMIO addresses with HIP\n",
           myrank, registered_count, created_count);

    // Barrier to sync all ranks
    PMI2_KVS_Fence();

    // Report results
    if (myrank == 0) {
        printf("\n========================================\n");
        printf("MMIO Registration Test Results:\n");
        printf("  Requested counters: %d\n", num_counters);
        printf("  Created counters: %d\n", created_count);
        printf("  Registered with HIP: %d\n", registered_count);
        printf("========================================\n");
    }

    // Cleanup
    for (int i = 0; i < created_count; i++) {
        if (counters[i].hip_registered) {
            hipHostUnregister(counters[i].mmio_addr);
        }
        if (counters[i].cntr) {
            fi_close(&counters[i].cntr->fid);
        }
    }

    fi_close(&domain->fid);
    fi_close(&fabric->fid);
    fi_freeinfo(info);

    PMI2_Finalize();

    printf("Rank %d: Done!\n", myrank);
    return 0;
}
