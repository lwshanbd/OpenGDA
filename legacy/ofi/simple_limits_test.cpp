/*
 * simple_limits_test.cpp - Simple Counter/DWQ limits test
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#include <cstring>
#include "hip_device_context.hpp"
#include "pmi_session.hpp"

int main() {
    unset_rocr_visible_devices();

    PmiSession pmi;

    // Select GPU based on local rank (0-7 -> GPU 0-7)
    int gpu_id = pmi.local_rank >= 0 ? pmi.local_rank : pmi.rank % 8;
    HipDeviceContext hip(gpu_id);

    // Each GPU should use its closest NIC
    // GPU 0,1 -> cxi0, GPU 2,3 -> cxi1, GPU 4,5 -> cxi2, GPU 6,7 -> cxi3
    int nic_id = gpu_id / 2;
    char cxi_domain[16];
    snprintf(cxi_domain, sizeof(cxi_domain), "cxi%d", nic_id);

    printf("Rank %d/%d: GPU=%d, target CXI=%s\n",
           pmi.rank, pmi.size, gpu_id, cxi_domain);

    pmi.barrier();

    // Initialize Fabric
    struct fi_info* hints = fi_allocinfo();
    hints->caps = FI_RMA | FI_MSG | FI_HMEM;
    hints->mode = FI_CONTEXT2;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                  FI_MR_PROV_KEY | FI_MR_LOCAL | FI_MR_ENDPOINT;

    struct fi_info* info = nullptr;
    int ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                         NULL, NULL, 0, hints, &info);
    fi_freeinfo(hints);

    if (ret) {
        fprintf(stderr, "Rank %d: fi_getinfo failed\n", pmi.rank);
        return 1;
    }

    // Find the specific CXI provider we want
    struct fi_info* cxi_info = nullptr;
    for (struct fi_info* cur = info; cur; cur = cur->next) {
        if (cur->fabric_attr && cur->fabric_attr->prov_name &&
            strcmp(cur->fabric_attr->prov_name, "cxi") == 0 &&
            cur->domain_attr && cur->domain_attr->name &&
            strcmp(cur->domain_attr->name, cxi_domain) == 0) {
            cxi_info = cur;
            break;
        }
    }
    if (!cxi_info) {
        // Fallback to first CXI
        for (struct fi_info* cur = info; cur; cur = cur->next) {
            if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
                cxi_info = cur;
                break;
            }
        }
    }
    if (!cxi_info) {
        fprintf(stderr, "Rank %d: CXI not found\n", pmi.rank);
        return 1;
    }

    printf("Rank %d: Using %s (wanted %s)\n", pmi.rank, cxi_info->domain_attr->name, cxi_domain);

    struct fid_fabric* fabric = nullptr;
    struct fid_domain* domain = nullptr;
    fi_fabric(cxi_info->fabric_attr, &fabric, NULL);
    fi_domain(fabric, cxi_info, &domain, NULL);

    // Test: Create counters until failure
    std::vector<struct fid_cntr*> counters;
    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;

    int max_test = 3000;
    int created = 0;

    for (int i = 0; i < max_test; i++) {
        struct fid_cntr* cntr = nullptr;
        ret = fi_cntr_open(domain, &cntr_attr, &cntr, NULL);
        if (ret != 0) {
            break;
        }
        counters.push_back(cntr);
        created++;
    }

    printf("Rank %d: Created %d counters (CXI=%s)\n",
           pmi.rank, created, cxi_info->domain_attr->name);

    // Cleanup
    for (auto* cntr : counters) {
        fi_close(&cntr->fid);
    }

    fi_close(&domain->fid);
    fi_close(&fabric->fid);
    fi_freeinfo(info);

    pmi.barrier();

    if (pmi.rank == 0) {
        printf("\nDone!\n");
    }

    return 0;
}
