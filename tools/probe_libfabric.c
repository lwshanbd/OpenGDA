/*
 * probe_libfabric.c - Verifies provider capabilities required by the CPU
 * Proxy path on the local node. Run inside a salloc allocation.
 */
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Renamed from `yn` to avoid colliding with libm's Bessel function `yn`. */
static const char* yesno(int x) { return x ? "yes" : "NO"; }

int main(void) {
    struct fi_info* hints = fi_allocinfo();
    if (!hints) { fprintf(stderr, "fi_allocinfo failed\n"); return 1; }

    hints->ep_attr->type    = FI_EP_RDM;
    hints->caps             = FI_RMA | FI_MSG | FI_HMEM;
    hints->mode             = FI_CONTEXT;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_PROV_KEY |
                                  FI_MR_HMEM | FI_MR_ENDPOINT;

    struct fi_info* info = NULL;
    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
    if (ret) {
        fprintf(stderr, "fi_getinfo failed: %s\n", fi_strerror(-ret));
        return 1;
    }

    printf("== probe_libfabric report ==\n");
    printf("provider:           %s\n", info->fabric_attr->prov_name);
    printf("ep type:            %s\n", info->ep_attr->type == FI_EP_RDM ? "FI_EP_RDM" : "other");
    printf("caps:               0x%llx\n", (unsigned long long)info->caps);
    printf("  FI_RMA:           %s\n", yesno(info->caps & FI_RMA));
    printf("  FI_MSG:           %s\n", yesno(info->caps & FI_MSG));
    printf("  FI_HMEM:          %s\n", yesno(info->caps & FI_HMEM));
    printf("  FI_ATOMIC:        %s\n", yesno(info->caps & FI_ATOMIC));
    printf("mr_mode:            0x%llx\n", (unsigned long long)info->domain_attr->mr_mode);
    printf("tx_attr.size:       %zu\n", info->tx_attr->size);
    printf("rx_attr.size:       %zu\n", info->rx_attr->size);
    printf("tx_attr.inject_size:%zu\n", info->tx_attr->inject_size);
    printf("max_msg_size:       %zu\n", info->ep_attr->max_msg_size);

    /* Try to open fabric/domain/two CQs on one EP. */
    struct fid_fabric* fabric = NULL;
    ret = fi_fabric(info->fabric_attr, &fabric, NULL);
    if (ret) { fprintf(stderr, "fi_fabric: %s\n", fi_strerror(-ret)); return 1; }

    struct fid_domain* domain = NULL;
    ret = fi_domain(fabric, info, &domain, NULL);
    if (ret) { fprintf(stderr, "fi_domain: %s\n", fi_strerror(-ret)); return 1; }

    struct fid_ep* ep = NULL;
    ret = fi_endpoint(domain, info, &ep, NULL);
    if (ret) { fprintf(stderr, "fi_endpoint: %s\n", fi_strerror(-ret)); return 1; }

    struct fi_cq_attr cq_attr1 = { .format = FI_CQ_FORMAT_CONTEXT, .size = 64 };
    struct fid_cq* cq1 = NULL;
    ret = fi_cq_open(domain, &cq_attr1, &cq1, NULL);
    if (ret) { fprintf(stderr, "cq1 open: %s\n", fi_strerror(-ret)); return 1; }
    ret = fi_ep_bind(ep, &cq1->fid, FI_TRANSMIT);
    if (ret) { fprintf(stderr, "ep_bind cq1: %s\n", fi_strerror(-ret)); return 1; }

    struct fi_cq_attr cq_attr2 = { .format = FI_CQ_FORMAT_CONTEXT, .size = 64 };
    struct fid_cq* cq2 = NULL;
    ret = fi_cq_open(domain, &cq_attr2, &cq2, NULL);
    int cq2_open_ok = (ret == 0);
    int cq2_bind_ok = 0;
    if (cq2_open_ok) {
        /* Provider may reject a 2nd TX CQ binding on the same EP. */
        ret = fi_ep_bind(ep, &cq2->fid, FI_TRANSMIT);
        cq2_bind_ok = (ret == 0);
    }
    printf("dual TX CQ:         %s (open=%s, bind=%s)\n",
           yesno(cq2_open_ok && cq2_bind_ok), yesno(cq2_open_ok), yesno(cq2_bind_ok));

    /* HMEM iface query: try CUDA first, then ROCR. */
#ifdef FI_HMEM_CUDA
    printf("FI_HMEM_CUDA enum:  defined\n");
#else
    printf("FI_HMEM_CUDA enum:  NOT defined in this libfabric\n");
#endif
#ifdef FI_HMEM_ROCR
    printf("FI_HMEM_ROCR enum:  defined\n");
#else
    printf("FI_HMEM_ROCR enum:  NOT defined in this libfabric\n");
#endif

    if (cq2) fi_close(&cq2->fid);
    fi_close(&cq1->fid);
    fi_close(&ep->fid);
    fi_close(&domain->fid);
    fi_close(&fabric->fid);
    fi_freeinfo(info);
    fi_freeinfo(hints);
    return 0;
}
