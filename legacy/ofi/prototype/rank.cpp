/*
 * rank.cpp - Two-Process RDMA Write Example using Libfabric
 *
 * This program demonstrates a simple RDMA write operation between two processes
 * using the CXI provider with Libfabric. Rank 0 writes a message to Rank 1's
 * memory using one-sided RDMA operations.
 *
 * Key Features:
 * - Uses PMI2 for process coordination and information exchange
 * - Handles CXI provider's lack of FI_MR_VIRT_ADDR support
 * - Uses fi_writemsg with FI_DELIVERY_COMPLETE for reliable transfers
 *
 * Based on LCI (Lightweight Communication Interface) OFI backend implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <stdint.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_trigger.h>
#include <rdma/fi_cxi_ext.h>
#include <rdma/fi_cm.h>
#include <unistd.h>
#include <pmi2.h>

#define CHECK(x, msg) do { \
    int ret = (x); \
    if (ret) { \
        fprintf(stderr, "Rank %d: %s failed: %s (%d)\n", myrank, msg, fi_strerror(-ret), ret); \
        PMI2_Finalize(); \
        exit(1); \
    } \
} while(0)

// =============================================================================
// Global Variables
// =============================================================================
int myrank = -1;  // Global for error handling

static uint64_t g_next_rdma_key = 0;

// =============================================================================
// Utility Functions
// =============================================================================

static void bytes_to_hex(const uint8_t *in, size_t len, char *out) {
    static const char *h = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        out[2*i]   = h[(in[i] >> 4) & 0xF];
        out[2*i+1] = h[in[i] & 0xF];
    }
    out[2*len] = '\0';
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
        int lo = hexval(in[i+1]);
        if (hi < 0 || lo < 0) return -1;
        out[i/2] = (uint8_t)((hi << 4) | lo);
    }
    return (int)(n/2);
}

// =============================================================================
// PMI2 Address Exchange
// =============================================================================

int exchange_addrs(int rank, int size, const char *my_hex, size_t hex_len, char *all_hex)
{
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

int register_memory_region(struct fid_domain *domain, 
                           struct fid_ep *ep,
                           struct fi_info *cxi_info,
                           void *buffer, 
                           size_t buffer_size,
                           struct fid_mr **mr_out) {
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
    mr_attr.iface = FI_HMEM_SYSTEM;
    
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

    // =============================================================================
    // STEP 2: Libfabric Provider Setup
    // =============================================================================
    struct fi_info *hints = fi_allocinfo();
    hints->caps = FI_RMA | FI_MSG;
    hints->mode = FI_CONTEXT;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_LOCAL | FI_MR_ENDPOINT;
    hints->domain_attr->threading = FI_THREAD_SAFE;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

    int ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                         NULL, NULL, 0, hints, &info);
    fi_freeinfo(hints);

    if (ret) {
        fprintf(stderr, "Rank %d: fi_getinfo failed: %s (%d)\n", myrank, fi_strerror(-ret), ret);
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

    printf("Rank %d: Using CXI provider %s (mr_mode=0x%lx)\n",
           myrank, cxi_info->domain_attr->name, (unsigned long)cxi_info->domain_attr->mr_mode);

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
    CHECK(exchange_addrs(myrank, size, my_hex, 2 * addrlen, all_hex), "exchange_addrs");

    // Convert from hex to binary
    uint8_t *all_bin = (uint8_t *)malloc(size * addrlen);
    for (int i = 0; i < size; i++) {
        const char *hex = all_hex + i * (2 * addrlen + 1);
        CHECK(hex_to_bytes(hex, all_bin + i * addrlen, addrlen) > 0 ? 0 : -1, "hex_to_bytes");
    }

    // Insert peer address into AV
    int peer = (myrank == 0) ? 1 : 0;
    void *peer_addr = all_bin + peer * addrlen;

    fi_addr_t peer_fi_addr = FI_ADDR_NOTAVAIL;
    int inserted = fi_av_insert(av, peer_addr, 1, &peer_fi_addr, 0, NULL);
    if (inserted != 1) {
        fprintf(stderr, "Rank %d: fi_av_insert failed, inserted=%d\n", myrank, inserted);
        PMI2_Finalize();
        exit(1);
    }
    if (peer_fi_addr == FI_ADDR_NOTAVAIL) {
        fprintf(stderr, "Rank %d: peer_fi_addr is FI_ADDR_NOTAVAIL!\n", myrank);
        PMI2_Finalize();
        exit(1);
    }

    // =============================================================================
    // STEP 8: Register Memory Regions
    // =============================================================================
    size_t buf_size = 64;
    char *local_buf = (char *)malloc(buf_size);
    char *remote_buf = (char *)malloc(buf_size);

    // Initialize buffers
    memset(local_buf, 0, buf_size);
    memset(remote_buf, 0, buf_size);
    snprintf(local_buf, buf_size, "Hello from rank %d!", myrank);

    // Register local buffer (for sending)
    struct fid_mr *mr_local;
    CHECK(register_memory_region(domain, ep, cxi_info, local_buf, buf_size, &mr_local),
          "register_memory_region(local)");

    // Register remote buffer (for receiving)
    struct fid_mr *mr_remote;
    CHECK(register_memory_region(domain, ep, cxi_info, remote_buf, buf_size, &mr_remote),
          "register_memory_region(remote)");

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
    my_rma_info.addr = (uint64_t)remote_buf;
    my_rma_info.key = remote_key;

    // Publish my RMA info to PMI2
    char rma_hex[128];
    bytes_to_hex((uint8_t*)&my_rma_info, sizeof(my_rma_info), rma_hex);

    char key[PMI2_MAX_KEYLEN];
    snprintf(key, sizeof(key), "rma-%d", myrank);
    CHECK(PMI2_KVS_Put(key, rma_hex), "PMI2_KVS_Put(rma)");
    CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(rma)");

    // Retrieve peer's RMA info
    snprintf(key, sizeof(key), "rma-%d", peer);
    char peer_rma_hex[PMI2_MAX_VALLEN];
    int vallen;
    CHECK(PMI2_KVS_Get(NULL, PMI2_ID_NULL, key, peer_rma_hex,
                       sizeof(peer_rma_hex), &vallen), "PMI2_KVS_Get(rma)");
    CHECK(hex_to_bytes(peer_rma_hex, (uint8_t*)&peer_rma_info, sizeof(peer_rma_info)) > 0 ? 0 : -1,
          "hex_to_bytes(rma)");

    uint64_t peer_remote_addr = peer_rma_info.addr;
    uint64_t peer_remote_key = peer_rma_info.key;

    // =============================================================================
    // STEP 10: Synchronize Before RDMA
    // =============================================================================
    CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(pre-write)");
    usleep(100000);  // 100ms delay for CXI provider readiness

    // =============================================================================
    // STEP 11: Perform RDMA Write Operation
    // =============================================================================
    if (myrank == 0) {
        printf("Rank 0: Writing \"%s\" to rank 1\n", local_buf);

        // Setup local buffer descriptor
        struct iovec iov;
        iov.iov_base = local_buf;
        iov.iov_len = buf_size;

        // Setup remote buffer descriptor
        // CRITICAL: CXI doesn't support FI_MR_VIRT_ADDR, must use offset mode
        uint64_t remote_addr_for_rma;
        if (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
            remote_addr_for_rma = peer_remote_addr;
        } else {
            remote_addr_for_rma = 0;  // Use offset from MR base
        }

        struct fi_rma_iov rma_iov;
        rma_iov.addr = remote_addr_for_rma;
        rma_iov.len = buf_size;
        rma_iov.key = peer_remote_key;

        // Setup RMA message
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

        // Initiate RDMA write
        ret = fi_writemsg(ep, &msg, FI_COMPLETION | FI_DELIVERY_COMPLETE);

        if (ret) {
            fprintf(stderr, "Rank 0: fi_writemsg immediately failed: %s (%d)\n", fi_strerror(-ret), ret);
        }

        // Poll for completion
        struct fi_cq_data_entry cqe;
        ssize_t rc;
        int retry = 0;
        do {
            rc = fi_cq_read(cq, &cqe, 1);
            if (rc == -FI_EAGAIN) {
                retry++;
                usleep(10);
            } else if (rc < 0) {
                break;
            }
        } while (rc == -FI_EAGAIN && retry < 100000);

        // Handle completion result
        if (rc < 0) {
            if (rc == -FI_EAVAIL) {
                struct fi_cq_err_entry err_entry;
                int err_ret = fi_cq_readerr(cq, &err_entry, 0);
                if (err_ret > 0) {
                    fprintf(stderr, "Rank 0: CQ error: err=%d (%s), prov_errno=%d\n",
                            err_entry.err, fi_strerror(err_entry.err), err_entry.prov_errno);
                } else {
                    fprintf(stderr, "Rank 0: fi_cq_readerr failed: %d\n", err_ret);
                }
            } else {
                fprintf(stderr, "Rank 0: fi_writemsg failed: %s (%ld)\n", fi_strerror(-rc), rc);
            }
        } else {
            printf("Rank 0: RDMA write completed successfully\n");
        }
    }

    // =============================================================================
    // STEP 12: Synchronize and Verify
    // =============================================================================
    CHECK(PMI2_KVS_Fence(), "PMI2_KVS_Fence(sync)");

    if (myrank == 1) {
        usleep(1000);  // Small delay for any in-flight data
        printf("Rank 1: Received: \"%s\"\n", remote_buf);
    }

    // =============================================================================
    // STEP 13: Cleanup Resources
    // =============================================================================
    
    fi_close(&mr_local->fid);
    fi_close(&mr_remote->fid);
    free(local_buf);
    free(remote_buf);
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
