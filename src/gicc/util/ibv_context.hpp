/**
 * ibv_context.hpp - InfiniBand Verbs context with RAII and GPU memory support
 *
 * Provides basic RDMA initialization:
 *   - IB device/context/PD/CQ creation
 *   - Per-peer RC QP setup (one QP per peer for reliable connection)
 *   - GPU memory registration via nvidia_peermem
 */
#pragma once

#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <arpa/inet.h>

namespace gicc {

// Connection info exchanged between peers
struct IbvConnInfo {
    uint32_t qp_num;
    uint16_t lid;
    uint8_t gid[16];
    uint32_t psn;
};

// Per-peer QP structure
struct IbvPeerQP {
    struct ibv_qp* qp;
    uint32_t psn;
    bool connected;
    IbvConnInfo remote_info;

    IbvPeerQP() : qp(nullptr), psn(0), connected(false) {
        memset(&remote_info, 0, sizeof(remote_info));
    }
};

class IbvContext {
public:
    // IB objects
    struct ibv_context* ctx;
    struct ibv_pd* pd;
    struct ibv_cq* send_cq;
    struct ibv_cq* recv_cq;
    struct ibv_port_attr port_attr;
    union ibv_gid gid;
    int port_num;
    int gid_index;  // Selected GID index (for RoCE)

    // Device info
    struct ibv_device_attr dev_attr;
    std::string dev_name;

    // Per-peer QPs
    std::vector<IbvPeerQP> peer_qps;

    // Connection info
    int rank;
    int size;

    IbvContext(int rank_, int size_, const char* device_name = nullptr, int port = 1)
        : ctx(nullptr), pd(nullptr), send_cq(nullptr), recv_cq(nullptr),
          port_num(port), gid_index(0), rank(rank_), size(size_)
    {
        memset(&port_attr, 0, sizeof(port_attr));
        memset(&gid, 0, sizeof(gid));
        memset(&dev_attr, 0, sizeof(dev_attr));

        peer_qps.resize(size_);

        init_device(device_name);
        init_pd_cq();

        // Create one QP per peer
        for (int i = 0; i < size_; i++) {
            if (i != rank_) {
                create_qp_for_peer(i);
            }
        }

    }

    ~IbvContext() {
        for (auto& pqp : peer_qps) {
            if (pqp.qp) ibv_destroy_qp(pqp.qp);
        }
        if (send_cq) ibv_destroy_cq(send_cq);
        if (recv_cq) ibv_destroy_cq(recv_cq);
        if (pd) ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }

    // No copy
    IbvContext(const IbvContext&) = delete;
    IbvContext& operator=(const IbvContext&) = delete;

    // Get local connection info for a peer
    IbvConnInfo get_local_info(int peer_rank) const {
        IbvConnInfo info;
        info.qp_num = peer_qps[peer_rank].qp->qp_num;
        info.lid = port_attr.lid;
        memcpy(info.gid, gid.raw, 16);
        info.psn = peer_qps[peer_rank].psn;

        // Warn if LID is 0 on InfiniBand (indicates SM not running)
        if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND && info.lid == 0) {
            fprintf(stderr, "Warning: Rank %d: LID is 0 on InfiniBand - is the subnet manager running?\n", rank);
        }

        return info;
    }

    // Store peer's connection info
    void set_peer_info(int peer_rank, const IbvConnInfo& info) {
        peer_qps[peer_rank].remote_info = info;
    }

    // Connect QP to peer (INIT -> RTR -> RTS)
    void connect_to_peer(int peer_rank) {
        if (peer_qps[peer_rank].connected) return;

        IbvPeerQP& pqp = peer_qps[peer_rank];
        const IbvConnInfo& peer = pqp.remote_info;

        // Transition to RTR
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTR;

        // Use the port's active MTU (what the network actually supports)
        // Don't try to exceed this - it will cause "transport retry exceeded" errors
        attr.path_mtu = port_attr.active_mtu;

        // Allow override via environment variable (but cap at active_mtu)
        const char* mtu_env = getenv("GICC_MTU");
        if (mtu_env) {
            int mtu_val = atoi(mtu_env);
            enum ibv_mtu requested_mtu;
            if (mtu_val <= 256) requested_mtu = IBV_MTU_256;
            else if (mtu_val <= 512) requested_mtu = IBV_MTU_512;
            else if (mtu_val <= 1024) requested_mtu = IBV_MTU_1024;
            else if (mtu_val <= 2048) requested_mtu = IBV_MTU_2048;
            else requested_mtu = IBV_MTU_4096;

            // Cap at active MTU
            if (requested_mtu <= port_attr.active_mtu) {
                attr.path_mtu = requested_mtu;
            }
        }

        attr.dest_qp_num = peer.qp_num;
        attr.rq_psn = peer.psn;
        attr.max_dest_rd_atomic = 16;
        attr.min_rnr_timer = 12;

        attr.ah_attr.dlid = peer.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = port_num;

        // Use GRH for RoCE or if LID is 0
        if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET || peer.lid == 0) {
            attr.ah_attr.is_global = 1;
            memcpy(&attr.ah_attr.grh.dgid, peer.gid, 16);
            attr.ah_attr.grh.flow_label = 0;
            attr.ah_attr.grh.hop_limit = 64;
            attr.ah_attr.grh.sgid_index = gid_index;  // Use selected GID index
            attr.ah_attr.grh.traffic_class = 0;
        }

        int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                    IBV_QP_MIN_RNR_TIMER;

        int ret = ibv_modify_qp(pqp.qp, &attr, flags);
        if (ret) {
            fprintf(stderr, "Rank %d: Failed to modify QP to RTR for peer %d: %s\n",
                    rank, peer_rank, strerror(ret));
            exit(1);
        }

        // Transition to RTS
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 18;      // Increased timeout (~1 second per retry)
        attr.retry_cnt = 7;     // Max retries
        attr.rnr_retry = 7;     // Max RNR retries
        attr.sq_psn = pqp.psn;
        attr.max_rd_atomic = 16;

        flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

        ret = ibv_modify_qp(pqp.qp, &attr, flags);
        if (ret) {
            fprintf(stderr, "Rank %d: Failed to modify QP to RTS for peer %d: %s\n",
                    rank, peer_rank, strerror(ret));
            exit(1);
        }

        pqp.connected = true;
    }

    // Poll CQ for completion
    int poll_cq(struct ibv_wc* wc, int max_entries = 1) {
        return ibv_poll_cq(send_cq, max_entries, wc);
    }

    // Post RDMA write to specific peer
    int post_rdma_write(int peer_rank, void* local_addr, uint32_t lkey, size_t len,
                        uint64_t remote_addr, uint32_t rkey, uint64_t wr_id,
                        bool signaled = true) {
        struct ibv_sge sge = {};
        sge.addr = (uint64_t)local_addr;
        sge.length = len;
        sge.lkey = lkey;

        struct ibv_send_wr wr = {};
        wr.wr_id = wr_id;
        wr.next = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;

        struct ibv_send_wr* bad_wr = nullptr;
        return ibv_post_send(peer_qps[peer_rank].qp, &wr, &bad_wr);
    }

    // Post RDMA read from specific peer
    int post_rdma_read(int peer_rank, void* local_addr, uint32_t lkey, size_t len,
                       uint64_t remote_addr, uint32_t rkey, uint64_t wr_id,
                       bool signaled = true) {
        struct ibv_sge sge = {};
        sge.addr = (uint64_t)local_addr;
        sge.length = len;
        sge.lkey = lkey;

        struct ibv_send_wr wr = {};
        wr.wr_id = wr_id;
        wr.next = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_READ;
        wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;

        struct ibv_send_wr* bad_wr = nullptr;
        return ibv_post_send(peer_qps[peer_rank].qp, &wr, &bad_wr);
    }

    // Post atomic fetch-and-add to specific peer
    int post_atomic_add(int peer_rank, void* local_addr, uint32_t lkey,
                        uint64_t remote_addr, uint32_t rkey,
                        uint64_t add_value, uint64_t wr_id,
                        bool signaled = true) {
        struct ibv_sge sge = {};
        sge.addr = (uint64_t)local_addr;
        sge.length = 8;
        sge.lkey = lkey;

        struct ibv_send_wr wr = {};
        wr.wr_id = wr_id;
        wr.next = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
        wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
        wr.wr.atomic.remote_addr = remote_addr;
        wr.wr.atomic.rkey = rkey;
        wr.wr.atomic.compare_add = add_value;

        struct ibv_send_wr* bad_wr = nullptr;
        return ibv_post_send(peer_qps[peer_rank].qp, &wr, &bad_wr);
    }

private:
    void check(int ret, const char* msg) {
        if (ret) {
            fprintf(stderr, "Rank %d: %s failed: %s\n", rank, msg, strerror(ret));
            exit(1);
        }
    }

    void init_device(const char* device_name) {
        int num_devices = 0;
        struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
        if (!dev_list || num_devices == 0) {
            fprintf(stderr, "Rank %d: No IB devices found\n", rank);
            exit(1);
        }

        // Check environment variable for device name override
        const char* env_dev = getenv("GICC_IB_DEV");
        if (env_dev) {
            device_name = env_dev;
        }

        struct ibv_device* dev = nullptr;
        if (device_name) {
            for (int i = 0; i < num_devices; i++) {
                if (strcmp(ibv_get_device_name(dev_list[i]), device_name) == 0) {
                    dev = dev_list[i];
                    break;
                }
            }
        } else {
            dev = dev_list[0];
        }

        if (!dev) {
            fprintf(stderr, "Rank %d: IB device '%s' not found\n",
                    rank, device_name ? device_name : "default");
            ibv_free_device_list(dev_list);
            exit(1);
        }

        dev_name = ibv_get_device_name(dev);

        ctx = ibv_open_device(dev);
        if (!ctx) {
            fprintf(stderr, "Rank %d: Failed to open IB device\n", rank);
            ibv_free_device_list(dev_list);
            exit(1);
        }

        ibv_free_device_list(dev_list);

        check(ibv_query_device(ctx, &dev_attr), "ibv_query_device");
        check(ibv_query_port(ctx, port_num, &port_attr), "ibv_query_port");

        // Verify port is active
        if (port_attr.state != IBV_PORT_ACTIVE) {
            fprintf(stderr, "Rank %d: IB port %d is not active (state=%d)\n",
                    rank, port_num, port_attr.state);
            // Try other ports
            for (int p = 1; p <= 2; p++) {
                if (p == port_num) continue;
                struct ibv_port_attr test_port;
                if (ibv_query_port(ctx, p, &test_port) == 0 &&
                    test_port.state == IBV_PORT_ACTIVE) {
                    port_num = p;
                    port_attr = test_port;
                    break;
                }
            }
            if (port_attr.state != IBV_PORT_ACTIVE) {
                fprintf(stderr, "Rank %d: No active IB port found\n", rank);
            }
        }

        // For RoCE, try to find a valid GID (prefer RoCEv2)
        // Check GID_INDEX env var first
        const char* gid_env = getenv("GICC_GID_INDEX");
        if (gid_env) {
            gid_index = atoi(gid_env);
        } else if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
            // For RoCE, scan GIDs to find a valid one (skip link-local)
            gid_index = find_best_gid();
        }

        check(ibv_query_gid(ctx, port_num, gid_index, &gid), "ibv_query_gid");
    }

    // Find best GID index for RoCE (prefer RoCEv2, avoid link-local)
    int find_best_gid() {
        union ibv_gid test_gid;
        int best_idx = 0;

        // Scan up to 16 GIDs looking for a valid one
        for (int i = 0; i < 16; i++) {
            if (ibv_query_gid(ctx, port_num, i, &test_gid) != 0) continue;

            // Skip all-zero GID
            bool all_zero = true;
            for (int j = 0; j < 16; j++) {
                if (test_gid.raw[j] != 0) { all_zero = false; break; }
            }
            if (all_zero) continue;

            // For RoCE, GIDs starting with fe80 are link-local (less preferred)
            // GIDs with ::ffff: prefix are IPv4-mapped (RoCEv2)
            if (test_gid.raw[0] == 0xfe && test_gid.raw[1] == 0x80) {
                // Link-local, use as fallback
                if (best_idx == 0) best_idx = i;
            } else {
                // Prefer non-link-local GIDs
                best_idx = i;
                break;
            }
        }

        return best_idx;
    }

    void init_pd_cq() {
        pd = ibv_alloc_pd(ctx);
        if (!pd) {
            fprintf(stderr, "Rank %d: Failed to allocate PD\n", rank);
            exit(1);
        }

        // Create separate CQs with enough entries for all peers
        int cq_size = 4096;  // Large enough for multiple outstanding ops
        send_cq = ibv_create_cq(ctx, cq_size, nullptr, nullptr, 0);
        if (!send_cq) {
            fprintf(stderr, "Rank %d: Failed to create send CQ\n", rank);
            exit(1);
        }

        recv_cq = ibv_create_cq(ctx, cq_size, nullptr, nullptr, 0);
        if (!recv_cq) {
            fprintf(stderr, "Rank %d: Failed to create recv CQ\n", rank);
            exit(1);
        }
    }

    void create_qp_for_peer(int peer_rank) {
        IbvPeerQP& pqp = peer_qps[peer_rank];

        // Random PSN for this QP
        pqp.psn = (rand() & 0xFFFFFF);

        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = send_cq;
        qp_init_attr.recv_cq = recv_cq;
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.cap.max_send_wr = 512;
        qp_init_attr.cap.max_recv_wr = 512;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.sq_sig_all = 0;

        pqp.qp = ibv_create_qp(pd, &qp_init_attr);
        if (!pqp.qp) {
            fprintf(stderr, "Rank %d: Failed to create QP for peer %d\n", rank, peer_rank);
            exit(1);
        }

        // Transition to INIT state
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = port_num;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ |
                               IBV_ACCESS_REMOTE_ATOMIC;

        int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        int ret = ibv_modify_qp(pqp.qp, &attr, flags);
        if (ret) {
            fprintf(stderr, "Rank %d: Failed to modify QP to INIT for peer %d: %s\n",
                    rank, peer_rank, strerror(ret));
            exit(1);
        }
    }
};

} // namespace gicc
