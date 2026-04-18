/**
 * devx_context.hpp - MLX5 DevX context for GPU-triggered RDMA
 *
 * Uses mlx5dv DevX API to allow GPU to directly:
 *   - Build and write WQEs to NIC-mapped memory
 *   - Ring doorbells to trigger RDMA operations
 *   - Poll CQ for completions
 *
 * This enables true GPU-initiated RDMA without CPU intervention.
 *
 * MULTI-QP SUPPORT: Creates one QP per peer for ring topology communication.
 */
#pragma once

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>

#include "gicc/bootstrap/bootstrap.hpp"

namespace gicc::mlx5 {

// MLX5 WQE sizes
#define MLX5_SEND_WQE_BB 64      // Work Queue Element Basic Block size
#define MLX5_SEND_WQE_SHIFT 6    // log2(64)
#define MLX5_CQE_SIZE 64         // Completion Queue Entry size

// WQE opcodes
#define MLX5_OPCODE_RDMA_WRITE 0x08
#define MLX5_OPCODE_RDMA_READ  0x10
#define MLX5_OPCODE_NOP        0x00

// WQE control flags
#define MLX5_WQE_CTRL_CQ_UPDATE (1 << 2)
#define MLX5_WQE_CTRL_FENCE     (1 << 5)

// Byte swap macros (assuming little-endian host)
#define HTOBE32(x) __builtin_bswap32(x)
#define HTOBE64(x) __builtin_bswap64(x)
#define BETOH32(x) __builtin_bswap32(x)
#define BETOH64(x) __builtin_bswap64(x)

// MLX5 WQE Control Segment (16 bytes)
struct Mlx5CtrlSeg {
    uint32_t opmod_idx_opcode;
    uint32_t qpn_ds;
    uint8_t  signature;
    uint8_t  rsvd[2];
    uint8_t  fm_ce_se;
    uint32_t imm;
} __attribute__((packed));

// MLX5 Remote Address Segment (16 bytes)
struct Mlx5RaddrSeg {
    uint64_t raddr;
    uint32_t rkey;
    uint32_t reserved;
} __attribute__((packed));

// MLX5 Data Segment (16 bytes)
struct Mlx5DataSeg {
    uint32_t byte_count;
    uint32_t lkey;
    uint64_t addr;
} __attribute__((packed));

// MLX5 CQE structure (64 bytes)
#ifndef MLX5_CQE64_DEFINED
#define MLX5_CQE64_DEFINED
struct Mlx5Cqe64 {
    uint8_t  rsvd0[2];
    uint16_t wqe_id;
    uint8_t  rsvd1[8];
    uint32_t srqn_uidx;
    uint32_t imm_inval_pkey;
    uint8_t  rsvd2[4];
    uint32_t byte_cnt;
    uint64_t timestamp;
    uint32_t sop_drop_qpn;
    uint16_t wqe_counter;
    uint8_t  signature;
    uint8_t  op_own;
} __attribute__((packed));
#endif

// GPU-accessible QP state
struct DeviceQp {
    uint32_t qpn;                    // QP number
    uint16_t nwqes;                  // Number of WQEs in queue

    // WQE buffer (GPU-accessible)
    void* wqe_buf;                   // WQE buffer base address
    uint32_t wqe_lkey;               // lkey for WQE buffer

    // Doorbell (GPU-writable)
    volatile uint32_t* dbrec;        // Doorbell record

    // Producer index (GPU updates this)
    volatile uint64_t* prod_idx;     // Producer index for GPU to update

    // CQ for this QP
    volatile Mlx5Cqe64* cqe;         // CQ entry buffer
    uint32_t cqn;                    // CQ number
    uint32_t ncqes;                  // Number of CQ entries
    volatile uint64_t* cq_cons_idx;  // CQ consumer index
    volatile uint32_t* cq_dbrec;     // CQ doorbell record
};

// Remote peer info for RDMA
struct RemotePeer {
    uint64_t buf_addr;               // Remote buffer address
    uint32_t rkey;                   // Remote key
    uint32_t qpn;                    // Remote QP number
    uint16_t lid;                    // Remote LID
    uint8_t  gid[16];                // Remote GID (for RoCE)
};

// Connection info for exchange
struct ConnInfo {
    uint32_t qpn;
    uint16_t lid;
    uint8_t  gid[16];
    uint32_t psn;
    uint64_t buf_addr;
    uint32_t rkey;
};

// Per-peer QP resources (for multi-QP support)
struct PerPeerQp {
    struct ibv_qp* qp;
    struct mlx5dv_qp qp_ex;
    uint32_t psn;
    bool connected;

    // GPU-mapped resources for this QP
    void* d_wqe_buf;
    volatile uint32_t* d_dbrec;
    volatile uint64_t* d_bf_reg;
    volatile uint64_t* d_prod_idx;
    uint64_t* h_prod_idx;

    // Remote peer info
    RemotePeer remote;

    PerPeerQp() : qp(nullptr), psn(0), connected(false),
                  d_wqe_buf(nullptr), d_dbrec(nullptr), d_bf_reg(nullptr),
                  d_prod_idx(nullptr), h_prod_idx(nullptr) {
        memset(&qp_ex, 0, sizeof(qp_ex));
        memset(&remote, 0, sizeof(remote));
    }
};

class DevxContext {
public:
    // IB objects
    struct ibv_context* ctx;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct mlx5dv_cq cq_ex;          // Extended CQ info from mlx5dv

    // Device info
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    union ibv_gid gid;
    int port_num;
    std::string dev_name;

    // Per-peer QPs (multi-QP support)
    std::map<int, PerPeerQp> peer_qps;
    std::vector<RemotePeer> remote_peers;

    // Bootstrap (MPI or PMI2, selected at build time)
    gicc::Bootstrap& boot;
    int rank;
    int size;

    // Shared CQ GPU resources
    volatile Mlx5Cqe64* d_cqe;       // GPU pointer to CQ entries

    // QP parameters
    uint32_t qp_depth;
    uint32_t cq_depth;

    // Legacy single-QP pointers (for backward compatibility)
    struct ibv_qp* qp;               // Points to first peer's QP
    struct mlx5dv_qp qp_ex;          // Points to first peer's qp_ex
    void* d_wqe_buf;
    volatile uint64_t* d_prod_idx;
    volatile uint32_t* d_dbrec;
    volatile uint64_t* d_bf_reg;
    void* h_wqe_buf;
    uint64_t* h_prod_idx;
    uint32_t psn;

    DevxContext(gicc::Bootstrap& boot_, const char* device_name = nullptr, int port = 1)
        : ctx(nullptr), pd(nullptr), cq(nullptr),
          port_num(port), boot(boot_), rank(boot_.rank()), size(boot_.size()),
          d_cqe(nullptr),
          qp_depth(256), cq_depth(512),
          qp(nullptr), d_wqe_buf(nullptr), d_prod_idx(nullptr),
          d_dbrec(nullptr), d_bf_reg(nullptr),
          h_wqe_buf(nullptr), h_prod_idx(nullptr), psn(0)
    {
        memset(&dev_attr, 0, sizeof(dev_attr));
        memset(&port_attr, 0, sizeof(port_attr));
        memset(&gid, 0, sizeof(gid));
        memset(&cq_ex, 0, sizeof(cq_ex));
        memset(&qp_ex, 0, sizeof(qp_ex));

        remote_peers.resize(size);

        init_device(device_name);
        init_pd_cq();
        allocate_cq_gpu_resources();
    }

    ~DevxContext() {
        // Free per-peer QP resources
        for (auto& kv : peer_qps) {
            free_peer_qp_gpu_resources(kv.second);
            if (kv.second.qp) ibv_destroy_qp(kv.second.qp);
        }

        // Free shared CQ resources
        if (d_cqe && cq_ex.buf) {
            cudaHostUnregister(cq_ex.buf);
        }

        if (cq) ibv_destroy_cq(cq);
        if (pd) ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }

    // No copy
    DevxContext(const DevxContext&) = delete;
    DevxContext& operator=(const DevxContext&) = delete;

    /**
     * Create a QP for communicating with a specific peer
     */
    void create_qp_for_peer(int peer_rank) {
        if (peer_rank == rank) return;
        if (peer_qps.find(peer_rank) != peer_qps.end()) return;  // Already created

        PerPeerQp& pqp = peer_qps[peer_rank];
        pqp.psn = (rand() & 0xFFFFFF);

        // Create QP
        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = cq;
        qp_init_attr.recv_cq = cq;
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.sq_sig_all = 0;
        qp_init_attr.cap.max_send_wr = qp_depth;
        qp_init_attr.cap.max_recv_wr = qp_depth;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.cap.max_inline_data = 64;

        pqp.qp = ibv_create_qp(pd, &qp_init_attr);
        if (!pqp.qp) {
            fprintf(stderr, "Rank %d: Failed to create QP for peer %d: %s\n",
                    rank, peer_rank, strerror(errno));
            exit(1);
        }

        // Transition to INIT
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

        // Query mlx5dv extended info
        struct mlx5dv_obj obj = {};
        obj.qp.in = pqp.qp;
        obj.qp.out = &pqp.qp_ex;

        ret = mlx5dv_init_obj(&obj, MLX5DV_OBJ_QP);
        if (ret) {
            fprintf(stderr, "Rank %d: Failed to query mlx5dv QP for peer %d: %s\n",
                    rank, peer_rank, strerror(ret));
            exit(1);
        }

        // Allocate GPU resources for this QP
        allocate_peer_qp_gpu_resources(pqp);

        if (rank == 0) {
            printf("Created QP for peer %d: QPN=%u\n", peer_rank, pqp.qp->qp_num);
        }
    }

    /**
     * Get local connection info for a specific peer's QP
     */
    ConnInfo get_local_info_for_peer(int peer_rank) {
        auto it = peer_qps.find(peer_rank);
        if (it == peer_qps.end()) {
            fprintf(stderr, "Rank %d: No QP for peer %d\n", rank, peer_rank);
            exit(1);
        }

        ConnInfo info;
        info.qpn = it->second.qp->qp_num;
        info.lid = port_attr.lid;
        memcpy(info.gid, gid.raw, 16);
        info.psn = it->second.psn;
        info.buf_addr = 0;
        info.rkey = 0;
        return info;
    }

    // Legacy: Get local connection info (uses first QP)
    ConnInfo get_local_info() const {
        if (peer_qps.empty()) {
            ConnInfo info = {};
            info.lid = port_attr.lid;
            memcpy(info.gid, gid.raw, 16);
            return info;
        }

        auto it = peer_qps.begin();
        ConnInfo info;
        info.qpn = it->second.qp->qp_num;
        info.lid = port_attr.lid;
        memcpy(info.gid, gid.raw, 16);
        info.psn = it->second.psn;
        info.buf_addr = 0;
        info.rkey = 0;
        return info;
    }

    /**
     * Connect QP to a specific peer
     */
    void connect_to_peer(int peer_rank, const ConnInfo& peer_info) {
        if (peer_rank == rank) return;

        auto it = peer_qps.find(peer_rank);
        if (it == peer_qps.end()) {
            fprintf(stderr, "Rank %d: No QP for peer %d, create it first\n", rank, peer_rank);
            exit(1);
        }

        PerPeerQp& pqp = it->second;
        if (pqp.connected) return;

        // Store peer info
        pqp.remote.qpn = peer_info.qpn;
        pqp.remote.lid = peer_info.lid;
        memcpy(pqp.remote.gid, peer_info.gid, 16);
        pqp.remote.buf_addr = peer_info.buf_addr;
        pqp.remote.rkey = peer_info.rkey;

        // Transition QP to RTR
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_4096;
        attr.dest_qp_num = peer_info.qpn;
        attr.rq_psn = peer_info.psn;
        attr.max_dest_rd_atomic = 16;
        attr.min_rnr_timer = 12;

        attr.ah_attr.dlid = peer_info.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = port_num;

        // Use GRH for RoCE or if LID is 0
        if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET || peer_info.lid == 0) {
            attr.ah_attr.is_global = 1;
            memcpy(&attr.ah_attr.grh.dgid, peer_info.gid, 16);
            attr.ah_attr.grh.flow_label = 0;
            attr.ah_attr.grh.hop_limit = 64;
            attr.ah_attr.grh.sgid_index = 0;
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
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
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

        if (rank == 0) {
            printf("Rank %d: Connected to peer %d (QPN=%u)\n",
                   rank, peer_rank, peer_info.qpn);
        }
    }

    /**
     * Get GPU-accessible state for a specific peer's QP
     */
    bool get_peer_gpu_state(int peer_rank, void** wqe_buf, volatile uint32_t** dbrec,
                            volatile uint64_t** bf_reg, volatile uint64_t** prod_idx,
                            uint32_t* qpn, uint32_t* depth) {
        auto it = peer_qps.find(peer_rank);
        if (it == peer_qps.end()) return false;

        const PerPeerQp& pqp = it->second;
        *wqe_buf = pqp.d_wqe_buf;
        *dbrec = pqp.d_dbrec;
        *bf_reg = pqp.d_bf_reg;
        *prod_idx = pqp.d_prod_idx;
        *qpn = pqp.qp->qp_num;
        *depth = qp_depth;
        return true;
    }

    // Legacy: Get GPU-accessible device QP info (uses first QP)
    DeviceQp get_device_qp() const {
        DeviceQp dqp;
        memset(&dqp, 0, sizeof(dqp));

        if (!peer_qps.empty()) {
            auto it = peer_qps.begin();
            const PerPeerQp& pqp = it->second;
            dqp.qpn = pqp.qp->qp_num;
            dqp.nwqes = qp_depth;
            dqp.wqe_buf = pqp.d_wqe_buf;
            dqp.wqe_lkey = 0;
            dqp.dbrec = pqp.d_dbrec;
            dqp.prod_idx = pqp.d_prod_idx;
        }

        dqp.cqe = d_cqe;
        dqp.cqn = cq_ex.cqn;
        dqp.ncqes = cq_depth;
        dqp.cq_cons_idx = nullptr;
        dqp.cq_dbrec = (volatile uint32_t*)cq_ex.dbrec;
        return dqp;
    }

    /**
     * Update legacy pointers to point to a specific peer's QP
     */
    void set_active_peer(int peer_rank) {
        auto it = peer_qps.find(peer_rank);
        if (it == peer_qps.end()) return;

        PerPeerQp& pqp = it->second;
        qp = pqp.qp;
        qp_ex = pqp.qp_ex;
        d_wqe_buf = pqp.d_wqe_buf;
        d_dbrec = pqp.d_dbrec;
        d_bf_reg = pqp.d_bf_reg;
        d_prod_idx = pqp.d_prod_idx;
        h_prod_idx = pqp.h_prod_idx;
        h_wqe_buf = pqp.qp_ex.sq.buf;
        psn = pqp.psn;
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
        check(ibv_query_gid(ctx, port_num, 0, &gid), "ibv_query_gid");
    }

    void init_pd_cq() {
        pd = ibv_alloc_pd(ctx);
        if (!pd) {
            fprintf(stderr, "Rank %d: Failed to allocate PD\n", rank);
            exit(1);
        }

        // Create CQ with mlx5dv for GPU access (shared by all QPs)
        struct ibv_cq_init_attr_ex cq_attr = {};
        cq_attr.cqe = cq_depth;
        cq_attr.channel = nullptr;
        cq_attr.comp_vector = 0;
        cq_attr.wc_flags = IBV_WC_STANDARD_FLAGS;

        struct mlx5dv_cq_init_attr mlx5_cq_attr = {};
        mlx5_cq_attr.comp_mask = 0;

        struct ibv_cq_ex* cq_ex_ptr = mlx5dv_create_cq(ctx, &cq_attr, &mlx5_cq_attr);
        if (!cq_ex_ptr) {
            fprintf(stderr, "Rank %d: Failed to create CQ with mlx5dv\n", rank);
            exit(1);
        }
        cq = ibv_cq_ex_to_cq(cq_ex_ptr);

        // Query CQ extended info
        struct mlx5dv_obj obj = {};
        obj.cq.in = cq;
        obj.cq.out = &cq_ex;
        int ret = mlx5dv_init_obj(&obj, MLX5DV_OBJ_CQ);
        if (ret) {
            fprintf(stderr, "Rank %d: Failed to query mlx5dv CQ: %s\n", rank, strerror(ret));
            exit(1);
        }
    }

    void allocate_cq_gpu_resources() {
        // Map CQ buffer to GPU (shared by all QPs)
        size_t cqe_buf_size = cq_ex.cqe_cnt * cq_ex.cqe_size;
        cudaError_t err = cudaHostRegister(cq_ex.buf, cqe_buf_size, cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostRegister for CQ buffer failed: %s\n",
                    rank, cudaGetErrorString(err));
            d_cqe = nullptr;
        } else {
            err = cudaHostGetDevicePointer((void**)&d_cqe, cq_ex.buf, 0);
            if (err != cudaSuccess) {
                d_cqe = nullptr;
            }
        }

        if (rank == 0) {
            printf("Shared CQ: cqn=%u, cqe_cnt=%u, d_cqe=%p\n",
                   cq_ex.cqn, cq_ex.cqe_cnt, (void*)d_cqe);
        }
    }

    void allocate_peer_qp_gpu_resources(PerPeerQp& pqp) {
        // Map NIC's WQE buffer to GPU
        size_t wqe_buf_size = pqp.qp_ex.sq.wqe_cnt * pqp.qp_ex.sq.stride;
        void* h_wqe = pqp.qp_ex.sq.buf;

        cudaError_t err = cudaHostRegister(h_wqe, wqe_buf_size, cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            pqp.d_wqe_buf = h_wqe;  // Fallback
        } else {
            err = cudaHostGetDevicePointer(&pqp.d_wqe_buf, h_wqe, 0);
            if (err != cudaSuccess) {
                pqp.d_wqe_buf = h_wqe;
            }
        }

        // Map doorbell
        volatile uint32_t* dbrec_base = pqp.qp_ex.dbrec;
        err = cudaHostRegister((void*)dbrec_base, 64,
            cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (err != cudaSuccess) {
            pqp.d_dbrec = dbrec_base;
        } else {
            err = cudaHostGetDevicePointer((void**)&pqp.d_dbrec, (void*)dbrec_base, 0);
            if (err != cudaSuccess) {
                pqp.d_dbrec = dbrec_base;
            }
        }

        // Map BlueFlame register
        if (pqp.qp_ex.bf.reg && pqp.qp_ex.bf.size > 0) {
            err = cudaHostRegister(pqp.qp_ex.bf.reg, pqp.qp_ex.bf.size,
                cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory);
            if (err != cudaSuccess) {
                pqp.d_bf_reg = (volatile uint64_t*)pqp.qp_ex.bf.reg;
            } else {
                err = cudaHostGetDevicePointer((void**)&pqp.d_bf_reg, pqp.qp_ex.bf.reg, 0);
                if (err != cudaSuccess) {
                    pqp.d_bf_reg = (volatile uint64_t*)pqp.qp_ex.bf.reg;
                }
            }
        }

        // Allocate producer index
        err = cudaHostAlloc((void**)&pqp.h_prod_idx, sizeof(uint64_t), cudaHostAllocMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostAlloc for prod_idx failed\n", rank);
            exit(1);
        }
        *pqp.h_prod_idx = 0;

        err = cudaHostGetDevicePointer((void**)&pqp.d_prod_idx, (void*)pqp.h_prod_idx, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for prod_idx failed\n", rank);
            exit(1);
        }
    }

    void free_peer_qp_gpu_resources(PerPeerQp& pqp) {
        if (pqp.h_prod_idx) cudaFreeHost((void*)pqp.h_prod_idx);
        if (pqp.qp_ex.sq.buf) cudaHostUnregister(pqp.qp_ex.sq.buf);
    }
};

}  // namespace gicc::mlx5
