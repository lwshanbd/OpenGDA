/**
 * mlx5_gda_context.hpp - MLX5 DevX context for GPU-triggered RDMA
 *
 * Uses mlx5dv DevX API to allow GPU to directly:
 *   - Build and write WQEs to NIC-mapped memory
 *   - Ring doorbells to trigger RDMA operations
 *   - Poll CQ for completions
 *
 * This enables true GPU-initiated RDMA without CPU intervention.
 */
#pragma once

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mpi_bootstrap.hpp"

namespace opengda {

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

// GPU-accessible QP state
struct GdaDeviceQp {
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
struct GdaRemotePeer {
    uint64_t buf_addr;               // Remote buffer address
    uint32_t rkey;                   // Remote key
    uint32_t qpn;                    // Remote QP number
    uint16_t lid;                    // Remote LID
    uint8_t  gid[16];                // Remote GID (for RoCE)
};

// Connection info for exchange
struct GdaConnInfo {
    uint32_t qpn;
    uint16_t lid;
    uint8_t  gid[16];
    uint32_t psn;
    uint64_t buf_addr;
    uint32_t rkey;
};

class Mlx5GdaContext {
public:
    // IB objects
    struct ibv_context* ctx;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct mlx5dv_qp qp_ex;          // Extended QP info from mlx5dv
    struct mlx5dv_cq cq_ex;          // Extended CQ info from mlx5dv

    // Device info
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    union ibv_gid gid;
    int port_num;
    std::string dev_name;

    // Per-peer state
    std::vector<GdaDeviceQp> peer_qps;
    std::vector<GdaRemotePeer> remote_peers;

    // MPI bootstrap
    MpiBootstrap& mpi;
    int rank;
    int size;

    // GPU-mapped resources
    void* d_wqe_buf;                 // GPU pointer to WQE buffer
    volatile uint64_t* d_prod_idx;   // GPU pointer to producer index
    volatile Mlx5Cqe64* d_cqe;       // GPU pointer to CQ entries

    // Host resources for GPU mapping
    void* h_wqe_buf;
    uint64_t* h_prod_idx;
    Mlx5Cqe64* h_cqe;

    // QP parameters
    uint32_t psn;
    uint32_t qp_depth;
    uint32_t cq_depth;

    Mlx5GdaContext(MpiBootstrap& mpi_, const char* device_name = nullptr, int port = 1)
        : ctx(nullptr), pd(nullptr), cq(nullptr), qp(nullptr),
          port_num(port), mpi(mpi_), rank(mpi_.rank), size(mpi_.size),
          d_wqe_buf(nullptr), d_prod_idx(nullptr), d_cqe(nullptr),
          h_wqe_buf(nullptr), h_prod_idx(nullptr), h_cqe(nullptr),
          qp_depth(256), cq_depth(512)
    {
        memset(&dev_attr, 0, sizeof(dev_attr));
        memset(&port_attr, 0, sizeof(port_attr));
        memset(&gid, 0, sizeof(gid));
        memset(&qp_ex, 0, sizeof(qp_ex));
        memset(&cq_ex, 0, sizeof(cq_ex));

        peer_qps.resize(size);
        remote_peers.resize(size);

        psn = (rand() & 0xFFFFFF);

        init_device(device_name);
        init_pd_cq();
        init_qp();
        query_mlx5dv_objects();
        allocate_gpu_resources();
    }

    ~Mlx5GdaContext() {
        free_gpu_resources();
        if (qp) ibv_destroy_qp(qp);
        if (cq) ibv_destroy_cq(cq);
        if (pd) ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }

    // No copy
    Mlx5GdaContext(const Mlx5GdaContext&) = delete;
    Mlx5GdaContext& operator=(const Mlx5GdaContext&) = delete;

    // Get local connection info
    GdaConnInfo get_local_info() const {
        GdaConnInfo info;
        info.qpn = qp->qp_num;
        info.lid = port_attr.lid;
        memcpy(info.gid, gid.raw, 16);
        info.psn = psn;
        info.buf_addr = 0;  // Will be set after buffer registration
        info.rkey = 0;
        return info;
    }

    // Connect to peer
    void connect_to_peer(int peer_rank, const GdaConnInfo& peer_info) {
        if (peer_rank == rank) return;

        // Store peer info
        remote_peers[peer_rank].qpn = peer_info.qpn;
        remote_peers[peer_rank].lid = peer_info.lid;
        memcpy(remote_peers[peer_rank].gid, peer_info.gid, 16);
        remote_peers[peer_rank].buf_addr = peer_info.buf_addr;
        remote_peers[peer_rank].rkey = peer_info.rkey;

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

        int ret = ibv_modify_qp(qp, &attr, flags);
        if (ret) {
            fprintf(stderr, "Rank %d: Failed to modify QP to RTR: %s\n",
                    rank, strerror(ret));
            exit(1);
        }

        // Transition to RTS
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = psn;
        attr.max_rd_atomic = 16;

        flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

        ret = ibv_modify_qp(qp, &attr, flags);
        if (ret) {
            fprintf(stderr, "Rank %d: Failed to modify QP to RTS: %s\n",
                    rank, strerror(ret));
            exit(1);
        }

        if (rank == 0) {
            printf("Rank %d: Connected to peer %d (QPN=%u)\n",
                   rank, peer_rank, peer_info.qpn);
        }
    }

    // Get GPU-accessible device QP info
    GdaDeviceQp get_device_qp() const {
        GdaDeviceQp dqp;
        dqp.qpn = qp->qp_num;
        dqp.nwqes = qp_depth;
        dqp.wqe_buf = d_wqe_buf;
        dqp.wqe_lkey = 0;  // TODO: Register WQE buffer
        dqp.dbrec = (volatile uint32_t*)qp_ex.dbrec;
        dqp.prod_idx = d_prod_idx;
        dqp.cqe = d_cqe;
        dqp.cqn = cq_ex.cqn;
        dqp.ncqes = cq_depth;
        dqp.cq_cons_idx = nullptr;  // TODO
        dqp.cq_dbrec = (volatile uint32_t*)cq_ex.dbrec;
        return dqp;
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

        // Create CQ with mlx5dv for GPU access
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
    }

    void init_qp() {
        // Create QP - standard ibv for compatibility
        // We'll use mlx5dv_init_obj later to query internal structures
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

        qp = ibv_create_qp(pd, &qp_init_attr);
        if (!qp) {
            fprintf(stderr, "Rank %d: Failed to create QP: %s\n",
                    rank, strerror(errno));
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
        int ret = ibv_modify_qp(qp, &attr, flags);
        if (ret) {
            fprintf(stderr, "Rank %d: Failed to modify QP to INIT: %s\n",
                    rank, strerror(ret));
            exit(1);
        }
    }

    void query_mlx5dv_objects() {
        // Query extended QP and CQ info
        struct mlx5dv_obj obj = {};
        obj.qp.in = qp;
        obj.qp.out = &qp_ex;
        obj.cq.in = cq;
        obj.cq.out = &cq_ex;

        int ret = mlx5dv_init_obj(&obj, MLX5DV_OBJ_QP | MLX5DV_OBJ_CQ);
        if (ret) {
            fprintf(stderr, "Rank %d: Failed to query mlx5dv objects: %s\n",
                    rank, strerror(ret));
            exit(1);
        }

        if (rank == 0) {
            printf("MLX5 QP info:\n");
            printf("  sq.buf = %p\n", qp_ex.sq.buf);
            printf("  sq.wqe_cnt = %u\n", qp_ex.sq.wqe_cnt);
            printf("  sq.stride = %u\n", qp_ex.sq.stride);
            printf("  dbrec = %p\n", qp_ex.dbrec);
            printf("  bf.reg = %p\n", qp_ex.bf.reg);
            printf("  bf.size = %u\n", qp_ex.bf.size);
            printf("MLX5 CQ info:\n");
            printf("  buf = %p\n", cq_ex.buf);
            printf("  cqe_cnt = %u\n", cq_ex.cqe_cnt);
            printf("  cqn = %u\n", cq_ex.cqn);
            printf("  dbrec = %p\n", cq_ex.dbrec);
            fflush(stdout);
        }
    }

    void allocate_gpu_resources() {
        // Map NIC's WQE buffer (qp_ex.sq.buf) to GPU
        // This is the key for GPU-triggered RDMA: GPU writes WQEs directly to NIC's buffer
        size_t wqe_buf_size = qp_ex.sq.wqe_cnt * qp_ex.sq.stride;

        h_wqe_buf = qp_ex.sq.buf;  // Use NIC's WQE buffer directly

        // Register NIC's WQE buffer with CUDA for GPU access
        cudaError_t err = cudaHostRegister(h_wqe_buf, wqe_buf_size, cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostRegister for NIC WQE buffer failed: %s\n",
                    rank, cudaGetErrorString(err));
            fprintf(stderr, "  WQE buffer address: %p, size: %zu\n", h_wqe_buf, wqe_buf_size);
            // Try without registration - direct access might work on some systems
            d_wqe_buf = h_wqe_buf;
        } else {
            err = cudaHostGetDevicePointer(&d_wqe_buf, h_wqe_buf, 0);
            if (err != cudaSuccess) {
                fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for WQE failed: %s\n",
                        rank, cudaGetErrorString(err));
                d_wqe_buf = h_wqe_buf;  // Fallback to host pointer
            }
        }

        // Also register doorbell for GPU access
        // Note: dbrec is a 64-bit region (SQ dbrec + RQ dbrec)
        volatile uint32_t* dbrec_base = qp_ex.dbrec;
        err = cudaHostRegister((void*)dbrec_base, 64, cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            if (rank == 0) {
                printf("Note: cudaHostRegister for dbrec failed: %s (using direct access)\n",
                       cudaGetErrorString(err));
            }
        }

        // Allocate producer index
        err = cudaHostAlloc((void**)&h_prod_idx, sizeof(uint64_t),
                            cudaHostAllocMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostAlloc for prod_idx failed\n", rank);
            exit(1);
        }
        *h_prod_idx = 0;

        err = cudaHostGetDevicePointer((void**)&d_prod_idx, (void*)h_prod_idx, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for prod_idx failed\n", rank);
            exit(1);
        }

        // Map CQ buffer to GPU
        size_t cqe_buf_size = cq_ex.cqe_cnt * cq_ex.cqe_size;
        err = cudaHostRegister(cq_ex.buf, cqe_buf_size, cudaHostRegisterDefault);
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
            printf("GPU resources:\n");
            printf("  NIC WQE buffer: %p (stride=%u, cnt=%u)\n",
                   qp_ex.sq.buf, qp_ex.sq.stride, qp_ex.sq.wqe_cnt);
            printf("  d_wqe_buf = %p\n", d_wqe_buf);
            printf("  dbrec = %p\n", (void*)qp_ex.dbrec);
            printf("  d_prod_idx = %p\n", d_prod_idx);
            printf("  d_cqe = %p\n", d_cqe);
            fflush(stdout);
        }
    }

    void free_gpu_resources() {
        if (d_cqe && cq_ex.buf) {
            cudaHostUnregister(cq_ex.buf);
        }
        if (h_prod_idx) cudaFreeHost((void*)h_prod_idx);
        // Don't free h_wqe_buf - it's the NIC's buffer (qp_ex.sq.buf)
        // Just unregister it
        if (h_wqe_buf && h_wqe_buf == qp_ex.sq.buf) {
            cudaHostUnregister(h_wqe_buf);
        }
    }
};

}  // namespace opengda
