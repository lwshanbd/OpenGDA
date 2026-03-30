/**
 * mlx5_devx_qp.hpp - DevX-based QP for GPU-triggered RDMA
 *
 * Creates QP using DevX API with GPU-accessible:
 *   - UAR (BlueFlame register)
 *   - WQE buffer
 *   - Doorbell record
 *
 * This enables true GPU-initiated RDMA without CPU intervention.
 */
#pragma once

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <cuda_runtime.h>
#include <endian.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// MLX5 IFC structures for DevX
#include "mlx5_ifc.h"
#include "mlx5_prm.h"

namespace gicc::mlx5 {

// Constants
#define MLX5_SEND_WQE_BB 64
#define MLX5_SEND_WQE_SHIFT 6
#define GPAGE_SIZE 65536  // 64KB alignment for NIC buffers
#define DBREC_SIZE 8      // 8 bytes for doorbell record
#define NC_UAR_SIZE 8     // Non-cached UAR write size

// QP states
enum {
    MLX5_QPC_ST_RC = 0x0,
    MLX5_QPC_ST_DCI = 0x5,
};

enum {
    MLX5_QPC_PM_STATE_MIGRATED = 0x3,
};

// DevX QP that provides GPU-accessible resources
class DevxQp {
public:
    // IB context
    struct ibv_context* ctx;
    struct ibv_pd* pd;

    // DevX UAR (GPU-accessible)
    struct mlx5dv_devx_uar* uar;
    void* h_uar_reg;        // Host pointer to UAR register
    void* d_uar_reg;        // GPU pointer to UAR register
    size_t uar_reg_size;

    // WQ buffer (GPU-accessible)
    struct mlx5dv_devx_umem* wq_umem;
    void* h_wq_buf;         // Host pointer to WQ buffer
    void* d_wq_buf;         // GPU pointer to WQ buffer
    size_t wq_buf_size;
    uint32_t log_wq_size;   // log2 of number of WQEs

    // Doorbell record (GPU-accessible)
    // Note: MLX5 doorbell record has two 32-bit values:
    //   offset 0: Receive Queue (RQ) doorbell
    //   offset 4: Send Queue (SQ) doorbell
    // We point to the SQ doorbell (offset 4)
    struct mlx5dv_devx_umem* dbr_umem;
    void* h_dbr_base;             // Base allocation (for CUDA unregister/free)
    volatile uint32_t* h_dbrec;   // Host pointer to SQ doorbell (offset 4)
    volatile uint32_t* d_dbrec;   // GPU pointer to SQ doorbell (offset 4)

    // DevX QP object
    struct mlx5dv_devx_obj* devx_qp;
    uint32_t qpn;           // QP number

    // DevX CQ (for send completions)
    struct mlx5dv_devx_obj* devx_cq;
    struct mlx5dv_devx_umem* cq_umem;
    struct mlx5dv_devx_umem* cq_dbr_umem;
    void* h_cq_buf;
    void* d_cq_buf;
    volatile uint32_t* h_cq_dbrec;
    volatile uint32_t* d_cq_dbrec;
    uint32_t cqn;
    uint32_t num_cqe;

    // PD number (from ibv_pd)
    int pdn;

    // SRQ (Shared Receive Queue) for RC QP
    struct ibv_srq* srq;
    uint32_t srqn;
    struct ibv_cq* recv_cq;

    // Producer index (GPU-managed)
    volatile uint64_t* h_prod_idx;
    volatile uint64_t* d_prod_idx;

    int rank;
    int port_num;

    DevxQp(struct ibv_context* ctx_, struct ibv_pd* pd_, int rank_, int port_ = 1,
           uint32_t qp_depth = 256, uint32_t cq_depth = 512)
        : ctx(ctx_), pd(pd_), rank(rank_), port_num(port_),
          uar(nullptr), h_uar_reg(nullptr), d_uar_reg(nullptr), uar_reg_size(0),
          wq_umem(nullptr), h_wq_buf(nullptr), d_wq_buf(nullptr), wq_buf_size(0),
          dbr_umem(nullptr), h_dbr_base(nullptr), h_dbrec(nullptr), d_dbrec(nullptr),
          devx_qp(nullptr), qpn(0),
          devx_cq(nullptr), cq_umem(nullptr), cq_dbr_umem(nullptr),
          h_cq_buf(nullptr), d_cq_buf(nullptr),
          h_cq_dbrec(nullptr), d_cq_dbrec(nullptr),
          cqn(0), num_cqe(cq_depth),
          pdn(0), srq(nullptr), srqn(0), recv_cq(nullptr),
          h_prod_idx(nullptr), d_prod_idx(nullptr)
    {
        // Get PD number
        struct mlx5dv_obj dv_obj = {};
        struct mlx5dv_pd dvpd = {};
        dv_obj.pd.in = pd;
        dv_obj.pd.out = &dvpd;
        int ret = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_PD);
        if (ret) {
            fprintf(stderr, "Rank %d: mlx5dv_init_obj for PD failed\n", rank);
            exit(1);
        }
        pdn = dvpd.pdn;

        // Calculate WQ size (power of 2)
        uint32_t num_wqe = 1;
        log_wq_size = 0;
        while (num_wqe < qp_depth) {
            num_wqe <<= 1;
            log_wq_size++;
        }
        wq_buf_size = num_wqe * MLX5_SEND_WQE_BB;

        // Allocate and map UAR
        allocate_uar();

        // Allocate and map WQ buffer
        allocate_wq_buffer();

        // Allocate and map doorbell record
        allocate_dbrec();

        // Allocate and map CQ
        allocate_cq();

        // Create SRQ for RC QP
        create_srq();

        // Create QP using DevX
        create_devx_qp();

        // Allocate producer index
        allocate_prod_idx();
    }

    ~DevxQp() {
        if (devx_qp) mlx5dv_devx_obj_destroy(devx_qp);
        if (devx_cq) mlx5dv_devx_obj_destroy(devx_cq);
        if (srq) ibv_destroy_srq(srq);
        if (recv_cq) ibv_destroy_cq(recv_cq);

        // Free GPU resources
        if (d_prod_idx) {
            cudaHostUnregister((void*)h_prod_idx);
            cudaFreeHost((void*)h_prod_idx);
        }
        if (d_cq_dbrec) cudaHostUnregister((void*)h_cq_dbrec);
        if (d_cq_buf) cudaHostUnregister(h_cq_buf);
        if (d_dbrec) cudaHostUnregister(h_dbr_base);
        if (d_wq_buf) cudaHostUnregister(h_wq_buf);
        if (d_uar_reg) cudaHostUnregister(h_uar_reg);

        // Free umems
        if (cq_dbr_umem) mlx5dv_devx_umem_dereg(cq_dbr_umem);
        if (cq_umem) mlx5dv_devx_umem_dereg(cq_umem);
        if (dbr_umem) mlx5dv_devx_umem_dereg(dbr_umem);
        if (wq_umem) mlx5dv_devx_umem_dereg(wq_umem);

        // Free host buffers
        if (h_cq_dbrec) free((void*)h_cq_dbrec);
        if (h_cq_buf) free(h_cq_buf);
        if (h_dbr_base) free(h_dbr_base);
        if (h_wq_buf) free(h_wq_buf);

        // Free UAR
        if (uar) mlx5dv_devx_free_uar(uar);
    }

    // Create dummy ibv_qp wrapper for using ibv_modify_qp
    // The QP was created with DevX but we need ibv_qp for state transitions
    struct ibv_qp* create_dummy_qp() {
        // For DevX QPs, we need to create a dummy ibv_qp pointing to our QPN
        // This is a bit of a hack but allows us to use ibv_modify_qp
        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = recv_cq;  // Use our recv_cq
        qp_init_attr.recv_cq = recv_cq;
        qp_init_attr.srq = srq;
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.cap.max_send_wr = 16;
        qp_init_attr.cap.max_recv_wr = 16;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;

        return ibv_create_qp(pd, &qp_init_attr);
    }

    // Transition QP RST -> INIT (DevX)
    void rst2init() {
        uint8_t cmd_in[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {0};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {0};

        DEVX_SET(rst2init_qp_in, cmd_in, opcode, MLX5_CMD_OP_RST2INIT_QP);
        DEVX_SET(rst2init_qp_in, cmd_in, qpn, qpn);

        void* qpc = DEVX_ADDR_OF(rst2init_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, port_num);
        DEVX_SET(qpc, qpc, primary_address_path.pkey_index, 0);
        DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
        DEVX_SET(qpc, qpc, rwe, 1);  // Remote write enable
        DEVX_SET(qpc, qpc, rre, 1);  // Remote read enable
        DEVX_SET(qpc, qpc, rae, 1);  // Remote atomic enable
        DEVX_SET(qpc, qpc, atomic_mode, 0x3);  // Up to 64-bit atomics

        int ret = mlx5dv_devx_obj_modify(devx_qp, cmd_in, sizeof(cmd_in),
                                         cmd_out, sizeof(cmd_out));
        if (ret) {
            uint32_t syndrome = DEVX_GET(rst2init_qp_out, cmd_out, syndrome);
            fprintf(stderr, "Rank %d: RST2INIT failed: %s (syndrome=0x%x)\n",
                    rank, strerror(errno), syndrome);
            exit(1);
        }
    }

    // Transition QP INIT -> RTR (InfiniBand)
    void init2rtr_ib(uint32_t dest_qpn, uint16_t dest_lid, uint8_t* dest_gid,
                     uint32_t remote_psn, int mtu = -1) {
        // Auto-detect MTU from port if not specified
        if (mtu < 0) {
            struct ibv_port_attr port_attr;
            if (ibv_query_port(ctx, port_num, &port_attr) == 0) {
                mtu = port_attr.active_mtu;
            } else {
                mtu = 3;  // Default to 1024 if query fails
            }
        }

        uint8_t cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {0};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {0};

        DEVX_SET(init2rtr_qp_in, cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
        DEVX_SET(init2rtr_qp_in, cmd_in, qpn, qpn);

        void* qpc = DEVX_ADDR_OF(init2rtr_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, mtu, mtu);
        DEVX_SET(qpc, qpc, log_msg_max, 30);
        DEVX_SET(qpc, qpc, remote_qpn, dest_qpn);
        DEVX_SET(qpc, qpc, min_rnr_nak, 12);
        DEVX_SET(qpc, qpc, log_rra_max, 4);
        DEVX_SET(qpc, qpc, next_rcv_psn, remote_psn);

        // IB address path
        DEVX_SET(qpc, qpc, primary_address_path.rlid, dest_lid);
        DEVX_SET(qpc, qpc, primary_address_path.grh, 0);
        DEVX_SET(qpc, qpc, primary_address_path.mlid, 0);
        DEVX_SET(qpc, qpc, primary_address_path.sl, 0);

        int ret = mlx5dv_devx_obj_modify(devx_qp, cmd_in, sizeof(cmd_in),
                                         cmd_out, sizeof(cmd_out));
        if (ret) {
            uint32_t syndrome = DEVX_GET(init2rtr_qp_out, cmd_out, syndrome);
            fprintf(stderr, "Rank %d: INIT2RTR (IB) failed: %s (syndrome=0x%x)\n",
                    rank, strerror(errno), syndrome);
            exit(1);
        }
    }

    // Find a valid RoCE v2 GID index
    int find_roce_gid_index() {
        struct ibv_port_attr port_attr;
        if (ibv_query_port(ctx, port_num, &port_attr)) {
            return 0;  // Default to 0
        }

        // Iterate through GIDs to find a RoCE v2 one
        for (int i = 0; i < port_attr.gid_tbl_len; i++) {
            union ibv_gid gid;
            if (ibv_query_gid(ctx, port_num, i, &gid)) continue;

            // Skip null GIDs
            if (gid.global.subnet_prefix == 0 && gid.global.interface_id == 0) continue;

            // Check RoCE version via sysfs
            char path[256];
            const char* dev_name = ibv_get_device_name(ctx->device);
            snprintf(path, sizeof(path),
                     "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d",
                     dev_name, port_num, i);

            FILE* f = fopen(path, "r");
            if (f) {
                char buf[32] = {0};
                if (fgets(buf, sizeof(buf), f)) {
                    fclose(f);
                    // Look for RoCE v2 (preferred) or RoCE v1
                    if (strstr(buf, "RoCE v2") || strstr(buf, "ROCEv2")) {
                        return i;
                    }
                } else {
                    fclose(f);
                }
            }
        }

        // Fallback: return first non-null GID
        for (int i = 0; i < port_attr.gid_tbl_len; i++) {
            union ibv_gid gid;
            if (ibv_query_gid(ctx, port_num, i, &gid)) continue;
            if (gid.global.subnet_prefix != 0 || gid.global.interface_id != 0) {
                return i;
            }
        }
        return 0;
    }

    // Transition QP INIT -> RTR (RoCE - following nvshmem's minimal approach)
    void init2rtr(uint32_t dest_qpn, uint16_t dest_lid, uint8_t* dest_gid,
                  uint32_t remote_psn, int mtu = -1, int gid_index = -1) {
        // Auto-detect GID index if not specified
        if (gid_index < 0) {
            gid_index = find_roce_gid_index();
        }

        // Query port to get active MTU
        struct ibv_port_attr port_attr;
        if (ibv_query_port(ctx, port_num, &port_attr)) {
            fprintf(stderr, "Rank %d: ibv_query_port failed\n", rank);
            exit(1);
        }

        // Auto-detect MTU from port if not specified
        if (mtu < 0) {
            // Use active_mtu from port (what the network actually supports)
            mtu = port_attr.active_mtu;  // ibv_mtu enum: 1=256, 2=512, 3=1024, 4=2048, 5=4096

            // Check for environment override
            const char* mtu_env = getenv("GDA_MTU");
            if (mtu_env) {
                int mtu_bytes = atoi(mtu_env);
                int requested_mtu;
                if (mtu_bytes <= 256) requested_mtu = 1;
                else if (mtu_bytes <= 512) requested_mtu = 2;
                else if (mtu_bytes <= 1024) requested_mtu = 3;
                else if (mtu_bytes <= 2048) requested_mtu = 4;
                else requested_mtu = 5;

                // Cap at active MTU
                if (requested_mtu <= (int)port_attr.active_mtu) {
                    mtu = requested_mtu;
                } else {
                    fprintf(stderr, "Rank %d: Requested MTU %d bytes exceeds active MTU, using active MTU %d\n",
                            rank, mtu_bytes, 256 << (port_attr.active_mtu - 1));
                }
            }
        }

        uint8_t cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {0};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {0};

        DEVX_SET(init2rtr_qp_in, cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
        DEVX_SET(init2rtr_qp_in, cmd_in, qpn, qpn);

        void* qpc = DEVX_ADDR_OF(init2rtr_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, mtu, mtu);
        DEVX_SET(qpc, qpc, log_msg_max, 30);
        DEVX_SET(qpc, qpc, remote_qpn, dest_qpn);
        DEVX_SET(qpc, qpc, min_rnr_nak, 12);
        DEVX_SET(qpc, qpc, log_rra_max, 4);
        // Note: nvshmem does NOT set next_rcv_psn

        // port_attr was already queried above for MTU detection

        if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
            // InfiniBand path
            DEVX_SET(qpc, qpc, primary_address_path.rlid, dest_lid);
            DEVX_SET(qpc, qpc, primary_address_path.mlid, 0);
            DEVX_SET(qpc, qpc, primary_address_path.sl, 0);
            DEVX_SET(qpc, qpc, primary_address_path.grh, 0);
        } else {
            // RoCE path - following nvshmem's ibgda_rc_init2rtr exactly
            struct ibv_ah_attr ah_attr = {};
            ah_attr.is_global = 1;
            ah_attr.port_num = port_num;
            ah_attr.grh.dgid.global.subnet_prefix = ((uint64_t*)dest_gid)[0];
            ah_attr.grh.dgid.global.interface_id = ((uint64_t*)dest_gid)[1];
            ah_attr.grh.sgid_index = gid_index;
            ah_attr.grh.traffic_class = 0;  // Default traffic class
            ah_attr.sl = 0;
            ah_attr.src_path_bits = 0;
            // nvshmem: ah_attr.dlid = port_attr->lid | IBGDA_ROCE_V2_UDP_SPORT_BASE
            ah_attr.dlid = port_attr.lid | 0xC000;  // RoCE v2 UDP sport

            struct ibv_ah* ah = ibv_create_ah(pd, &ah_attr);
            if (!ah) {
                fprintf(stderr, "Rank %d: ibv_create_ah failed: %s\n",
                        rank, strerror(errno));
                exit(1);
            }

            // Get AH internal data
            struct mlx5dv_obj dv = {};
            struct mlx5dv_ah dah = {};
            dv.ah.in = ah;
            dv.ah.out = &dah;
            int dv_ret = mlx5dv_init_obj(&dv, MLX5DV_OBJ_AH);
            if (dv_ret) {
                fprintf(stderr, "Rank %d: mlx5dv_init_obj for AH failed: %d\n", rank, dv_ret);
                exit(1);
            }

            // Set RoCE address path - ONLY these fields per nvshmem
            memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32),
                   &dah.av->rmac, 6);
            DEVX_SET(qpc, qpc, primary_address_path.hop_limit, 255);  // nvshmem uses 255
            DEVX_SET(qpc, qpc, primary_address_path.src_addr_index, gid_index);
            DEVX_SET(qpc, qpc, primary_address_path.eth_prio, 0);  // sl
            DEVX_SET(qpc, qpc, primary_address_path.udp_sport, ah_attr.dlid);  // from nvshmem
            DEVX_SET(qpc, qpc, primary_address_path.dscp, 0);  // traffic_class >> 2
            memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip),
                   &dah.av->rgid, 16);

            ibv_destroy_ah(ah);
        }

        int ret = mlx5dv_devx_obj_modify(devx_qp, cmd_in, sizeof(cmd_in),
                                         cmd_out, sizeof(cmd_out));
        if (ret) {
            uint32_t syndrome = DEVX_GET(init2rtr_qp_out, cmd_out, syndrome);
            fprintf(stderr, "Rank %d: INIT2RTR failed: %s (syndrome=0x%x)\n",
                    rank, strerror(errno), syndrome);
            fprintf(stderr, "  dest_qpn=%u, mtu=%d, link_layer=%d\n",
                    dest_qpn, mtu, port_attr.link_layer);
            exit(1);
        }
    }

    // Transition QP RTR -> RTS
    void rtr2rts(uint32_t local_psn) {
        uint8_t cmd_in[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {0};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {0};

        DEVX_SET(rtr2rts_qp_in, cmd_in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
        DEVX_SET(rtr2rts_qp_in, cmd_in, qpn, qpn);

        void* qpc = DEVX_ADDR_OF(rtr2rts_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, log_ack_req_freq, 0);  // ACK every packet
        DEVX_SET(qpc, qpc, log_sra_max, 4);       // Max 16 outstanding reads
        DEVX_SET(qpc, qpc, next_send_psn, local_psn);
        DEVX_SET(qpc, qpc, retry_count, 7);
        DEVX_SET(qpc, qpc, rnr_retry, 7);
        DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, 14);

        int ret = mlx5dv_devx_obj_modify(devx_qp, cmd_in, sizeof(cmd_in),
                                         cmd_out, sizeof(cmd_out));
        if (ret) {
            fprintf(stderr, "Rank %d: RTR2RTS failed: %s\n", rank, strerror(errno));
            exit(1);
        }
    }

private:
    void allocate_uar() {
        // Allocate DevX UAR - this will be used for BlueFlame doorbell
        // Try dedicated NC UAR first, then fall back to BF UAR
#ifdef MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED
        uar = mlx5dv_devx_alloc_uar(ctx, MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED);
        if (uar) {
            uar_reg_size = NC_UAR_SIZE;
        }
#endif
        if (!uar) {
            uar = mlx5dv_devx_alloc_uar(ctx, MLX5DV_UAR_ALLOC_TYPE_BF);
            if (!uar) {
                fprintf(stderr, "Rank %d: Failed to allocate DevX UAR\n", rank);
                exit(1);
            }
            // Query BF size from HCA capabilities
            uar_reg_size = 512;  // Default BF size
        }

        h_uar_reg = uar->reg_addr;

        // Register UAR with CUDA for GPU access
        cudaError_t err = cudaHostRegister(h_uar_reg, uar_reg_size,
            cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostRegister for UAR failed: %s\n",
                    rank, cudaGetErrorString(err));
            exit(1);
        }

        err = cudaHostGetDevicePointer(&d_uar_reg, h_uar_reg, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for UAR failed: %s\n",
                    rank, cudaGetErrorString(err));
            exit(1);
        }
    }

    void allocate_wq_buffer() {
        // Allocate aligned host memory for WQ buffer
        size_t aligned_size = ((wq_buf_size + GPAGE_SIZE - 1) / GPAGE_SIZE) * GPAGE_SIZE;
        int ret = posix_memalign(&h_wq_buf, GPAGE_SIZE, aligned_size);
        if (ret) {
            fprintf(stderr, "Rank %d: posix_memalign for WQ failed\n", rank);
            exit(1);
        }
        memset(h_wq_buf, 0, aligned_size);

        // Register with NIC
        wq_umem = mlx5dv_devx_umem_reg(ctx, h_wq_buf, aligned_size, IBV_ACCESS_LOCAL_WRITE);
        if (!wq_umem) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_umem_reg for WQ failed\n", rank);
            exit(1);
        }

        // Register with CUDA for GPU access
        cudaError_t err = cudaHostRegister(h_wq_buf, aligned_size,
            cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostRegister for WQ failed: %s\n",
                    rank, cudaGetErrorString(err));
            exit(1);
        }

        err = cudaHostGetDevicePointer(&d_wq_buf, h_wq_buf, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for WQ failed: %s\n",
                    rank, cudaGetErrorString(err));
            exit(1);
        }
    }

    void allocate_dbrec() {
        // Allocate aligned host memory for doorbell record
        // Layout: [RQ doorbell (4 bytes)][SQ doorbell (4 bytes)]
        int ret = posix_memalign(&h_dbr_base, GPAGE_SIZE, GPAGE_SIZE);
        if (ret) {
            fprintf(stderr, "Rank %d: posix_memalign for DBR failed\n", rank);
            exit(1);
        }
        memset(h_dbr_base, 0, GPAGE_SIZE);

        // Register with NIC
        dbr_umem = mlx5dv_devx_umem_reg(ctx, h_dbr_base, GPAGE_SIZE, IBV_ACCESS_LOCAL_WRITE);
        if (!dbr_umem) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_umem_reg for DBR failed\n", rank);
            exit(1);
        }

        // Register with CUDA for GPU access
        cudaError_t err = cudaHostRegister(h_dbr_base, GPAGE_SIZE,
            cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostRegister for DBR failed: %s\n",
                    rank, cudaGetErrorString(err));
            exit(1);
        }

        void* d_dbr_base;
        err = cudaHostGetDevicePointer(&d_dbr_base, h_dbr_base, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for DBR failed: %s\n",
                    rank, cudaGetErrorString(err));
            exit(1);
        }

        // Point to SQ doorbell at offset 4 (not RQ doorbell at offset 0)
        h_dbrec = (volatile uint32_t*)((uintptr_t)h_dbr_base + sizeof(uint32_t));
        d_dbrec = (volatile uint32_t*)((uintptr_t)d_dbr_base + sizeof(uint32_t));
    }

    void allocate_cq() {
        // Allocate CQ buffer
        size_t cq_buf_size = num_cqe * 64;  // 64 bytes per CQE
        size_t aligned_size = ((cq_buf_size + GPAGE_SIZE - 1) / GPAGE_SIZE) * GPAGE_SIZE;

        int ret = posix_memalign(&h_cq_buf, GPAGE_SIZE, aligned_size);
        if (ret) {
            fprintf(stderr, "Rank %d: posix_memalign for CQ failed\n", rank);
            exit(1);
        }
        // Initialize CQ with 0xFF (owner bit = 1 initially)
        memset(h_cq_buf, 0xFF, aligned_size);

        // Register CQ buffer with NIC
        cq_umem = mlx5dv_devx_umem_reg(ctx, h_cq_buf, aligned_size, IBV_ACCESS_LOCAL_WRITE);
        if (!cq_umem) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_umem_reg for CQ failed\n", rank);
            exit(1);
        }

        // Allocate CQ doorbell record
        ret = posix_memalign((void**)&h_cq_dbrec, GPAGE_SIZE, GPAGE_SIZE);
        if (ret) {
            fprintf(stderr, "Rank %d: posix_memalign for CQ DBR failed\n", rank);
            exit(1);
        }
        memset((void*)h_cq_dbrec, 0, GPAGE_SIZE);

        // Register CQ DBR with NIC
        cq_dbr_umem = mlx5dv_devx_umem_reg(ctx, (void*)h_cq_dbrec, GPAGE_SIZE, IBV_ACCESS_LOCAL_WRITE);
        if (!cq_dbr_umem) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_umem_reg for CQ DBR failed\n", rank);
            exit(1);
        }

        // Register with CUDA
        cudaError_t err = cudaHostRegister(h_cq_buf, aligned_size,
            cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostRegister for CQ failed: %s\n",
                    rank, cudaGetErrorString(err));
        } else {
            cudaHostGetDevicePointer(&d_cq_buf, h_cq_buf, 0);
        }

        err = cudaHostRegister((void*)h_cq_dbrec, GPAGE_SIZE,
            cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostRegister for CQ DBR failed: %s\n",
                    rank, cudaGetErrorString(err));
        } else {
            cudaHostGetDevicePointer((void**)&d_cq_dbrec, (void*)h_cq_dbrec, 0);
        }

        // Query EQN
        uint32_t eqn;
        ret = mlx5dv_devx_query_eqn(ctx, 0, &eqn);
        if (ret) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_query_eqn failed\n", rank);
            exit(1);
        }

        // Create CQ with DevX
        uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_cq_in)] = {0};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_cq_out)] = {0};

        uint32_t log_cq_size = 0;
        uint32_t ncqe = 1;
        while (ncqe < num_cqe) {
            ncqe <<= 1;
            log_cq_size++;
        }

        DEVX_SET(create_cq_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_CQ);
        DEVX_SET(create_cq_in, cmd_in, cq_umem_id, cq_umem->umem_id);
        DEVX_SET(create_cq_in, cmd_in, cq_umem_valid, 1);
        DEVX_SET64(create_cq_in, cmd_in, cq_umem_offset, 0);

        void* cqc = DEVX_ADDR_OF(create_cq_in, cmd_in, cq_context);
        DEVX_SET(cqc, cqc, dbr_umem_valid, 1);
        DEVX_SET(cqc, cqc, cqe_sz, 0);  // 64 byte CQEs
        DEVX_SET(cqc, cqc, cc, 0);      // Non-collapsed CQ (each WQE gets its own CQE slot)
        DEVX_SET(cqc, cqc, oi, 1);      // Overrun ignore
        DEVX_SET(cqc, cqc, scqe_break_moderation_en, 0);  // Disable moderation
        DEVX_SET(cqc, cqc, dbr_umem_id, cq_dbr_umem->umem_id);
        DEVX_SET(cqc, cqc, log_cq_size, log_cq_size);
        DEVX_SET(cqc, cqc, uar_page, uar->page_id);
        DEVX_SET(cqc, cqc, c_eqn, eqn);
        DEVX_SET(cqc, cqc, log_page_size, 16 - 12);  // 64KB pages
        DEVX_SET64(cqc, cqc, dbr_addr, 0);

        devx_cq = mlx5dv_devx_obj_create(ctx, cmd_in, sizeof(cmd_in),
                                          cmd_out, sizeof(cmd_out));
        if (!devx_cq) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_obj_create for CQ failed: %s\n",
                    rank, strerror(errno));
            exit(1);
        }

        cqn = DEVX_GET(create_cq_out, cmd_out, cqn);
    }

    void create_srq() {
        // Create a receive CQ for SRQ
        recv_cq = ibv_create_cq(ctx, 64, nullptr, nullptr, 0);
        if (!recv_cq) {
            fprintf(stderr, "Rank %d: ibv_create_cq for recv failed: %s\n",
                    rank, strerror(errno));
            exit(1);
        }

        // Create SRQ
        struct ibv_srq_init_attr srq_init_attr = {};
        srq_init_attr.attr.max_wr = 64;
        srq_init_attr.attr.max_sge = 1;
        srq = ibv_create_srq(pd, &srq_init_attr);
        if (!srq) {
            fprintf(stderr, "Rank %d: ibv_create_srq failed: %s\n",
                    rank, strerror(errno));
            exit(1);
        }

        // Get SRQ number using mlx5dv
        struct mlx5dv_obj dv = {};
        struct mlx5dv_srq dvsrq = {};
        memset(&dv, 0, sizeof(dv));
        memset(&dvsrq, 0, sizeof(dvsrq));
        dvsrq.comp_mask = MLX5DV_SRQ_MASK_SRQN;  // Request SRQN field
        dv.srq.in = srq;
        dv.srq.out = &dvsrq;
        int ret = mlx5dv_init_obj(&dv, MLX5DV_OBJ_SRQ);
        if (ret) {
            fprintf(stderr, "Rank %d: mlx5dv_init_obj for SRQ failed: %d\n", rank, ret);
            exit(1);
        }
        srqn = dvsrq.srqn;
    }

    void create_devx_qp() {
        uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_qp_in)] = {0};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_qp_out)] = {0};

        DEVX_SET(create_qp_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_QP);
        DEVX_SET(create_qp_in, cmd_in, wq_umem_id, wq_umem->umem_id);
        DEVX_SET(create_qp_in, cmd_in, wq_umem_valid, 1);
        DEVX_SET64(create_qp_in, cmd_in, wq_umem_offset, 0);

        void* qpc = DEVX_ADDR_OF(create_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, st, MLX5_QPC_ST_RC);           // RC QP
        DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
        DEVX_SET(qpc, qpc, pd, pdn);
        DEVX_SET(qpc, qpc, uar_page, uar->page_id);       // Our GPU-accessible UAR!
        DEVX_SET(qpc, qpc, cqn_snd, cqn);                 // Send CQ (DevX)
        // For receive, we use the SRQ's CQ
        struct mlx5dv_obj dv_rcq = {};
        struct mlx5dv_cq dv_recv_cq = {};
        dv_rcq.cq.in = recv_cq;
        dv_rcq.cq.out = &dv_recv_cq;
        mlx5dv_init_obj(&dv_rcq, MLX5DV_OBJ_CQ);
        DEVX_SET(qpc, qpc, cqn_rcv, dv_recv_cq.cqn);     // Receive CQ (ibv)
        DEVX_SET(qpc, qpc, log_sq_size, log_wq_size);
        DEVX_SET(qpc, qpc, log_rq_size, 0);               // No RQ, using SRQ
        DEVX_SET(qpc, qpc, rq_type, 1);                   // 1 = SRQ
        DEVX_SET(qpc, qpc, srqn_rmpn_xrqn, srqn);         // Reference the SRQ
        DEVX_SET(qpc, qpc, no_sq, 0);                     // We have SQ
        DEVX_SET(qpc, qpc, cs_req, 0);
        DEVX_SET(qpc, qpc, cs_res, 0);
        DEVX_SET(qpc, qpc, user_index, 0);
        DEVX_SET(qpc, qpc, page_offset, 0);
        DEVX_SET(qpc, qpc, dbr_umem_valid, 1);
        DEVX_SET(qpc, qpc, dbr_umem_id, dbr_umem->umem_id);
        DEVX_SET64(qpc, qpc, dbr_addr, 0);                // Offset 0 in the umem

        devx_qp = mlx5dv_devx_obj_create(ctx, cmd_in, sizeof(cmd_in),
                                          cmd_out, sizeof(cmd_out));
        if (!devx_qp) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_obj_create for QP failed: %s\n",
                    rank, strerror(errno));
            uint32_t syndrome = DEVX_GET(create_qp_out, cmd_out, syndrome);
            fprintf(stderr, "  syndrome = 0x%x\n", syndrome);
            exit(1);
        }

        qpn = DEVX_GET(create_qp_out, cmd_out, qpn);
    }

    void allocate_prod_idx() {
        cudaError_t err = cudaHostAlloc((void**)&h_prod_idx, sizeof(uint64_t),
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
    }
};

}  // namespace gicc::mlx5
