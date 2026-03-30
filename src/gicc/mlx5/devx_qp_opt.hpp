/**
 * mlx5_devx_qp_opt.hpp - Optimized DevX QP for GPU-triggered RDMA
 *
 * Enhancements over basic version:
 *   1. BlueFlame register mapping to GPU
 *   2. Separated producer indices (resv_head, ready_head, prod_idx)
 *   3. Batch configuration support
 *   4. Improved memory layout for GPU access
 */
#pragma once

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <cuda_runtime.h>
#include <endian.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mlx5_ifc.h"
#include "mlx5_prm.h"
#include "device_opt.cuh"

namespace gicc::mlx5 {

// Constants
#define GPAGE_SIZE 65536  // 64KB alignment for NIC buffers
#define DBREC_SIZE 8
#define NC_UAR_SIZE 8
#define BF_UAR_SIZE 512

// QP states
enum {
    MLX5_QPC_ST_RC = 0x0,
    MLX5_QPC_ST_DCI = 0x5,
};

enum {
    MLX5_QPC_PM_STATE_MIGRATED = 0x3,
};

/**
 * Optimized DevX QP with BlueFlame support
 */
class DevxQpOpt {
public:
    // IB context
    struct ibv_context* ctx;
    struct ibv_pd* pd;

    // DevX UAR
    struct mlx5dv_devx_uar* uar;
    void* h_uar_page;       // Host pointer to UAR page
    void* d_uar_page;       // GPU pointer to UAR page
    void* h_bf_reg;         // Host pointer to BlueFlame register
    void* d_bf_reg;         // GPU pointer to BlueFlame register
    size_t uar_size;
    bool use_blueflame;

    // WQ buffer
    struct mlx5dv_devx_umem* wq_umem;
    void* h_wq_buf;
    void* d_wq_buf;
    size_t wq_buf_size;
    uint32_t log_wq_size;
    uint32_t num_wqes;

    // Doorbell record
    struct mlx5dv_devx_umem* dbr_umem;
    volatile uint32_t* h_dbrec;
    volatile uint32_t* d_dbrec;

    // DevX QP object
    struct mlx5dv_devx_obj* devx_qp;
    uint32_t qpn;

    // DevX CQ
    struct mlx5dv_devx_obj* devx_cq;
    struct mlx5dv_devx_umem* cq_umem;
    struct mlx5dv_devx_umem* cq_dbr_umem;
    void* h_cq_buf;
    void* d_cq_buf;
    volatile uint32_t* h_cq_dbrec;
    volatile uint32_t* d_cq_dbrec;
    uint32_t cqn;
    uint32_t num_cqe;

    int pdn;

    // Producer indices (nvshmem-style)
    volatile uint64_t* h_resv_head;     // Reserved slots
    volatile uint64_t* d_resv_head;
    volatile uint64_t* h_ready_head;    // Ready to post
    volatile uint64_t* d_ready_head;
    volatile uint64_t* h_prod_idx;      // Posted to hardware
    volatile uint64_t* d_prod_idx;

    // Completion counter
    volatile uint64_t* h_num_completions;
    volatile uint64_t* d_num_completions;

    // Batch configuration
    uint32_t batch_size;

    int rank;
    int port_num;

    // Device state for GPU kernels
    GdaDeviceStateOpt* h_device_state;
    GdaDeviceStateOpt* d_device_state;

    DevxQpOpt(struct ibv_context* ctx_, struct ibv_pd* pd_, int rank_, int port_ = 1,
              uint32_t qp_depth = 256, uint32_t cq_depth = 512, uint32_t batch = 32)
        : ctx(ctx_), pd(pd_), rank(rank_), port_num(port_), batch_size(batch),
          uar(nullptr), h_uar_page(nullptr), d_uar_page(nullptr),
          h_bf_reg(nullptr), d_bf_reg(nullptr), uar_size(0), use_blueflame(false),
          wq_umem(nullptr), h_wq_buf(nullptr), d_wq_buf(nullptr), wq_buf_size(0),
          dbr_umem(nullptr), h_dbrec(nullptr), d_dbrec(nullptr),
          devx_qp(nullptr), qpn(0),
          devx_cq(nullptr), cq_umem(nullptr), cq_dbr_umem(nullptr),
          h_cq_buf(nullptr), d_cq_buf(nullptr),
          h_cq_dbrec(nullptr), d_cq_dbrec(nullptr),
          cqn(0), num_cqe(cq_depth),
          pdn(0),
          h_resv_head(nullptr), d_resv_head(nullptr),
          h_ready_head(nullptr), d_ready_head(nullptr),
          h_prod_idx(nullptr), d_prod_idx(nullptr),
          h_num_completions(nullptr), d_num_completions(nullptr),
          h_device_state(nullptr), d_device_state(nullptr)
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
        num_wqes = 1;
        log_wq_size = 0;
        while (num_wqes < qp_depth) {
            num_wqes <<= 1;
            log_wq_size++;
        }
        wq_buf_size = num_wqes * MLX5_SEND_WQE_BB;

        // Ensure batch_size is power of 2 and <= num_wqes
        if (batch_size > num_wqes) batch_size = num_wqes;
        uint32_t bs = 1;
        while (bs < batch_size) bs <<= 1;
        batch_size = bs;

        // Allocate all resources
        allocate_uar_blueflame();
        allocate_wq_buffer();
        allocate_dbrec();
        allocate_cq();
        create_devx_qp();
        allocate_producer_indices();
        allocate_device_state();

        if (rank == 0) {
            printf("DevxQpOpt created:\n");
            printf("  QPN = %u, CQN = %u\n", qpn, cqn);
            printf("  UAR page_id = %u (BlueFlame: %s)\n",
                   uar->page_id, use_blueflame ? "YES" : "NO");
            printf("  WQ: %u WQEs, batch_size = %u\n", num_wqes, batch_size);
            printf("  d_bf_reg = %p\n", d_bf_reg);
            printf("  d_wq_buf = %p\n", d_wq_buf);
            printf("  d_dbrec = %p\n", (void*)d_dbrec);
            fflush(stdout);
        }
    }

    ~DevxQpOpt() {
        if (devx_qp) mlx5dv_devx_obj_destroy(devx_qp);
        if (devx_cq) mlx5dv_devx_obj_destroy(devx_cq);

        // Free device state
        if (d_device_state) cudaFree(d_device_state);
        if (h_device_state) cudaFreeHost(h_device_state);

        // Free producer indices
        if (h_resv_head) { cudaHostUnregister((void*)h_resv_head); cudaFreeHost((void*)h_resv_head); }
        if (h_ready_head) { cudaHostUnregister((void*)h_ready_head); cudaFreeHost((void*)h_ready_head); }
        if (h_prod_idx) { cudaHostUnregister((void*)h_prod_idx); cudaFreeHost((void*)h_prod_idx); }
        if (h_num_completions) { cudaHostUnregister((void*)h_num_completions); cudaFreeHost((void*)h_num_completions); }

        // Free CQ resources
        if (d_cq_dbrec) cudaHostUnregister((void*)h_cq_dbrec);
        if (d_cq_buf) cudaHostUnregister(h_cq_buf);
        if (d_dbrec) cudaHostUnregister((void*)h_dbrec);
        if (d_wq_buf) cudaHostUnregister(h_wq_buf);
        if (d_uar_page) cudaHostUnregister(h_uar_page);

        if (cq_dbr_umem) mlx5dv_devx_umem_dereg(cq_dbr_umem);
        if (cq_umem) mlx5dv_devx_umem_dereg(cq_umem);
        if (dbr_umem) mlx5dv_devx_umem_dereg(dbr_umem);
        if (wq_umem) mlx5dv_devx_umem_dereg(wq_umem);

        if (h_cq_dbrec) free((void*)h_cq_dbrec);
        if (h_cq_buf) free(h_cq_buf);
        if (h_dbrec) free((void*)h_dbrec);
        if (h_wq_buf) free(h_wq_buf);

        if (uar) mlx5dv_devx_free_uar(uar);
    }

    // Get device state for passing to GPU kernels
    GdaDeviceStateOpt* get_device_state() { return d_device_state; }

    // Update remote target info
    void set_remote_target(uint64_t remote_addr, uint32_t remote_rkey) {
        h_device_state->remote_addr = remote_addr;
        h_device_state->remote_rkey = remote_rkey;
        // Copy to GPU
        cudaMemcpy(d_device_state, h_device_state, sizeof(GdaDeviceStateOpt),
                   cudaMemcpyHostToDevice);
    }

    // QP state transitions
    void rst2init() {
        uint8_t cmd_in[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {0};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {0};

        DEVX_SET(rst2init_qp_in, cmd_in, opcode, MLX5_CMD_OP_RST2INIT_QP);
        DEVX_SET(rst2init_qp_in, cmd_in, qpn, qpn);

        void* qpc = DEVX_ADDR_OF(rst2init_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, port_num);
        DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
        DEVX_SET(qpc, qpc, rwe, 1);
        DEVX_SET(qpc, qpc, rre, 1);
        DEVX_SET(qpc, qpc, rae, 1);
        DEVX_SET(qpc, qpc, atomic_mode, 0x3);

        int ret = mlx5dv_devx_obj_modify(devx_qp, cmd_in, sizeof(cmd_in),
                                         cmd_out, sizeof(cmd_out));
        if (ret) {
            fprintf(stderr, "Rank %d: RST2INIT failed: %s\n", rank, strerror(errno));
            exit(1);
        }
    }

    void init2rtr(uint32_t dest_qpn, uint16_t dest_lid, uint8_t* dest_gid,
                  uint32_t remote_psn, int mtu = 5) {
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

        DEVX_SET(qpc, qpc, primary_address_path.rlid, dest_lid);
        DEVX_SET(qpc, qpc, primary_address_path.grh, dest_lid == 0 ? 1 : 0);
        if (dest_gid && dest_lid == 0) {
            memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip), dest_gid, 16);
            DEVX_SET(qpc, qpc, primary_address_path.hop_limit, 64);
            DEVX_SET(qpc, qpc, primary_address_path.src_addr_index, 0);
        }

        int ret = mlx5dv_devx_obj_modify(devx_qp, cmd_in, sizeof(cmd_in),
                                         cmd_out, sizeof(cmd_out));
        if (ret) {
            fprintf(stderr, "Rank %d: INIT2RTR failed: %s\n", rank, strerror(errno));
            exit(1);
        }
    }

    void rtr2rts(uint32_t local_psn) {
        uint8_t cmd_in[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {0};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {0};

        DEVX_SET(rtr2rts_qp_in, cmd_in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
        DEVX_SET(rtr2rts_qp_in, cmd_in, qpn, qpn);

        void* qpc = DEVX_ADDR_OF(rtr2rts_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, log_ack_req_freq, 0);
        DEVX_SET(qpc, qpc, log_sra_max, 4);
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
    void allocate_uar_blueflame() {
        // Try BlueFlame UAR first for lower latency
        uar = mlx5dv_devx_alloc_uar(ctx, MLX5DV_UAR_ALLOC_TYPE_BF);
        if (uar) {
            use_blueflame = true;
            uar_size = BF_UAR_SIZE;
        } else {
#ifdef MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED
            // Fall back to NC dedicated
            uar = mlx5dv_devx_alloc_uar(ctx, MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED);
            if (uar) {
                use_blueflame = false;
                uar_size = NC_UAR_SIZE;
            }
#endif
        }

        if (!uar) {
            // Last resort: regular NC
            uar = mlx5dv_devx_alloc_uar(ctx, MLX5DV_UAR_ALLOC_TYPE_NC);
            if (!uar) {
                fprintf(stderr, "Rank %d: Failed to allocate DevX UAR\n", rank);
                exit(1);
            }
            use_blueflame = false;
            uar_size = NC_UAR_SIZE;
        }

        h_uar_page = uar->reg_addr;

        // BlueFlame register is at offset 0x800 from UAR base for BF type
        if (use_blueflame) {
            h_bf_reg = (void*)((uintptr_t)uar->base_addr + 0x800);
        } else {
            h_bf_reg = uar->reg_addr;  // Use doorbell register
        }

        // Register UAR with CUDA for GPU access
        cudaError_t err = cudaHostRegister(h_uar_page, uar_size,
            cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostRegister for UAR failed: %s\n",
                    rank, cudaGetErrorString(err));
            exit(1);
        }

        err = cudaHostGetDevicePointer(&d_uar_page, h_uar_page, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for UAR failed: %s\n",
                    rank, cudaGetErrorString(err));
            exit(1);
        }

        // Map BlueFlame register to GPU
        if (use_blueflame && h_bf_reg != h_uar_page) {
            err = cudaHostRegister(h_bf_reg, 8,
                cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory);
            if (err != cudaSuccess) {
                // BlueFlame might overlap with UAR page, use offset instead
                d_bf_reg = (void*)((uintptr_t)d_uar_page +
                                   ((uintptr_t)h_bf_reg - (uintptr_t)h_uar_page));
            } else {
                err = cudaHostGetDevicePointer(&d_bf_reg, h_bf_reg, 0);
                if (err != cudaSuccess) {
                    d_bf_reg = nullptr;
                    use_blueflame = false;
                }
            }
        } else {
            d_bf_reg = d_uar_page;
        }
    }

    void allocate_wq_buffer() {
        size_t aligned_size = ((wq_buf_size + GPAGE_SIZE - 1) / GPAGE_SIZE) * GPAGE_SIZE;
        int ret = posix_memalign(&h_wq_buf, GPAGE_SIZE, aligned_size);
        if (ret) {
            fprintf(stderr, "Rank %d: posix_memalign for WQ failed\n", rank);
            exit(1);
        }
        memset(h_wq_buf, 0, aligned_size);

        wq_umem = mlx5dv_devx_umem_reg(ctx, h_wq_buf, aligned_size, IBV_ACCESS_LOCAL_WRITE);
        if (!wq_umem) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_umem_reg for WQ failed\n", rank);
            exit(1);
        }

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
        int ret = posix_memalign((void**)&h_dbrec, GPAGE_SIZE, GPAGE_SIZE);
        if (ret) {
            fprintf(stderr, "Rank %d: posix_memalign for DBR failed\n", rank);
            exit(1);
        }
        memset((void*)h_dbrec, 0, GPAGE_SIZE);

        dbr_umem = mlx5dv_devx_umem_reg(ctx, (void*)h_dbrec, GPAGE_SIZE, IBV_ACCESS_LOCAL_WRITE);
        if (!dbr_umem) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_umem_reg for DBR failed\n", rank);
            exit(1);
        }

        cudaError_t err = cudaHostRegister((void*)h_dbrec, GPAGE_SIZE,
            cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostRegister for DBR failed: %s\n",
                    rank, cudaGetErrorString(err));
            exit(1);
        }

        err = cudaHostGetDevicePointer((void**)&d_dbrec, (void*)h_dbrec, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for DBR failed: %s\n",
                    rank, cudaGetErrorString(err));
            exit(1);
        }
    }

    void allocate_cq() {
        size_t cq_buf_size = num_cqe * 64;
        size_t aligned_size = ((cq_buf_size + GPAGE_SIZE - 1) / GPAGE_SIZE) * GPAGE_SIZE;

        int ret = posix_memalign(&h_cq_buf, GPAGE_SIZE, aligned_size);
        if (ret) {
            fprintf(stderr, "Rank %d: posix_memalign for CQ failed\n", rank);
            exit(1);
        }
        memset(h_cq_buf, 0xFF, aligned_size);

        cq_umem = mlx5dv_devx_umem_reg(ctx, h_cq_buf, aligned_size, IBV_ACCESS_LOCAL_WRITE);
        if (!cq_umem) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_umem_reg for CQ failed\n", rank);
            exit(1);
        }

        ret = posix_memalign((void**)&h_cq_dbrec, GPAGE_SIZE, GPAGE_SIZE);
        if (ret) {
            fprintf(stderr, "Rank %d: posix_memalign for CQ DBR failed\n", rank);
            exit(1);
        }
        memset((void*)h_cq_dbrec, 0, GPAGE_SIZE);

        cq_dbr_umem = mlx5dv_devx_umem_reg(ctx, (void*)h_cq_dbrec, GPAGE_SIZE, IBV_ACCESS_LOCAL_WRITE);
        if (!cq_dbr_umem) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_umem_reg for CQ DBR failed\n", rank);
            exit(1);
        }

        cudaError_t err = cudaHostRegister(h_cq_buf, aligned_size,
            cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (err == cudaSuccess) {
            cudaHostGetDevicePointer(&d_cq_buf, h_cq_buf, 0);
        }

        err = cudaHostRegister((void*)h_cq_dbrec, GPAGE_SIZE,
            cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (err == cudaSuccess) {
            cudaHostGetDevicePointer((void**)&d_cq_dbrec, (void*)h_cq_dbrec, 0);
        }

        // Query EQN
        uint32_t eqn;
        ret = mlx5dv_devx_query_eqn(ctx, 0, &eqn);
        if (ret) {
            fprintf(stderr, "Rank %d: mlx5dv_devx_query_eqn failed\n", rank);
            exit(1);
        }

        // Create CQ
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
        DEVX_SET(cqc, cqc, cqe_sz, 0);
        DEVX_SET(cqc, cqc, cc, 1);
        DEVX_SET(cqc, cqc, oi, 1);
        DEVX_SET(cqc, cqc, dbr_umem_id, cq_dbr_umem->umem_id);
        DEVX_SET(cqc, cqc, log_cq_size, log_cq_size);
        DEVX_SET(cqc, cqc, uar_page, uar->page_id);
        DEVX_SET(cqc, cqc, c_eqn, eqn);
        DEVX_SET(cqc, cqc, log_page_size, 16 - 12);
        DEVX_SET64(cqc, cqc, dbr_addr, 0);

        devx_cq = mlx5dv_devx_obj_create(ctx, cmd_in, sizeof(cmd_in),
                                          cmd_out, sizeof(cmd_out));
        if (!devx_cq) {
            fprintf(stderr, "Rank %d: Create CQ failed: %s\n", rank, strerror(errno));
            exit(1);
        }

        cqn = DEVX_GET(create_cq_out, cmd_out, cqn);
    }

    void create_devx_qp() {
        uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_qp_in)] = {0};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_qp_out)] = {0};

        DEVX_SET(create_qp_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_QP);
        DEVX_SET(create_qp_in, cmd_in, wq_umem_id, wq_umem->umem_id);
        DEVX_SET(create_qp_in, cmd_in, wq_umem_valid, 1);
        DEVX_SET64(create_qp_in, cmd_in, wq_umem_offset, 0);

        void* qpc = DEVX_ADDR_OF(create_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, st, MLX5_QPC_ST_RC);
        DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
        DEVX_SET(qpc, qpc, pd, pdn);
        DEVX_SET(qpc, qpc, uar_page, uar->page_id);
        DEVX_SET(qpc, qpc, cqn_snd, cqn);
        DEVX_SET(qpc, qpc, cqn_rcv, cqn);
        DEVX_SET(qpc, qpc, log_sq_size, log_wq_size);
        DEVX_SET(qpc, qpc, log_rq_size, 0);
        DEVX_SET(qpc, qpc, no_sq, 0);
        DEVX_SET(qpc, qpc, cs_req, 0);
        DEVX_SET(qpc, qpc, cs_res, 0);
        DEVX_SET(qpc, qpc, dbr_umem_valid, 1);
        DEVX_SET(qpc, qpc, dbr_umem_id, dbr_umem->umem_id);
        DEVX_SET64(qpc, qpc, dbr_addr, 0);

        devx_qp = mlx5dv_devx_obj_create(ctx, cmd_in, sizeof(cmd_in),
                                          cmd_out, sizeof(cmd_out));
        if (!devx_qp) {
            fprintf(stderr, "Rank %d: Create QP failed: %s\n", rank, strerror(errno));
            exit(1);
        }

        qpn = DEVX_GET(create_qp_out, cmd_out, qpn);
    }

    void allocate_producer_indices() {
        // Allocate all indices in one contiguous block for better cache behavior
        size_t total_size = 4 * sizeof(uint64_t);  // resv, ready, prod, completions

        cudaError_t err = cudaHostAlloc((void**)&h_resv_head, total_size, cudaHostAllocMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostAlloc for indices failed\n", rank);
            exit(1);
        }

        h_ready_head = h_resv_head + 1;
        h_prod_idx = h_resv_head + 2;
        h_num_completions = h_resv_head + 3;

        *h_resv_head = 0;
        *h_ready_head = 0;
        *h_prod_idx = 0;
        *h_num_completions = 0;

        err = cudaHostGetDevicePointer((void**)&d_resv_head, (void*)h_resv_head, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostGetDevicePointer for indices failed\n", rank);
            exit(1);
        }

        d_ready_head = d_resv_head + 1;
        d_prod_idx = d_resv_head + 2;
        d_num_completions = d_resv_head + 3;
    }

    void allocate_device_state() {
        // Allocate host-side state
        cudaError_t err = cudaHostAlloc((void**)&h_device_state, sizeof(GdaDeviceStateOpt),
                                         cudaHostAllocMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaHostAlloc for device state failed\n", rank);
            exit(1);
        }

        // Initialize state
        h_device_state->qpn = qpn;
        h_device_state->nwqes = num_wqes;
        h_device_state->nwqes_mask = num_wqes - 1;
        h_device_state->wqe_buf = d_wq_buf;
        h_device_state->wqe_lkey = 0;  // Will be set when MR is registered
        h_device_state->dbrec = d_dbrec;
        h_device_state->bf_reg = use_blueflame ? (volatile uint64_t*)d_bf_reg : nullptr;
        h_device_state->resv_head = d_resv_head;
        h_device_state->ready_head = d_ready_head;
        h_device_state->prod_idx = d_prod_idx;
        h_device_state->cqe = (volatile GdaCqe64*)d_cq_buf;
        h_device_state->ncqes = num_cqe;
        h_device_state->ncqes_mask = num_cqe - 1;
        h_device_state->cq_cons_idx = nullptr;  // TODO
        h_device_state->cq_dbrec = d_cq_dbrec;
        h_device_state->remote_addr = 0;
        h_device_state->remote_rkey = 0;
        h_device_state->num_completions = d_num_completions;
        h_device_state->batch_size = batch_size;
        h_device_state->batch_mask = batch_size - 1;

        // Allocate device copy
        err = cudaMalloc(&d_device_state, sizeof(GdaDeviceStateOpt));
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaMalloc for device state failed\n", rank);
            exit(1);
        }

        // Copy to device
        err = cudaMemcpy(d_device_state, h_device_state, sizeof(GdaDeviceStateOpt),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: cudaMemcpy for device state failed\n", rank);
            exit(1);
        }
    }
};

}  // namespace gicc::mlx5
