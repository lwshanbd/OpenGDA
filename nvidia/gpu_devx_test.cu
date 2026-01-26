/**
 * gpu_devx_test.cu - Test DevX-based GPU-Triggered RDMA
 *
 * Uses mlx5_devx_qp.hpp for proper GPU-accessible UAR
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "mpi_bootstrap.hpp"
#include "mlx5_devx_qp.hpp"

using namespace opengda;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// MLX5 WQE structures (same as gda_device.cuh but for host)
struct HostCtrlSeg {
    uint32_t opmod_idx_opcode;
    uint32_t qpn_ds;
    uint8_t  signature;
    uint8_t  rsvd[2];
    uint8_t  fm_ce_se;
    uint32_t imm;
} __attribute__((packed));

struct HostRaddrSeg {
    uint64_t raddr;
    uint32_t rkey;
    uint32_t reserved;
} __attribute__((packed));

struct HostDataSeg {
    uint32_t byte_count;
    uint32_t lkey;
    uint64_t addr;
} __attribute__((packed));

// GPU kernel to build WQE and ring doorbell
__global__ void gpu_rdma_write_kernel(
    void* wqe_buf,
    volatile uint32_t* dbrec,
    volatile void* bf_reg,
    volatile uint64_t* prod_idx,
    uint32_t qpn,
    uint16_t nwqes_mask,
    uint64_t local_addr,
    uint32_t local_lkey,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint32_t size,
    int* result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Get current producer index
    uint64_t prod = *prod_idx;
    uint16_t wqe_slot = (uint16_t)(prod & 0xFFFF);
    uint16_t new_prod = (uint16_t)((prod + 1) & 0xFFFF);

    printf("GPU: prod=%lu, wqe_slot=%u, new_prod=%u\n", prod, wqe_slot, new_prod);
    printf("GPU: wqe_buf=%p, dbrec=%p, bf_reg=%p\n", wqe_buf, dbrec, bf_reg);
    printf("GPU: local_addr=0x%lx, remote_addr=0x%lx\n", local_addr, remote_addr);

    // Get WQE pointer
    void* wqe_ptr = (void*)((uintptr_t)wqe_buf + ((wqe_slot & nwqes_mask) << 6));

    // Build WQE
    uint32_t* wqe32 = (uint32_t*)wqe_ptr;

    // Control segment (16 bytes)
    // opmod_idx_opcode: opmod(8) | wqe_idx(16) | opcode(8)
    wqe32[0] = __byte_perm((wqe_slot << 8) | 0x08, 0, 0x0123);  // RDMA_WRITE = 0x08
    // qpn_ds: qpn(24) | ds(8) = 3 for RDMA WRITE
    wqe32[1] = __byte_perm((qpn << 8) | 3, 0, 0x0123);
    wqe32[2] = 0;
    wqe32[3] = 0x08000000;  // fm_ce_se = signaled (bit 2)

    // Remote address segment (16 bytes)
    uint64_t raddr_be = __byte_perm(remote_addr >> 32, 0, 0x0123);
    raddr_be = (raddr_be << 32) | __byte_perm(remote_addr & 0xFFFFFFFF, 0, 0x0123);
    wqe32[4] = (uint32_t)(raddr_be >> 32);
    wqe32[5] = (uint32_t)raddr_be;
    wqe32[6] = __byte_perm(remote_rkey, 0, 0x0123);
    wqe32[7] = 0;

    // Data segment (16 bytes)
    wqe32[8] = __byte_perm(size, 0, 0x0123);
    wqe32[9] = __byte_perm(local_lkey, 0, 0x0123);
    uint64_t laddr_be = __byte_perm(local_addr >> 32, 0, 0x0123);
    laddr_be = (laddr_be << 32) | __byte_perm(local_addr & 0xFFFFFFFF, 0, 0x0123);
    wqe32[10] = (uint32_t)(laddr_be >> 32);
    wqe32[11] = (uint32_t)laddr_be;

    // Memory fence
    __threadfence_system();

    // Update producer index
    *prod_idx = new_prod;

    // Ring doorbell with release semantics
    uint32_t dbrec_val = __byte_perm(new_prod & 0xFFFF, 0, 0x0123);
#if __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.L1::no_allocate.b32 [%0], %1;"
                 : : "l"(dbrec), "r"(dbrec_val) : "memory");
#else
    __threadfence_system();
    *dbrec = dbrec_val;
    __threadfence_system();
#endif

    printf("GPU: wrote dbrec = 0x%08x\n", dbrec_val);

    // BlueFlame doorbell (8 bytes)
    __threadfence_system();

    if (bf_reg) {
        // Format: opmod_idx_opcode (4B) + qpn_ds (4B)
        uint32_t bf_lo = __byte_perm((new_prod & 0xFFFF) << 8, 0, 0x0123);
        uint32_t bf_hi = __byte_perm(qpn << 8, 0, 0x0123);
        uint64_t bf_val = ((uint64_t)bf_lo) | ((uint64_t)bf_hi << 32);

        printf("GPU: writing BF 0x%016lx to %p\n", bf_val, bf_reg);

#if __CUDA_ARCH__ >= 700
        asm volatile("st.release.sys.global.L1::no_allocate.b64 [%0], %1;"
                     : : "l"(bf_reg), "l"(bf_val) : "memory");
#else
        __threadfence_system();
        *(volatile uint64_t*)bf_reg = bf_val;
        __threadfence_system();
#endif

        printf("GPU: BF write done\n");
    }

    __threadfence_system();

    *result = 0;
}

// Connection info for exchange
struct ConnInfo {
    uint32_t qpn;
    uint16_t lid;
    uint8_t gid[16];
    uint32_t psn;
    uint64_t buf_addr;
    uint32_t rkey;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_size != 2) {
        if (mpi_rank == 0) {
            fprintf(stderr, "This test requires exactly 2 ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Initialize GPU - use local rank or default to 0
    int gpu_id = 0;
    const char* local_rank_str = getenv("SLURM_LOCALID");
    if (local_rank_str) {
        gpu_id = atoi(local_rank_str);
    }
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (gpu_id >= num_gpus) gpu_id = 0;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    if (mpi_rank == 0) {
        printf("=== GPU DevX RDMA Test ===\n\n");
        fflush(stdout);
    }

    // Find and open IB device
    int num_devices = 0;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "No IB devices found\n");
        MPI_Finalize();
        return 1;
    }

    struct ibv_context* ctx = ibv_open_device(dev_list[0]);
    if (!ctx) {
        fprintf(stderr, "Failed to open IB device\n");
        MPI_Finalize();
        return 1;
    }

    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
        fprintf(stderr, "Failed to allocate PD\n");
        MPI_Finalize();
        return 1;
    }

    // Query port
    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, 1, &port_attr);

    union ibv_gid gid;
    ibv_query_gid(ctx, 1, 0, &gid);

    if (mpi_rank == 0) {
        printf("Rank %d: Using device %s, port 1, LID=%d\n",
               mpi_rank, ibv_get_device_name(dev_list[0]), port_attr.lid);
        fflush(stdout);
    }

    ibv_free_device_list(dev_list);

    // Create DevX QP
    DevxQp devx_qp(ctx, pd, mpi_rank, 1, 256, 512);

    // Allocate data buffers
    size_t msg_size = 64;
    void* d_send_buf = nullptr;
    void* d_recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_buf, msg_size));
    CUDA_CHECK(cudaMalloc(&d_recv_buf, msg_size));
    CUDA_CHECK(cudaMemset(d_send_buf, 0xAB + mpi_rank, msg_size));
    CUDA_CHECK(cudaMemset(d_recv_buf, 0x00, msg_size));

    // Register buffers with IB
    struct ibv_mr* send_mr = ibv_reg_mr(pd, d_send_buf, msg_size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    struct ibv_mr* recv_mr = ibv_reg_mr(pd, d_recv_buf, msg_size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

    if (!send_mr || !recv_mr) {
        fprintf(stderr, "Rank %d: Failed to register MR\n", mpi_rank);
        MPI_Finalize();
        return 1;
    }

    // Prepare connection info
    ConnInfo local_info;
    local_info.qpn = devx_qp.qpn;
    local_info.lid = port_attr.lid;
    memcpy(local_info.gid, gid.raw, 16);
    local_info.psn = rand() & 0xFFFFFF;
    local_info.buf_addr = (uint64_t)d_recv_buf;  // Receive buffer for remote writes
    local_info.rkey = recv_mr->rkey;

    // Exchange connection info
    ConnInfo remote_info;
    int peer = (mpi_rank == 0) ? 1 : 0;
    MPI_Sendrecv(&local_info, sizeof(ConnInfo), MPI_BYTE, peer, 0,
                 &remote_info, sizeof(ConnInfo), MPI_BYTE, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (mpi_rank == 0) {
        printf("Connection info exchanged:\n");
        printf("  Local: QPN=%u, LID=%u, rkey=0x%x\n",
               local_info.qpn, local_info.lid, local_info.rkey);
        printf("  Remote: QPN=%u, LID=%u, rkey=0x%x\n",
               remote_info.qpn, remote_info.lid, remote_info.rkey);
        fflush(stdout);
    }

    // Connect QP
    devx_qp.rst2init();
    MPI_Barrier(MPI_COMM_WORLD);

    devx_qp.init2rtr(remote_info.qpn, remote_info.lid, remote_info.gid,
                     remote_info.psn, 5);
    MPI_Barrier(MPI_COMM_WORLD);

    devx_qp.rtr2rts(local_info.psn);
    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("\nQP connected successfully!\n\n");
        fflush(stdout);
    }

    // Allocate result
    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    // Only rank 0 sends
    if (mpi_rank == 0) {
        printf("Rank 0: Launching GPU RDMA kernel...\n");
        fflush(stdout);

        uint16_t nwqes_mask = (1 << devx_qp.log_wq_size) - 1;

        gpu_rdma_write_kernel<<<1, 1>>>(
            devx_qp.d_wq_buf,
            devx_qp.d_dbrec,
            devx_qp.d_uar_reg,
            devx_qp.d_prod_idx,
            devx_qp.qpn,
            nwqes_mask,
            (uint64_t)d_send_buf,
            send_mr->lkey,
            remote_info.buf_addr,
            remote_info.rkey,
            msg_size,
            d_result
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Rank 0: Kernel completed\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(100000);  // Wait 100ms for NIC

    // Verify
    if (mpi_rank == 1) {
        uint8_t buf[64];
        CUDA_CHECK(cudaMemcpy(buf, d_recv_buf, msg_size, cudaMemcpyDeviceToHost));
        printf("Rank 1: Received: 0x%02x (expected 0xAB)\n", buf[0]);
        if (buf[0] == 0xAB) {
            printf("SUCCESS: GPU-triggered RDMA worked!\n");
        } else {
            printf("FAILED: Data not received correctly\n");
        }
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_send_buf));
    CUDA_CHECK(cudaFree(d_recv_buf));
    ibv_dereg_mr(send_mr);
    ibv_dereg_mr(recv_mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);

    MPI_Finalize();
    return 0;
}
