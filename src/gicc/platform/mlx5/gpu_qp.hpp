/*
 * gpu_qp.hpp - host-side construction of one GPU-driven RC queue pair.
 *
 * The QP is created through DevX so that every resource the posting thread
 * touches can be placed where the GPU reaches it:
 *
 *   send queue      host memory, registered with CUDA (GPU writes WQEs)
 *   doorbell record host memory, registered with CUDA (GPU writes)
 *   doorbell (UAR)  NIC MMIO page, registered with CUDA as I/O memory
 *   completion      one collapsed CQE in GPU memory (GPU polls)
 *
 * The CQE lives in GPU memory on purpose: it is written by the NIC after the
 * data of an RDMA READ, and keeping both on the same side of the host bridge
 * is what lets a GPU thread that saw the CQE trust the data behind it.
 *
 * Receive work requests are never posted (RDMA WRITE and READ do not
 * consume them); the QP points at a shared, empty SRQ.
 */
#pragma once

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <cuda_runtime.h>

#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <map>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gicc/platform/mlx5/device_ctx.hpp"
#include "gicc/platform/mlx5/mlx5_ifc.h"
#include "gicc/platform/mlx5/mlx5_prm.h"

namespace gicc::mlx5 {

// Addressing information a peer needs to connect to one of our QPs.
struct IbPortAddr {
    uint16_t lid;
    uint8_t  link_layer;     // IBV_LINK_LAYER_*
    uint8_t  gid_index;
    uint8_t  gid[16];
};

[[noreturn]] inline void die(const char* what, int err = errno) {
    std::fprintf(stderr, "[gicc] %s failed: %s\n", what, std::strerror(err));
    std::abort();
}

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[gicc] %s failed: %s\n", what, cudaGetErrorString(err));
        std::abort();
    }
}

// Register memory with the NIC in 4 KB pages. The QP and CQ contexts are
// created with log_page_size = 0, so their umem must use 4 KB pages too; left
// to itself mlx5dv_devx_umem_reg picks the largest page the region allows,
// and a CQ that sits at a 4 KB offset of, say, a 512 KB region is then
// rejected (CREATE_CQ fails with syndrome 0x357275 at 128 and 256 QPs).
inline mlx5dv_devx_umem* umem_reg(ibv_context* ctx, void* addr, size_t size) {
    mlx5dv_devx_umem_in in = {};
    in.addr = addr;
    in.size = size;
    in.access = IBV_ACCESS_LOCAL_WRITE;
    in.pgsz_bitmap = 1ull << 12;
    return mlx5dv_devx_umem_reg_ex(ctx, &in);
}

// Port MTU, capped by GICC_MTU (bytes) when set.
inline int port_mtu(const ibv_port_attr& port) {
    int mtu = port.active_mtu;
    if (const char* e = std::getenv("GICC_MTU")) {
        const int bytes = std::atoi(e);
        int want = bytes <= 256 ? 1 : bytes <= 512 ? 2 : bytes <= 1024 ? 3
                 : bytes <= 2048 ? 4 : 5;
        if (want < mtu) mtu = want;
    }
    return mtu;
}

// Fill a QP context's primary address path towards `remote`. RoCE needs the
// peer MAC, which only an address handle resolves.
inline void set_address_path(void* qpc, ibv_pd* pd, uint8_t port_num,
                         const IbPortAddr& remote) {
    if (remote.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        DEVX_SET(qpc, qpc, primary_address_path.rlid, remote.lid);
        DEVX_SET(qpc, qpc, primary_address_path.grh, 0);
        DEVX_SET(qpc, qpc, primary_address_path.sl, 0);
        return;
    }
    ibv_ah_attr ah = {};
    ah.is_global = 1;
    ah.port_num = port_num;
    std::memcpy(ah.grh.dgid.raw, remote.gid, 16);
    ah.grh.sgid_index = remote.gid_index;
    ah.dlid = 0xC000;                                  // RoCE v2 UDP sport base
    ibv_ah* handle = ibv_create_ah(pd, &ah);
    if (!handle) die("ibv_create_ah");
    mlx5dv_obj dv = {};
    mlx5dv_ah dah = {};
    dv.ah.in = handle;
    dv.ah.out = &dah;
    if (mlx5dv_init_obj(&dv, MLX5DV_OBJ_AH)) die("mlx5dv_init_obj(AH)");
    std::memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32),
                &dah.av->rmac, 6);
    std::memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip),
                &dah.av->rgid, 16);
    DEVX_SET(qpc, qpc, primary_address_path.hop_limit, 255);
    DEVX_SET(qpc, qpc, primary_address_path.src_addr_index, remote.gid_index);
    DEVX_SET(qpc, qpc, primary_address_path.udp_sport, 0xC000);
    ibv_destroy_ah(handle);
}

// Map an MMIO address (a UAR doorbell) into the GPU. Several UARs can share
// one system page (64 KB kernels hold sixteen 4 KB UAR pages), and CUDA
// refuses to register a page twice, so registrations are shared per page.
class MmioMap {
public:
    static void* map(void* addr) {
        auto& self = instance();
        std::lock_guard<std::mutex> g(self.mu_);
        const uintptr_t page = (uintptr_t)addr & ~(self.page_ - 1);
        auto it = self.pages_.find(page);
        if (it == self.pages_.end()) {
            void* dev = nullptr;
            cuda_check(cudaHostRegister((void*)page, self.page_,
                                      cudaHostRegisterPortable | cudaHostRegisterMapped |
                                      cudaHostRegisterIoMemory),
                     "cudaHostRegister(UAR)");
            cuda_check(cudaHostGetDevicePointer(&dev, (void*)page, 0),
                     "cudaHostGetDevicePointer(UAR)");
            it = self.pages_.emplace(page, Entry{(char*)dev, 0}).first;
        }
        ++it->second.refs;
        return it->second.dev + ((uintptr_t)addr - page);
    }

    static void unmap(void* addr) {
        auto& self = instance();
        std::lock_guard<std::mutex> g(self.mu_);
        const uintptr_t page = (uintptr_t)addr & ~(self.page_ - 1);
        auto it = self.pages_.find(page);
        if (it == self.pages_.end()) return;
        if (--it->second.refs == 0) {
            cudaHostUnregister((void*)page);
            self.pages_.erase(it);
        }
    }

private:
    struct Entry { char* dev; int refs; };
    std::mutex mu_;
    std::map<uintptr_t, Entry> pages_;
    uintptr_t page_ = (uintptr_t)sysconf(_SC_PAGESIZE);

    static MmioMap& instance() {
        static MmioMap m;
        return m;
    }
};

class GpuQp {
public:
    // `cqe_slot` is the CQ's single 64-byte entry, in GPU memory, at the
    // 4 KB-aligned byte offset `cq_umem_off` of the registered umem
    // `cq_umem`. The engine packs every QP's CQ into one allocation so the
    // GPU memory costs one registration.
    GpuQp(ibv_context* ctx, ibv_pd* pd, uint8_t port, uint32_t nwqes,
              uint32_t srqn, uint32_t recv_cqn, uint32_t eqn,
              mlx5dv_devx_umem* cq_umem, uint64_t cq_umem_off, void* cqe_slot)
        : ctx_(ctx), pd_(pd), port_(port), nwqes_(nwqes), cqe_slot_(cqe_slot) {
        mlx5dv_obj obj = {};
        mlx5dv_pd dvpd = {};
        obj.pd.in = pd;
        obj.pd.out = &dvpd;
        if (mlx5dv_init_obj(&obj, MLX5DV_OBJ_PD)) die("mlx5dv_init_obj(PD)");

        alloc_uar();
        alloc_host(&wq_, (size_t)nwqes * 64, &wq_umem_, &d_wq_);
        std::memset(wq_, 0, (size_t)nwqes * 64);
        alloc_host(&dbr_, 64, &dbr_umem_, &d_dbr_);
        std::memset(dbr_, 0, 64);
        alloc_host(&cq_dbr_, 64, &cq_dbr_umem_, nullptr);
        std::memset(cq_dbr_, 0, 64);
        create_cq(eqn, cq_umem, cq_umem_off);
        create_qp(dvpd.pdn, srqn, recv_cqn);
    }

    ~GpuQp() {
        if (qp_) mlx5dv_devx_obj_destroy(qp_);
        if (cq_) mlx5dv_devx_obj_destroy(cq_);
        free_host(wq_, wq_umem_);
        free_host(dbr_, dbr_umem_);
        free_host(cq_dbr_, cq_dbr_umem_);
        if (uar_) {
            MmioMap::unmap(uar_->reg_addr);
            mlx5dv_devx_free_uar(uar_);
        }
    }

    GpuQp(const GpuQp&) = delete;
    GpuQp& operator=(const GpuQp&) = delete;

    uint32_t qpn() const { return qpn_; }

    // RST -> INIT -> RTR -> RTS against the remote QP `remote_qpn`.
    void connect(uint32_t remote_qpn, const IbPortAddr& remote) {
        ibv_port_attr port = {};
        if (ibv_query_port(ctx_, port_, &port)) die("ibv_query_port");
        {
            uint8_t in[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {};
            uint8_t out[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {};
            DEVX_SET(rst2init_qp_in, in, opcode, MLX5_CMD_OP_RST2INIT_QP);
            DEVX_SET(rst2init_qp_in, in, qpn, qpn_);
            void* qpc = DEVX_ADDR_OF(rst2init_qp_in, in, qpc);
            DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, port_);
            DEVX_SET(qpc, qpc, pm_state, 0x3);         // migrated
            DEVX_SET(qpc, qpc, rwe, 1);
            DEVX_SET(qpc, qpc, rre, 1);
            DEVX_SET(qpc, qpc, rae, 1);
            DEVX_SET(qpc, qpc, atomic_mode, 0x3);
            modify(in, sizeof(in), out, sizeof(out), "RST2INIT");
        }
        {
            uint8_t in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {};
            uint8_t out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {};
            DEVX_SET(init2rtr_qp_in, in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
            DEVX_SET(init2rtr_qp_in, in, qpn, qpn_);
            void* qpc = DEVX_ADDR_OF(init2rtr_qp_in, in, qpc);
            DEVX_SET(qpc, qpc, mtu, port_mtu(port));
            DEVX_SET(qpc, qpc, log_msg_max, 30);
            DEVX_SET(qpc, qpc, remote_qpn, remote_qpn);
            DEVX_SET(qpc, qpc, min_rnr_nak, 12);
            DEVX_SET(qpc, qpc, log_rra_max, 4);
            DEVX_SET(qpc, qpc, next_rcv_psn, 0);
            DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, port_);
            set_address_path(qpc, pd_, port_, remote);
            modify(in, sizeof(in), out, sizeof(out), "INIT2RTR");
        }
        {
            uint8_t in[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {};
            uint8_t out[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {};
            DEVX_SET(rtr2rts_qp_in, in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
            DEVX_SET(rtr2rts_qp_in, in, qpn, qpn_);
            void* qpc = DEVX_ADDR_OF(rtr2rts_qp_in, in, qpc);
            DEVX_SET(qpc, qpc, log_ack_req_freq, 0);
            DEVX_SET(qpc, qpc, log_sra_max, 4);
            DEVX_SET(qpc, qpc, next_send_psn, 0);
            DEVX_SET(qpc, qpc, retry_count, 7);
            DEVX_SET(qpc, qpc, rnr_retry, 7);
            DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, 14);
            modify(in, sizeof(in), out, sizeof(out), "RTR2RTS");
        }
    }

    // The device view; `counters` points at this QP's four counters in GPU
    // memory: resv, then (16-byte aligned) ready and rung, read as one pair.
    QpView device_view(uint64_t* counters) const {
        QpView q = {};
        q.wq       = static_cast<uint8_t*>(d_wq_);
        q.dbrec    = reinterpret_cast<volatile uint32_t*>(
                         static_cast<char*>(d_dbr_) + 4);   // SQ half of the record
        q.uar      = static_cast<uint64_t*>(d_uar_);
        q.cqe_tail = reinterpret_cast<const uint32_t*>(
                         static_cast<char*>(cqe_slot_) + 60);
        q.resv     = counters;
        q.ready    = counters + 2;
        q.rung     = counters + 3;
        q.qpn      = qpn_;
        q.nwqes    = nwqes_;
        return q;
    }

private:
    ibv_context* ctx_;
    ibv_pd*      pd_;
    uint8_t      port_;
    uint32_t     nwqes_;
    void*        cqe_slot_;

    mlx5dv_devx_uar*  uar_ = nullptr;
    void*             d_uar_ = nullptr;
    void*             wq_ = nullptr;
    void*             d_wq_ = nullptr;
    mlx5dv_devx_umem* wq_umem_ = nullptr;
    void*             dbr_ = nullptr;
    void*             d_dbr_ = nullptr;
    mlx5dv_devx_umem* dbr_umem_ = nullptr;
    void*             cq_dbr_ = nullptr;
    mlx5dv_devx_umem* cq_dbr_umem_ = nullptr;
    mlx5dv_devx_obj*  cq_ = nullptr;
    mlx5dv_devx_obj*  qp_ = nullptr;
    uint32_t          cqn_ = 0;
    uint32_t          qpn_ = 0;

    static constexpr size_t kPage = 65536;

    void alloc_uar() {
        // Non-cached UAR: an 8-byte doorbell store, no BlueFlame buffer.
        uar_ = mlx5dv_devx_alloc_uar(ctx_, MLX5DV_UAR_ALLOC_TYPE_NC);
        if (!uar_) uar_ = mlx5dv_devx_alloc_uar(ctx_, MLX5DV_UAR_ALLOC_TYPE_BF);
        if (!uar_) die("mlx5dv_devx_alloc_uar");
        d_uar_ = MmioMap::map(uar_->reg_addr);
    }

    // Page-aligned host memory, registered with the NIC and (when d_ptr is
    // given) mapped into the GPU.
    void alloc_host(void** ptr, size_t bytes, mlx5dv_devx_umem** umem, void** d_ptr) {
        const size_t size = (bytes + kPage - 1) / kPage * kPage;
        if (posix_memalign(ptr, kPage, size)) die("posix_memalign", ENOMEM);
        std::memset(*ptr, 0, size);
        *umem = umem_reg(ctx_, *ptr, size);
        if (!*umem) die("mlx5dv_devx_umem_reg(host)");
        if (d_ptr) {
            cuda_check(cudaHostRegister(*ptr, size, cudaHostRegisterPortable |
                                                  cudaHostRegisterMapped),
                     "cudaHostRegister(queue)");
            cuda_check(cudaHostGetDevicePointer(d_ptr, *ptr, 0),
                     "cudaHostGetDevicePointer(queue)");
        }
    }

    void free_host(void* ptr, mlx5dv_devx_umem* umem) {
        if (!ptr) return;
        if (ptr == wq_ || ptr == dbr_) cudaHostUnregister(ptr);
        if (umem) mlx5dv_devx_umem_dereg(umem);
        std::free(ptr);
    }

    void modify(void* in, size_t in_len, void* out, size_t out_len, const char* what) {
        if (mlx5dv_devx_obj_modify(qp_, in, in_len, out, out_len)) {
            std::fprintf(stderr, "[gicc] QP 0x%x %s failed: %s (syndrome 0x%x)\n",
                         qpn_, what, std::strerror(errno),
                         DEVX_GET(rst2init_qp_out, out, syndrome));
            std::abort();
        }
    }

    // Collapsed CQ (cc=1): the NIC writes every CQE into entry 0, so a
    // single 64-byte slot is all the GPU polls. oi=1 lifts the overrun check,
    // which is what lets nobody ever update the consumer index.
    void create_cq(uint32_t eqn, mlx5dv_devx_umem* umem, uint64_t off) {
        uint8_t in[DEVX_ST_SZ_BYTES(create_cq_in)] = {};
        uint8_t out[DEVX_ST_SZ_BYTES(create_cq_out)] = {};
        DEVX_SET(create_cq_in, in, opcode, MLX5_CMD_OP_CREATE_CQ);
        DEVX_SET(create_cq_in, in, cq_umem_id, umem->umem_id);
        DEVX_SET(create_cq_in, in, cq_umem_valid, 1);
        DEVX_SET64(create_cq_in, in, cq_umem_offset, off);
        void* cqc = DEVX_ADDR_OF(create_cq_in, in, cq_context);
        DEVX_SET(cqc, cqc, dbr_umem_valid, 1);
        DEVX_SET(cqc, cqc, dbr_umem_id, cq_dbr_umem_->umem_id);
        DEVX_SET64(cqc, cqc, dbr_addr, 0);
        DEVX_SET(cqc, cqc, cqe_sz, 0);                  // 64-byte CQEs
        DEVX_SET(cqc, cqc, cc, 1);
        DEVX_SET(cqc, cqc, oi, 1);
        DEVX_SET(cqc, cqc, log_cq_size, 0);
        DEVX_SET(cqc, cqc, uar_page, uar_->page_id);
        DEVX_SET(cqc, cqc, c_eqn, eqn);
        DEVX_SET(cqc, cqc, log_page_size, 0);           // 4 KB adapter pages
        cq_ = mlx5dv_devx_obj_create(ctx_, in, sizeof(in), out, sizeof(out));
        if (!cq_) {
            std::fprintf(stderr, "[gicc] DevX CREATE_CQ failed: %s (syndrome 0x%x)\n",
                         std::strerror(errno),
                         DEVX_GET(create_qp_out, out, syndrome));  // same header
            std::abort();
        }
        cqn_ = DEVX_GET(create_cq_out, out, cqn);
    }

    void create_qp(uint32_t pdn, uint32_t srqn, uint32_t recv_cqn) {
        uint32_t log_sq = 0;
        while ((1u << log_sq) < nwqes_) ++log_sq;
        uint8_t in[DEVX_ST_SZ_BYTES(create_qp_in)] = {};
        uint8_t out[DEVX_ST_SZ_BYTES(create_qp_out)] = {};
        DEVX_SET(create_qp_in, in, opcode, MLX5_CMD_OP_CREATE_QP);
        DEVX_SET(create_qp_in, in, wq_umem_id, wq_umem_->umem_id);
        DEVX_SET(create_qp_in, in, wq_umem_valid, 1);
        DEVX_SET64(create_qp_in, in, wq_umem_offset, 0);
        void* qpc = DEVX_ADDR_OF(create_qp_in, in, qpc);
        DEVX_SET(qpc, qpc, st, 0x0);                    // RC
        DEVX_SET(qpc, qpc, pm_state, 0x3);
        DEVX_SET(qpc, qpc, pd, pdn);
        DEVX_SET(qpc, qpc, uar_page, uar_->page_id);
        DEVX_SET(qpc, qpc, cqn_snd, cqn_);
        DEVX_SET(qpc, qpc, cqn_rcv, recv_cqn);
        DEVX_SET(qpc, qpc, log_sq_size, log_sq);
        DEVX_SET(qpc, qpc, log_rq_size, 0);
        DEVX_SET(qpc, qpc, rq_type, 1);                 // SRQ
        DEVX_SET(qpc, qpc, srqn_rmpn_xrqn, srqn);
        DEVX_SET(qpc, qpc, dbr_umem_valid, 1);
        DEVX_SET(qpc, qpc, dbr_umem_id, dbr_umem_->umem_id);
        DEVX_SET64(qpc, qpc, dbr_addr, 0);
        qp_ = mlx5dv_devx_obj_create(ctx_, in, sizeof(in), out, sizeof(out));
        if (!qp_) {
            std::fprintf(stderr, "[gicc] DevX CREATE_QP failed: %s (syndrome 0x%x)\n",
                         std::strerror(errno), DEVX_GET(create_qp_out, out, syndrome));
            std::abort();
        }
        qpn_ = DEVX_GET(create_qp_out, out, qpn);
    }
};

} // namespace gicc::mlx5
