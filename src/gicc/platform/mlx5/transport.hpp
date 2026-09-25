/*
 * transport.hpp - the InfiniBand transport behind the GPU-driven API.
 *
 * Two kinds of RC queue pairs, both on the Runtime's PD:
 *
 *   GPU QPs   one per (peer, lane), created through DevX (gpu_qp.hpp). GPU
 *             threads post to them directly: a put inside a kernel or an
 *             OpenMP target region is a WQE write plus a doorbell, with no
 *             trigger and no proxy.
 *   host QPs  one per peer, plain verbs. Transfers the host issues itself
 *             (ompx_put from host code, Runtime::put) go here, so the CPU
 *             never races a GPU thread for a send queue.
 *
 * A rank also connects to itself, so a put to one's own rank takes the same
 * path as any other.
 *
 * Everything collective happens in the constructor: one allgather carries
 * every QP number, after which each QP connects to its mirror on the peer.
 */
#pragma once

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gicc/bootstrap/bootstrap.hpp"
#include "gicc/platform/mlx5/gpu_qp.hpp"
#include "gicc/platform/mlx5/device_ctx.hpp"

#if defined(__CUDACC__)
#include "gicc/platform/mlx5/mlx5_device.hpp"
#endif

namespace gicc::mlx5 {

#if defined(__CUDACC__)
// Host-side quiet for GPU-posted work: the completion state lives in GPU
// memory, so the cheapest reader is one GPU thread running the same code
// the kernels use. A template keeps one definition across TUs.
template <int = 0>
__global__ void quiet_kernel(const DeviceCtx* ctx) {
    dev::quiet_all(ctx);
}
#endif

// Open the HCA with DevX enabled. GICC_IB_DEV names the device; otherwise
// the first one whose port 1 is an active InfiniBand port wins, then the
// first active port of any kind (RoCE).
inline ibv_context* open_device(int rank) {
    int n = 0;
    ibv_device** list = ibv_get_device_list(&n);
    if (!list || n == 0) {
        std::fprintf(stderr, "[gicc] rank %d: no RDMA devices\n", rank);
        std::abort();
    }
    const char* want = std::getenv("GICC_IB_DEV");
    ibv_device* pick = nullptr;
    ibv_device* fallback = nullptr;
    for (int i = 0; i < n && !pick; ++i) {
        const char* name = ibv_get_device_name(list[i]);
        if (want) {
            if (std::strcmp(name, want) == 0) pick = list[i];
            continue;
        }
        ibv_context* probe = ibv_open_device(list[i]);
        if (!probe) continue;
        ibv_port_attr port = {};
        if (ibv_query_port(probe, 1, &port) == 0 && port.state == IBV_PORT_ACTIVE) {
            if (port.link_layer == IBV_LINK_LAYER_INFINIBAND) pick = list[i];
            else if (!fallback) fallback = list[i];
        }
        ibv_close_device(probe);
    }
    if (!pick) pick = fallback;
    if (!pick) {
        std::fprintf(stderr, "[gicc] rank %d: no active RDMA port%s%s\n", rank,
                     want ? " on GICC_IB_DEV=" : "", want ? want : "");
        std::abort();
    }
    mlx5dv_context_attr attr = {};
    attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
    ibv_context* ctx = mlx5dv_open_device(pick, &attr);
    if (!ctx) {
        std::fprintf(stderr, "[gicc] rank %d: mlx5dv_open_device(%s, DEVX) failed: %s\n",
                     rank, ibv_get_device_name(pick), std::strerror(errno));
        std::abort();
    }
    ibv_free_device_list(list);
    return ctx;
}

// Our address on `port`. RoCE addresses by GID, so pick a RoCE v2 entry.
inline IbPortAddr port_addr(ibv_context* ctx, uint8_t port_num) {
    ibv_port_attr port = {};
    if (ibv_query_port(ctx, port_num, &port)) die("ibv_query_port");
    IbPortAddr a = {};
    a.lid = port.lid;
    a.link_layer = port.link_layer;
    if (port.link_layer != IBV_LINK_LAYER_INFINIBAND) {
        int index = -1;
        for (int i = 0; i < port.gid_tbl_len && index < 0; ++i) {
            char path[256];
            std::snprintf(path, sizeof(path),
                          "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d",
                          ibv_get_device_name(ctx->device), port_num, i);
            FILE* f = std::fopen(path, "r");
            if (!f) continue;
            char type[32] = {};
            if (std::fgets(type, sizeof(type), f) &&
                (std::strstr(type, "RoCE v2") || std::strstr(type, "ROCEv2"))) {
                index = i;
            }
            std::fclose(f);
        }
        a.gid_index = (uint8_t)(index < 0 ? 0 : index);
    }
    ibv_gid gid;
    if (ibv_query_gid(ctx, port_num, a.gid_index, &gid)) die("ibv_query_gid");
    std::memcpy(a.gid, gid.raw, 16);
    return a;
}

class Transport {
public:
    static constexpr uint8_t kPort = 1;
    static constexpr size_t  kCqStride = 4096;

    // Collective: every rank constructs its engine at the same point.
    Transport(Bootstrap& boot, ibv_context* ctx, ibv_pd* pd, int lanes, uint32_t depth)
        : boot_(boot), ctx_(ctx), pd_(pd),
          rank_(boot.rank()), nranks_(boot.size()),
          lanes_(lanes < 1 ? 1 : lanes) {
        uint32_t d = 1;
        while (d < depth) d <<= 1;
        depth_ = d > kMaxWqes ? kMaxWqes : d;
        nqp_ = nranks_ * lanes_;

        // Shared receive side: nothing is ever received into it.
        recv_cq_ = ibv_create_cq(ctx_, 16, nullptr, nullptr, 0);
        if (!recv_cq_) die("ibv_create_cq(recv)");
        ibv_srq_init_attr sattr = {};
        sattr.attr.max_wr = 16;
        sattr.attr.max_sge = 1;
        srq_ = ibv_create_srq(pd_, &sattr);
        if (!srq_) die("ibv_create_srq");
        uint32_t srqn = 0, recv_cqn = 0;
        {
            mlx5dv_obj obj = {};
            mlx5dv_srq dsrq = {};
            dsrq.comp_mask = MLX5DV_SRQ_MASK_SRQN;
            obj.srq.in = srq_;
            obj.srq.out = &dsrq;
            mlx5dv_cq dcq = {};
            obj.cq.in = recv_cq_;
            obj.cq.out = &dcq;
            if (mlx5dv_init_obj(&obj, MLX5DV_OBJ_SRQ | MLX5DV_OBJ_CQ))
                die("mlx5dv_init_obj(SRQ, CQ)");
            srqn = dsrq.srqn;
            recv_cqn = dcq.cqn;
        }
        uint32_t eqn = 0;
        if (mlx5dv_devx_query_eqn(ctx_, 0, &eqn)) die("mlx5dv_devx_query_eqn");

        // Every GPU QP's CQ in one GPU allocation, one registration.
        const size_t cq_bytes = (size_t)nqp_ * kCqStride;
        cuda_check(cudaMalloc(&cq_raw_, cq_bytes + 65536), "cudaMalloc(CQ)");
        cq_base_ = reinterpret_cast<char*>(
            ((uintptr_t)cq_raw_ + 65535) & ~(uintptr_t)65535);
        cuda_check(cudaMemset(cq_base_, 0xFF, cq_bytes), "cudaMemset(CQ)");
        cq_umem_ = mlx5dv_devx_umem_reg(ctx_, cq_base_, cq_bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!cq_umem_) die("mlx5dv_devx_umem_reg(GPU CQ)");

        cuda_check(cudaMalloc(&d_counters_, (size_t)nqp_ * 2 * sizeof(uint64_t)),
                 "cudaMalloc(counters)");
        cuda_check(cudaMemset(d_counters_, 0, (size_t)nqp_ * 2 * sizeof(uint64_t)),
                 "cudaMemset(counters)");

        for (int q = 0; q < nqp_; ++q) {
            gpu_qps_.push_back(std::make_unique<GpuQp>(
                ctx_, pd_, kPort, depth_, srqn, recv_cqn, eqn,
                cq_umem_, (uint64_t)q * kCqStride, cq_base_ + (size_t)q * kCqStride));
        }
        create_host_qps();
        connect_all();

        std::vector<QpView> views(nqp_);
        for (int q = 0; q < nqp_; ++q)
            views[q] = gpu_qps_[q]->device_view(d_counters_ + 2 * q);
        cuda_check(cudaMalloc(&d_qps_, sizeof(QpView) * nqp_), "cudaMalloc(QP views)");
        cuda_check(cudaMemcpy(d_qps_, views.data(), sizeof(QpView) * nqp_,
                            cudaMemcpyHostToDevice), "upload QP views");
        cuda_check(cudaMalloc(&d_ctx_, sizeof(DeviceCtx)), "cudaMalloc(DeviceCtx)");
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                 "cudaStreamCreate");
        h_ctx_.my_rank = rank_;
        h_ctx_.nranks  = nranks_;
        h_ctx_.nlanes  = lanes_;
        h_ctx_.qps     = d_qps_;
        upload_ctx();
        boot_.barrier();
    }

    ~Transport() {
        for (auto& h : host_qps_) if (h.qp) ibv_destroy_qp(h.qp);
        if (host_cq_) ibv_destroy_cq(host_cq_);
        gpu_qps_.clear();
        if (cq_umem_) mlx5dv_devx_umem_dereg(cq_umem_);
        if (srq_) ibv_destroy_srq(srq_);
        if (recv_cq_) ibv_destroy_cq(recv_cq_);
        cudaFree(cq_raw_);
        cudaFree(d_counters_);
        cudaFree(d_qps_);
        cudaFree(d_ctx_);
        free_tables();
        if (stream_) cudaStreamDestroy(stream_);
    }

    Transport(const Transport&) = delete;
    Transport& operator=(const Transport&) = delete;

    int lanes() const { return lanes_; }
    uint32_t depth() const { return depth_; }

    // The device context. Its address never changes; set_* and set_tables
    // update the contents in place.
    DeviceCtx* device_ctx() const { return d_ctx_; }

    // Address book: `laddr/lkey` per local buffer, `raddr/rkey` indexed
    // [peer * nbufs + buf].
    void set_tables(const std::vector<uint64_t>& laddr, const std::vector<uint32_t>& lkey,
                    const std::vector<uint64_t>& raddr, const std::vector<uint32_t>& rkey) {
        free_tables();
        const int nbufs = (int)laddr.size();
        upload(&d_laddr_, laddr);
        upload(&d_lkey_, lkey);
        upload(&d_raddr_, raddr);
        upload(&d_rkey_, rkey);
        h_ctx_.nbufs = nbufs;
        h_ctx_.lbuf_addr = d_laddr_;
        h_ctx_.lbuf_lkey = d_lkey_;
        h_ctx_.rbuf_addr = d_raddr_;
        h_ctx_.rbuf_rkey = d_rkey_;
        upload_ctx();
    }

    void set_heap(void* base, int index) {
        h_ctx_.heap_base = base;
        h_ctx_.heap_buf  = index;
        upload_ctx();
    }

    void set_signals(uint64_t* base, int index) {
        h_ctx_.sig_base = base;
        h_ctx_.sig_buf  = index;
        upload_ctx();
    }

    //--------------------------------------------------------------------------
    // Host-issued transfers (host QPs)
    //--------------------------------------------------------------------------

    void host_write(int peer, uint64_t laddr, uint32_t lkey,
                    uint64_t raddr, uint32_t rkey, size_t bytes) {
        host_post(peer, IBV_WR_RDMA_WRITE, laddr, lkey, raddr, rkey, bytes);
    }

    void host_read(int peer, uint64_t laddr, uint32_t lkey,
                   uint64_t raddr, uint32_t rkey, size_t bytes) {
        host_post(peer, IBV_WR_RDMA_READ, laddr, lkey, raddr, rkey, bytes);
    }

    // An 8-byte RDMA WRITE carried inline, behind everything already posted
    // to `peer` on the host QP.
    void host_write_u64(int peer, uint64_t raddr, uint32_t rkey, uint64_t value) {
        HostQp& h = host_qps_[peer];
        make_room(h);
        ibv_sge sge = {};
        sge.addr = (uint64_t)(uintptr_t)&value;
        sge.length = sizeof(value);
        ibv_send_wr wr = {}, *bad = nullptr;
        wr.wr_id = (uint64_t)peer;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.wr.rdma.remote_addr = raddr;
        wr.wr.rdma.rkey = rkey;
        if (int err = ibv_post_send(h.qp, &wr, &bad)) die("ibv_post_send(inline)", err);
        ++h.outstanding;
        ++host_outstanding_;
    }

    // Reap host-QP completions without waiting.
    void host_progress() {
        ibv_wc wc[32];
        int n;
        while ((n = ibv_poll_cq(host_cq_, 32, wc)) > 0) {
            for (int i = 0; i < n; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    std::fprintf(stderr, "[gicc] rank %d: host RDMA to %d failed: %s\n",
                                 rank_, (int)wc[i].wr_id,
                                 ibv_wc_status_str(wc[i].status));
                    std::abort();
                }
                --host_qps_[wc[i].wr_id].outstanding;
                --host_outstanding_;
            }
        }
        if (n < 0) die("ibv_poll_cq");
    }

    void host_quiet() {
        while (host_outstanding_ > 0) host_progress();
    }

#if defined(__CUDACC__)
    // Wait for every GPU-posted transfer that has been rung so far.
    void device_quiet() {
        quiet_kernel<<<1, 1, 0, stream_>>>(d_ctx_);
        cuda_check(cudaGetLastError(), "launch quiet_kernel");
        cuda_check(cudaStreamSynchronize(stream_), "quiet_kernel");
    }
#endif

private:
    struct HostQp {
        ibv_qp* qp = nullptr;
        int     outstanding = 0;
    };

    // What one rank publishes: its port address, then the QP numbers it
    // created for each peer, host QPs first.
    struct ConnHeader {
        IbPortAddr addr;
        int32_t    lanes;
    };

    Bootstrap&   boot_;
    ibv_context* ctx_;
    ibv_pd*      pd_;
    int          rank_, nranks_, lanes_, nqp_ = 0;
    uint32_t     depth_ = 0;

    ibv_cq*  recv_cq_ = nullptr;
    ibv_srq* srq_ = nullptr;
    void*    cq_raw_ = nullptr;
    char*    cq_base_ = nullptr;
    mlx5dv_devx_umem* cq_umem_ = nullptr;
    uint64_t* d_counters_ = nullptr;
    std::vector<std::unique_ptr<GpuQp>> gpu_qps_;

    ibv_cq*  host_cq_ = nullptr;
    std::vector<HostQp> host_qps_;
    long     host_outstanding_ = 0;
    uint32_t host_depth_ = 256;

    DeviceCtx   h_ctx_{};
    DeviceCtx*  d_ctx_ = nullptr;
    QpView*   d_qps_ = nullptr;
    uint64_t* d_laddr_ = nullptr;
    uint32_t* d_lkey_ = nullptr;
    uint64_t* d_raddr_ = nullptr;
    uint32_t* d_rkey_ = nullptr;
    cudaStream_t stream_ = nullptr;

    template <class T>
    static void upload(T** dst, const std::vector<T>& v) {
        if (v.empty()) { *dst = nullptr; return; }
        cuda_check(cudaMalloc(dst, sizeof(T) * v.size()), "cudaMalloc(table)");
        cuda_check(cudaMemcpy(*dst, v.data(), sizeof(T) * v.size(),
                            cudaMemcpyHostToDevice), "upload table");
    }

    void free_tables() {
        cudaFree(d_laddr_); cudaFree(d_lkey_); cudaFree(d_raddr_); cudaFree(d_rkey_);
        d_laddr_ = nullptr; d_lkey_ = nullptr; d_raddr_ = nullptr; d_rkey_ = nullptr;
    }

    void upload_ctx() {
        cuda_check(cudaMemcpy(d_ctx_, &h_ctx_, sizeof(DeviceCtx), cudaMemcpyHostToDevice),
                 "upload DeviceCtx");
    }

    void create_host_qps() {
        host_cq_ = ibv_create_cq(ctx_, (int)(host_depth_ * nranks_), nullptr, nullptr, 0);
        if (!host_cq_) die("ibv_create_cq(host)");
        host_qps_.resize(nranks_);
        for (int p = 0; p < nranks_; ++p) {
            ibv_qp_init_attr attr = {};
            attr.send_cq = host_cq_;
            attr.recv_cq = recv_cq_;
            attr.srq = srq_;
            attr.qp_type = IBV_QPT_RC;
            attr.sq_sig_all = 0;
            attr.cap.max_send_wr = host_depth_;
            attr.cap.max_send_sge = 1;
            attr.cap.max_inline_data = 16;
            host_qps_[p].qp = ibv_create_qp(pd_, &attr);
            if (!host_qps_[p].qp) die("ibv_create_qp(host)");
        }
    }

    void connect_host_qp(ibv_qp* qp, uint32_t remote_qpn, const IbPortAddr& remote) {
        ibv_port_attr port = {};
        if (ibv_query_port(ctx_, kPort, &port)) die("ibv_query_port");
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = kPort;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
        if (int e = ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                          IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            die("ibv_modify_qp(INIT)", e);

        a = {};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = (ibv_mtu)port_mtu(port);
        a.dest_qp_num = remote_qpn;
        a.rq_psn = 0;
        a.max_dest_rd_atomic = 16;
        a.min_rnr_timer = 12;
        a.ah_attr.port_num = kPort;
        if (remote.link_layer == IBV_LINK_LAYER_INFINIBAND) {
            a.ah_attr.dlid = remote.lid;
        } else {
            a.ah_attr.is_global = 1;
            std::memcpy(a.ah_attr.grh.dgid.raw, remote.gid, 16);
            a.ah_attr.grh.sgid_index = remote.gid_index;
            a.ah_attr.grh.hop_limit = 255;
        }
        if (int e = ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                          IBV_QP_MAX_DEST_RD_ATOMIC |
                                          IBV_QP_MIN_RNR_TIMER))
            die("ibv_modify_qp(RTR)", e);

        a = {};
        a.qp_state = IBV_QPS_RTS;
        a.timeout = 14;
        a.retry_cnt = 7;
        a.rnr_retry = 7;
        a.sq_psn = 0;
        a.max_rd_atomic = 16;
        if (int e = ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT |
                                          IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
            die("ibv_modify_qp(RTS)", e);
    }

    // One allgather, then every QP connects to the mirror QP the peer
    // created for us: host QP p <-> peer's host QP rank_, GPU QP (p, lane) <->
    // peer's GPU QP (rank_, lane).
    void connect_all() {
        const size_t nq = (size_t)nranks_ + (size_t)nqp_;
        std::vector<uint8_t> mine(sizeof(ConnHeader) + nq * sizeof(uint32_t));
        ConnHeader hdr = {};
        hdr.addr = port_addr(ctx_, kPort);
        hdr.lanes = lanes_;
        std::memcpy(mine.data(), &hdr, sizeof(hdr));
        uint32_t* qpns = reinterpret_cast<uint32_t*>(mine.data() + sizeof(hdr));
        for (int p = 0; p < nranks_; ++p) qpns[p] = host_qps_[p].qp->qp_num;
        for (int q = 0; q < nqp_; ++q) qpns[nranks_ + q] = gpu_qps_[q]->qpn();

        auto all = boot_.allgather(mine.data(), (int)mine.size());
        for (int p = 0; p < nranks_; ++p) {
            if (all[p].size() != mine.size()) {
                std::fprintf(stderr, "[gicc] rank %d: peer %d uses a different "
                             "GICC_IB_LANES\n", rank_, p);
                std::abort();
            }
            ConnHeader peer;
            std::memcpy(&peer, all[p].data(), sizeof(peer));
            const uint32_t* pq =
                reinterpret_cast<const uint32_t*>(all[p].data() + sizeof(peer));
            connect_host_qp(host_qps_[p].qp, pq[rank_], peer.addr);
            for (int l = 0; l < lanes_; ++l) {
                gpu_qps_[p * lanes_ + l]->connect(pq[nranks_ + rank_ * lanes_ + l],
                                                  peer.addr);
            }
        }
    }

    void make_room(HostQp& h) {
        while (h.outstanding >= (int)host_depth_) host_progress();
    }

    void host_post(int peer, ibv_wr_opcode op, uint64_t laddr, uint32_t lkey,
                   uint64_t raddr, uint32_t rkey, size_t bytes) {
        HostQp& h = host_qps_[peer];
        for (size_t done = 0; done < bytes; done += kMaxMsg) {
            const size_t len = bytes - done < kMaxMsg ? bytes - done : kMaxMsg;
            make_room(h);
            ibv_sge sge = {};
            sge.addr = laddr + done;
            sge.length = (uint32_t)len;
            sge.lkey = lkey;
            ibv_send_wr wr = {}, *bad = nullptr;
            wr.wr_id = (uint64_t)peer;
            wr.opcode = op;
            wr.send_flags = IBV_SEND_SIGNALED;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.wr.rdma.remote_addr = raddr + done;
            wr.wr.rdma.rkey = rkey;
            if (int err = ibv_post_send(h.qp, &wr, &bad)) die("ibv_post_send", err);
            ++h.outstanding;
            ++host_outstanding_;
        }
    }
};

} // namespace gicc::mlx5
