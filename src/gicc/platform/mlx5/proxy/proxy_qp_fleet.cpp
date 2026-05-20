/*
 * proxy_qp_fleet.cpp - construct + connect N_peers-1 RC QPs for one proxy
 * thread; tear them down on destruction.
 *
 * QP state machine: RESET -> INIT -> RTR -> RTS via ibv_modify_qp.
 *
 * Peer QPN/LID/GID exchange uses Bootstrap::sendrecv with a tag derived
 * from thread_idx so parallel proxy threads on the same rank don't alias
 * each other during the handshake.
 */
#include "proxy_qp_fleet.hpp"

#include "gicc/bootstrap/bootstrap.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gicc::mlx5::proxy {

namespace {

struct ConnInfo {
    uint32_t qpn;
    uint16_t lid;
    uint8_t  gid[16];
    uint32_t psn;
};

constexpr uint8_t kIbPort      = 1;
constexpr uint8_t kGidIndex    = 0;
constexpr uint32_t kStartingPsn = 0;

void modify_to_init(ibv_qp* qp) {
    ibv_qp_attr a{};
    a.qp_state        = IBV_QPS_INIT;
    a.pkey_index      = 0;
    a.port_num        = kIbPort;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE  |
                        IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ  |
                        IBV_ACCESS_REMOTE_ATOMIC;
    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS;
    int ret = ibv_modify_qp(qp, &a, flags);
    if (ret) {
        fprintf(stderr,
            "ProxyQpFleet: ibv_modify_qp INIT failed: %s (%d)\n",
            strerror(ret), ret);
        std::abort();
    }
}

void modify_to_rtr(ibv_qp* qp, const ConnInfo& peer) {
    ibv_qp_attr a{};
    a.qp_state           = IBV_QPS_RTR;
    a.path_mtu           = IBV_MTU_1024;
    a.dest_qp_num        = peer.qpn;
    a.rq_psn             = peer.psn;
    a.max_dest_rd_atomic = 16;
    a.min_rnr_timer      = 12;

    a.ah_attr.is_global    = 1;
    a.ah_attr.dlid         = peer.lid;
    a.ah_attr.sl           = 0;
    a.ah_attr.src_path_bits = 0;
    a.ah_attr.port_num     = kIbPort;
    a.ah_attr.grh.hop_limit  = 1;
    a.ah_attr.grh.sgid_index = kGidIndex;
    a.ah_attr.grh.traffic_class = 0;
    a.ah_attr.grh.flow_label    = 0;
    memcpy(&a.ah_attr.grh.dgid, peer.gid, 16);

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    int ret = ibv_modify_qp(qp, &a, flags);
    if (ret) {
        fprintf(stderr,
            "ProxyQpFleet: ibv_modify_qp RTR failed: %s (%d)\n",
            strerror(ret), ret);
        std::abort();
    }
}

void modify_to_rts(ibv_qp* qp, uint32_t my_psn) {
    ibv_qp_attr a{};
    a.qp_state      = IBV_QPS_RTS;
    a.timeout       = 14;
    a.retry_cnt     = 7;
    a.rnr_retry     = 7;
    a.sq_psn        = my_psn;
    a.max_rd_atomic = 16;

    int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    int ret = ibv_modify_qp(qp, &a, flags);
    if (ret) {
        fprintf(stderr,
            "ProxyQpFleet: ibv_modify_qp RTS failed: %s (%d)\n",
            strerror(ret), ret);
        std::abort();
    }
}

} // anon namespace

ProxyQpFleet::ProxyQpFleet(ibv_context* ctx,
                           ibv_pd*      pd,
                           Bootstrap&   boot,
                           int          thread_idx,
                           uint32_t     sq_depth,
                           uint32_t     cq_depth)
    : ctx_(ctx), pd_(pd), thread_idx_(thread_idx)
{
    cq_ = ibv_create_cq(ctx_, static_cast<int>(cq_depth),
                        /*cq_context=*/nullptr,
                        /*channel=*/nullptr,
                        /*comp_vector=*/0);
    if (!cq_) {
        fprintf(stderr,
            "ProxyQpFleet[%d]: ibv_create_cq failed: %s\n",
            thread_idx_, strerror(errno));
        std::abort();
    }

    const int N = boot.size();
    qps_.assign(N, nullptr);

    // 1. Create QPs (all peers except self) in RESET state.
    for (int peer = 0; peer < N; ++peer) {
        if (peer == boot.rank()) continue;

        ibv_qp_init_attr qp_attr{};
        qp_attr.send_cq          = cq_;
        qp_attr.recv_cq          = cq_;
        qp_attr.qp_type          = IBV_QPT_RC;
        qp_attr.sq_sig_all       = 0;            // signal only when WR.send_flags asks
        qp_attr.cap.max_send_wr  = sq_depth;
        qp_attr.cap.max_recv_wr  = 1;            // we don't recv but verbs requires > 0
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.cap.max_inline_data = 0;

        ibv_qp* qp = ibv_create_qp(pd_, &qp_attr);
        if (!qp) {
            fprintf(stderr,
                "ProxyQpFleet[%d]: ibv_create_qp(peer=%d) failed: %s\n",
                thread_idx_, peer, strerror(errno));
            std::abort();
        }
        modify_to_init(qp);
        qps_[peer] = qp;
    }

    // 2. Exchange ConnInfo with each peer. Tag includes thread_idx so
    //    parallel fleets don't alias on a shared MPI tag space.
    ibv_port_attr port_attr{};
    ibv_query_port(ctx_, kIbPort, &port_attr);
    ibv_gid my_gid{};
    ibv_query_gid(ctx_, kIbPort, kGidIndex, &my_gid);

    // Tag must be symmetric across the (rank, peer) pair — MPI_Sendrecv
    // uses the same tag for send + receive on both sides. Encoding only
    // thread_idx keeps parallel proxy-fleet handshakes on different
    // threads from aliasing each other while staying symmetric.
    const int xchg_tag = 0x5000 + thread_idx_;

    for (int peer = 0; peer < N; ++peer) {
        if (peer == boot.rank()) continue;
        ConnInfo me{}, them{};
        me.qpn = qps_[peer]->qp_num;
        me.lid = port_attr.lid;
        memcpy(me.gid, &my_gid, 16);
        me.psn = kStartingPsn;
        boot.sendrecv(&me, &them, sizeof(ConnInfo), peer, xchg_tag);

        modify_to_rtr(qps_[peer], them);
        modify_to_rts(qps_[peer], me.psn);
    }

    boot.barrier();
}

ProxyQpFleet::~ProxyQpFleet() {
    for (auto* qp : qps_) {
        if (qp) ibv_destroy_qp(qp);
    }
    qps_.clear();
    if (cq_) {
        ibv_destroy_cq(cq_);
        cq_ = nullptr;
    }
}

} // namespace gicc::mlx5::proxy
