/*
 * proxy_verbs.cpp - submit and poll RDMA over verbs for the MLX5 proxy.
 */
// Suppress the inline __global__ kernels in device_opt.cuh — this is a
// pure host TU; g++ would otherwise try to parse blockIdx / clock64 /
// __syncthreads at namespace scope and fail.
#define GICC_DEVICE_OPT_SUPPRESS_KERNELS 1

#include "proxy_verbs.hpp"
#include "proxy_qp_fleet.hpp"

#include "gicc/platform/mlx5/mlx5_runtime.hpp"

#include <infiniband/verbs.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gicc::mlx5::proxy {

namespace {

// One per-Verbs scratch landing for IBV_WR_ATOMIC_FETCH_AND_ADD's
// returned previous value. We only need any GPU-reachable 8-byte slot
// registered in pd_; the value itself is discarded (we want the *remote*
// side-effect, not the returned pre-add). Allocated lazily in the first
// submit_atomic_add call on each ProxyVerbs.
struct AtomicScratch {
    uint64_t* host_ptr = nullptr;
    ibv_mr*   mr       = nullptr;
};

} // anon namespace

ProxyVerbs::ProxyVerbs(::gicc::Runtime& rt, ProxyQpFleet& fleet)
    : rt_(rt), fleet_(fleet) {}

int ProxyVerbs::submit_write(const ::gicc::proxy::TransferCmd& c, uint64_t slot)
{
    ibv_qp* qp = fleet_.qp(c.dst_rank);
    if (!qp) {
        fprintf(stderr,
            "ProxyVerbs::submit_write: no QP for dst_rank=%u (self or out-of-range)\n",
            (unsigned)c.dst_rank);
        std::abort();
    }

    const auto& lb = rt_.proxy_local_buf(c.src_buf);
    const auto& rb = rt_.proxy_remote_buf(c.dst_rank, c.dst_buf);

    ibv_sge sge{};
    sge.addr   = lb.addr + c.src_offset;
    sge.length = c.bytes;
    sge.lkey   = lb.lkey;

    ibv_send_wr wr{};
    wr.wr_id      = slot;
    wr.next       = nullptr;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.opcode     = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;            // one CQE per submit; ack drives tail.
    wr.wr.rdma.remote_addr = rb.addr + c.dst_offset;
    wr.wr.rdma.rkey        = rb.rkey;

    ibv_send_wr* bad = nullptr;
    int ret = ibv_post_send(qp, &wr, &bad);
    if (ret == ENOMEM) return -ENOMEM;
    if (ret != 0) {
        fprintf(stderr,
            "ProxyVerbs::submit_write: ibv_post_send failed: %s (%d) bad=%p\n",
            strerror(ret), ret, (void*)bad);
        std::abort();
    }
    return 0;
}

int ProxyVerbs::submit_read(const ::gicc::proxy::TransferCmd& c, uint64_t slot)
{
    ibv_qp* qp = fleet_.qp(c.dst_rank);
    if (!qp) {
        fprintf(stderr,
            "ProxyVerbs::submit_read: no QP for dst_rank=%u (self or out-of-range)\n",
            (unsigned)c.dst_rank);
        std::abort();
    }

    // Cmd-struct convention: src_* = LOCAL landing, dst_* = REMOTE source
    // on dst_rank. Same accessors as submit_write but the data flow is
    // reversed (NIC -> local).
    const auto& lb = rt_.proxy_local_buf(c.src_buf);
    const auto& rb = rt_.proxy_remote_buf(c.dst_rank, c.dst_buf);

    ibv_sge sge{};
    sge.addr   = lb.addr + c.src_offset;
    sge.length = c.bytes;
    sge.lkey   = lb.lkey;

    ibv_send_wr wr{};
    wr.wr_id      = slot;
    wr.next       = nullptr;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.opcode     = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;            // one CQE per submit; ack drives tail.
    wr.wr.rdma.remote_addr = rb.addr + c.dst_offset;
    wr.wr.rdma.rkey        = rb.rkey;

    ibv_send_wr* bad = nullptr;
    int ret = ibv_post_send(qp, &wr, &bad);
    if (ret == ENOMEM) return -ENOMEM;
    if (ret != 0) {
        fprintf(stderr,
            "ProxyVerbs::submit_read: ibv_post_send failed: %s (%d) bad=%p\n",
            strerror(ret), ret, (void*)bad);
        std::abort();
    }
    return 0;
}

int ProxyVerbs::submit_atomic_add(const ::gicc::proxy::TransferCmd& c, uint64_t slot)
{
    // Lazy-allocate one 8-byte scratch landing per ProxyVerbs to absorb
    // IBV_WR_ATOMIC_FETCH_AND_ADD's returned previous value. Registered
    // in the Runtime's shared PD.
    static thread_local AtomicScratch scratch;
    if (!scratch.mr) {
        // 64 bytes (cache line) so neighbouring submissions don't false-share.
        scratch.host_ptr = static_cast<uint64_t*>(aligned_alloc(64, 64));
        if (!scratch.host_ptr) {
            fprintf(stderr,
                "ProxyVerbs::submit_atomic_add: aligned_alloc(64,64) failed\n");
            std::abort();
        }
        *scratch.host_ptr = 0;
        scratch.mr = rt_.proxy_register_host(scratch.host_ptr, 64);
        if (!scratch.mr) {
            fprintf(stderr,
                "ProxyVerbs::submit_atomic_add: register scratch failed\n");
            std::abort();
        }
    }

    ibv_qp* qp = fleet_.qp(c.dst_rank);
    if (!qp) {
        fprintf(stderr,
            "ProxyVerbs::submit_atomic_add: no QP for dst_rank=%u\n",
            (unsigned)c.dst_rank);
        std::abort();
    }

    // The user-supplied "value to add" lives at (src_buf, src_offset)
    // as a uint32 in the OFI semantics; verbs IBV_WR_ATOMIC_FETCH_AND_ADD
    // is 64-bit and reads the addend from the WR.compare_add field, not
    // from a memory operand. Snapshot the 4 bytes into a uint64_t addend.
    const auto& lb = rt_.proxy_local_buf(c.src_buf);
    uint32_t addend32 = 0;
    std::memcpy(&addend32,
                reinterpret_cast<const char*>(lb.addr) + c.src_offset,
                sizeof(uint32_t));

    const auto& rb = rt_.proxy_remote_buf(c.dst_rank, c.dst_buf);

    ibv_sge sge{};
    sge.addr   = reinterpret_cast<uintptr_t>(scratch.host_ptr);
    sge.length = 8;
    sge.lkey   = scratch.mr->lkey;

    ibv_send_wr wr{};
    wr.wr_id      = slot;
    wr.next       = nullptr;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.opcode     = IBV_WR_ATOMIC_FETCH_AND_ADD;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.atomic.remote_addr = rb.addr + c.dst_offset;
    wr.wr.atomic.rkey        = rb.rkey;
    wr.wr.atomic.compare_add = static_cast<uint64_t>(addend32);

    ibv_send_wr* bad = nullptr;
    int ret = ibv_post_send(qp, &wr, &bad);
    if (ret == ENOMEM) return -ENOMEM;
    if (ret != 0) {
        fprintf(stderr,
            "ProxyVerbs::submit_atomic_add: ibv_post_send failed: %s (%d)\n",
            strerror(ret), ret);
        std::abort();
    }
    return 0;
}

int ProxyVerbs::poll(Completion* out, int max)
{
    constexpr int kMaxBatch = 256;
    ibv_wc wcs[kMaxBatch];
    int want = std::min(max, kMaxBatch);
    int n = ibv_poll_cq(fleet_.cq(), want, wcs);
    if (n < 0) {
        fprintf(stderr,
            "ProxyVerbs::poll: ibv_poll_cq returned %d\n", n);
        std::abort();
    }
    for (int i = 0; i < n; ++i) {
        if (wcs[i].status != IBV_WC_SUCCESS) {
            fprintf(stderr,
                "ProxyVerbs::poll: CQ error wr_id=%lu status=%s (%d) vendor=0x%x\n",
                (unsigned long)wcs[i].wr_id,
                ibv_wc_status_str(wcs[i].status), wcs[i].status,
                wcs[i].vendor_err);
            std::abort();
        }
        out[i].context = reinterpret_cast<void*>(wcs[i].wr_id);
    }
    return n;
}

} // namespace gicc::mlx5::proxy
