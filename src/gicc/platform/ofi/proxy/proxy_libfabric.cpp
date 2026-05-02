/*
 * proxy_libfabric.cpp - host-only libfabric submission / poll wrapper.
 *
 * The constructor first attempts to open a dedicated TX completion queue
 * for the proxy and bind it to the existing Fabric endpoint with the
 * FI_TRANSMIT flag. If either fi_cq_open or fi_ep_bind fails — most
 * notably on providers that reject a second TX CQ on an already-enabled
 * endpoint — we fall back to sharing Fabric's CQ and clear own_cq_ so the
 * destructor does not double-close it.
 */
#include "proxy_libfabric.hpp"

#include "gicc/platform/ofi/internal/fabric.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"

#include <rdma/fi_rma.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace gicc {
namespace proxy {

ProxyLibfabric::ProxyLibfabric(::gicc::Fabric& fab, ::gicc::Runtime& rt)
    : fab_(fab), rt_(rt), proxy_cq_(nullptr), own_cq_(false)
{
    // Attempt to open a private TX CQ. Format CONTEXT is sufficient — we only
    // need the op_context pointer back (which we use as the slot index).
    struct fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.size   = 4096;

    int ret = fi_cq_open(fab_.fabric->domain, &cq_attr, &proxy_cq_, NULL);
    if (ret != 0) {
        fprintf(stderr,
            "ProxyLibfabric: fi_cq_open failed (%s); sharing Fabric CQ\n",
            fi_strerror(-ret));
        proxy_cq_ = fab_.fabric->cq;
        own_cq_   = false;
        return;
    }

    ret = fi_ep_bind(fab_.fabric->ep, &proxy_cq_->fid, FI_TRANSMIT);
    if (ret != 0) {
        fprintf(stderr,
            "ProxyLibfabric: ep_bind 2nd TX CQ failed (%s); sharing Fabric CQ\n",
            fi_strerror(-ret));
        fi_close(&proxy_cq_->fid);
        proxy_cq_ = fab_.fabric->cq;
        own_cq_   = false;
        return;
    }

    own_cq_ = true;
    fprintf(stderr,
        "ProxyLibfabric: dual TX CQ active (provider accepted second CQ)\n");
}

ProxyLibfabric::~ProxyLibfabric()
{
    if (own_cq_ && proxy_cq_) {
        fi_close(&proxy_cq_->fid);
    }
    proxy_cq_ = nullptr;
}

int ProxyLibfabric::submit_write(const TransferCmd& c, uint64_t slot)
{
    // Local buffer (src) metadata lives on Runtime; remote (rma_addr,
    // rma_key, base_addr) is also cached on Runtime by exchange().
    auto        lb   = rt_.local_buf_view(c.src_buf);
    const auto& ri   = rt_.remote_info(c.dst_rank, c.dst_buf);
    fi_addr_t   peer = rt_.av_addr(c.dst_rank);

    char*    src   = static_cast<char*>(lb.ptr) + c.src_offset;
    void*    desc  = lb.desc;
    uint64_t raddr = rt_.is_virt_addr_mode()
                       ? (ri.rma_addr + c.dst_offset)
                       : (ri.rma_addr - ri.base_addr) + c.dst_offset;
    uint64_t rkey  = ri.rma_key;

    int ret = fi_write(fab_.fabric->ep,
                       src, c.bytes, desc,
                       peer, raddr, rkey,
                       reinterpret_cast<void*>(slot));
    if (ret == -FI_EAGAIN) {
        return -FI_EAGAIN;
    }
    if (ret != 0) {
        fprintf(stderr,
            "ProxyLibfabric::submit_write: fi_write failed: %s\n",
            fi_strerror(-ret));
        std::abort();
    }
    return 0;
}


int ProxyLibfabric::poll(Completion* out, int max)
{
    // 256 fi_cq_entry slots = 4 KiB on stack. UCCL-EP polls 2048 per call;
    // 256 is the minimum bump that absorbs a typical 50-cmd bench burst in
    // one syscall while keeping the on-stack array modest.
    constexpr int kMaxBatch = 256;
    struct fi_cq_entry entries[kMaxBatch];

    int n = (int)fi_cq_read(proxy_cq_, entries,
                            std::min(max, kMaxBatch));
    if (n == -FI_EAGAIN) {
        return 0;
    }
    if (n == -FI_EAVAIL) {
        struct fi_cq_err_entry err = {};
        ssize_t r = fi_cq_readerr(proxy_cq_, &err, 0);
        if (r > 0) {
            fprintf(stderr,
                "ProxyLibfabric::poll: CQ error: %s (prov_errno=%d)\n",
                fi_strerror(err.err), err.prov_errno);
        }
        std::abort();
    }
    if (n < 0) {
        fprintf(stderr,
            "ProxyLibfabric::poll: fi_cq_read failed: %s\n",
            fi_strerror(-n));
        std::abort();
    }
    for (int i = 0; i < n; ++i) {
        out[i].context = entries[i].op_context;
    }
    return n;
}

} // namespace proxy
} // namespace gicc
