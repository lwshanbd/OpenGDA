// examples/omp/e0b_ring_push.cpp
// E0b: omp target region pushes TransferCmds into a real ProxyRing; the CPU
// pops them with the ring's host-side pop(). Verifies the gicc::omp port writes
// a byte-compatible, correctly-ordered slot. Single process, no NIC.
#include <hip/hip_runtime.h>
#include <omp.h>
#include <cstdio>
#include <cstdint>

// gicc_omp_device.hpp brings: ProxyRing/TransferCmd/CmdType (via proxy_ring_defs.hpp
// -> d2h_ring.cuh, incl. the host allocator + pop()), the HIP-free DeviceCtx
// (via device_ctx.hpp), and the gicc::omp::put declare-target function.
#include "gicc/platform/ofi/gicc_omp_device.hpp"

using gicc::proxy::ProxyRing;
using gicc::proxy::TransferCmd;
using gicc::proxy::CmdType;

#define NPUSH 100

int main() {
    hipSetDevice(0); omp_set_default_device(0);

    ProxyRing* dev_ring = nullptr;
    ProxyRing* host_ring =
        gicc::proxy::allocate_d2h_ring_host<gicc::proxy::kProxyRingCapacity>(&dev_ring);

    // Minimal DeviceCtx pointing at the single ring (pinned/mapped).
    gicc::DeviceCtx* host_ctx = nullptr;
    hipHostMalloc((void**)&host_ctx, sizeof(gicc::DeviceCtx), hipHostMallocMapped);
    *host_ctx = gicc::DeviceCtx{};
    host_ctx->proxy_ring = dev_ring;          // single-ring fallback path
    host_ctx->proxy_rings_arr = nullptr;
    host_ctx->num_proxy_rings = 0;
    gicc::DeviceCtx* dev_ctx = nullptr;
    hipHostGetDevicePointer((void**)&dev_ctx, host_ctx, 0);

    // Producer: NPUSH WRITE commands with distinct, checkable fields, single
    // device thread (1 team / 1 thread) to keep slot order deterministic.
    #pragma omp target is_device_ptr(dev_ctx)
    {
        for (int i = 0; i < NPUSH; ++i)
            gicc::omp::put(dev_ctx, /*rank=*/1, /*dbuf=*/2, /*doff=*/(size_t)i,
                                    /*sbuf=*/3, /*soff=*/(size_t)(i * 10),
                                    /*bytes=*/(size_t)(i + 1));
    }

    // Consumer: pop all NPUSH and verify fields + order.
    int got = 0; bool ok = true;
    while (got < NPUSH) {
        TransferCmd c; uint64_t slot;
        if (!host_ring->pop(c, &slot)) continue;
        if (c.cmd_type != CmdType::WRITE || c.dst_rank != 1 || c.dst_buf != 2 ||
            c.src_buf != 3 || c.dst_offset != (uint64_t)got ||
            c.src_offset != (uint64_t)(got * 10) || c.bytes != (uint32_t)(got + 1)) {
            printf("FAIL at %d: type=%d rank=%d dbuf=%d sbuf=%d doff=%llu soff=%llu bytes=%u\n",
                   got, (int)c.cmd_type, c.dst_rank, c.dst_buf, c.src_buf,
                   (unsigned long long)c.dst_offset, (unsigned long long)c.src_offset, c.bytes);
            ok = false; break;
        }
        host_ring->mark_acked(slot);
        host_ring->advance_tail_from_mask();
        ++got;
    }
    printf("popped=%d/%d : %s\n", got, NPUSH, (ok && got == NPUSH) ? "PASS" : "FAIL");
    gicc::proxy::free_d2h_ring_host<gicc::proxy::kProxyRingCapacity>(host_ring);
    hipHostFree(host_ctx);
    return (ok && got == NPUSH) ? 0 : 2;
}
