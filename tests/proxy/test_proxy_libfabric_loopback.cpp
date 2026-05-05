/*
 * test_proxy_libfabric_loopback.cpp - Single-rank loopback for ProxyLibfabric.
 *
 * Registers a single GPU buffer, fi_write's onto itself via the proxy
 * wrapper, then polls until the completion fires. Asserts that the
 * op_context returned through the CQ entry equals the slot we passed.
 *
 * This is the smallest end-to-end exercise of the dual-CQ-attempt-then-
 * fall-back constructor and the submit/poll path. Whether the constructor
 * binds its own TX CQ or shares Fabric's is logged at first run — that is
 * the answer Task 3's libfabric probe could not provide.
 */
#include "gicc/platform/ofi/proxy/proxy_libfabric.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"

#include <cassert>
#include <cstdio>
#include <cstdint>

using namespace gicc;
using namespace gicc::proxy;

int main() {
    Runtime rt;

    constexpr size_t SZ = 4096;
    void* dev_buf = nullptr;
    if (gpuMalloc(&dev_buf, SZ) != GPU_SUCCESS) {
        fprintf(stderr, "gpuMalloc failed\n");
        return 1;
    }

    auto h_local = rt.register_buffer(dev_buf, SZ, /*is_device=*/true);
    rt.exchange();   // single rank: collective is just a self allgather

    // Use proxy EP 0 — Runtime opens the fleet eagerly in its ctor.
    ProxyLibfabric pl(rt.fabric(), rt, /*ep_idx=*/0);

    TransferCmd c{};
    c.cmd_type   = CmdType::WRITE;
    c.dst_rank   = (uint8_t)rt.rank();   // self
    c.src_buf    = (uint8_t)h_local.index;
    c.dst_buf    = (uint8_t)h_local.index;
    c.bytes      = 1024;
    c.src_offset = 0;
    c.dst_offset = 2048;

    int ret = pl.submit_write(c, /*slot=*/42);
    assert(ret == 0);

    Completion comp{};
    int polled = 0;
    while (polled == 0) {
        polled = pl.poll(&comp, 1);
    }
    assert(comp.context == reinterpret_cast<void*>(static_cast<uintptr_t>(42)));

    printf("test_proxy_libfabric_loopback: PASS\n");
    (void)gpuFree(dev_buf);
    return 0;
}
