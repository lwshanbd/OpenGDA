/*
 * test_proxy_thread.cpp - Single-rank end-to-end test for ProxyThread.
 *
 * Pushes 100 WRITE commands into the host side of the proxy ring (the
 * push helper mirrors the device-side atomic_push, minus the CAS — the
 * test is single-threaded), spins up the proxy worker, and waits for tail
 * to reach 100. The proxy must submit each WRITE via libfabric, observe
 * the CQ completions, mark them acked, and advance tail.
 *
 * Like test_proxy_libfabric_loopback, this test transitively pulls in
 * ofi_runtime.hpp; the CUDA build target therefore fails until Task 7
 * splits the GPU API out of Runtime. The HIP build is the one that
 * exercises the proxy worker today.
 */
#include "gicc/platform/ofi/proxy/proxy_thread.hpp"
#include "gicc/platform/ofi/proxy/d2h_ring.cuh"
#include "gicc/platform/ofi/proxy/transfer_cmd.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace gicc;
using namespace gicc::proxy;

// Host-side push that mimics the device-side atomic_push without the CAS:
// the test driver is single-threaded so a relaxed read + release-store of
// `head` is enough to publish the slot to the proxy worker on another core.
static uint64_t host_push(ProxyRing* r, const TransferCmd& c) {
    uint64_t h   = __atomic_load_n(&r->head, __ATOMIC_RELAXED);
    uint32_t idx = static_cast<uint32_t>(h) & ProxyRing::mask();
    r->buf[idx]  = c;
    __atomic_store_n(&r->head, h + 1, __ATOMIC_RELEASE);
    return h;
}

int main() {
    Runtime rt;

    constexpr size_t SZ = 65536;
    void* dev_buf = nullptr;
    if (gpuMalloc(&dev_buf, SZ) != GPU_SUCCESS) {
        fprintf(stderr, "test_proxy_thread: gpuMalloc failed\n");
        return 1;
    }

    auto h_local = rt.register_buffer(dev_buf, SZ, /*is_device=*/true);
    rt.exchange();   // single rank: collective is just a self allgather

    ProxyThread pt(rt);
    pt.start();

    constexpr int N = 100;
    for (int i = 0; i < N; ++i) {
        TransferCmd c{};
        c.cmd_type   = CmdType::WRITE;
        c.dst_rank   = (uint8_t)rt.rank();
        c.src_buf    = (uint8_t)h_local.index;
        c.dst_buf    = (uint8_t)h_local.index;
        c.bytes      = 64;
        c.src_offset = 0;
        c.dst_offset = (uint64_t)(i * 64);
        host_push(pt.ring_host(), c);
    }

    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::seconds(10);
    while (pt.ring_host()->tail_volatile() < (uint64_t)N) {
        if (std::chrono::steady_clock::now() > deadline) {
            fprintf(stderr,
                    "test_proxy_thread: timeout, tail=%lu\n",
                    (unsigned long)pt.ring_host()->tail_volatile());
            std::abort();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    pt.stop();
    printf("test_proxy_thread: PASS\n");
    (void)gpuFree(dev_buf);
    return 0;
}
