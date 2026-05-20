/*
 * test_d2h_ring.hip.cpp - GPU pushes 100 WRITE cmds; CPU pops them in order.
 *
 * Microtest for the SPSC D2HRing<Capacity> template under HIP. Byte-identical
 * to test_d2h_ring.cu apart from the runtime sync call and the printf tag.
 */
#include "gicc/proxy/common/d2h_ring.cuh"
#include <cstdio>
#include <cassert>

using namespace gicc::proxy;

constexpr uint32_t kCap = 128;
using Ring = D2HRing<kCap>;

__global__ void producer_kernel(Ring* ring, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < n; ++i) {
        TransferCmd c;
        c.cmd_type   = CmdType::WRITE;
        c.dst_rank   = (uint8_t)(i & 0xFF);
        c.src_buf    = 0;
        c.dst_buf    = 0;
        c.bytes      = (uint32_t)i;
        c.src_offset = (uint64_t)i * 100;
        c.dst_offset = (uint64_t)i * 200;
        ring->atomic_push(c);
    }
}

int main() {
    Ring* dev = nullptr;
    Ring* host = allocate_d2h_ring_host<kCap>(&dev);

    producer_kernel<<<1, 1>>>(dev, 100);
    auto sync_err = hipDeviceSynchronize();
    assert(sync_err == hipSuccess);

    int popped = 0;
    while (popped < 100) {
        TransferCmd c;
        uint64_t slot;
        if (!host->pop(c, &slot)) continue;
        assert(c.cmd_type == CmdType::WRITE);
        assert(c.dst_rank == (uint8_t)(popped & 0xFF));
        assert(c.bytes == (uint32_t)popped);
        assert(c.src_offset == (uint64_t)popped * 100);
        assert(c.dst_offset == (uint64_t)popped * 200);
        host->mark_acked(slot);
        host->advance_tail_from_mask();
        ++popped;
    }
    assert(host->head_volatile() == 100);
    assert(host->tail_volatile() == 100);
    printf("test_d2h_ring (HIP): PASS\n");

    free_d2h_ring_host<kCap>(host);
    return 0;
}
