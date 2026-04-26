#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
        return 1; \
    } \
} while (0)

int main() {
    gicc::Runtime rt;
    if (rt.size() != 2) return 1;

    void *p1, *p2;
    hipMalloc(&p1, 1024);
    hipMalloc(&p2, 2048);
    auto b1 = rt.register_buffer(p1, 1024, true);
    auto b2 = rt.register_buffer(p2, 2048, true);
    rt.exchange();

    // Local lookup
    CHECK(rt.buffer_by_lkey(b1.lkey).addr == b1.addr);
    CHECK(rt.buffer_by_lkey(b2.lkey).addr == b2.addr);
    CHECK(rt.buffer_by_lkey(b1.lkey).index == 0);
    CHECK(rt.buffer_by_lkey(b2.lkey).index == 1);

    // Remote lookup matches remote_buffer
    int peer = 1 - rt.rank();
    auto rinfo = rt.remote_buffer(peer, b1.index);
    CHECK(rt.peer_buffer_base(peer, b1.lkey) == rinfo.addr);

    rt.boot().barrier();
    if (rt.rank() == 0) printf("lookup: PASS\n");
    return 0;
}
