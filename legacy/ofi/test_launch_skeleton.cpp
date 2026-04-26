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

__global__ void k(gicc::DeviceCtx* ctx) {
    if (threadIdx.x == 0) {
        gicc::flush(ctx);   // no queued ops; trigger fires for nothing
        gicc::quiet(ctx);   // no slots to wait on; returns immediately
    }
}

int main() {
    gicc::Runtime rt;
    if (rt.size() != 2) return 1;
    int peer = 1 - rt.rank();

    void* d;
    (void)hipMalloc(&d, 4096);
    auto b = rt.register_buffer(d, 4096, true);
    rt.exchange();
    rt.boot().barrier();

    // gicc::launch: should call detail::kernel_trace (default no-op),
    // then prepare(), then launch the kernel. The kernel is a non-type
    // template parameter so the OFI backend's kernel_trace<Kernel> can
    // specialize on it.
    gicc::launch<k>(rt, dim3(1), dim3(1), peer, b.index);
    CHECK(hipDeviceSynchronize() == hipSuccess);
    rt.reset();

    rt.boot().barrier();
    if (rt.rank() == 0) printf("launch_skeleton: PASS\n");
    return 0;
}
