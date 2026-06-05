// e2_put_quiet_loop.cpp - OpenMP-target backpressure + quiet() liveness test.
// rank 0 issues NITER puts (NITER >> ring capacity 4096, forcing ring-full
// backoff) then a device-side gicc::omp::quiet() — all from inside one omp
// target region. The proxy thread drains concurrently. The test asserts
// LIVENESS (completes well under the time limit, no deadlock) and that the
// final payload still lands correctly on the peer.
#include <omp.h>
#include <cstdio>
#include <cstddef>

#include "gicc/platform/ofi/gicc_omp_device.hpp"   // gicc::omp::put/quiet + DeviceCtx
#include "examples/omp/gicc_omp_bridge.hpp"

int main() {
    const size_t bytes = 4096;       // small: this is a backpressure test, not bandwidth
    const int    NITER = 20000;      // > 4096 ring capacity -> forces ring-full backoff
    gicc_omp_bridge::init(bytes);

    int my = gicc_omp_bridge::rank();
    int nr = gicc_omp_bridge::nranks();
    if (nr != 2) {
        if (my == 0) fprintf(stderr, "need 2 ranks\n");
        gicc_omp_bridge::finalize();
        return 1;
    }
    int peer = my ^ 1;
    int bidx = gicc_omp_bridge::buf_index();

    gicc_omp_bridge::fill_buffer(my == 0 ? 0xAB : 0x00, bytes);

    gicc::DeviceCtx* d_ctx = gicc_omp_bridge::prepare();

    if (my == 0) {
        #pragma omp target is_device_ptr(d_ctx)
        {
            for (int i = 0; i < NITER; ++i)
                gicc::omp::put(d_ctx, peer, bidx, /*dst_off=*/0,
                                      bidx, /*src_off=*/0, bytes);
            gicc::omp::quiet(d_ctx);   // device-side fence: must drain before returning
        }
        gicc_omp_bridge::reset();
    } else {
        gicc_omp_bridge::reset();
    }
    gicc_omp_bridge::barrier();

    int rc = 0;
    if (my == 1) {
        size_t bad = gicc_omp_bridge::count_mismatches(0xAB, bytes);
        printf("recv: bad_bytes=%zu/%zu after %d puts : %s\n",
               bad, bytes, NITER, bad == 0 ? "PASS" : "FAIL");
        rc = (bad == 0) ? 0 : 2;
    }
    gicc_omp_bridge::barrier();
    gicc_omp_bridge::finalize();
    return rc;
}
