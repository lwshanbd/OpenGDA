// e2_put_quiet_loop.cpp - OpenMP-target backpressure + quiet() liveness test.
// rank 0 issues NITER puts (NITER >> ring capacity 4096, forcing ring-full
// backoff) then a device-side gicc::omp::quiet() — all from inside one omp
// target region. The proxy thread drains concurrently. The test asserts
// LIVENESS (completes well under the time limit, no deadlock) and that the
// final payload still lands correctly on the peer.
#include <omp.h>
#include <cstdio>
#include <cstddef>

#include "gicc/omp.h"
#include "examples/omp/giomp_example_utils.hpp"

int main() {
    const size_t bytes = 4096;       // small: this is a backpressure test, not bandwidth
    const int    NITER = 20000;      // > 4096 ring capacity -> forces ring-full backoff
    ompx_init();
    ompx_buffer buffer = ompx_alloc(bytes);
    ompx_exchange();

    int my = omp_get_rank_num();
    int nr = omp_get_num_ranks();
    if (nr != 2) {
        if (my == 0) fprintf(stderr, "need 2 ranks\n");
        ompx_free(buffer);
        ompx_finalize();
        return 1;
    }
    int peer = my ^ 1;
    int bidx = buffer.index;

    giomp_example::fill_buffer(buffer, my == 0 ? 0xAB : 0x00);
    ompx_barrier();

    gicc::DeviceCtx* d_ctx = ompx_prepare();

    if (my == 0) {
        #pragma omp target is_device_ptr(d_ctx)
        {
            for (int i = 0; i < NITER; ++i)
                ompx_put_proxy(d_ctx, peer, bidx, /*dst_off=*/0,
                               bidx, /*src_off=*/0, bytes);
            ompx_quiet(d_ctx);   // device-side fence: must drain before returning
        }
        ompx_quiet_host();
    } else {
        ompx_quiet_host();
    }
    ompx_barrier();

    int rc = 0;
    if (my == 1) {
        size_t bad = giomp_example::count_mismatches(buffer, 0xAB);
        printf("recv: bad_bytes=%zu/%zu after %d puts : %s\n",
               bad, bytes, NITER, bad == 0 ? "PASS" : "FAIL");
        rc = (bad == 0) ? 0 : 2;
    }
    ompx_barrier();
    ompx_free(buffer);
    ompx_finalize();
    return rc;
}
