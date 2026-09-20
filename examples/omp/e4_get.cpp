// e4_get.cpp - OpenMP-target ompx_get_dev() correctness test (RMA read).
// rank 1 (source) holds a known pattern; rank 0 (puller) issues get() to pull
// that data into its own buffer from inside an omp target region, quiets, and
// then verifies the landed bytes ON-DEVICE (after quiet) — exercising the
// device-side read-after-quiet ordering path. A host cross-check is secondary.
#include <omp.h>
#include <cstdio>
#include <cstddef>

#include "gicc/omp.h"
#include "examples/omp/giomp_example_utils.hpp"

int main() {
    const size_t bytes = 4096;
    ompx_init();
    unsigned char* buffer = static_cast<unsigned char*>(ompx_alloc(bytes));

    int my = ompx_get_rank_num();
    int nr = ompx_get_num_ranks();
    if (nr != 2) {
        if (my == 0) fprintf(stderr, "need 2 ranks\n");
        ompx_free(buffer);
        ompx_finalize();
        return 1;
    }
    int peer = my ^ 1;

    // Source (rank 1) holds 0xCD; puller (rank 0) starts at 0x00.
    giomp_example::fill_buffer(buffer, bytes, my == 1 ? 0xCD : 0x00);

    ompx_ctx* d_ctx = ompx_prepare();
    ompx_barrier();   // ensure source's fill is done before the pull

    int rc = 0;
    if (my == 0) {
        int dev_bad = 0;
        #pragma omp target is_device_ptr(d_ctx, buffer) map(tofrom: dev_bad)
        {
            ompx_get_dev(d_ctx, peer, /*dst=*/buffer, /*src=*/buffer, bytes);
            ompx_quiet_dev(d_ctx);
            // Read landed data ON-DEVICE, after quiet(). This is the path under test.
            int b = 0;
            for (size_t i = 0; i < bytes; ++i)
                if (buffer[i] != (unsigned char)0xCD) ++b;
            dev_bad = b;
        }
        ompx_quiet();
        size_t host_bad = giomp_example::count_mismatches(buffer, bytes, 0xCD);  // secondary
        int ok = (dev_bad == 0 && host_bad == 0);
        printf("get: ondevice_bad=%d host_bad=%zu : %s\n",
               dev_bad, host_bad, ok ? "PASS" : "FAIL");
        rc = ok ? 0 : 2;
    } else {
        ompx_quiet();   // source stays alive to serve the RMA read
    }
    ompx_barrier();
    ompx_free(buffer);
    ompx_finalize();
    return rc;
}
