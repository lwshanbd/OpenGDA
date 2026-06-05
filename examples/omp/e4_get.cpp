// e4_get.cpp - OpenMP-target gicc::omp::get() correctness test (RMA read).
// rank 1 (source) holds a known pattern; rank 0 (puller) issues get() to pull
// that data into its own buffer from inside an omp target region, quiets, and
// then verifies the landed bytes ON-DEVICE (after quiet) — exercising the
// device-side read-after-quiet ordering path. A host cross-check is secondary.
#include <omp.h>
#include <cstdio>
#include <cstddef>

#include "gicc/platform/ofi/gicc_omp_device.hpp"
#include "examples/omp/gicc_omp_bridge.hpp"

int main() {
    const size_t bytes = 4096;
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

    // Source (rank 1) holds 0xCD; puller (rank 0) starts at 0x00.
    gicc_omp_bridge::fill_buffer(my == 1 ? 0xCD : 0x00, bytes);

    gicc::DeviceCtx* d_ctx = gicc_omp_bridge::prepare();
    gicc_omp_bridge::barrier();   // ensure source's fill is done before the pull

    int rc = 0;
    if (my == 0) {
        unsigned char* dbuf = static_cast<unsigned char*>(gicc_omp_bridge::device_buffer());
        int dev_bad = 0;
        #pragma omp target is_device_ptr(d_ctx) is_device_ptr(dbuf) map(tofrom: dev_bad)
        {
            gicc::omp::get(d_ctx, peer, bidx, /*src_off=*/0, bidx, /*dst_off=*/0, bytes);
            gicc::omp::quiet(d_ctx);
            // Read landed data ON-DEVICE, after quiet(). This is the path under test.
            int b = 0;
            for (size_t i = 0; i < bytes; ++i)
                if (dbuf[i] != (unsigned char)0xCD) ++b;
            dev_bad = b;
        }
        gicc_omp_bridge::reset();
        size_t host_bad = gicc_omp_bridge::count_mismatches(0xCD, bytes);  // secondary
        int ok = (dev_bad == 0 && host_bad == 0);
        printf("get: ondevice_bad=%d host_bad=%zu : %s\n",
               dev_bad, host_bad, ok ? "PASS" : "FAIL");
        rc = ok ? 0 : 2;
    } else {
        gicc_omp_bridge::reset();   // source stays alive to serve the RMA read
    }
    gicc_omp_bridge::barrier();
    gicc_omp_bridge::finalize();
    return rc;
}
