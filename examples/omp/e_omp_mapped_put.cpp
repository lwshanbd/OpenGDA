// e_omp_mapped_put.cpp - make-or-break for the minimod port.
//
// Registers an OpenMP target-mapped buffer's device pointer with GICC (via
// `use_device_ptr`) and delivers one face through gicc::omp::put from inside an
// omp target region. This mirrors exactly how a real OpenMP-offload stencil
// (minimod) would hand its omp-mapped field to GICC for halo exchange: the
// field lives in OpenMP-managed device memory, and GICC must be able to
// register + RMA into it. Cross-node (proxy/NIC) per srun layout.
#include <omp.h>
#include <cstdio>
#include <cstddef>
#include <cstdlib>

#include "gicc/platform/ofi/gicc_omp_device.hpp"   // gicc::omp::put + DeviceCtx
#include "examples/omp/gicc_omp_bridge.hpp"

int main() {
    const size_t n = 65536;                       // bytes in the "field"
    unsigned char* v = (unsigned char*)malloc(n);

    gicc_omp_bridge::init_runtime_only();
    int my = gicc_omp_bridge::rank();
    int nr = gicc_omp_bridge::nranks();
    if (nr != 2) { if (!my) fprintf(stderr, "need 2 ranks\n"); gicc_omp_bridge::finalize(); return 1; }
    int peer = my ^ 1;

    for (size_t i = 0; i < n; ++i) v[i] = (my == 0) ? 0xAB : 0x00;

    // OpenMP target-map the field (allocate device memory + copy host->device),
    // exactly like minimod's `#pragma omp target enter data map(to: v)`.
    #pragma omp target enter data map(to: v[0:n])

    // Register the OMP-managed device pointer with GICC for RMA.
    int bidx = -1;
    #pragma omp target data use_device_ptr(v)
    {
        bidx = gicc_omp_bridge::register_external(v, n);
    }
    gicc_omp_bridge::exchange_buffers();

    gicc::DeviceCtx* d_ctx = gicc_omp_bridge::prepare();

    if (my == 0) {
        #pragma omp target is_device_ptr(d_ctx)
        {
            gicc::omp::put(d_ctx, peer, bidx, /*dst_off=*/0, bidx, /*src_off=*/0, n);
        }
        gicc_omp_bridge::reset();
    } else {
        gicc_omp_bridge::reset();
    }
    gicc_omp_bridge::barrier();

    // Copy the OMP-mapped device field back to host and verify on the receiver.
    #pragma omp target exit data map(from: v[0:n])

    int rc = 0;
    if (my == 1) {
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) if (v[i] != 0xAB) ++bad;
        printf("omp-mapped put: bad_bytes=%zu/%zu : %s\n", bad, n, bad == 0 ? "PASS" : "FAIL");
        rc = (bad == 0) ? 0 : 2;
    }
    gicc_omp_bridge::barrier();
    free(v);
    gicc_omp_bridge::finalize();
    return rc;
}
