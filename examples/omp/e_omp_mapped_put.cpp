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

#include "gicc/omp.h"

int main() {
    const size_t n = 65536;                       // bytes in the "field"
    unsigned char* v = (unsigned char*)malloc(n);

    ompx_init();
    int my = omp_get_rank_num();
    int nr = omp_get_num_ranks();
    if (nr != 2) { if (!my) fprintf(stderr, "need 2 ranks\n"); ompx_finalize(); return 1; }
    int peer = my ^ 1;

    for (size_t i = 0; i < n; ++i) v[i] = (my == 0) ? 0xAB : 0x00;

    // OpenMP target-map the field (allocate device memory + copy host->device),
    // exactly like minimod's `#pragma omp target enter data map(to: v)`.
    #pragma omp target enter data map(to: v[0:n])

    // Register the OMP-managed device pointer with GICC for RMA.
    int bidx = -1;
    #pragma omp target data use_device_ptr(v)
    {
        bidx = ompx_register(v, n);
    }
    ompx_exchange();

    gicc::DeviceCtx* d_ctx = ompx_prepare();

    if (my == 0) {
        #pragma omp target is_device_ptr(d_ctx)
        {
            ompx_put_proxy(d_ctx, peer, bidx, /*dst_off=*/0, bidx, /*src_off=*/0, n);
        }
        ompx_quiet_host();
    } else {
        ompx_quiet_host();
    }
    ompx_barrier();

    // Copy the OMP-mapped device field back to host and verify on the receiver.
    #pragma omp target exit data map(from: v[0:n])

    int rc = 0;
    if (my == 1) {
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) if (v[i] != 0xAB) ++bad;
        printf("omp-mapped put: bad_bytes=%zu/%zu : %s\n", bad, n, bad == 0 ? "PASS" : "FAIL");
        rc = (bad == 0) ? 0 : 2;
    }
    ompx_barrier();
    free(v);
    ompx_finalize();
    return rc;
}
