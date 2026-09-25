// e_omp_mapped_put.cpp - make-or-break for the minimod port.
//
// Binds an application field to the symmetric heap with ompx_bind and delivers
// one face through ompx_put from inside an omp target region. This mirrors
// exactly how a real OpenMP-offload stencil (minimod) hands its omp-mapped
// field to GICC for halo exchange: the field keeps its host pointer and its
// `map` clauses, and GICC can RMA into it. Cross-node (proxy/NIC) per srun
// layout.
#include <omp.h>
#include <cstdio>
#include <cstddef>
#include <cstdlib>

#include "gicc/omp.h"

int main() {
    const size_t n = 65536;                       // bytes in the "field"
    unsigned char* v = (unsigned char*)malloc(n);

    ompx_init();
    int my = ompx_get_rank_num();
    int nr = ompx_get_num_ranks();
    if (nr != 2) { if (!my) fprintf(stderr, "need 2 ranks\n"); ompx_finalize(); return 1; }
    int peer = my ^ 1;

    for (size_t i = 0; i < n; ++i) v[i] = (my == 0) ? 0xAB : 0x00;

    // Give the field a heap home: ompx_bind allocates from the symmetric heap,
    // associates it with the host pointer and copies the data in, so the
    // application's own `map` clauses keep working on the same memory.
    void* dv = ompx_bind(v, n);

    // OpenMP target-map the field, exactly like minimod's
    // `#pragma omp target enter data map(to: v)`; the binding above makes this
    // resolve to the heap allocation instead of a fresh device buffer.
    #pragma omp target enter data map(to: v[0:n])

    // The receiver's ompx_bind copies its initial field into the heap; a put
    // that arrives before that copy is overwritten by it. Wait for every rank.
    ompx_barrier();

    ompx_ctx* d_ctx = ompx_prepare();

    if (my == 0) {
        #pragma omp target is_device_ptr(d_ctx, dv)
        {
            ompx_put(peer, dv, dv, n);
        }
        ompx_quiet();
    } else {
        ompx_quiet();
    }
    ompx_barrier();

    // Copy the OMP-mapped device field back to host and verify on the receiver.
    #pragma omp target update from(v[0:n])

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
