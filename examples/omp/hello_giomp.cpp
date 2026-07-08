// hello_giomp.cpp - uses ONLY the public gicc/omp.h surface. If this compiles
// and links via find_package(gicc-omp), the packaging chain works end to end.
#include "gicc/omp.h"
#include <cstdio>

int main() {
    ompx_init();
    int me = omp_get_rank_num();
    int np = omp_get_num_ranks();
    ompx_buffer b = ompx_alloc(1024 * sizeof(float));
    ompx_exchange();
    gicc::DeviceCtx* c = ompx_prepare();
    const int peer = (me + 1) % np;
    // Smart host-side put: picks IPC (same-node) or proxy (cross-node) itself;
    // issues the omp target region internally, so no #pragma omp target here.
    ompx_put(c, peer, b.index, 0, b.index, 0, 1024 * sizeof(float));
    ompx_quiet_host();
    ompx_barrier();
    if (me == 0) printf("hello_giomp: %d ranks, ok\n", np);
    ompx_free(b);
    ompx_finalize();
    return 0;
}
