// hello_giomp.cpp - uses ONLY the public gicc/omp.h surface. If this compiles
// and links via find_package(gicc-omp), the packaging chain works end to end.
#include "gicc/omp.h"
#include <cstdio>

int main() {
    ompx_init();
    int me = ompx_get_rank_num();
    int np = ompx_get_num_ranks();
    const size_t bytes = 1024 * sizeof(float);
    void* b = ompx_alloc(bytes);
    const int peer = (me + 1) % np;
    // Smart host-side put: picks IPC (same-node) or proxy (cross-node) itself;
    // issues the omp target region internally, so no #pragma omp target here.
    ompx_put(peer, /*dst=*/b, /*src=*/b, bytes);
    ompx_fence();
    if (me == 0) printf("hello_giomp: %d ranks, ok\n", np);
    ompx_free(b);
    ompx_finalize();
    return 0;
}
