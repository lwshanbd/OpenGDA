// e1_single_put.cpp - OpenMP-target single-put end-to-end test.
// rank 0 issues one put to rank 1 from inside an omp target region; rank 1
// verifies the bytes landed. Intra-node (IPC) or cross-node (NIC) per srun layout.
#include <omp.h>
#include <cstdio>
#include <cstddef>

#include "gicc/omp.h"
#include "examples/omp/giomp_example_utils.hpp"

int main() {
    const size_t bytes = 65536;
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

    // Sender stamps 0xAB; receiver stays 0x00.
    giomp_example::fill_buffer(buffer, bytes, my == 0 ? 0xAB : 0x00);
    ompx_barrier();

    ompx_ctx* d_ctx = ompx_prepare();

    if (my == 0) {
        #pragma omp target is_device_ptr(d_ctx, buffer)
        {
            ompx_put(peer, /*dst=*/buffer, /*src=*/buffer, bytes);
        }
    }
    ompx_fence();   // drain proxy + completion, then rendezvous

    int rc = 0;
    if (my == 1) {
        size_t bad = giomp_example::count_mismatches(buffer, bytes, 0xAB);
        printf("recv: bad_bytes=%zu/%zu : %s\n", bad, bytes, bad == 0 ? "PASS" : "FAIL");
        rc = (bad == 0) ? 0 : 2;
    }
    ompx_barrier();
    ompx_free(buffer);
    ompx_finalize();
    return rc;
}
