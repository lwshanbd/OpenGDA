// e3_halo.cpp - OpenMP-target 1-D ring halo exchange correctness test.
// Each rank sends its edge faces to left/right neighbors via ompx_put_dev
// from inside an omp target region, then verifies the received faces.
#include <omp.h>
#include <cstdio>
#include <cstddef>

#include "gicc/omp.h"
#include "examples/omp/giomp_example_utils.hpp"

int main() {
    const size_t face = 4096;
    const size_t nbuf = 4 * face;
    const size_t off_lsend = 0, off_rsend = face, off_lrecv = 2 * face, off_rrecv = 3 * face;

    ompx_init();
    unsigned char* buffer = static_cast<unsigned char*>(ompx_alloc(nbuf));
    int my = ompx_get_rank_num();
    int nr = ompx_get_num_ranks();
    if (nr < 2) {
        if (my == 0) fprintf(stderr, "need >=2 ranks\n");
        ompx_free(buffer);
        ompx_finalize();
        return 1;
    }
    int left  = (my - 1 + nr) % nr;
    int right = (my + 1) % nr;

    // Stamp send faces; zero recv faces.
    unsigned char lsv = (unsigned char)(0x10 + my);
    unsigned char rsv = (unsigned char)(0x80 + my);
    giomp_example::fill_region(buffer, off_lsend, lsv,  face);
    giomp_example::fill_region(buffer, off_rsend, rsv,  face);
    giomp_example::fill_region(buffer, off_lrecv, 0x00, face);
    giomp_example::fill_region(buffer, off_rrecv, 0x00, face);
    ompx_barrier();

    ompx_ctx* d_ctx = ompx_prepare();
    #pragma omp target is_device_ptr(d_ctx, buffer)
    {
        // my right_send -> right neighbor's left_recv  (lane 0)
        ompx_put_dev(d_ctx, right, buffer + off_lrecv, buffer + off_rsend, face, /*lane=*/0);
        // my left_send  -> left neighbor's right_recv  (lane 1)
        ompx_put_dev(d_ctx, left,  buffer + off_rrecv, buffer + off_lsend, face, /*lane=*/1);
        ompx_quiet_dev(d_ctx, 0);
        ompx_quiet_dev(d_ctx, 1);
    }
    ompx_fence();

    unsigned char exp_lrecv = (unsigned char)(0x80 + left);
    unsigned char exp_rrecv = (unsigned char)(0x10 + right);
    size_t bad_l = giomp_example::count_region_mismatches(buffer, off_lrecv, exp_lrecv, face);
    size_t bad_r = giomp_example::count_region_mismatches(buffer, off_rrecv, exp_rrecv, face);
    int ok = (bad_l == 0 && bad_r == 0);
    printf("rank %d halo: lrecv_bad=%zu rrecv_bad=%zu : %s\n",
           my, bad_l, bad_r, ok ? "PASS" : "FAIL");

    ompx_barrier();
    ompx_free(buffer);
    ompx_finalize();
    return ok ? 0 : 2;
}
