// e3_halo.cpp - OpenMP-target 1-D ring halo exchange correctness test.
// Each rank sends its edge faces to left/right neighbors via gicc::omp::put
// from inside an omp target region, then verifies the received faces.
#include <omp.h>
#include <cstdio>
#include <cstddef>

#include "gicc/platform/ofi/gicc_omp_device.hpp"
#include "examples/omp/gicc_omp_bridge.hpp"

int main() {
    const size_t face = 4096;
    const size_t nbuf = 4 * face;
    const size_t off_lsend = 0, off_rsend = face, off_lrecv = 2 * face, off_rrecv = 3 * face;

    gicc_omp_bridge::init(nbuf);
    int my = gicc_omp_bridge::rank();
    int nr = gicc_omp_bridge::nranks();
    if (nr < 2) {
        if (my == 0) fprintf(stderr, "need >=2 ranks\n");
        gicc_omp_bridge::finalize();
        return 1;
    }
    int left  = (my - 1 + nr) % nr;
    int right = (my + 1) % nr;
    int bidx  = gicc_omp_bridge::buf_index();

    // Stamp send faces; zero recv faces.
    unsigned char lsv = (unsigned char)(0x10 + my);
    unsigned char rsv = (unsigned char)(0x80 + my);
    gicc_omp_bridge::fill_region(off_lsend, lsv,  face);
    gicc_omp_bridge::fill_region(off_rsend, rsv,  face);
    gicc_omp_bridge::fill_region(off_lrecv, 0x00, face);
    gicc_omp_bridge::fill_region(off_rrecv, 0x00, face);

    gicc::DeviceCtx* d_ctx = gicc_omp_bridge::prepare();
    #pragma omp target is_device_ptr(d_ctx)
    {
        // my right_send -> right neighbor's left_recv  (lane 0)
        gicc::omp::put(d_ctx, right, bidx, off_lrecv, bidx, off_rsend, face, /*lane=*/0);
        // my left_send  -> left neighbor's right_recv  (lane 1)
        gicc::omp::put(d_ctx, left,  bidx, off_rrecv, bidx, off_lsend, face, /*lane=*/1);
        gicc::omp::quiet(d_ctx, 0);
        gicc::omp::quiet(d_ctx, 1);
    }
    gicc_omp_bridge::reset();
    gicc_omp_bridge::barrier();

    unsigned char exp_lrecv = (unsigned char)(0x80 + left);
    unsigned char exp_rrecv = (unsigned char)(0x10 + right);
    size_t bad_l = gicc_omp_bridge::count_region_mismatches(off_lrecv, exp_lrecv, face);
    size_t bad_r = gicc_omp_bridge::count_region_mismatches(off_rrecv, exp_rrecv, face);
    int ok = (bad_l == 0 && bad_r == 0);
    printf("rank %d halo: lrecv_bad=%zu rrecv_bad=%zu : %s\n",
           my, bad_l, bad_r, ok ? "PASS" : "FAIL");

    gicc_omp_bridge::barrier();
    gicc_omp_bridge::finalize();
    return ok ? 0 : 2;
}
