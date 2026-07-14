// halo_omp_bench.cpp - GICC-from-OpenMP halo exchange bandwidth/latency sweep.
// Each rank exchanges `size`-byte faces with its left/right ring neighbors via
// gicc::omp::put issued from an omp target region. Times the full halo round
// (region + host drain + barrier), sweeping message size. Run cross-node
// (-N2 -n2 1/node) for the NIC path or intra-node (-N1 -n2) for IPC.
#include <omp.h>
#include <cstdio>
#include <cstddef>
#include <cstdlib>

#include "gicc/omp.h"

static int env_int(const char* k, int d) {
    if (const char* v = std::getenv(k)) { int x = atoi(v); if (x > 0) return x; }
    return d;
}

int main() {
    const size_t sizes[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304};
    const int    nsz   = sizeof(sizes) / sizeof(sizes[0]);
    const size_t maxface = sizes[nsz - 1];
    const size_t nbuf  = 4 * maxface;
    const int    kIters = env_int("GICC_HALO_ITERS", 100);
    const int    kWarm  = 10;

    ompx_init();
    ompx_buffer buffer = ompx_alloc(nbuf);
    ompx_exchange();
    int my = omp_get_rank_num();
    int nr = omp_get_num_ranks();
    if (nr < 2) { if (!my) fprintf(stderr, "need >=2 ranks\n"); ompx_free(buffer); ompx_finalize(); return 1; }
    int left  = (my - 1 + nr) % nr;
    int right = (my + 1) % nr;
    int bidx  = buffer.index;

    const size_t off_lsend = 0, off_rsend = maxface, off_lrecv = 2 * maxface, off_rrecv = 3 * maxface;

    if (my == 0) {
        printf("# GICC-from-OpenMP halo: ranks=%d iters=%d\n", nr, kIters);
        printf("# size_bytes,us_per_halo,GBps\n");
    }

    for (int s = 0; s < nsz; ++s) {
        size_t sz = sizes[s];
        gicc::DeviceCtx* d_ctx = ompx_prepare();

        for (int it = 0; it < kWarm; ++it) {
            #pragma omp target is_device_ptr(d_ctx)
            {
                ompx_put_proxy(d_ctx, right, bidx, off_lrecv, bidx, off_rsend, sz, 0);
                ompx_put_proxy(d_ctx, left,  bidx, off_rrecv, bidx, off_lsend, sz, 1);
                ompx_quiet(d_ctx, 0);
                ompx_quiet(d_ctx, 1);
            }
            ompx_quiet_host();
            ompx_barrier();
        }

        ompx_barrier();
        double t0 = omp_get_wtime();
        for (int it = 0; it < kIters; ++it) {
            #pragma omp target is_device_ptr(d_ctx)
            {
                ompx_put_proxy(d_ctx, right, bidx, off_lrecv, bidx, off_rsend, sz, 0);
                ompx_put_proxy(d_ctx, left,  bidx, off_rrecv, bidx, off_lsend, sz, 1);
                ompx_quiet(d_ctx, 0);
                ompx_quiet(d_ctx, 1);
            }
            ompx_quiet_host();
            ompx_barrier();
        }
        double t1 = omp_get_wtime();

        if (my == 0) {
            double us = (t1 - t0) * 1e6 / kIters;          // per halo round
            double gbps = (2.0 * sz) / (us * 1e3);          // two faces sent per rank
            printf("%zu,%.3f,%.2f\n", sz, us, gbps);
            fflush(stdout);
        }
        ompx_barrier();
    }

    ompx_free(buffer);
    ompx_finalize();
    return 0;
}
