// halo_omp_bench.cpp - GICC-from-OpenMP halo exchange bandwidth/latency sweep.
// Each rank exchanges `size`-byte faces with its left/right ring neighbors via
// ompx_put issued from an omp target region. Times the full halo round
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
    void* buffer = ompx_alloc(nbuf);
    int my = ompx_get_rank_num();
    int nr = ompx_get_num_ranks();
    if (nr < 2) { if (!my) fprintf(stderr, "need >=2 ranks\n"); ompx_free(buffer); ompx_finalize(); return 1; }
    int left  = (my - 1 + nr) % nr;
    int right = (my + 1) % nr;

    // The heap is symmetric, so these local face addresses also name the
    // neighbor's faces on the receiving side.
    char* base  = (char*)buffer;
    char* lsend = base;
    char* rsend = base + maxface;
    char* lrecv = base + 2 * maxface;
    char* rrecv = base + 3 * maxface;

    if (my == 0) {
        printf("# GICC-from-OpenMP halo: ranks=%d iters=%d\n", nr, kIters);
        printf("# size_bytes,us_per_halo,GBps\n");
    }

    for (int s = 0; s < nsz; ++s) {
        size_t sz = sizes[s];
        ompx_ctx* d_ctx = ompx_prepare();

        for (int it = 0; it < kWarm; ++it) {
            #pragma omp target is_device_ptr(d_ctx, lsend, rsend, lrecv, rrecv)
            {
                ompx_put(right, lrecv, rsend, sz, 0);
                ompx_put(left,  rrecv, lsend, sz, 1);
                ompx_quiet(0);
                ompx_quiet(1);
            }
            ompx_fence();
        }

        ompx_barrier();
        double t0 = omp_get_wtime();
        for (int it = 0; it < kIters; ++it) {
            #pragma omp target is_device_ptr(d_ctx, lsend, rsend, lrecv, rrecv)
            {
                ompx_put(right, lrecv, rsend, sz, 0);
                ompx_put(left,  rrecv, lsend, sz, 1);
                ompx_quiet(0);
                ompx_quiet(1);
            }
            ompx_fence();
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
