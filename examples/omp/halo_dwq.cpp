// halo_dwq.cpp - OpenMP-target ring halo over the DWQ transport (Phase-2 M5).
// DWQ is cross-node only, so run with every neighbor on another node
// (ring of 2: -N2 -n2 --ntasks-per-node=1). Correctness verified against the
// Phase-1 proxy halo (e3_halo) math; also sweeps message size for a DWQ-vs-proxy
// perf datapoint.
#include <omp.h>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include "gicc/omp.h"
#include "examples/omp/giomp_example_utils.hpp"

static int env_int(const char* k, int d) {
    if (const char* v = std::getenv(k)) { int x = atoi(v); if (x > 0) return x; }
    return d;
}

int main() {
    const size_t sizes[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576};
    const int nsz = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const size_t maxface = sizes[nsz - 1];
    const size_t nbuf = 4 * maxface;
    const int kIters = env_int("GICC_HALO_ITERS", 50), kWarm = 5;

    ompx_init();
    ompx_buffer buffer = ompx_alloc(nbuf);
    ompx_exchange();
    int my = omp_get_rank_num();
    int nr = omp_get_num_ranks();
    if (nr < 2) {
        if (!my) fprintf(stderr, "need >=2 ranks\n");
        ompx_free(buffer);
        ompx_finalize();
        return 1;
    }
    int left  = (my - 1 + nr) % nr;
    int right = (my + 1) % nr;
    int bidx  = buffer.index;

    const size_t off_ls = 0, off_rs = maxface, off_lr = 2 * maxface, off_rr = 3 * maxface;

    // --- correctness at one size (face = 4096) ---
    {
        const size_t face = 4096;
        unsigned char lsv = (unsigned char)(0x10 + my);
        unsigned char rsv = (unsigned char)(0x80 + my);
        giomp_example::fill_region(buffer, off_ls, lsv,  face);
        giomp_example::fill_region(buffer, off_rs, rsv,  face);
        giomp_example::fill_region(buffer, off_lr, 0x00, face);
        giomp_example::fill_region(buffer, off_rr, 0x00, face);
        ompx_barrier();

        gicc::DeviceCtx* d = ompx_prepare();
        #pragma omp target is_device_ptr(d)
        {
            ompx_dwq_put(d, right, bidx, off_lr, bidx, off_rs, face);
            ompx_dwq_put(d, left,  bidx, off_rr, bidx, off_ls, face);
            ompx_dwq_flush(d);
        }
        ompx_quiet_host();
        ompx_barrier();

        unsigned char el = (unsigned char)(0x80 + left);
        unsigned char er = (unsigned char)(0x10 + right);
        size_t bl = giomp_example::count_region_mismatches(buffer, off_lr, el, face);
        size_t br = giomp_example::count_region_mismatches(buffer, off_rr, er, face);
        printf("rank %d dwq-halo: lrecv_bad=%zu rrecv_bad=%zu : %s\n",
               my, bl, br, (bl == 0 && br == 0) ? "PASS" : "FAIL");
        fflush(stdout);
        ompx_barrier();
    }

    // --- perf sweep ---
    if (my == 0)
        printf("# DWQ-from-OpenMP halo: ranks=%d iters=%d\n# size_bytes,us_per_halo,GBps\n",
               nr, kIters);

    for (int s = 0; s < nsz; ++s) {
        size_t sz = sizes[s];
        gicc::DeviceCtx* d = ompx_prepare();

        // Warmup
        for (int it = 0; it < kWarm; ++it) {
            #pragma omp target is_device_ptr(d)
            {
                ompx_dwq_put(d, right, bidx, off_lr, bidx, off_rs, sz);
                ompx_dwq_put(d, left,  bidx, off_rr, bidx, off_ls, sz);
                ompx_dwq_flush(d);
            }
            ompx_quiet_host();
            ompx_barrier();
        }

        ompx_barrier();
        double t0 = omp_get_wtime();
        for (int it = 0; it < kIters; ++it) {
            #pragma omp target is_device_ptr(d)
            {
                ompx_dwq_put(d, right, bidx, off_lr, bidx, off_rs, sz);
                ompx_dwq_put(d, left,  bidx, off_rr, bidx, off_ls, sz);
                ompx_dwq_flush(d);
            }
            ompx_quiet_host();
            ompx_barrier();
        }
        double t1 = omp_get_wtime();

        if (my == 0) {
            double us   = (t1 - t0) * 1e6 / kIters;
            double gbps = (2.0 * sz) / (us * 1e3);
            printf("%zu,%.3f,%.2f\n", sz, us, gbps);
            fflush(stdout);
        }
        ompx_barrier();
    }

    ompx_free(buffer);
    ompx_finalize();
    return 0;
}
