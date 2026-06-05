// halo_dwq.cpp - OpenMP-target ring halo over the DWQ transport (Phase-2 M5).
// DWQ is cross-node only, so run with every neighbor on another node
// (ring of 2: -N2 -n2 --ntasks-per-node=1). Correctness verified against the
// Phase-1 proxy halo (e3_halo) math; also sweeps message size for a DWQ-vs-proxy
// perf datapoint.
#include <omp.h>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include "examples/omp/gicc_omp_dwq.hpp"
#include "examples/omp/gicc_omp_bridge.hpp"

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

    gicc_omp_bridge::init(nbuf);
    int my = gicc_omp_bridge::rank();
    int nr = gicc_omp_bridge::nranks();
    if (nr < 2) {
        if (!my) fprintf(stderr, "need >=2 ranks\n");
        gicc_omp_bridge::finalize();
        return 1;
    }
    int left  = (my - 1 + nr) % nr;
    int right = (my + 1) % nr;
    int bidx  = gicc_omp_bridge::buf_index();

    const size_t off_ls = 0, off_rs = maxface, off_lr = 2 * maxface, off_rr = 3 * maxface;

    // --- correctness at one size (face = 4096) ---
    {
        const size_t face = 4096;
        unsigned char lsv = (unsigned char)(0x10 + my);
        unsigned char rsv = (unsigned char)(0x80 + my);
        gicc_omp_bridge::fill_region(off_ls, lsv,  face);
        gicc_omp_bridge::fill_region(off_rs, rsv,  face);
        gicc_omp_bridge::fill_region(off_lr, 0x00, face);
        gicc_omp_bridge::fill_region(off_rr, 0x00, face);

        gicc::DeviceCtx* d = gicc_omp_bridge::prepare();
        #pragma omp target is_device_ptr(d)
        {
            gicc::omp_dwq::put(d, right, bidx, off_lr, bidx, off_rs, face);
            gicc::omp_dwq::put(d, left,  bidx, off_rr, bidx, off_ls, face);
            gicc::omp_dwq::flush(d);
        }
        gicc_omp_bridge::reset();
        gicc_omp_bridge::barrier();

        unsigned char el = (unsigned char)(0x80 + left);
        unsigned char er = (unsigned char)(0x10 + right);
        size_t bl = gicc_omp_bridge::count_region_mismatches(off_lr, el, face);
        size_t br = gicc_omp_bridge::count_region_mismatches(off_rr, er, face);
        printf("rank %d dwq-halo: lrecv_bad=%zu rrecv_bad=%zu : %s\n",
               my, bl, br, (bl == 0 && br == 0) ? "PASS" : "FAIL");
        fflush(stdout);
        gicc_omp_bridge::barrier();
    }

    // --- perf sweep ---
    if (my == 0)
        printf("# DWQ-from-OpenMP halo: ranks=%d iters=%d\n# size_bytes,us_per_halo,GBps\n",
               nr, kIters);

    for (int s = 0; s < nsz; ++s) {
        size_t sz = sizes[s];
        gicc::DeviceCtx* d = gicc_omp_bridge::prepare();

        // Warmup
        for (int it = 0; it < kWarm; ++it) {
            #pragma omp target is_device_ptr(d)
            {
                gicc::omp_dwq::put(d, right, bidx, off_lr, bidx, off_rs, sz);
                gicc::omp_dwq::put(d, left,  bidx, off_rr, bidx, off_ls, sz);
                gicc::omp_dwq::flush(d);
            }
            gicc_omp_bridge::reset();
            gicc_omp_bridge::barrier();
        }

        gicc_omp_bridge::barrier();
        double t0 = gicc_omp_bridge::wtime();
        for (int it = 0; it < kIters; ++it) {
            #pragma omp target is_device_ptr(d)
            {
                gicc::omp_dwq::put(d, right, bidx, off_lr, bidx, off_rs, sz);
                gicc::omp_dwq::put(d, left,  bidx, off_rr, bidx, off_ls, sz);
                gicc::omp_dwq::flush(d);
            }
            gicc_omp_bridge::reset();
            gicc_omp_bridge::barrier();
        }
        double t1 = gicc_omp_bridge::wtime();

        if (my == 0) {
            double us   = (t1 - t0) * 1e6 / kIters;
            double gbps = (2.0 * sz) / (us * 1e3);
            printf("%zu,%.3f,%.2f\n", sz, us, gbps);
            fflush(stdout);
        }
        gicc_omp_bridge::barrier();
    }

    gicc_omp_bridge::finalize();
    return 0;
}
