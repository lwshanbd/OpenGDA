// pipeline_after_bench - what moving the puts after a kernel into it costs a
// short kernel.
//
// A CloverLeaf-shaped step: int loop variables, collapse(2) over the interior
// of padded 2-D fields (x contiguous, two ghost cells a side), two fields read
// and two written. After it the host puts rows of the written fields to both
// ring neighbours, interior columns only, one put per row:
//
//   two    1 field  x 2 neighbours x 1 row = 2 puts
//   eight  2 fields x 2 neighbours x 2 rows = 8 puts
//
// Built plain, with the gicc-passes plugin but nothing discovered (the
// control: the same kernel code), or in the plugin's two passes
// (build_giomp_after.sh), where each kernel sends its puts itself. Only the
// kernel and its puts are timed; the fence that ends each step is not.
//
// Every value is an integer, so the rows each rank receives in the last step
// are checked word for word.
//
// Usage: pipeline_after_bench [x_max y_max iters]
#include "gicc/omp.h"
#include "gicc/omp_pipeline.h"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct Grid {
    int x_max, y_max;
    size_t sx, n;        // row length, words per field
};

// The receive slot of (field, side, row): side 0 holds the neighbour below's
// top rows, side 1 the neighbour above's bottom rows. A slot keeps the row
// layout, so a row lands at the same columns.
// Scalar arguments only: the plugin takes a put's arguments from values the
// kernel cannot change, never from memory.
double* slot(double* rcv, size_t sx, int f, int side, int r) {
    return rcv + ((size_t)((f * 2 + side) * 2 + r)) * sx;
}

// The inputs d (which 0) and e (which 1), and p after step `it`.
double input(int rank, int which, size_t k) { return (double)rank * 1e7 + which * 1e6 + (double)k; }
double first(int it, int rank, size_t k) { return input(rank, 0, k) + input(rank, 1, k) + it; }

}  // namespace

// p = d + e + it, q = p - e: integer-valued, so exact.
static void step_two(double* p, double* q, const double* d, const double* e, int x_min, int x_max,
                     int y_min, int y_max, size_t sx, int it, int up, int down, double* rcv) {
    #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(p, q, d, e)
    for (int j = y_min + 1; j < y_max + 2; j++)
        for (int i = x_min + 1; i < x_max + 2; i++) {
            const size_t k = i + j * sx;
            p[k] = d[k] + e[k] + it;
            q[k] = p[k] - e[k];
        }
    const size_t row = (size_t)x_max * sizeof(double);
    ompx_put(up, slot(rcv, sx, 0, 0, 0) + 2, p + (size_t)(y_max + 1) * sx + 2, row);
    ompx_put(down, slot(rcv, sx, 0, 1, 0) + 2, p + (size_t)2 * sx + 2, row);
}

static void step_eight(double* p, double* q, const double* d, const double* e, int x_min,
                       int x_max, int y_min, int y_max, size_t sx, int it, int up, int down,
                       double* rcv) {
    #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(p, q, d, e)
    for (int j = y_min + 1; j < y_max + 2; j++)
        for (int i = x_min + 1; i < x_max + 2; i++) {
            const size_t k = i + j * sx;
            p[k] = d[k] + e[k] + it;
            q[k] = p[k] - e[k];
        }
    const size_t row = (size_t)x_max * sizeof(double);
    ompx_put(up, slot(rcv, sx, 0, 0, 0) + 2, p + (size_t)y_max * sx + 2, row);
    ompx_put(up, slot(rcv, sx, 0, 0, 1) + 2, p + (size_t)(y_max + 1) * sx + 2, row);
    ompx_put(up, slot(rcv, sx, 1, 0, 0) + 2, q + (size_t)y_max * sx + 2, row);
    ompx_put(up, slot(rcv, sx, 1, 0, 1) + 2, q + (size_t)(y_max + 1) * sx + 2, row);
    ompx_put(down, slot(rcv, sx, 0, 1, 0) + 2, p + (size_t)2 * sx + 2, row);
    ompx_put(down, slot(rcv, sx, 0, 1, 1) + 2, p + (size_t)3 * sx + 2, row);
    ompx_put(down, slot(rcv, sx, 1, 1, 0) + 2, q + (size_t)2 * sx + 2, row);
    ompx_put(down, slot(rcv, sx, 1, 1, 1) + 2, q + (size_t)3 * sx + 2, row);
}

// Words of the receive slots that differ from what the neighbours sent in
// step `it`; `rows` rows of `fields` fields were put.
static long check(double* rcv, const Grid& g, int it, int up, int down, int fields, int rows) {
    const size_t words = 8 * g.sx;
    std::vector<double> h(words);
    omp_target_memcpy(h.data(), rcv, words * sizeof(double), 0, 0, omp_get_initial_device(),
                      omp_get_default_device());
    long bad = 0;
    for (int f = 0; f < fields; ++f)
        for (int side = 0; side < 2; ++side)
            for (int r = 0; r < rows; ++r) {
                const int from = side == 0 ? down : up;
                const int j = side == 0 ? g.y_max + 2 - rows + r : 2 + r;
                const double* got = slot(h.data(), g.sx, f, side, r);
                for (int i = 2; i < g.x_max + 2; ++i) {
                    const size_t k = i + j * g.sx;
                    const double p = first(it, from, k);
                    const double want = f == 0 ? p : p - input(from, 1, k);
                    bad += got[i] != want;
                }
            }
    return bad;
}

int main(int argc, char** argv) {
    Grid g;
    g.x_max = argc > 1 ? std::atoi(argv[1]) : 960;
    g.y_max = argc > 2 ? std::atoi(argv[2]) : 1920;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 200;
    g.sx = (size_t)g.x_max + 4;
    g.n = g.sx * ((size_t)g.y_max + 4);
    ompx_init();
    const int rank = ompx_get_rank_num(), n = ompx_get_num_ranks();
    const int up = (rank + 1) % n, down = (rank + n - 1) % n;

    double* d = (double*)ompx_alloc(g.n * sizeof(double));
    double* e = (double*)ompx_alloc(g.n * sizeof(double));
    double* p = (double*)ompx_alloc(g.n * sizeof(double));
    double* q = (double*)ompx_alloc(g.n * sizeof(double));
    double* rcv2 = (double*)ompx_alloc(8 * g.sx * sizeof(double));
    double* rcv8 = (double*)ompx_alloc(8 * g.sx * sizeof(double));
    const size_t words = g.n;
    #pragma omp target teams distribute parallel for is_device_ptr(d, e, p, q) firstprivate(rank)
    for (size_t k = 0; k < words; ++k) {
        d[k] = (double)rank * 1e7 + (double)k;             // input(rank, 0, k)
        e[k] = (double)rank * 1e7 + 1e6 + (double)k;       // input(rank, 1, k)
        p[k] = q[k] = 0;
    }
    ompx_prepare();
    ompx_fence();

    const int x_min = 1, y_min = 1;
    double t2 = 0, t8 = 0;
    for (int it = -5; it < iters; ++it) {   // five untimed warm-up steps
        double t0 = omp_get_wtime();
        step_two(p, q, d, e, x_min, g.x_max, y_min, g.y_max, g.sx, it, up, down, rcv2);
        double t1 = omp_get_wtime();
        ompx_fence();
        double t3 = omp_get_wtime();
        step_eight(p, q, d, e, x_min, g.x_max, y_min, g.y_max, g.sx, it, up, down, rcv8);
        double t4 = omp_get_wtime();
        ompx_fence();
        if (it >= 0) {
            t2 += t1 - t0;
            t8 += t4 - t3;
        }
    }
    const int last = iters - 1;
    long bad = check(rcv2, g, last, up, down, 1, 1) + check(rcv8, g, last, up, down, 2, 2);
    double t[2] = {t2 / iters * 1e6, t8 / iters * 1e6}, tmax[2];
    long all = 0;
    MPI_Reduce(t, tmax, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bad, &all, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("pipeline_after_bench: %d ranks %dx%d, %d iters: two puts %.1f us, eight "
                    "puts %.1f us per step (max over ranks); %ld words wrong: %s\n",
                    n, g.x_max, g.y_max, iters, tmax[0], tmax[1], all, all == 0 ? "PASS" : "FAIL");
    ompx_finalize();
    return all == 0 ? 0 : 1;
}
