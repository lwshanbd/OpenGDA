// putbw: the halo exchange of himeno alone. Every rank puts BYTES to each
// neighbour in a line of ranks and fences, ITERS times; prints the mean
// time per exchange, split into the quiet and the barrier.
//
// Usage: putbw [BYTES] [ITERS] [PAD] [TOUCH] [PLANES] [SRCSHIFT] [DSTSHIFT]
//   SRCSHIFT, DSTSHIFT  bytes added to every source / destination address
//   PAD    bytes of heap allocated first
//   TOUCH  1: before each exchange a kernel writes PAD floats' worth of the
//          heap and then the two source blocks, as himeno's copy does
//   PLANES >0: himeno's layout instead: an array of PLANES blocks of BYTES,
//          block 1 into the left neighbour's last block and block PLANES-2
//          into the right neighbour's block 0
#include "gicc/omp.h"

#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    ompx_init();
    const int rank = ompx_get_rank_num(), n = ompx_get_num_ranks();
    const size_t bytes = argc > 1 ? strtoull(argv[1], nullptr, 0) : (1 << 20);
    const int iters = argc > 2 ? atoi(argv[2]) : 1000;
    const size_t pad = argc > 3 ? strtoull(argv[3], nullptr, 0) : 0;
    const int touch = argc > 4 ? atoi(argv[4]) : 0;
    float* scratch = pad ? static_cast<float*>(ompx_alloc(pad)) : nullptr;
    const int planes = argc > 5 ? atoi(argv[5]) : 0;
    char* buf = static_cast<char*>(ompx_alloc((planes > 0 ? planes : 4) * bytes + 4096));
    char *lsrc = buf, *ldst = buf + 2 * bytes, *rsrc = buf + bytes, *rdst = buf + 3 * bytes;
    const long ss = argc > 6 ? atol(argv[6]) : 0, ds = argc > 7 ? atol(argv[7]) : 0;
    if (planes > 0) {
        lsrc = buf + bytes;
        ldst = buf + (planes - 1) * bytes;
        rsrc = buf + (planes - 2) * bytes;
        rdst = buf;
    }
    lsrc += ss, rsrc += ss, ldst += ds, rdst += ds;
    const int left = rank > 0 ? rank - 1 : -1, right = rank < n - 1 ? rank + 1 : -1;
    double tq = 0, tb = 0;
    for (int it = -10; it < iters; ++it) {
        if (touch && scratch) {
            const long nf = (long)(pad / sizeof(float)), nb = (long)(2 * bytes / sizeof(float));
            float* src = reinterpret_cast<float*>(buf);
            #pragma omp target teams distribute parallel for is_device_ptr(scratch, src)
            for (long i = 0; i < nf + nb; ++i) {
                if (i < nf) scratch[i] = (float)(i + it);
                else src[i - nf] = (float)(i + it);
            }
        }
        const double t0 = MPI_Wtime();
        if (left >= 0) ompx_put(left, ldst, lsrc, bytes);
        if (right >= 0) ompx_put(right, rdst, rsrc, bytes);
        ompx_quiet();
        const double t1 = MPI_Wtime();
        ompx_barrier();
        const double t2 = MPI_Wtime();
        if (it >= 0) tq += t1 - t0, tb += t2 - t1;
    }
    double l[2] = {tq, tb}, m[2];
    MPI_Reduce(l, m, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    for (int r = 0; r < n; ++r) {
        if (r == rank)
            printf("rank %d: %zu B: quiet %.1f us, barrier %.1f us per exchange\n", rank, bytes,
                   tq / iters * 1e6, tb / iters * 1e6);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    ompx_finalize();
    return 0;
}
