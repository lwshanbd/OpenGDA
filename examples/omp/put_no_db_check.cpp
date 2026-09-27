// put_no_db_check - does ompx_put_no_db deliver what an ompx_put issued at the
// quiet would?
//
// Ranks form a ring. Every step each rank posts its source `a` to the right
// neighbour's `h` and then writes `a` in the ways a program might: a
// collapse(2) loop that leaves padding columns alone, a later kernel that
// overwrites part of it, a single-thread update, an atomic, a byte store,
// or writes made before the post. Right before the fence the sender copies
// `a` to `snap`; after it the receiver fetches the sender's `snap` and
// compares it with `h` word for word.
//
// Built with the gicc-passes plugin (GICC_MODE=chunk-lower), stores into
// `a` are mirrored into the neighbour as they happen and the fence sends
// the rest; built without it, the fence sends everything. Both must pass.
// Runs past 255 steps so the epoch counter wraps.
//
// Usage: put_no_db_check [steps]
#include "gicc/omp.h"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kRows = 256, kCols = 1000, kLd = 1024;   // 24 padding columns
constexpr size_t kWords = (size_t)kRows * kLd;

}  // namespace

#pragma omp declare target
static inline unsigned dvalue(int step, int rank, size_t i, unsigned salt) {
    return (unsigned)(step * 2654435761u) ^ (unsigned)(rank << 24) ^ (unsigned)i ^ salt;
}
#pragma omp end declare target

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 300;
    ompx_init();
    const int rank = ompx_get_rank_num(), n = ompx_get_num_ranks();
    const int right = (rank + 1) % n, left = (rank + n - 1) % n;

    unsigned* a    = static_cast<unsigned*>(ompx_alloc(kWords * 4));
    unsigned* h    = static_cast<unsigned*>(ompx_alloc(kWords * 4));
    unsigned* snap = static_cast<unsigned*>(ompx_alloc(kWords * 4));
    unsigned* chk  = static_cast<unsigned*>(ompx_alloc(kWords * 4));

    // Padding columns get a value once, here, and are never written again:
    // the fence has to send them every step.
    #pragma omp target teams distribute parallel for is_device_ptr(a)
    for (size_t i = 0; i < kWords; ++i) a[i] = dvalue(0, rank, i, 0xabcd);
    ompx_fence();

    long bad_total = 0;
    int bad_steps = 0;
    for (int s = 1; s <= steps; ++s) {
        const int kind = s % 6;

        if (kind == 5) {   // written before the post, so never mirrored
            #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a)
            for (int r = 0; r < kRows; ++r)
                for (int c = 0; c < kCols; ++c)
                    a[(size_t)r * kLd + c] = dvalue(s, rank, (size_t)r * kLd + c, 5);
        }

        ompx_put_no_db(right, h, a, kWords * 4);

        if (kind != 5) {
            #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a)
            for (int r = 0; r < kRows; ++r)
                for (int c = 0; c < kCols; ++c)
                    a[(size_t)r * kLd + c] = dvalue(s, rank, (size_t)r * kLd + c, 1);
        }
        if (kind == 1) {   // a second writer overwrites half of it
            #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a)
            for (int r = 0; r < kRows / 2; ++r)
                for (int c = 0; c < kCols; c += 3)
                    a[(size_t)r * kLd + c] = dvalue(s, rank, (size_t)r * kLd + c, 2);
        }
        if (kind == 2) {   // a single-thread update, like adding a source term
            #pragma omp target is_device_ptr(a)
            { a[kLd + 5] += 7u; }
        }
        if (kind == 3) {   // an atomic: not mirrored, so the fence sends everything
            #pragma omp target teams distribute parallel for is_device_ptr(a)
            for (int i = 0; i < 64; ++i) {
                #pragma omp atomic update
                a[7] += 1u;
            }
        }
        if (kind == 4) {   // a byte store into a word the loop above marked
            #pragma omp target is_device_ptr(a)
            { reinterpret_cast<unsigned char*>(a)[4 * 9 + 1] = (unsigned char)s; }
        }

        #pragma omp target teams distribute parallel for is_device_ptr(a, snap)
        for (size_t i = 0; i < kWords; ++i) snap[i] = a[i];
        ompx_fence();

        // h now holds what `left` posted; compare it with left's snapshot.
        ompx_get(left, chk, snap, kWords * 4);
        ompx_quiet();
        unsigned bad = 0;
        #pragma omp target teams distribute parallel for reduction(+ : bad) is_device_ptr(h, chk)
        for (size_t i = 0; i < kWords; ++i) bad += h[i] != chk[i];
        if (bad != 0) {
            ++bad_steps;
            bad_total += bad;
            if (bad_steps <= 5)
                std::printf("rank %d step %d kind %d: %u words differ\n", rank, s, kind, bad);
        }
        ompx_barrier();   // left may overwrite snap only after this
    }

    long all = 0;
    MPI_Allreduce(&bad_total, &all, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("put_no_db_check: %d ranks, %d steps, %ld words wrong: %s\n",
                    n, steps, all, all == 0 ? "PASS" : "FAIL");
    ompx_finalize();
    return all == 0 ? 0 : 1;
}
