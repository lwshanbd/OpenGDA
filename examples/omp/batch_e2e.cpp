// batch_e2e.cpp - does aggregation favor the compiler-lowered trigger?
// ONE target region issues up to 64 independent puts to the peer as guarded
// straight-line sites (if (i < n) dput...), then one flush. ONE source, TWO
// builds (as jacobi_e2e):
//   proxy build (no pass):  n in-kernel ring pushes, drained by the CPU
//     worker one operation at a time.
//   dwq build (LTO pass):   the pass erases the dput sites, the synthesized
//     host trace evaluates each guard as a host-side branch and stages the n
//     live descriptors before launch, and the single in-region flush
//     releases them all with one MMIO trigger.
// Guards and constant*formal offsets exercise the guard row of the lowering
// map. (The canonical-loop form records a degraded IV in the template and
// does not stage correctly in the current omp prototype, so this benchmark
// uses guarded sites.)
// Cross-node only: run 2 ranks on 2 nodes, same env recipes as jacobi_e2e.
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "gicc/omp.h"

static int env_int(const char* k, int d) {
    if (const char* v = std::getenv(k)) { int x = atoi(v); if (x > 0) return x; }
    return d;
}

// 64 guarded straight-line dput sites; the constant I keeps every operand a
// constant*formal expression the trace can evaluate on the host.
#define DPUT(I)                                                     \
    if ((I) < n)                                                    \
        ompx_dput(ctx, peer, bidx, roff + (size_t)(I) * msg,        \
                        bidx, (size_t)(I) * msg, msg);
#define DPUT8(B) DPUT(B) DPUT(B+1) DPUT(B+2) DPUT(B+3) \
                 DPUT(B+4) DPUT(B+5) DPUT(B+6) DPUT(B+7)
#define DPUT64 DPUT8(0) DPUT8(8) DPUT8(16) DPUT8(24) \
               DPUT8(32) DPUT8(40) DPUT8(48) DPUT8(56)

int main() {
    const int ITERS = env_int("BAT_ITERS", 200);
    const int WARM  = env_int("BAT_WARM", 20);
    const size_t MSG = (size_t)env_int("BAT_MSG", 256);
    const int NMAX = 64;

    ompx_init();
    int my = omp_get_rank_num(), nr = omp_get_num_ranks();
    if (nr != 2) { if (!my) fprintf(stderr, "need 2 ranks\n"); ompx_finalize(); return 1; }
    int peer = my ^ 1;

    // Segments [0, NMAX*MSG) are send slots, [NMAX*MSG, 2*NMAX*MSG) receive.
    ompx_buffer b = ompx_alloc(2 * (size_t)NMAX * MSG);
    ompx_exchange();
    int bidx = b.index;
    unsigned char* p = (unsigned char*)b.ptr;
    size_t roff = (size_t)NMAX * MSG;
    gicc::DeviceCtx* ctx = ompx_prepare();

    if (my == 0)
        printf("# batch_e2e ranks=%d msg=%zu iters=%d\n# nmsg,us_per_exchange,verify\n",
               nr, MSG, ITERS);

    for (int nmsg = 1; nmsg <= NMAX; nmsg *= 2) {
        // Sender stamps segment i with (0x20+i); receiver slots start zeroed.
        unsigned char base = (unsigned char)(0x20 + my);
        size_t total = (size_t)NMAX * MSG;
        #pragma omp target teams distribute parallel for is_device_ptr(p) \
            firstprivate(total, base, MSG)
        for (size_t k = 0; k < 2 * total; ++k)
            p[k] = (k < total) ? (unsigned char)(base + k / MSG) : 0;
        ompx_barrier();

        // Correctness pass: one exchange, then verify each received segment.
        int n = nmsg; size_t msg = MSG;
        #pragma omp target is_device_ptr(ctx) firstprivate(peer, bidx, n, msg, roff)
        {
            DPUT64
            ompx_flush(ctx);
        }
        ompx_quiet_host();
        ompx_barrier();

        long bad = 0;
        unsigned char pbase = (unsigned char)(0x20 + peer);
        size_t check = (size_t)nmsg * MSG;
        #pragma omp target teams distribute parallel for reduction(+:bad) \
            is_device_ptr(p) firstprivate(check, roff, pbase, MSG)
        for (size_t k = 0; k < check; ++k)
            if (p[roff + k] != (unsigned char)(pbase + k / MSG)) bad++;

        // Timed exchanges.
        ompx_barrier();
        double t0 = omp_get_wtime();
        for (int it = 0; it < WARM + ITERS; ++it) {
            if (it == WARM) { ompx_barrier(); t0 = omp_get_wtime(); }
            #pragma omp target is_device_ptr(ctx) firstprivate(peer, bidx, n, msg, roff)
            {
                DPUT64
                ompx_flush(ctx);
            }
            ompx_quiet_host();
            ompx_barrier();
        }
        double t1 = omp_get_wtime();

        if (my == 0) {
            printf("%d,%.3f,%s\n", nmsg, (t1 - t0) * 1e6 / ITERS,
                   bad == 0 ? "PASS" : "FAIL");
            fflush(stdout);
        }
        ompx_barrier();
    }
    ompx_free(b);
    ompx_finalize();
    return 0;
}
