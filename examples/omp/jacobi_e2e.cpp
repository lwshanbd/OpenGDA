// jacobi_e2e.cpp - end-to-end compiler-lowered halo exchange inside a real
// Jacobi iteration. ONE source, ONE API (ompx_dwq_put_dev/ompx_dwq_flush_dev),
// TWO builds:
//   proxy build (no pass):  the marker bodies execute the Proxy enqueue and
//     the flush marker is a no-op.
//   dwq build (LTO pass, GICC_MODE=omp-dwq): the pass erases the put sites,
//     synthesizes the host trace that pre-stages the descriptors before each
//     launch, and lowers the flush to the lead-thread MMIO trigger.
// The source is IDENTICAL for both builds (no defines), so the per-rank
// field hash must match bit-exactly between transports at every scale.
//
// Memory is the GiOMP symmetric heap: ompx_alloc carves from one registered
// allocation, so the markers name every operand as (heap index, heap offset)
// -- the buffer-index form the LTO pass analyses, obtained on the host with
// ompx_heap_index() / ompx_heap_offset_of().
//
// Ring (periodic) 1-D decomposition, R x C interior per rank, ghost rows 0 and
// R+1, ping-pong buffers. DWQ is NIC-only: run 1 rank per node, >=2 nodes.
//   proxy: GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 HSA_XNACK=1
//   dwq:   GICC_HALO_DWQ=1 HSA_XNACK=1        (no GICC_SKIP_DWQ_INIT)
// Output: correctness hash, time/iter + comm time, and a launch-overhead
// microbench (halo region launched with zero-iteration guards).
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#define GIOMP_ENABLE_DWQ 1          // expose the compiler-facing DWQ markers
#include "gicc/omp.h"

static int env_int(const char* k, int d) {
    if (const char* v = std::getenv(k)) { int x = atoi(v); if (x > 0) return x; }
    return d;
}

int main(int argc, char** argv) {
    const int R = env_int("JAC_ROWS", 2048);
    const int C = env_int("JAC_COLS", 256);
    const int ITERS = env_int("JAC_ITERS", 200);
    const int WARM = env_int("JAC_WARM", 20);
    const int LAUNCH_N = env_int("JAC_LAUNCH_N", 2000);

    ompx_init();
    int my = ompx_get_rank_num();
    int nr = ompx_get_num_ranks();
    if (nr < 2) { if (!my) fprintf(stderr, "need >=2 ranks\n"); ompx_finalize(); return 1; }
    int up   = (my - 1 + nr) % nr;   // owns the rows above mine (ring)
    int down = (my + 1) % nr;

    const size_t row_bytes = (size_t)C * sizeof(float);
    const size_t buf_bytes = (size_t)(R + 2) * row_bytes;
    float* buf[2] = { (float*)ompx_alloc(buf_bytes), (float*)ompx_alloc(buf_bytes) };
    // One heap, one address-book entry; the two buffers differ only by offset.
    const int bidx = ompx_heap_index();
    const size_t hbase[2] = { ompx_heap_offset_of(buf[0]),
                              ompx_heap_offset_of(buf[1]) };

    // Deterministic init of interior rows on the device; ghosts start 0.
    for (int b = 0; b < 2; ++b) {
        float* p = buf[b];
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(p) firstprivate(R, C, my)
        for (int i = 0; i <= R + 1; ++i)
            for (int j = 0; j < C; ++j) {
                float v = 0.f;
                if (i >= 1 && i <= R) {
                    long g = (long)my * R + (i - 1);
                    v = (float)((g * 131 + j * 17) % 1024) * 0.001f;
                }
                p[(size_t)i * C + j] = v;
            }
    }
    ompx_barrier();

    const size_t rel_top_int = row_bytes;              // my row 1
    const size_t rel_bot_int = (size_t)R * row_bytes;  // my row R
    const size_t rel_top_gho = 0;                      // ghost row 0
    const size_t rel_bot_gho = (size_t)(R + 1) * row_bytes;

    double t_comm = 0.0, t_total = 0.0;
    int src = 0;
    ompx_ctx* ctx = ompx_prepare();

    for (int it = 0; it < WARM + ITERS; ++it) {
        if (it == WARM) { ompx_barrier(); t_total = omp_get_wtime(); }
        float* pa = buf[src];
        float* pb = buf[1 - src];
        const size_t hb = hbase[1 - src];   // we send rows of the buffer we WROTE

        // Real compute: 5-point Jacobi update of every interior row.
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(pa, pb) firstprivate(R, C)
        for (int i = 1; i <= R; ++i)
            for (int j = 0; j < C; ++j) {
                int jl = j > 0 ? j - 1 : j, jr = j < C - 1 ? j + 1 : j;
                pb[(size_t)i * C + j] = 0.25f *
                    (pa[(size_t)(i - 1) * C + j] + pa[(size_t)(i + 1) * C + j] +
                     pa[(size_t)i * C + jl]      + pa[(size_t)i * C + jr]);
            }

        // Halo exchange: a real target region; args are captured formals so
        // the DWQ build's pass can synthesize the host trace for this launch.
        double c0 = omp_get_wtime();
        size_t oti = hb + rel_top_int, obi = hb + rel_bot_int;
        size_t otg = hb + rel_top_gho, obg = hb + rel_bot_gho;
        size_t rb = row_bytes;
        #pragma omp target is_device_ptr(ctx) \
            firstprivate(up, down, bidx, oti, obi, otg, obg, rb)
        {
            ompx_dwq_put_dev(ctx, up,   bidx, obg, bidx, oti, rb);
            ompx_dwq_put_dev(ctx, down, bidx, otg, bidx, obi, rb);
            ompx_dwq_flush_dev(ctx);
        }
        ompx_quiet();
        if (it >= WARM) t_comm += omp_get_wtime() - c0;
        // The attribution timer stops at local communication completion.
        // Keep the iteration barrier outside it: otherwise rank-to-rank
        // variation in the preceding Jacobi kernel is charged to the rank
        // that reaches communication first, which can dominate large-face
        // measurements even though the communication path is unchanged.
        ompx_barrier();
        src = 1 - src;
    }
    t_total = omp_get_wtime() - t_total;

    // Order-independent per-rank hash of the interior + a value sum.
    float* pf = buf[src];
    unsigned long long hx = 0; double sum = 0.0;
    #pragma omp target teams distribute parallel for collapse(2) \
        reduction(^:hx) reduction(+:sum) is_device_ptr(pf) firstprivate(R, C)
    for (int i = 1; i <= R; ++i)
        for (int j = 0; j < C; ++j) {
            union { float f; unsigned int u; } cv;
            cv.f = pf[(size_t)i * C + j];
            hx ^= ((unsigned long long)cv.u << ((j % 5) * 8));
            sum += cv.f;
        }
    printf("rank %d field_hash=%016llx sum=%.6f\n", my, hx, sum);
    fflush(stdout);
    ompx_barrier();

    if (my == 0)
        printf("RESULT ranks=%d R=%d C=%d iters=%d us_per_iter=%.3f comm_us_per_iter=%.3f\n",
               nr, R, C, ITERS, t_total * 1e6 / ITERS, t_comm * 1e6 / ITERS);

    // Launch-overhead microbench. Baseline: an empty target region (bare
    // OpenMP launch). Marker loop: the same halo region with 4-byte puts.
    // In the DWQ build every marker-region launch first runs the synthesized
    // host trace (argument evaluation + descriptor staging), so the increment
    // over the empty launch bounds the pre-launch cost (plus one 4-byte
    // triggered transfer and its completion wait).
    {
        ompx_barrier();
        double e0 = omp_get_wtime();
        for (int k = 0; k < LAUNCH_N; ++k) {
            #pragma omp target
            { asm volatile("" ::: "memory"); }
        }
        double e1 = omp_get_wtime();

        size_t z4 = 4, zo = hbase[src], zo2 = hbase[src] + 8;
        int b0 = bidx;
        ompx_barrier();
        double l0 = omp_get_wtime();
        for (int k = 0; k < LAUNCH_N; ++k) {
            #pragma omp target is_device_ptr(ctx) firstprivate(up, down, b0, zo, zo2, z4)
            {
                ompx_dwq_put_dev(ctx, up,   b0, zo,  b0, zo2, z4);
                ompx_dwq_put_dev(ctx, down, b0, zo2, b0, zo,  z4);
                ompx_dwq_flush_dev(ctx);
            }
            ompx_quiet();
        }
        double l1 = omp_get_wtime();
        if (my == 0)
            printf("LAUNCH n=%d empty_us=%.3f halo4B_us=%.3f delta_us=%.3f\n",
                   LAUNCH_N,
                   (e1 - e0) * 1e6 / LAUNCH_N,
                   (l1 - l0) * 1e6 / LAUNCH_N,
                   ((l1 - l0) - (e1 - e0)) * 1e6 / LAUNCH_N);
    }
    ompx_barrier();
    ompx_free(buf[0]); ompx_free(buf[1]);
    ompx_finalize();
    return 0;
}
