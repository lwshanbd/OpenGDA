// pipeline_after_check - ordinary puts after a kernel, which the gicc-passes
// plugin moves into the kernel: does the peer get exactly what the puts
// would have sent?
//
// The kernels of pipeline_box_check, written without any marker: each is a
// combined construct, and the puts come after it on the host, the way a
// program is written without the pipeline in mind --
//
//   slab   collapse(3) over the interior of padded planes; each face is
//          whole planes, ghost cells included, put only if the neighbour
//          exists (an open chain).
//   grid   collapse(2) over padded rows; the range starts and ends mid-row.
//   trans  collapse(2) with the loop order transposed to the layout.
//
// Built in two passes (build_giomp_after.sh): GICC_MODE=put-discover finds
// the puts after each launch, and GICC_MODE=chunk-lower has each kernel send
// them itself -- stores repeated into a same-node peer, the range sent by
// count to any other -- while the host skips a put the kernel made. Built
// without the plugin it is a plain program with the same output.
//
// After the fence each rank checks its receive buffers word for word,
// including the words just outside every range, which must still hold their
// sentinel.
//
// Usage: pipeline_after_check [steps]
#include "gicc/omp.h"
#include "gicc/omp_pipeline.h"

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int64_t NX = 24, NY = 40, NZ = 56, G = 3;   // slab, ghost width
constexpr int64_t PY = NY + 2 * G, PZ = NZ + 2 * G, PLANE = PY * PZ, W = 4;
constexpr int64_t NR = 70, NC = 90, LD = 101;          // grid / trans rows
constexpr int64_t GRID_LO = 3 * LD + 5, GRID_HI = (NR - 4) * LD + NC - 7;
constexpr int64_t TRANS_WORDS = (NC - 1) * LD + NR;    // trans's written span
constexpr uint32_t SENTINEL = 0x5e5e5e5eu;

}  // namespace

#pragma omp declare target
static inline uint32_t val(int step, int rank, int64_t obj, int64_t i) {
    return (uint32_t)(step * 2654435761u) ^ (uint32_t)(rank << 20) ^ (uint32_t)(obj << 28) ^
           (uint32_t)i;
}
static inline uint32_t ghost(int rank, int64_t obj, int64_t i) {
    return 0xa0000000u ^ (uint32_t)(rank << 16) ^ (uint32_t)(obj << 12) ^ (uint32_t)i;
}
#pragma omp end declare target

static void slab_step(uint32_t* a, int step, int rank, int left, uint32_t* to_left, int right,
                      uint32_t* to_right) {
    const size_t bytes = W * PLANE * 4;
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(a) \
            firstprivate(step, rank)
    for (int64_t i = 0; i < NX; ++i)
        for (int64_t j = G; j < G + NY; ++j)
            for (int64_t k = G; k < G + NZ; ++k)
                a[(i * PY + j) * PZ + k] = val(step, rank, 0, (i * PY + j) * PZ + k);
    if (left >= 0) ompx_put(left, to_left, a, bytes);
    if (right >= 0) ompx_put(right, to_right, a + (NX - W) * PLANE, bytes);
}

static void grid_step(uint32_t* b, int step, int rank, int right, uint32_t* to) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(b) \
            firstprivate(step, rank)
    for (int64_t i = 0; i < NR; ++i)
        for (int64_t j = 0; j < NC; ++j)
            b[i * LD + j] = val(step, rank, 1, i * LD + j);
    ompx_put(right, to, b + GRID_LO, (GRID_HI - GRID_LO) * 4);
}

static void trans_step(uint32_t* c, int step, int rank, int right, uint32_t* to) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(c) \
            firstprivate(step, rank)
    for (int64_t i = 0; i < NR; ++i)
        for (int64_t j = 0; j < NC; ++j)
            c[j * LD + i] = val(step, rank, 2, j * LD + i);
    ompx_put(right, to, c, TRANS_WORDS * 4);
}

// Words of `got` that differ from what `sender` produced in `step`. `box`
// tells a written index from a ghost one; [lo, hi) is the range the sender
// put, and outside it `got` must still hold the sentinel. sender < 0: no
// sender, the whole buffer must be sentinel.
template <class Box>
static long count_bad(const uint32_t* got, int64_t words, int64_t lo, int64_t hi, int step,
                      int sender, int64_t obj, int64_t base, Box box) {
    long bad = 0;
    #pragma omp target teams distribute parallel for reduction(+ : bad) is_device_ptr(got) \
            firstprivate(lo, hi, step, sender, obj, base, box)
    for (int64_t x = 0; x < words; ++x) {
        uint32_t want = SENTINEL;
        if (sender >= 0 && x >= lo && x < hi) {
            const int64_t i = base + x;
            want = box(i) ? val(step, sender, obj, i) : ghost(sender, obj, i);
        }
        bad += got[x] != want;
    }
    return bad;
}

struct SlabBox {
    bool operator()(int64_t idx) const {
        const int64_t k = idx % PZ, j = idx / PZ % PY;
        return j >= G && j < G + NY && k >= G && k < G + NZ;
    }
};
struct RowBox {   // grid and trans: rows of LD words, the first `n` written
    int64_t n;
    bool operator()(int64_t idx) const { return idx % LD < n; }
};

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 20;
    ompx_init();
    const int rank = ompx_get_rank_num(), n = ompx_get_num_ranks();
    const int left = rank > 0 ? rank - 1 : -1, right = rank < n - 1 ? rank + 1 : -1;
    const int ring_right = (rank + 1) % n, ring_left = (rank + n - 1) % n;

    const int64_t slab_w = NX * PLANE, face_w = W * PLANE, grid_w = NR * LD, trans_w = NC * LD;
    uint32_t* a      = (uint32_t*)ompx_alloc(slab_w * 4);
    uint32_t* recv_l = (uint32_t*)ompx_alloc(face_w * 4);   // from the left neighbour
    uint32_t* recv_r = (uint32_t*)ompx_alloc(face_w * 4);   // from the right neighbour
    uint32_t* b      = (uint32_t*)ompx_alloc(grid_w * 4);
    uint32_t* recv_b = (uint32_t*)ompx_alloc(grid_w * 4);
    uint32_t* c      = (uint32_t*)ompx_alloc(trans_w * 4);
    uint32_t* recv_c = (uint32_t*)ompx_alloc(trans_w * 4);

    // Ghost cells get their value once; receive buffers start as sentinel.
    #pragma omp target teams distribute parallel for is_device_ptr(a) firstprivate(rank)
    for (int64_t x = 0; x < slab_w; ++x) a[x] = ghost(rank, 0, x);
    #pragma omp target teams distribute parallel for is_device_ptr(b, recv_b) firstprivate(rank)
    for (int64_t x = 0; x < grid_w; ++x) { b[x] = ghost(rank, 1, x); recv_b[x] = SENTINEL; }
    #pragma omp target teams distribute parallel for is_device_ptr(c, recv_c) firstprivate(rank)
    for (int64_t x = 0; x < trans_w; ++x) { c[x] = ghost(rank, 2, x); recv_c[x] = SENTINEL; }
    #pragma omp target teams distribute parallel for is_device_ptr(recv_l, recv_r)
    for (int64_t x = 0; x < face_w; ++x) { recv_l[x] = SENTINEL; recv_r[x] = SENTINEL; }
    ompx_prepare();
    ompx_fence();

    long bad_total = 0;
    int bad_steps = 0;
    for (int s = 1; s <= steps; ++s) {
        slab_step(a, s, rank, left, recv_r, right, recv_l);
        grid_step(b, s, rank, ring_right, recv_b + GRID_LO);
        trans_step(c, s, rank, ring_right, recv_c);
        ompx_fence();

        // recv_l holds the left neighbour's right face, recv_r the right
        // neighbour's left face; at the chain's ends nothing may arrive.
        const long bl = count_bad(recv_l, face_w, 0, face_w, s, left, 0, (NX - W) * PLANE, SlabBox{});
        const long br = count_bad(recv_r, face_w, 0, face_w, s, right, 0, 0, SlabBox{});
        const long bb = count_bad(recv_b, grid_w, GRID_LO, GRID_HI, s, ring_left, 1, 0, RowBox{NC});
        const long bc = count_bad(recv_c, trans_w, 0, TRANS_WORDS, s, ring_left, 2, 0, RowBox{NR});
        const long bad = bl + br + bb + bc;
        if (bad != 0) {
            ++bad_steps;
            bad_total += bad;
            if (bad_steps <= 3)
                std::printf("rank %d step %d: %ld words wrong (slab from left %ld, slab from "
                            "right %ld, grid %ld, trans %ld)\n", rank, s, bad, bl, br, bb, bc);
        }
        ompx_barrier();   // the next step may overwrite what was just checked
    }

    long all = 0;
    MPI_Allreduce(&bad_total, &all, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("pipeline_after_check: %d ranks, %d steps, %ld words wrong: %s\n", n, steps,
                    all, all == 0 ? "PASS" : "FAIL");
    ompx_finalize();
    return all == 0 ? 0 : 1;
}
