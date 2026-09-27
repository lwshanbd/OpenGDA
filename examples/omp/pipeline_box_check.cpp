// pipeline_box_check - does a pipelined put of a multi-dimensional write
// deliver exactly what a put after the kernel would?
//
// Every step each rank rewrites three objects in collapse loops and states,
// at the end of each kernel, a pipelined put of a contiguous range of it:
//
//   slab   collapse(3) over the interior of padded planes; the left and
//          right faces are whole planes, ghost cells included, so part of
//          each range is never written by the kernel. Open chain: the end
//          ranks pass a negative peer, which must send nothing.
//   grid   collapse(2) over padded rows; the range starts and ends mid-row.
//   trans  collapse(2) with the loop order transposed to the layout.
//
// Values depend on the step, the sender and the index; ghost cells keep
// the value they were given once at the start. After the fence each rank
// checks its receive buffers word for word, including the words just
// outside every range, which must still hold their sentinel.
//
// A neighbour that is not IPC-mapped cannot take a multi-dimensional
// pipelined put; the kernel then gets a negative peer and the host puts the
// range after it, so the check also runs across nodes.
//
// Build with GICC_MODE=chunk-lower and -foffload-lto -fpass-plugin=...
// Usage: pipeline_box_check [steps]
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

static void slab_kernel(uint32_t* a, int step, int rank, int left, uint32_t* to_left,
                        int right, uint32_t* to_right) {
    const int64_t bytes = W * PLANE * 4, right_off = (NX - W) * PLANE;
    #pragma omp target teams is_device_ptr(a, to_left, to_right) \
            firstprivate(step, rank, left, right, bytes, right_off)
    {
        #pragma omp distribute parallel for collapse(3)
        for (int64_t i = 0; i < NX; ++i)
            for (int64_t j = G; j < G + NY; ++j)
                for (int64_t k = G; k < G + NZ; ++k)
                    a[(i * PY + j) * PZ + k] = val(step, rank, 0, (i * PY + j) * PZ + k);
        ompx_pipelined_put(left, to_left, a, bytes);
        ompx_pipelined_put(right, to_right, a + right_off, bytes);
    }
}

static void grid_kernel(uint32_t* b, int step, int rank, int right, uint32_t* to) {
    const int64_t bytes = (GRID_HI - GRID_LO) * 4;
    #pragma omp target teams is_device_ptr(b, to) firstprivate(step, rank, right, bytes)
    {
        #pragma omp distribute parallel for collapse(2)
        for (int64_t i = 0; i < NR; ++i)
            for (int64_t j = 0; j < NC; ++j)
                b[i * LD + j] = val(step, rank, 1, i * LD + j);
        ompx_pipelined_put(right, to, b + GRID_LO, bytes);
    }
}

static void trans_kernel(uint32_t* c, int step, int rank, int right, uint32_t* to) {
    const int64_t bytes = TRANS_WORDS * 4;
    #pragma omp target teams is_device_ptr(c, to) firstprivate(step, rank, right, bytes)
    {
        #pragma omp distribute parallel for collapse(2)
        for (int64_t i = 0; i < NR; ++i)
            for (int64_t j = 0; j < NC; ++j)
                c[j * LD + i] = val(step, rank, 2, j * LD + i);
        ompx_pipelined_put(right, to, c, bytes);
    }
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

    // Same-node neighbours take the pipelined put; the others the host put.
    auto mapped = [&](int peer, void* obj) { return peer >= 0 && ompx_peer_ptr(peer, obj); };
    const bool pl = mapped(left, recv_r), pr = mapped(right, recv_l);
    const bool pring = mapped(ring_right, recv_b);

    long bad_total = 0;
    int bad_steps = 0;
    for (int s = 1; s <= steps; ++s) {
        slab_kernel(a, s, rank, pl ? left : -1, recv_r, pr ? right : -1, recv_l);
        if (left >= 0 && !pl) ompx_put(left, recv_r, a, face_w * 4);
        if (right >= 0 && !pr) ompx_put(right, recv_l, a + (NX - W) * PLANE, face_w * 4);
        grid_kernel(b, s, rank, pring ? ring_right : -1, recv_b + GRID_LO);
        if (!pring) ompx_put(ring_right, recv_b + GRID_LO, b + GRID_LO, (GRID_HI - GRID_LO) * 4);
        trans_kernel(c, s, rank, pring ? ring_right : -1, recv_c);
        if (!pring) ompx_put(ring_right, recv_c, c, TRANS_WORDS * 4);
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
    int pipelined = (int)pl + (int)pr + 2 * (int)pring, total = 0;
    MPI_Allreduce(&bad_total, &all, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &pipelined, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    total = 2 * (n - 1) + 2 * n;
    if (rank == 0)
        std::printf("pipeline_box_check: %d ranks, %d steps, %d of %d sends pipelined, "
                    "%ld words wrong: %s\n", n, steps, pipelined, total, all,
                    all == 0 ? "PASS" : "FAIL");
    ompx_finalize();
    return all == 0 ? 0 : 1;
}
