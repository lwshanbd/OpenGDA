// gicc/omp_box.h - the box arithmetic gicc/omp_pipeline_impl.h and the
// runtime share: which iterations of a kernel's collapsed loop reach a byte
// range. Internal; not for application code.
#pragma once

#include <cstddef>
#include <cstdint>

static inline int64_t ompx__floor_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
static inline int64_t ompx__ceil_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) == (b < 0))) ? q + 1 : q;
}

// The iterations [lo, hi] of a collapsed loop whose accesses through the
// box reach [src, src + bytes); hi < lo when none does. `box` is the address
// with every digit 0, the dims (stride in bytes, extent in points) as the
// loop nests them, outermost first: the first loop_dims of them the loop's
// digits, the rest sequential loops inside it. Iteration x has the digits
// x = q_0 * R_1 + ..., R_1 the product of the extents of digits 1 to
// loop_dims - 1, and the interval is the slabs of q_0 whose accesses reach
// the range.
static inline void ompx__hull(const void* src, size_t bytes, const void* box, int dims,
                              int loop_dims, const int64_t* stride, const int64_t* extent,
                              int64_t elem, int64_t& lo, int64_t& hi) {
    lo = 0;
    hi = -1;
    if (dims <= 0 || bytes == 0 || extent[0] <= 0) return;
    int64_t inner = 1, min_rest = 0, max_rest = 0;
    for (int d = 1; d < dims; ++d) {
        if (extent[d] <= 0) return;
        // A dim past loop_dims is a sequential loop each iteration runs
        // whole: it widens a slab's accesses, not its iterations.
        if (d < loop_dims) inner *= extent[d];
        const int64_t span = stride[d] * (extent[d] - 1);
        (span < 0 ? min_rest : max_rest) += span;
    }
    // Slab q reaches [S0*q + min_rest, S0*q + max_rest + elem) from the
    // box's origin; it meets the range [s, e) when S0*q lies in [a, b].
    const int64_t s = static_cast<const char*>(src) - static_cast<const char*>(box);
    const int64_t e = s + static_cast<int64_t>(bytes);
    const int64_t a = s - elem - max_rest + 1, b = e - min_rest - 1;
    const int64_t s0 = stride[0];
    int64_t q_lo = 0, q_hi = extent[0] - 1;
    if (s0 > 0) {
        q_lo = ompx__ceil_div(a, s0);
        q_hi = ompx__floor_div(b, s0);
    } else if (s0 < 0) {
        q_lo = ompx__ceil_div(b, s0);
        q_hi = ompx__floor_div(a, s0);
    } else if (a > 0 || b < 0) {
        return;
    }
    if (q_lo < 0) q_lo = 0;
    if (q_hi > extent[0] - 1) q_hi = extent[0] - 1;
    if (q_lo > q_hi) return;
    lo = q_lo * inner;
    hi = (q_hi + 1) * inner - 1;
}
