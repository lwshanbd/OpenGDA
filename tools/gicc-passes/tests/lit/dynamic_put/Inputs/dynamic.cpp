// Puts after a kernel whose stores land where data says (GICCDynamicPut.h).
#include "gicc/omp.h"
#include "gicc/omp_pipeline.h"

#include <cstdint>

// Rows scattered by an index the kernel reads: no affine write set, so the
// chunk analysis cannot split the put, and only run-time counting sends it
// from the kernel -- payload and signal both.
void scatter(float* out, const float* in, const int* idx, int64_t rows, int64_t cols, int peer,
             float* dst, unsigned long long step) {
    #pragma omp target teams distribute parallel for is_device_ptr(out, in, idx) \
            firstprivate(rows, cols)
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
            out[idx[r] * cols + c] = in[r * cols + c];
    ompx_put_signal(peer, dst, out, rows * cols * sizeof(float), 3, step);
}

// An atomic writes the source: counting cannot see it, so the put stays on
// the host.
void tally(float* out, const int* idx, int64_t n, int peer, float* dst) {
    #pragma omp target teams distribute parallel for is_device_ptr(out, idx) firstprivate(n)
    for (int64_t i = 0; i < n; ++i) {
        #pragma omp atomic
        out[idx[i]] += 1.f;
    }
    ompx_put(peer, dst, out, n * sizeof(float));
}
