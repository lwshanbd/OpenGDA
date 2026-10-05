// In-loop pipelined puts the host posts before each launch (GICCAfterPut.h).
#include "gicc/omp.h"
#include "gicc/omp_pipeline.h"

#include <cstdint>

// A call in the loop's body: ROCm clang emits a loop without one as a
// specialized kernel (big-jump-loop / no-loop) the pass does not read.
#pragma omp declare target
static float wave(float s, int64_t i) { return s * (float)i; }
#pragma omp end declare target

// The last four rows of a 2-D write, stated in the loop's body: everything
// the put is computed from reaches the kernel as the launch hands it over,
// so the host posts it.
void face(float* v, float* to, int64_t nx, int64_t ny, int right, float s) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(v, to) \
            firstprivate(nx, ny, right, s)
    for (int64_t i = 0; i < nx; ++i)
        for (int64_t j = 0; j < ny; ++j) {
            v[i * ny + j] = wave(s, i + j);
            ompx_pipelined_put(right, to, v + (nx - 4) * ny, 4 * ny * sizeof(float));
        }
}

// The same from a mapped array: the kernel is handed its device copy, an
// address the host cannot compute, so the host posts nothing.
void mapped(float* v, float* to, int64_t nx, int64_t ny, int right) {
    #pragma omp target teams distribute parallel for collapse(2) map(tofrom: v[0:nx * ny]) \
            is_device_ptr(to) firstprivate(nx, ny, right)
    for (int64_t i = 0; i < nx; ++i)
        for (int64_t j = 0; j < ny; ++j) {
            v[i * ny + j] = wave(2.f, i + j);
            ompx_pipelined_put(right, to, v + (nx - 4) * ny, 4 * ny * sizeof(float));
        }
}

// A launch that does not wait for the kernel: nothing could ring the put's
// groups once the kernel is over, so the host posts nothing.
void later(float* v, float* to, int64_t nx, int64_t ny, int right) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(v, to) \
            firstprivate(nx, ny, right) nowait
    for (int64_t i = 0; i < nx; ++i)
        for (int64_t j = 0; j < ny; ++j) {
            v[i * ny + j] = wave(3.f, i + j);
            ompx_pipelined_put(right, to, v + (nx - 4) * ny, 4 * ny * sizeof(float));
        }
    #pragma omp taskwait
}
