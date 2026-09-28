// Puts after a kernel launch (GICCAfterPut.h).
#include "gicc/omp.h"
#include "gicc/omp_pipeline.h"

#include <cstdint>

// A call in the loop's body: ROCm clang emits a loop without one as a
// specialized kernel (big-jump-loop / no-loop) the pass does not read.
#pragma omp declare target
static float wave(float s, int64_t i) { return s * (float)i; }
#pragma omp end declare target

// Two faces of a 2-D write put after the kernel, each only when the
// neighbour exists: both can go from the kernel.
void step(float* v, int64_t nx, int64_t ny, int left, int right, float s) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(v) \
            firstprivate(nx, ny, s)
    for (int64_t i = 0; i < nx; ++i)
        for (int64_t j = 0; j < ny; ++j)
            v[i * ny + j] = wave(s, i + j);
    if (left >= 0) ompx_put(left, v, v, 4 * ny * sizeof(float));
    if (right >= 0) ompx_put(right, v + (nx - 4) * ny, v + (nx - 4) * ny, 4 * ny * sizeof(float));
}

// A barrier between the kernel and the put may be what the peer waits for
// before it leaves dst alone: the put stays on the host. (Were it posted,
// the kernel could not send it either: it does not store every word.)
void fenced(float* v, int64_t n, int peer) {
    #pragma omp target teams distribute parallel for is_device_ptr(v) firstprivate(n)
    for (int64_t i = 0; i < n; ++i)
        if (i % 3 != 0) v[i] = wave(1.f, i);
    ompx_barrier();
    ompx_put(peer, v, v, n * sizeof(float));
}
