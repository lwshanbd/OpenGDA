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

// A region that may run on the host (an if clause), over int loop
// variables: the launch no longer dominates the put, yet every path to it
// runs the region first; the inner index is lb + digit narrowed to int with
// no flags. The last row's interior can go from the kernel.
void gated(float* v, int x_min, int x_max, int y_min, int y_max, int peer, bool on) {
    const size_t sx = x_max + 4;
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(v) if(target: on)
    for (int j = y_min + 1; j < y_max + 2; ++j)
        for (int i = x_min + 1; i < x_max + 2; ++i)
            v[i + j * sx] = wave(3.f, i + j);
    if (peer >= 0) ompx_put(peer, v + (y_max + 1) * sx + 2, v + (y_max + 1) * sx + 2, x_max * sizeof(float));
}
