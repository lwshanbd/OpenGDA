// Loops of puts and nowait regions before puts (GICCAfterPut.h).
#include "gicc/omp.h"
#include "gicc/omp_pipeline.h"

#include <cstdint>

#pragma omp declare target
static float wave(float s, int64_t i) { return s * (float)i; }
#pragma omp end declare target

// A neighbour list: one put per neighbour, from host arrays the kernel is
// not handed. The first six iterations are posted (their peer and dst
// loaded before the launch), the rest made by the host as before.
void neighbours(float* v, int64_t n, int64_t w, int nn, const int* nbr, float* const* dst) {
    #pragma omp target teams distribute parallel for is_device_ptr(v) firstprivate(n)
    for (int64_t i = 0; i < n; ++i)
        v[i] = wave(2.f, i);
    for (int k = 0; k < nn; ++k) ompx_put(nbr[k], dst[k], v + k * w, w * sizeof(float));
}

// The same after a region that copies a map(from) back: the host arrays
// may change at the launch, so nothing is loaded before it.
void copied(float* v, float* h, int64_t n, int64_t w, int nn, const int* nbr, float* const* dst) {
    #pragma omp target teams distribute parallel for is_device_ptr(v) map(from: h[0:n]) \
            firstprivate(n)
    for (int64_t i = 0; i < n; ++i) {
        v[i] = wave(2.f, i);
        h[i] = v[i];
    }
    for (int k = 0; k < nn; ++k) ompx_put(nbr[k], dst[k], v + k * w, w * sizeof(float));
}

// A nowait region the host waits for with a taskwait: the put is posted
// before the region's task is created and made after the taskwait only if
// the kernel did not.
void waited(float* v, int64_t n, int peer, float* dst) {
    #pragma omp target teams distribute parallel for is_device_ptr(v) firstprivate(n) nowait
    for (int64_t i = 0; i < n; ++i)
        v[i] = wave(3.f, i);
    #pragma omp taskwait
    if (peer >= 0) ompx_put(peer, dst, v + n - 64, 64 * sizeof(float));
}

// A call between the task and the taskwait may be what the peer waits
// for: the put stays on the host.
void busy(float* v, int64_t n, int peer, float* dst) {
    #pragma omp target teams distribute parallel for is_device_ptr(v) firstprivate(n) nowait
    for (int64_t i = 0; i < n; ++i)
        v[i] = wave(3.f, i);
    ompx_barrier();
    #pragma omp taskwait
    if (peer >= 0) ompx_put(peer, dst, v + n - 64, 64 * sizeof(float));
}

// A nowait region with a depend clause, on a buffer named by a global the
// host loads again for the put: the same value, nothing writing it between.
float* g_halo;
void depended(int64_t n, int peer, float* dst) {
    float* v = g_halo;
    #pragma omp target teams distribute parallel for is_device_ptr(v) firstprivate(n) \
            nowait depend(inout: g_halo)
    for (int64_t i = 0; i < n; ++i)
        v[i] = wave(3.f, i);
    #pragma omp taskwait
    if (peer >= 0) ompx_put(peer, dst, g_halo + n - 64, 64 * sizeof(float));
}
