// The distribute loop runs once per time step; the put sends the last
// step's data once. Per-block sends would go out every step, racing the
// next step's writes.
#include <cstddef>
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* src, float* dst, int n, int steps, int peer, float a) {
    #pragma omp target teams is_device_ptr(src, dst)
    {
        for (int t = 0; t < steps; ++t) {
            #pragma omp distribute parallel for
            for (int i = 0; i < n; ++i)
                src[i] = src[i] * a + (float)t;
        }
        ompx_pipelined_put(peer, dst, src, n * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_{{.*}}_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: ILLEGAL: the distribute loop is inside a sequential loop of the teams region: its blocks would be sent once per trip
