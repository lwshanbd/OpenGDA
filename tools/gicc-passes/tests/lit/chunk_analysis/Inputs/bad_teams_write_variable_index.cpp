// The team's main thread patches one word after the loop, at an index the
// analysis cannot place: the blocks would already be on their way.
#include <cstddef>
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* src, float* dst, int n, int peer, float a, int k) {
    #pragma omp target teams is_device_ptr(src, dst)
    {
        #pragma omp distribute parallel for
        for (int i = 0; i < n; ++i)
            src[i] = a * (float)i;
        src[k] = 0.0f;
        ompx_pipelined_put(peer, dst, src, n * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_{{.*}}_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: ILLEGAL: 'arg#2' is written in the teams region outside the worksharing loop
