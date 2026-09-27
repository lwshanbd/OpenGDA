// A put in the loop's body that does not run on every iteration.
#include <cstddef>
typedef long long llint;
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* a, llint n, int peer, float s) {
    #pragma omp target teams distribute parallel for is_device_ptr(a)
    for (llint i = 0; i < n; ++i) {
        a[i] = s;
        if (i == 0) ompx_pipelined_put(peer, a, a, n * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfxif_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: ILLEGAL: the put does not run on every iteration (it is conditional)
