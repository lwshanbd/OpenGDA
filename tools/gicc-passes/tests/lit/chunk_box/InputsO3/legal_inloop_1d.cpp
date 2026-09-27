// A 1-D loop with the put in its body takes the box form too.
#include <cstddef>
typedef long long llint;
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* src, float* dst, llint n, int peer, float s) {
    #pragma omp target teams distribute parallel for is_device_ptr(src, dst)
    for (llint i = 0; i < n; ++i) {
        src[i] = s * (float)i;
        ompx_pipelined_put(peer, dst, src, n * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_xif_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: LEGAL
// CHECK-NEXT: box: 1-D extent=[%1] stride=[4] offset=0
// CHECK-NEXT: range: [arg#2 + 0, + (4 * %1)): stores inside it are mirrored, the rest is sent at kernel start
