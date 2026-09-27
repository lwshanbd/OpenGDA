// The put runs only when has_right is set (a halo's usual shape). Blocks
// sent as they complete would go out whatever has_right is.
#include <cstddef>
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* src, float* dst, int n, int peer, float a, int has_right) {
    #pragma omp target teams is_device_ptr(src, dst)
    {
        #pragma omp distribute parallel for
        for (int i = 0; i < n; ++i)
            src[i] = a * (float)i;
        if (has_right) ompx_pipelined_put(peer, dst, src, n * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_{{.*}}_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: ILLEGAL: the put does not run on every path after the distribute loop (it is conditional); guard the whole kernel instead
