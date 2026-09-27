// On the zero-trip path the put sends 16 bytes; lowered blocks send none.
#include <cstddef>
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* src, float* dst, int n, int peer, float a) {
    #pragma omp target teams is_device_ptr(src, dst)
    {
        size_t bytes = 16;
        if (n > 0) {
            #pragma omp distribute parallel for
            for (int i = 0; i < n; ++i)
                src[i] = a * (float)i;
            bytes = n * sizeof(float);
        }
        ompx_pipelined_put(peer, dst, src, bytes);
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_{{.*}}_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: ILLEGAL: the put sends bytes on the path that skips the loop
