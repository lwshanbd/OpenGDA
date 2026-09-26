// A plain put in the teams region runs once per team, before other teams finish: a race even unchunked.
#include <cstddef>
#pragma omp declare target
extern "C" void ompx_put(int peer, void* dst, const void* src, size_t bytes);
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
extern "C" void opaque(float* p, int i);
#pragma omp end declare target

void step(float* src, float* dst, int n, int peer, float a) {
    #pragma omp target teams is_device_ptr(src, dst)
    {
        #pragma omp distribute parallel for
        for (int i = 0; i < n; ++i)
            src[i] = a * (float)i;
        ompx_put(peer, dst, src, n * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_iif_l{{[0-9]+}}
// CHECK-NEXT: ompx_put: RACE: ompx_put in the teams region runs once per team, after only that team's blocks of 'arg#2' are written
