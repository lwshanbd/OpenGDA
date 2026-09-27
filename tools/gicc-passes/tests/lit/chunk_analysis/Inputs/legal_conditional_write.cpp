// Only some words are written. The put still sends all of them; blocks
// sent whole are right, mirroring only the stores that run is not.
#include <cstddef>
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* src, float* dst, const int* mask, int n, int peer, float a) {
    #pragma omp target teams is_device_ptr(src, dst, mask)
    {
        #pragma omp distribute parallel for
        for (int i = 0; i < n; ++i)
            if (mask[i]) src[i] = a * (float)i;
        ompx_pipelined_put(peer, dst, src, n * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_{{.*}}_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: LEGAL
// CHECK-NEXT: blocks: distribute sched=92 over [0, {{.*}}]
// CHECK-NEXT: block [LB,UB]: put(peer, dst + 4*(LB - 0), arg#3 + 4*LB + 0, 4*(UB - LB + 1)) after the block's parallel region joins
