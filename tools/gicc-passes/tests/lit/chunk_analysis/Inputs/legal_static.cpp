// One block per team; each block owns src[LB..UB] and the put covers all of them.
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
        ompx_pipelined_put(peer, dst, src, n * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_iif_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: LEGAL
// CHECK-NEXT: blocks: distribute sched=92 over [0, (sext i32 (-1 + (trunc i64 %1 to i32)) to i64)]
// CHECK-NEXT: block [LB,UB]: put(peer, dst + 4*(LB - 0), arg#2 + 4*LB + 0, 4*(UB - LB + 1)) after the block's parallel region joins
