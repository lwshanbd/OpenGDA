// The benchmark's shape: a size_t loop (the _8u runtime entry points and an
// "i.next < UB + 1" exit), a runtime dist_schedule chunk, an inner compute
// loop around each store, and n * size folded to 0 on the zero-trip path.
#include <cstddef>
#include <cstdint>
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
static inline uint32_t produce(int rank, size_t i, int work) {
    uint32_t x = (uint32_t)rank * 0x9E3779B1u ^ (uint32_t)i;
    for (int k = 0; k < work; ++k) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; }
    return x;
}
#pragma omp end declare target

void step(uint32_t* src, uint32_t* dst, size_t n, int me, int work, size_t chunk, int right) {
    #pragma omp target teams is_device_ptr(src, dst) firstprivate(n, me, work, chunk, right)
    {
        #pragma omp distribute parallel for dist_schedule(static, chunk)
        for (size_t i = 0; i < n; ++i) src[i] = produce(me, i, work);
        ompx_pipelined_put(right, dst, src, n * sizeof(uint32_t));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPjS_miimi_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: LEGAL
// CHECK-NEXT: blocks: distribute sched=91 chunk={{%[0-9]+}} over [0, (-1 + {{%[0-9]+}})]
// CHECK-NEXT: block [LB,UB]: put(peer, dst + 4*(LB - 0), arg#{{[0-9]+}} + 4*LB + 0, 4*(UB - LB + 1)) after the block's parallel region joins
