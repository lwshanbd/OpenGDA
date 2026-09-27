// A 1-D loop with a constant trip count: the distribute fini and the put
// share a block, which the put is still after.
#include <cstddef>
typedef long long llint;
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
extern "C" int omp_get_team_num();
#pragma omp end declare target

void step(float* src, float* dst, int peer, float s) {
    #pragma omp target teams is_device_ptr(src, dst) firstprivate(peer, s)
    {
        #pragma omp distribute parallel for
        for (int i = 0; i < 4096; ++i) src[i] = s * (float)i;
        ompx_pipelined_put(peer, dst, src, 4096 * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_if_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: LEGAL
// CHECK-NEXT: blocks: distribute sched=92 over [0, 4095]
