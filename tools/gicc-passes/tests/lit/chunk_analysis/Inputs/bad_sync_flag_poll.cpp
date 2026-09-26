// Hand-rolled synchronization: spin on a flag with an acquire load before
// the put. Same hazard as waiting on a signal.
#include <cstddef>
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* src, float* dst, int n, int peer, float a, int* flag) {
    #pragma omp target teams is_device_ptr(src, dst, flag)
    {
        #pragma omp distribute parallel for
        for (int i = 0; i < n; ++i)
            src[i] = a * (float)i;
        while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) == 0) {}
        ompx_pipelined_put(peer, dst, src, n * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_iifPi_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: ILLEGAL: synchronization (atomic or volatile load) between the first block and the put: early blocks would reach the peer before it
