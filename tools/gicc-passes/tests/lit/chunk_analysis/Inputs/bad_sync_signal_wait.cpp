// The team waits for the peer's "done reading dst" signal before the put.
// Early blocks would reach the peer before that wait: a race the original
// program does not have.
#include <cstddef>
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
extern "C" void ompx_signal_wait(int sig, unsigned long long ge);
#pragma omp end declare target

void step(float* src, float* dst, int n, int peer, float a) {
    #pragma omp target teams is_device_ptr(src, dst)
    {
        #pragma omp distribute parallel for
        for (int i = 0; i < n; ++i)
            src[i] = a * (float)i;
        ompx_signal_wait(0, 1);
        ompx_pipelined_put(peer, dst, src, n * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfS_iif_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: ILLEGAL: synchronization (ompx_signal_wait) between the first block and the put: early blocks would reach the peer before it
