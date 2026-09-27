// The peer is read from memory, so it may differ between teams.
#include <cstddef>
typedef long long llint;
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* a, const int* peers, llint nx, llint ny, float s) {
    #pragma omp target teams is_device_ptr(a, peers) firstprivate(nx, ny, s)
    {
        #pragma omp distribute parallel for collapse(2)
        for (llint i = 0; i < nx; ++i)
            for (llint j = 0; j < ny; ++j)
                a[i * ny + j] = s;
        ompx_pipelined_put(peers[0], a, a, nx * ny * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfPKixxf_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: ILLEGAL: peer or dst differs between teams
