// Two stores to the object, each a box of its own: the two halves.
#include <cstddef>
typedef long long llint;
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* a, llint nx, llint ny, int peer, float s) {
    #pragma omp target teams is_device_ptr(a) firstprivate(nx, ny, peer, s)
    {
        #pragma omp distribute parallel for collapse(2)
        for (llint i = 0; i < nx; ++i)
            for (llint j = 0; j < ny; ++j) {
                a[i * ny + j] = s;
                a[nx * ny + i * ny + j] = -s;
            }
        ompx_pipelined_put(peer, a, a, 2 * nx * ny * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfxxif_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: ILLEGAL: object 'arg#3' is written with more than one access pattern
