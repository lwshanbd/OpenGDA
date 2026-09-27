// The face put stated in the body of a combined construct: the kernel stays
// SPMD, and the put takes the box form.
#include <cstddef>
typedef long long llint;
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
#pragma omp end declare target

void step(float* a, llint nx, llint ny, llint nz, llint g, int peer, float s) {
    const llint py = ny + 2 * g, pz = nz + 2 * g;
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(a)
    for (llint i = 0; i < nx; ++i)
        for (llint j = g; j < g + ny; ++j)
            for (llint k = g; k < g + nz; ++k) {
                a[(i * py + j) * pz + k] = s * (float)k;
                ompx_pipelined_put(peer, a, a, py * pz * sizeof(float));
            }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfxxxxif_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: LEGAL
// CHECK-NEXT: box: 3-D extent=[%1, %2, %4] stride=[(4 * %6 * %7), (4 * %7), 4] offset=((4 + (4 * %7)) * %3)
// CHECK-NEXT: range: [arg#5 + 0, + (4 * %6 * %7)): stores inside it are mirrored, the rest is sent at kernel start
