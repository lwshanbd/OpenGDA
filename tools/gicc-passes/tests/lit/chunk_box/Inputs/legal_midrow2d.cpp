// collapse(2) over padded rows; the range starts and ends mid-row, and a
// formal offset places it.
#include <cstddef>
typedef long long llint;
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
extern "C" int omp_get_team_num();
#pragma omp end declare target

void step(int* b, llint nr, llint nc, llint ld, llint lo, llint hi, int peer, int* to) {
    #pragma omp target teams is_device_ptr(b, to) firstprivate(nr, nc, ld, lo, hi, peer)
    {
        #pragma omp distribute parallel for collapse(2)
        for (llint i = 0; i < nr; ++i)
            for (llint j = 0; j < nc; ++j)
                b[i * ld + j] = (int)(i ^ j);
        ompx_pipelined_put(peer, to, b + lo, (hi - lo) * sizeof(int));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPixxxxxiS__l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: LEGAL
// CHECK-NEXT: box: 2-D extent=[%1, %2] stride=[(4 * %4), 4] offset=0
// CHECK-NEXT: range: [arg#3 + (4 * %7){{(<nsw>)?}}, + ((4 * %8) + (-4 * %7))): stores inside it are mirrored, the rest is sent at kernel start
