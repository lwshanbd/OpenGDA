// collapse(2) with the loop order transposed to the layout: the outer digit
// has the smaller stride.
#include <cstddef>
typedef long long llint;
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
extern "C" int omp_get_team_num();
#pragma omp end declare target

void step(int* c, llint nr, llint nc, llint ld, int peer) {
    #pragma omp target teams is_device_ptr(c) firstprivate(nr, nc, ld, peer)
    {
        #pragma omp distribute parallel for collapse(2)
        for (llint i = 0; i < nr; ++i)
            for (llint j = 0; j < nc; ++j)
                c[j * ld + i] = (int)(i + j);
        ompx_pipelined_put(peer, c, c, ((nc - 1) * ld + nr) * sizeof(int));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPixxxi_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: LEGAL
// CHECK-NEXT: box: 2-D extent=[%1, %2] stride=[4, (4 * %4)] offset=0
// CHECK-NEXT: range: [arg#3 + 0, + {{.*}}): stores inside it are mirrored, the rest is sent at kernel start
