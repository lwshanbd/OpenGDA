// Only some points of the box are stored: a mirror of the stores would
// leave the others unsent, and they are not unwritten either.
#include <cstddef>
typedef long long llint;
#pragma omp declare target
extern "C" void ompx_pipelined_put(int peer, void* dst, const void* src, size_t bytes);
extern "C" int omp_get_team_num();
#pragma omp end declare target

void step(float* a, llint nx, llint ny, int peer, float s) {
    #pragma omp target teams is_device_ptr(a) firstprivate(nx, ny, peer, s)
    {
        #pragma omp distribute parallel for collapse(2)
        for (llint i = 0; i < nx; ++i)
            for (llint j = 0; j < ny; ++j)
                if ((i + j) % 3 == 0) a[i * ny + j] = s;
        ompx_pipelined_put(peer, a, a, nx * ny * sizeof(float));
    }
}

// CHECK: [gicc-chunk] kernel __omp_offloading_{{.*}}_Z4stepPfxxif_l{{[0-9]+}}
// CHECK-NEXT: ompx_pipelined_put: ILLEGAL: no store to 'arg#3' runs on every iteration: a put of a multi-dimensional write sends each word as it is stored, so the loop must store every point of its box
