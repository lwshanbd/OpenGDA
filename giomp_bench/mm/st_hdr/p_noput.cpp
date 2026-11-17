#include "gicc/omp.h"
#include <omp.h>
#include <cstddef>
extern "C" void halo_put_dev(float* u, int left, size_t left_off, int right, size_t right_off, size_t bytes) {
    ompx_prepare();
    #pragma omp target teams distribute parallel for num_teams(1) thread_limit(2) firstprivate(left, left_off, right, right_off, bytes)
    for (int t = 0; t < 2; ++t) { char* base = reinterpret_cast<char*>(u); if (t == 0 && left >= 0) base[left_off] = 1; }
}
