#include "gicc/omp.h"
#include <omp.h>
#include <cstddef>
extern "C" void halo_put_dev(float* u, int left, size_t left_off, int right, size_t right_off, size_t bytes) {
    #pragma omp target teams distribute parallel for num_teams(1) thread_limit(2)
    for (int t = 0; t < 2; ++t) { if (t == 0) ompx_quiet(0); }
}
