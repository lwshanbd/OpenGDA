#include <cstdio>
extern "C" void halo_put_dev(int* p, int x) {
    #pragma omp target teams distribute parallel for num_teams(1) thread_limit(2) is_device_ptr(p) firstprivate(x)
    for (int t = 0; t < 2; ++t) {
        if (x == 12345) {
            #pragma omp assume no_openmp
            printf("x %d\n", x);
        }
    }
}
