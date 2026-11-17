#include <cstdio>
extern "C" void halo_put_dev(int* p, int x) {
    #pragma omp target teams distribute parallel for num_teams(1) thread_limit(2) is_device_ptr(p) firstprivate(x)
    for (int t = 0; t < 2; ++t) {
#ifdef __NVPTX__
        if (x == 12345) p[0] = (int)__nvvm_read_ptx_sreg_globaltimer();
#endif
    }
}
