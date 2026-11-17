// halo_dev.cpp - issue the halo faces from a GPU thread.
//
// On InfiniBand a GPU thread writes the work request and rings the NIC
// doorbell itself, so the face transfers can start from inside a target
// region with no host involvement. The device form of ompx_put exists only in
// C++, hence this small TU next to the C sources.
//
// `u` is the application's host pointer, bound to the symmetric heap by
// ompx_bind; inside the target region it resolves to the heap address, which
// is what the device ompx_put expects.
#include "gicc/omp.h"
#include <omp.h>
#include <cstddef>

extern "C" void halo_put_dev(float* u, int left, size_t left_off,
                             int right, size_t right_off, size_t bytes)
{
    ompx_prepare();
    // One thread per face; the lanes let the two faces ride separate QPs
    // when GICC_IB_LANES > 1.
    #pragma omp target teams distribute parallel for num_teams(1) thread_limit(2) \
            firstprivate(left, left_off, right, right_off, bytes)
    for (int t = 0; t < 2; ++t) {
        char* base = reinterpret_cast<char*>(u);
        if (t == 0 && left >= 0)
            ompx_put(left, base + left_off, base + left_off, bytes, 0);
        if (t == 1 && right >= 0)
            ompx_put(right, base + right_off, base + right_off, bytes, 1);
    }
}
