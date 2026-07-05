// ompx_host.cpp - GiOMP host-side ompx_* symbols. Thin wrappers over the
// existing gicc_omp_bridge:: runtime, compiled -x hip into libgicc_omp.
#include <omp.h>
#include "gicc/omp.h"
#include "examples/omp/gicc_omp_bridge.hpp"   // internal: gicc_omp_bridge::*

void ompx_init() {
    gicc_omp_bridge::init_runtime_only();
    // Remove the historical footgun: pin OpenMP to the same device the runtime
    // selected, so the app never has to remember omp_set_default_device().
    omp_set_default_device(gicc_omp_bridge::local_device());
}

void ompx_finalize() { gicc_omp_bridge::finalize(); }

int  omp_get_rank_num()  { return gicc_omp_bridge::rank(); }
int  omp_get_num_ranks() { return gicc_omp_bridge::nranks(); }

ompx_buffer ompx_alloc(size_t bytes) {
    void* p = gicc_omp_bridge::device_alloc(bytes);
    int   i = gicc_omp_bridge::register_external(p, bytes);
    return ompx_buffer{p, i, bytes};
}

int  ompx_register(void* dev_ptr, size_t bytes) {
    return gicc_omp_bridge::register_external(dev_ptr, bytes);
}

void ompx_free(ompx_buffer b) { gicc_omp_bridge::device_free(b.ptr); }

void ompx_exchange() { gicc_omp_bridge::exchange_buffers(); }

gicc::DeviceCtx* ompx_prepare() { return gicc_omp_bridge::prepare(); }

void ompx_barrier()    { gicc_omp_bridge::barrier(); }
void ompx_quiet_host() { gicc_omp_bridge::reset(); }
