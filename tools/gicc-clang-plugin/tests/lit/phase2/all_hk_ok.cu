// RUN: %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(gicc::DeviceCtx* ctx, int target, int dst_buf,
                  size_t dst_off, int src_buf, size_t src_off,
                  size_t sz) {
    gicc::put_no_db(ctx, target, dst_buf, dst_off, src_buf, src_off, sz);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
