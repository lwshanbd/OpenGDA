// RUN: not %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: error: kernel 'k' must take gicc::DeviceCtx* as its first parameter
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(int x, gicc::DeviceCtx* ctx) {
    (void)x;
    gicc::put_no_db(ctx, 0, 0, 0, 0, 0, 0, 0);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
