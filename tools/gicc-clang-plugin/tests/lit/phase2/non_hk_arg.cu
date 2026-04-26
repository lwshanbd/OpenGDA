// RUN: not %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: error: gicc::put_no_db argument 'remote_addr' is not host-known
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(gicc::DeviceCtx* ctx, uint64_t* dev_arr) {
    gicc::put_no_db(ctx, 0, 0, dev_arr[0], 0, 0);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
