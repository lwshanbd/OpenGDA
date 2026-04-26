// RUN: not %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: error: gicc::put cannot be lifted to OFI
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
// Shim — gicc::put is the auto-doorbell variant in the MLX5 backend; the
// OFI device header omits it. Declare a same-name __device__ stub here
// so the kernel below compiles, then verify the plugin rejects the call
// because put is not in the v1 lift whitelist.
namespace gicc {
__device__ __forceinline__ void put(DeviceCtx*) {}
}
__global__ void k(gicc::DeviceCtx* ctx) {
    gicc::put(ctx);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
