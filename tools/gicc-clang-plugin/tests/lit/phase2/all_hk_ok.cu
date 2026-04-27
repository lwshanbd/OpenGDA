// RUN: %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(gicc::DeviceCtx* ctx, int peer, int dst_buf,
                  uint64_t la, uint32_t lk,
                  uint64_t ra, uint32_t rk, uint32_t s) {
    gicc::put_no_db(ctx, peer, dst_buf, la, lk, ra, rk, s);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
