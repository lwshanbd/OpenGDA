// RUN: %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1
// RUN: FileCheck --input-file=/tmp/lift_loop.gicc.cpp %s
//
// Validates Phase F.3: a put_no_db inside an HK for-loop is wrapped in
// the matching host-side for-loop in the sidecar trace specialization.
// The loop variable `i` is HK because its init is a literal and the
// cond uses kernel param N (HK), so the put's args (la + i*s, etc.) are
// all HK and pass E3.
//
// CHECK: for (int i = 0; i < N; i++)
// CHECK: rt.put_no_db
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(gicc::DeviceCtx* ctx, int N, uint64_t la, uint32_t lkey,
                  uint64_t ra, uint32_t rkey, uint32_t s) {
    for (int i = 0; i < N; i++) {
        gicc::put_no_db(ctx, la + i*s, lkey, ra + i*s, rkey, s);
    }
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
