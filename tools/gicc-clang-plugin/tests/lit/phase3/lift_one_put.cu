// RUN: %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1
// RUN: FileCheck --input-file=/tmp/lift_one_put.gicc.cpp %s
//
// Validates Phase F.2: a single straight-line gicc::put_no_db call in
// the kernel is lifted to a host-side rt.buffer_by_lkey + peer_buffer_base
// + rt.put_no_db sequence in the sidecar's kernel_trace specialization.
// All six args are kernel parameters (HK by definition), satisfying the
// E3 validator.
//
// CHECK: rt.buffer_by_lkey
// CHECK: rt.peer_buffer_base(peer
// CHECK: rt.put_no_db(_gicc_src, peer
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(gicc::DeviceCtx* ctx, uint64_t la, uint32_t lkey,
                  uint64_t ra, uint32_t rkey, uint32_t s) {
    gicc::put_no_db(ctx, la, lkey, ra, rkey, s);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
