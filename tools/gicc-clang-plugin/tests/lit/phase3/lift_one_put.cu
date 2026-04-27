// RUN: %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1
// RUN: FileCheck --input-file=/tmp/lift_one_put.gicc.cpp %s
//
// Validates Phase F.2: a single straight-line gicc::put_no_db call in
// the kernel is lifted to a host-side rt.buffer_by_lkey + rt.put_no_db
// sequence in the sidecar's kernel_trace specialization. All args are
// kernel parameters (HK by definition), satisfying E3.
//
// CHECK: rt.buffer_by_lkey
// CHECK: rt.put_no_db(_gicc_src
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(gicc::DeviceCtx* ctx, int target, int dst_buf,
                  size_t dst_off, int src_buf, size_t src_off, size_t s) {
    gicc::put_no_db(ctx, target, dst_buf, dst_off, src_buf, src_off, s);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
