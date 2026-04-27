// RUN: %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1
// RUN: FileCheck --input-file=/tmp/grid_expand.gicc.cpp %s
//
// Validates Phase F.4: a put_no_db whose args transitively reference
// blockIdx.x (here via local var i) is wrapped in a synthetic
// host-side for-loop over the x grid dimension. The HK local i is
// re-emitted inside the loop with blockIdx.x textually rewritten to
// _gicc_bx, and the put references i naturally.
//
// CHECK: for (uint32_t _gicc_bx = 0; _gicc_bx < grid.x; _gicc_bx++)
// CHECK: int i = _gicc_bx;
// CHECK: rt.put_no_db
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(gicc::DeviceCtx* ctx, int target, int dst_buf,
                  int src_buf, size_t s) {
    int i = blockIdx.x;
    gicc::put_no_db(ctx, target, dst_buf, i*s, src_buf, i*s, s);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
