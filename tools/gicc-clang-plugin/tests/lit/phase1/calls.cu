// RUN: %clang -std=c++17 -fplugin=%plugin -x hip -I%gicc_src/src \
// RUN:   -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 -c %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s
// CHECK: [gicc-plugin] kernel: k
// CHECK-NEXT: [gicc-plugin]   call: gicc::put_no_db
// CHECK-NEXT: [gicc-plugin]   call: gicc::flush
// CHECK-NEXT: [gicc-plugin]   call: gicc::quiet
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(gicc::DeviceCtx* ctx) {
    gicc::put_no_db(ctx, /*peer=*/0, /*dst_buf=*/0,
                    /*la=*/0, /*lk=*/0, /*ra=*/0, /*rk=*/0, /*sz=*/0);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
