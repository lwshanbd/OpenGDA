// RUN: %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1
// RUN: FileCheck --input-file=/tmp/trace_stub.gicc.cpp %s
//
// Validates Phase F.1: the plugin writes a sidecar file with a stub
// kernel_trace specialization for every validated kernel that calls
// into gicc::. This kernel has no put_no_db, so the body is empty —
// but the structural skeleton (namespace open, struct, run() method,
// namespace close) must still be present.
//
// CHECK: namespace gicc {
// CHECK: namespace detail {
// CHECK: struct kernel_trace
// CHECK: static void run(
// CHECK: } // namespace detail
// CHECK: } // namespace gicc
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(gicc::DeviceCtx* ctx) {
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
