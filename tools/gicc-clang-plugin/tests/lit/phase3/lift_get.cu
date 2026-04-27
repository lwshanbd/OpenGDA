// RUN: %clang -std=c++17 -fplugin=%plugin -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1
// RUN: FileCheck --input-file=/tmp/lift_get.gicc.cpp %s
//
// Validates Phase F.5: a single straight-line gicc::get_no_db call
// in the kernel is lifted to a host-side rt.buffer_by_lkey +
// peer_buffer_base + rt.get_no_db sequence in the sidecar trace.
// Same lookup machinery as put_no_db; the variable names reflect that
// the local buffer is the read DESTINATION (not the source) for a get.
//
// CHECK: rt.buffer_by_lkey
// CHECK: rt.peer_buffer_base(_gicc_peer
// CHECK: rt.get_no_db
#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"
__global__ void k(gicc::DeviceCtx* ctx, int peer, int src_buf,
                  uint64_t la, uint32_t lkey,
                  uint64_t ra, uint32_t rkey, uint32_t s) {
    gicc::get_no_db(ctx, peer, src_buf, la, lkey, ra, rkey, s);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
