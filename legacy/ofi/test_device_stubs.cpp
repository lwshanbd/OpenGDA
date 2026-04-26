#include <hip/hip_runtime.h>
#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"

__global__ void k(gicc::DeviceCtx* ctx) {
    gicc::put_no_db(ctx, 0, 0, 0, 0, 0, false);
    gicc::get_no_db(ctx, 0, 0, 0, 0, 0, false);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

int main() { return 0; }
