// 02_hk_error_diagnostics: kernel with a put_no_db whose size argument
// depends on threadIdx.x. HKAnalysis must report a clear diagnostic
// pointing at threadIdx so the user knows which value is non-HK.
#include <hip/hip_runtime.h>

#include "gicc/gicc_types.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

__global__ void bad_kernel(gicc::DeviceCtx* ctx, int peer) {
    // size depends on threadIdx → not host-knowable
    size_t bad_size = (size_t)threadIdx.x * 16;
    gicc::put_no_db(ctx, peer, /*dst_buf=*/0, /*dst_off=*/0,
                              /*src_buf=*/0, /*src_off=*/0, bad_size);
}

int main() { return 0; }
