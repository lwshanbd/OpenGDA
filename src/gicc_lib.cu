/**
 * gicc_lib.cu - GICC shared library compilation unit
 *
 * Includes all GICC headers to compile device code into libgicc.so.
 * Host-side classes are header-only but device code (__global__ kernels)
 * needs to be compiled with separable compilation for linking.
 */

// Suppress kernel definitions in user code to avoid ODR violations;
// they are compiled only here, inside the library.
// (This file is the canonical compilation unit for the kernels.)

#include "gicc/mlx5/device_opt.cuh"
#include "gicc/mlx5/device.cuh"

// Force instantiation of key types to ensure they are available in the library
namespace gicc::mlx5 {

// Ensure the GdaDeviceStateOpt struct is emitted
__device__ void _gicc_force_link_device_opt(GdaDeviceStateOpt* s) {
    (void)s;
}

} // namespace gicc::mlx5
