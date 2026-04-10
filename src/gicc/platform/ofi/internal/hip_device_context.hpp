/*
 * hip_device_context.hpp - Thin HIP device context wrapper
 */
#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

// CRITICAL: Call this BEFORE any HIP/PMI2 initialization
// Must be called at the very start of main()
inline void unset_rocr_visible_devices() {
    unsetenv("ROCR_VISIBLE_DEVICES");
}

class HipDeviceContext {
public:
    int gpu_id;
    int device_count;
    double clock_mhz;  // GPU clock in MHz for timing conversion
    hipDeviceProp_t prop;

    explicit HipDeviceContext(int gpu_id_) : gpu_id(gpu_id_), device_count(0), clock_mhz(0.0) {
        hipError_t err;

        err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess) {
            fprintf(stderr, "hipGetDeviceCount failed: %s\n", hipGetErrorString(err));
            exit(1);
        }

        err = hipSetDevice(gpu_id);
        if (err != hipSuccess) {
            fprintf(stderr, "hipSetDevice(%d) failed: %s\n", gpu_id, hipGetErrorString(err));
            exit(1);
        }

        err = hipGetDeviceProperties(&prop, gpu_id);
        if (err != hipSuccess) {
            fprintf(stderr, "hipGetDeviceProperties failed: %s\n", hipGetErrorString(err));
            exit(1);
        }

        // GPU clock rate in MHz (clockRate is in kHz)
        clock_mhz = prop.clockRate / 1000.0;
    }

    // No copy/move
    HipDeviceContext(const HipDeviceContext&) = delete;
    HipDeviceContext& operator=(const HipDeviceContext&) = delete;
};
