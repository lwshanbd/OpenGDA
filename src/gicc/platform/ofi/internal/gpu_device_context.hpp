/*
 * gpu_device_context.hpp - Thin GPU device context wrapper (HIP or CUDA)
 *
 * Selects between the AMD/HIP and NVIDIA/CUDA runtimes via the
 * GICC_GPU_HIP / GICC_GPU_CUDA preprocessor switches set by the
 * top-level CMakeLists.txt (option GICC_GPU_API). Existing call sites
 * that referenced HipDeviceContext continue to compile via the
 * `using HipDeviceContext = GpuDeviceContext` alias at the bottom of
 * this header.
 */
#pragma once

#include <cstdio>
#include <cstdlib>

#if defined(GICC_GPU_HIP)
#include <hip/hip_runtime.h>
using GpuError      = hipError_t;
using GpuDeviceProp = hipDeviceProp_t;
#define GPU_SUCCESS              hipSuccess
#define gpuGetDeviceCount        hipGetDeviceCount
#define gpuSetDevice             hipSetDevice
#define gpuGetDeviceProperties   hipGetDeviceProperties
#define gpuGetErrorString        hipGetErrorString

// CRITICAL: Call this BEFORE any HIP/PMI2 initialization
// Must be called at the very start of main()
inline void unset_rocr_visible_devices() { unsetenv("ROCR_VISIBLE_DEVICES"); }

#elif defined(GICC_GPU_CUDA)
#include <cuda_runtime.h>
using GpuError      = cudaError_t;
using GpuDeviceProp = cudaDeviceProp;
#define GPU_SUCCESS              cudaSuccess
#define gpuGetDeviceCount        cudaGetDeviceCount
#define gpuSetDevice             cudaSetDevice
#define gpuGetDeviceProperties   cudaGetDeviceProperties
#define gpuGetErrorString        cudaGetErrorString

// CUDA path has no equivalent of ROCR_VISIBLE_DEVICES; keep the symbol
// available so call sites can stay GPU-agnostic.
inline void unset_rocr_visible_devices() {}

#else
#error "GICC_GPU_HIP or GICC_GPU_CUDA must be defined"
#endif

class GpuDeviceContext {
public:
    int gpu_id;
    int device_count;
    double clock_mhz;  // GPU clock in MHz for timing conversion
    GpuDeviceProp prop;

    explicit GpuDeviceContext(int gpu_id_)
        : gpu_id(gpu_id_), device_count(0), clock_mhz(0.0) {
        GpuError err = gpuGetDeviceCount(&device_count);
        if (err != GPU_SUCCESS) {
            fprintf(stderr, "gpuGetDeviceCount failed: %s\n",
                    gpuGetErrorString(err));
            exit(1);
        }
        err = gpuSetDevice(gpu_id);
        if (err != GPU_SUCCESS) {
            fprintf(stderr, "gpuSetDevice(%d) failed: %s\n", gpu_id,
                    gpuGetErrorString(err));
            exit(1);
        }
        err = gpuGetDeviceProperties(&prop, gpu_id);
        if (err != GPU_SUCCESS) {
            fprintf(stderr, "gpuGetDeviceProperties failed: %s\n",
                    gpuGetErrorString(err));
            exit(1);
        }
        // GPU clock rate in MHz (clockRate is in kHz on both runtimes).
        clock_mhz = prop.clockRate / 1000.0;
    }

    // No copy/move
    GpuDeviceContext(const GpuDeviceContext&) = delete;
    GpuDeviceContext& operator=(const GpuDeviceContext&) = delete;
};

// Backwards-compat alias so existing call sites keep compiling.
using HipDeviceContext = GpuDeviceContext;
