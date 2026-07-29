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
using GpuStream_t   = hipStream_t;
using GpuEvent_t    = hipEvent_t;
using GpuIpcMemHandle_t = hipIpcMemHandle_t;
#define GPU_SUCCESS              hipSuccess
#define gpuGetDeviceCount        hipGetDeviceCount
#define gpuSetDevice             hipSetDevice
#define gpuGetDeviceProperties   hipGetDeviceProperties
#define gpuGetErrorString        hipGetErrorString

// Memory management aliases (used by Fabric and friends).
#define gpuMalloc                hipMalloc
#define gpuMallocManaged         hipMallocManaged
#define gpuFree                  hipFree
#define gpuMemcpy                hipMemcpy
#define gpuMemcpyAsync           hipMemcpyAsync
#define gpuMemcpyHostToDevice    hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice  hipMemcpyDeviceToDevice
#define gpuMemcpyDefault         hipMemcpyDefault
#define gpuMemset                hipMemset
#define gpuDeviceSynchronize     hipDeviceSynchronize
#define gpuDeviceCanAccessPeer   hipDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define gpuGetLastError          hipGetLastError
#define gpuErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled

// Host-pinned memory (mapped into the device address space).
#define gpuHostMalloc            hipHostMalloc
#define gpuHostFree              hipHostFree
#define gpuHostMallocMapped      hipHostMallocMapped
#define gpuHostMallocDefault     hipHostMallocDefault
#define gpuHostGetDevicePointer  hipHostGetDevicePointer
#define gpuHostRegister          hipHostRegister
#define gpuHostUnregister        hipHostUnregister
#define gpuHostRegisterMapped    hipHostRegisterMapped

// Flag for registering a NIC BAR page (the CXI trigger/completion counter
// doorbells) so a kernel can store to it. ROCr accepts the plain mapped flag
// for device MMIO, so this is the historical Tioga behaviour unchanged.
#define gpuHostRegisterMmio      hipHostRegisterMapped

// Streams.
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamCreate          hipStreamCreate
#define gpuStreamDestroy         hipStreamDestroy
#define gpuStreamSynchronize     hipStreamSynchronize
#define gpuStreamWaitEvent       hipStreamWaitEvent
#define gpuStreamNonBlocking     hipStreamNonBlocking

// Events.
#define gpuEventCreate           hipEventCreate
#define gpuEventCreateWithFlags  hipEventCreateWithFlags
#define gpuEventDestroy          hipEventDestroy
#define gpuEventRecord           hipEventRecord
#define gpuEventSynchronize      hipEventSynchronize
#define gpuEventDisableTiming    hipEventDisableTiming

// IPC handles (same-node peer mapping).
#define gpuIpcGetMemHandle       hipIpcGetMemHandle
#define gpuIpcOpenMemHandle      hipIpcOpenMemHandle
#define gpuIpcCloseMemHandle     hipIpcCloseMemHandle
#define gpuIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess

// Device PCI helpers.
#define gpuDeviceGetPCIBusId     hipDeviceGetPCIBusId

// Kernel launch wrapper. Expands to the exact same hipLaunchKernelGGL
// invocation under HIP, and to the CUDA triple-chevron form under CUDA.
#define gpuLaunchKernel(kernel, grid, block, shmem, stream, ...) \
    hipLaunchKernelGGL(kernel, grid, block, shmem, stream, __VA_ARGS__)

// CRITICAL: Call this BEFORE any HIP/PMI2 initialization
// Must be called at the very start of main()
inline void unset_rocr_visible_devices() { unsetenv("ROCR_VISIBLE_DEVICES"); }

#elif defined(GICC_GPU_CUDA)
#include <cuda_runtime.h>
using GpuError      = cudaError_t;
using GpuDeviceProp = cudaDeviceProp;
using GpuStream_t   = cudaStream_t;
using GpuEvent_t    = cudaEvent_t;
using GpuIpcMemHandle_t = cudaIpcMemHandle_t;
#define GPU_SUCCESS              cudaSuccess
#define gpuGetDeviceCount        cudaGetDeviceCount
#define gpuSetDevice             cudaSetDevice
#define gpuGetDeviceProperties   cudaGetDeviceProperties
#define gpuGetErrorString        cudaGetErrorString

// Memory management aliases (used by Fabric and friends).
#define gpuMalloc                cudaMalloc
#define gpuMallocManaged         cudaMallocManaged
#define gpuFree                  cudaFree
#define gpuMemcpy                cudaMemcpy
#define gpuMemcpyAsync           cudaMemcpyAsync
#define gpuMemcpyHostToDevice    cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost    cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice  cudaMemcpyDeviceToDevice
#define gpuMemcpyDefault         cudaMemcpyDefault
#define gpuMemset                cudaMemset
#define gpuDeviceSynchronize     cudaDeviceSynchronize
#define gpuDeviceCanAccessPeer   cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define gpuGetLastError          cudaGetLastError
#define gpuErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled

// Host-pinned memory (mapped into the device address space). CUDA's
// equivalent of hipHostMalloc(p, n, hipHostMallocMapped) is
// cudaHostAlloc(p, n, cudaHostAllocMapped). The flag names also differ.
#define gpuHostMalloc            cudaHostAlloc
#define gpuHostFree              cudaFreeHost
#define gpuHostMallocMapped      cudaHostAllocMapped
#define gpuHostMallocDefault     cudaHostAllocDefault
#define gpuHostGetDevicePointer  cudaHostGetDevicePointer
#define gpuHostRegister          cudaHostRegister
#define gpuHostUnregister        cudaHostUnregister
#define gpuHostRegisterMapped    cudaHostRegisterMapped

// Flag for registering a NIC BAR page (the CXI trigger/completion counter
// doorbells) so a kernel can store to it. CUDA rejects cudaHostRegisterMapped
// on I/O memory with cudaErrorInvalidValue -- device MMIO needs
// cudaHostRegisterIoMemory, the same flag NVSHMEM's IBGDA path uses for the
// mlx5 doorbell. Verified on GH200 + Slingshot: with Mapped the registration
// fails outright; with IoMemory a kernel store reaches the CXI counter.
#define gpuHostRegisterMmio      cudaHostRegisterIoMemory

// Streams.
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamCreate          cudaStreamCreate
#define gpuStreamDestroy         cudaStreamDestroy
#define gpuStreamSynchronize     cudaStreamSynchronize
#define gpuStreamWaitEvent       cudaStreamWaitEvent
#define gpuStreamNonBlocking     cudaStreamNonBlocking

// Events.
#define gpuEventCreate           cudaEventCreate
#define gpuEventCreateWithFlags  cudaEventCreateWithFlags
#define gpuEventDestroy          cudaEventDestroy
#define gpuEventRecord           cudaEventRecord
#define gpuEventSynchronize      cudaEventSynchronize
#define gpuEventDisableTiming    cudaEventDisableTiming

// IPC handles (same-node peer mapping).
#define gpuIpcGetMemHandle       cudaIpcGetMemHandle
#define gpuIpcOpenMemHandle      cudaIpcOpenMemHandle
#define gpuIpcCloseMemHandle     cudaIpcCloseMemHandle
#define gpuIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess

// Device PCI helpers.
#define gpuDeviceGetPCIBusId     cudaDeviceGetPCIBusId

// Kernel launch wrapper. CUDA uses triple-chevron syntax; nvcc parses this
// just like a normal kernel launch.
#define gpuLaunchKernel(kernel, grid, block, shmem, stream, ...) \
    kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)

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
