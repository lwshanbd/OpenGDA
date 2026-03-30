/**
 * cuda_device_context.hpp - CUDA GPU context with RAII
 *
 * Initializes CUDA device and provides GPU properties.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace gicc {

class CudaDeviceContext {
public:
    int gpu_id;
    int clock_mhz;
    cudaDeviceProp props;

    explicit CudaDeviceContext(int device_id = 0) : gpu_id(device_id), clock_mhz(0) {
        cudaError_t err = cudaSetDevice(gpu_id);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice(%d) failed: %s\n",
                    gpu_id, cudaGetErrorString(err));
            exit(1);
        }

        err = cudaGetDeviceProperties(&props, gpu_id);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed: %s\n",
                    cudaGetErrorString(err));
            exit(1);
        }

        clock_mhz = props.clockRate / 1000;  // Convert kHz to MHz
    }

    ~CudaDeviceContext() {
        // CUDA device context is automatically managed
    }

    // No copy/move
    CudaDeviceContext(const CudaDeviceContext&) = delete;
    CudaDeviceContext& operator=(const CudaDeviceContext&) = delete;

    // Check if peer access is possible
    bool can_access_peer(int peer_device) const {
        int can_access = 0;
        cudaDeviceCanAccessPeer(&can_access, gpu_id, peer_device);
        return can_access != 0;
    }

    // Enable peer access to another GPU
    void enable_peer_access(int peer_device) {
        cudaError_t err = cudaDeviceEnablePeerAccess(peer_device, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            fprintf(stderr, "cudaDeviceEnablePeerAccess(%d) failed: %s\n",
                    peer_device, cudaGetErrorString(err));
        }
    }
};

} // namespace gicc
