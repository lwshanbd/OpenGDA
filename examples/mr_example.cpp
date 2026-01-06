/**
 * mr_example.cpp - Memory registration API
 *
 * Demonstrates how to register and use memory regions
 * for both host and GPU memory.
 *
 * Run:
 *   srun -n 1 ./mr_example
 */

#include <gda.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_AMDGPU
#include <hip/hip_runtime.h>
#endif

int main(void) {
    // Initialize OpenGDA
    if (gda_init() != 0) {
        fprintf(stderr, "Failed to initialize OpenGDA\n");
        return 1;
    }

    printf("OpenGDA version: %s\n", gda_get_version());

    // Example 1: Register host memory
    size_t host_size = 4096;  // 4KB
    void* host_buf = malloc(host_size);
    if (!host_buf) {
        fprintf(stderr, "Failed to allocate host memory\n");
        gda_finalize();
        return 1;
    }

    gda_mr_t* host_mr = gda_register_memory(host_buf, host_size, 0);
    if (!host_mr) {
        fprintf(stderr, "Failed to register host memory\n");
        free(host_buf);
        gda_finalize();
        return 1;
    }

    uint64_t host_key = gda_mr_get_key(host_mr);
    void* host_desc = gda_mr_get_desc(host_mr);
    printf("Host memory registered: buf=%p, size=%zu, key=0x%lx, desc=%p\n",
           host_buf, host_size, host_key, host_desc);

#ifdef USE_AMDGPU
    // Example 2: Register GPU memory
    size_t gpu_size = 1024 * 1024;  // 1MB
    void* gpu_buf = NULL;
    hipError_t hip_err = hipMalloc(&gpu_buf, gpu_size);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory: %s\n",
                hipGetErrorString(hip_err));
        gda_deregister_memory(host_mr);
        free(host_buf);
        gda_finalize();
        return 1;
    }

    gda_mr_t* gpu_mr = gda_register_memory(gpu_buf, gpu_size, 1);
    if (!gpu_mr) {
        fprintf(stderr, "Failed to register GPU memory\n");
        hipFree(gpu_buf);
        gda_deregister_memory(host_mr);
        free(host_buf);
        gda_finalize();
        return 1;
    }

    uint64_t gpu_key = gda_mr_get_key(gpu_mr);
    void* gpu_desc = gda_mr_get_desc(gpu_mr);
    printf("GPU memory registered: buf=%p, size=%zu, key=0x%lx, desc=%p\n",
           gpu_buf, gpu_size, gpu_key, gpu_desc);

    // Cleanup GPU resources
    gda_deregister_memory(gpu_mr);
    hipFree(gpu_buf);
#endif

    // Cleanup host resources
    gda_deregister_memory(host_mr);
    free(host_buf);

    // Finalize OpenGDA
    if (gda_finalize() != 0) {
        fprintf(stderr, "Failed to finalize OpenGDA\n");
        return 1;
    }

    printf("OpenGDA memory registration example completed successfully\n");
    return 0;
}