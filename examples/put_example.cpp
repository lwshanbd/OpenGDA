/**
 * put_example.cpp - GPU-triggered RDMA put/get operations
 *
 * Demonstrates the simplified OpenGDA API for GPU-triggered RDMA.
 *
 * What this does:
 * - Rank 0 sends 4KB data to Rank 1 using GPU-triggered RDMA
 * - 10 iterations, each with NUM_OPS concurrent puts
 *
 * Run:
 *   srun -n 2 ./put_example
 */

#include "gda.h"
#include <iostream>
#include <vector>
#include <cstring>

#ifdef USE_AMDGPU
#include <hip/hip_runtime.h>
#endif

// Configuration
constexpr int NUM_OPS = 20;           // Concurrent operations per iteration
constexpr int NUM_ITERATIONS = 10;    // Number of iterations
constexpr size_t DATA_SIZE = 4096;    // 4KB per operation

#ifdef USE_AMDGPU

/**
 * GPU kernel: Initialize data buffer
 */
__global__ void init_data(uint8_t* buf, size_t size, int op_id, int iter) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = (uint8_t)((op_id ^ iter ^ (idx & 0xFF)) & 0xFF);
    }
}

/**
 * GPU kernel: Trigger operations and wait for completion
 * Each thread handles one operation
 */
__global__ void trigger_and_wait(gda_gpu_handle_t* handles, int num_ops) {
    int tid = threadIdx.x;
    if (tid < num_ops) {
        // Trigger the RDMA operation
        gda_gpu_trigger(handles[tid]);

        // Wait for completion
        gda_gpu_wait(handles[tid]);
    }
}

/**
 * GPU kernel: Verify received data
 */
__global__ void verify_data(uint8_t* buf, size_t size_per_op, int num_ops,
                            int iter, int* error_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = size_per_op * num_ops;
    if (idx < total) {
        int op_id = idx / size_per_op;
        size_t local_idx = idx % size_per_op;
        uint8_t expected = (uint8_t)((op_id ^ iter ^ (local_idx & 0xFF)) & 0xFF);
        if (buf[idx] != expected) {
            atomicAdd(error_count, 1);
        }
    }
}

#endif // USE_AMDGPU

int main(int argc, char** argv) {
    std::cout << "=== Simple Put Example (New API) ===" << std::endl;

    // ========================================================================
    // Step 1: Initialize OpenGDA
    // ========================================================================

    if (gda_init() != 0) {
        std::cerr << "Failed to initialize OpenGDA" << std::endl;
        return 1;
    }

    int rank = gda_rank();
    int size = gda_size();
    std::cout << "Rank " << rank << "/" << size << " initialized" << std::endl;

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "This example requires exactly 2 ranks" << std::endl;
        }
        gda_finalize();
        return 1;
    }

#ifndef USE_AMDGPU
    if (rank == 0) {
        std::cerr << "This example requires USE_AMDGPU" << std::endl;
    }
    gda_finalize();
    return 1;
#else

    // ========================================================================
    // Step 2: Get the default GPU buffer (already registered!)
    // ========================================================================

    uint8_t* gpu_buf = (uint8_t*)gda_gpu_buf();
    size_t gpu_buf_size = gda_gpu_buf_size();
    if (!gpu_buf) {
        std::cerr << "Rank " << rank << ": No GPU buffer available" << std::endl;
        gda_finalize();
        return 1;
    }
    std::cout << "Rank " << rank << ": GPU buffer at " << (void*)gpu_buf
              << " (" << gpu_buf_size << " bytes)" << std::endl;

    // Clear buffer
    hipMemset(gpu_buf, 0, NUM_OPS * DATA_SIZE * NUM_ITERATIONS * 2);
    hipDeviceSynchronize();

    // Allocate GPU memory for handles and error count
    gda_gpu_handle_t* d_handles = nullptr;
    int* d_errors = nullptr;
    hipMalloc(&d_handles, NUM_OPS * sizeof(gda_gpu_handle_t));
    hipMalloc(&d_errors, sizeof(int));

    // ========================================================================
    // Step 3: Create reusable handles (just once!)
    // ========================================================================

    std::vector<gda_handle_t*> handles(NUM_OPS);
    std::vector<gda_gpu_handle_t> host_gpu_handles(NUM_OPS);

    gda_barrier();
    int total_errors = 0;

    // ========================================================================
    // Step 4: Run iterations
    // ========================================================================

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        if (rank == 0) {
            // Sender: create put operations
            size_t iter_offset = iter * NUM_OPS * DATA_SIZE;

            for (int i = 0; i < NUM_OPS; i++) {
                uint8_t* local_ptr = gpu_buf + i * DATA_SIZE;
                size_t remote_offset = gpu_buf_size / 2 + iter_offset + i * DATA_SIZE;

                // Initialize data on GPU
                int blocks = (DATA_SIZE + 255) / 256;
                hipLaunchKernelGGL(init_data, dim3(blocks), dim3(256), 0, 0,
                                   local_ptr, DATA_SIZE, i, iter);

                // Create put operation - THIS IS THE NEW SIMPLE API!
                handles[i] = gda_put(local_ptr, DATA_SIZE, 1, remote_offset);
                if (!handles[i]) {
                    std::cerr << "Failed to create put " << i << std::endl;
                    break;
                }

                // Copy GPU handle for kernel
                host_gpu_handles[i] = handles[i]->gpu;
            }
            hipDeviceSynchronize();

            // Copy handles to GPU
            hipMemcpy(d_handles, host_gpu_handles.data(),
                      NUM_OPS * sizeof(gda_gpu_handle_t), hipMemcpyHostToDevice);

            // Launch kernel to trigger all operations
            hipLaunchKernelGGL(trigger_and_wait, dim3(1), dim3(NUM_OPS),
                               0, 0, d_handles, NUM_OPS);
            hipDeviceSynchronize();

            // Wait for all operations (CPU side)
            for (int i = 0; i < NUM_OPS; i++) {
                gda_wait(handles[i]);
            }

            std::cout << "Rank 0: Iteration " << iter << " - sent "
                      << NUM_OPS << " puts" << std::endl;

            // Flush BEFORE freeing handles
            gda_flush();

            // Now free handles
            for (int i = 0; i < NUM_OPS; i++) {
                gda_free(handles[i]);
            }
        }

        gda_barrier();

        // Receiver: verify data
        if (rank == 1) {
            uint8_t* recv_buf = gpu_buf + gpu_buf_size / 2;
            size_t iter_offset = iter * NUM_OPS * DATA_SIZE;

            hipMemset(d_errors, 0, sizeof(int));
            int blocks = (NUM_OPS * DATA_SIZE + 255) / 256;
            hipLaunchKernelGGL(verify_data, dim3(blocks), dim3(256), 0, 0,
                               recv_buf + iter_offset, DATA_SIZE, NUM_OPS,
                               iter, d_errors);
            hipDeviceSynchronize();

            int errors = 0;
            hipMemcpy(&errors, d_errors, sizeof(int), hipMemcpyDeviceToHost);

            if (errors > 0) {
                std::cout << "Rank 1: Iteration " << iter << " FAILED - "
                          << errors << " errors" << std::endl;
                total_errors += errors;
            } else {
                std::cout << "Rank 1: Iteration " << iter << " OK" << std::endl;
            }
        }

        gda_barrier();
    }

    // ========================================================================
    // Step 5: Summary
    // ========================================================================

    gda_barrier();
    if (rank == 1) {
        if (total_errors == 0) {
            std::cout << "\n*** SUCCESS: All data verified! ***" << std::endl;
        } else {
            std::cout << "\n*** FAILED: " << total_errors << " errors ***" << std::endl;
        }
    }

    // Cleanup
    hipFree(d_handles);
    hipFree(d_errors);

#endif // USE_AMDGPU

    gda_finalize();
    std::cout << "Rank " << rank << ": Done!" << std::endl;
    return (rank == 1 && total_errors > 0) ? 1 : 0;
}
