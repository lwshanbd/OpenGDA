/**
 * test_example.cpp - Non-blocking completion detection API
 *
 * Demonstrates gda_test/gda_test_any/gda_test_all:
 * - gda_test()     - Test single operation completion
 * - gda_test_any() - Test if any operation completed
 * - gda_test_all() - Test if all operations completed
 *
 * Run:
 *   srun -n 2 ./test_example
 */

#include "gda.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>

#ifdef USE_AMDGPU
#include <hip/hip_runtime.h>
#endif

constexpr int NUM_OPS = 4;
constexpr size_t DATA_SIZE = 4096;

#ifdef USE_AMDGPU

__global__ void init_data(uint8_t* buf, size_t size, int pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = (uint8_t)((pattern + idx) & 0xFF);
    }
}

__global__ void trigger_ops(gda_gpu_handle_t* handles, int num_ops) {
    int tid = threadIdx.x;
    if (tid < num_ops) {
        gda_gpu_trigger(handles[tid]);
        gda_gpu_wait(handles[tid]);
    }
}

__global__ void verify_data(uint8_t* buf, size_t size, int pattern, int* errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint8_t expected = (uint8_t)((pattern + idx) & 0xFF);
        if (buf[idx] != expected) {
            atomicAdd(errors, 1);
        }
    }
}

#endif

void test_gda_test_single(int rank, uint8_t* gpu_buf, size_t gpu_buf_size) {
    std::cout << "\n--- Test 1: gda_test() single operation ---" << std::endl;

    if (rank == 0) {
        uint8_t* local = gpu_buf;
        size_t remote_offset = gpu_buf_size / 2;

        hipLaunchKernelGGL(init_data, dim3(16), dim3(256), 0, 0,
                           local, DATA_SIZE, 0x42);
        hipDeviceSynchronize();

        gda_handle_t* h = gda_put(local, DATA_SIZE, 1, remote_offset);
        if (!h) {
            std::cerr << "Failed to create put" << std::endl;
            return;
        }

        // Copy handle to GPU and trigger
        gda_gpu_handle_t* d_handle;
        hipMalloc(&d_handle, sizeof(gda_gpu_handle_t));
        hipMemcpy(d_handle, &h->gpu, sizeof(gda_gpu_handle_t), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(trigger_ops, dim3(1), dim3(1), 0, 0, d_handle, 1);
        hipDeviceSynchronize();

        // Use gda_test() to poll for completion
        int poll_count = 0;
        while (gda_test(h) == 0) {
            poll_count++;
            // Busy wait
        }
        std::cout << "Rank 0: gda_test() returned completed after "
                  << poll_count << " polls" << std::endl;

        gda_flush();
        gda_free(h);
        hipFree(d_handle);
    }

    gda_barrier();

    if (rank == 1) {
        uint8_t* recv = gpu_buf + gpu_buf_size / 2;
        int* d_errors;
        hipMalloc(&d_errors, sizeof(int));
        hipMemset(d_errors, 0, sizeof(int));

        hipLaunchKernelGGL(verify_data, dim3(16), dim3(256), 0, 0,
                           recv, DATA_SIZE, 0x42, d_errors);
        hipDeviceSynchronize();

        int errors = 0;
        hipMemcpy(&errors, d_errors, sizeof(int), hipMemcpyDeviceToHost);
        hipFree(d_errors);

        if (errors == 0) {
            std::cout << "Rank 1: Test 1 PASSED" << std::endl;
        } else {
            std::cout << "Rank 1: Test 1 FAILED (" << errors << " errors)" << std::endl;
        }
    }

    gda_barrier();
}

void test_gda_test_any(int rank, uint8_t* gpu_buf, size_t gpu_buf_size) {
    std::cout << "\n--- Test 2: gda_test_any() multiple operations ---" << std::endl;

    if (rank == 0) {
        std::vector<gda_handle_t*> handles(NUM_OPS);
        std::vector<gda_gpu_handle_t> gpu_handles(NUM_OPS);

        // Create multiple put operations
        for (int i = 0; i < NUM_OPS; i++) {
            uint8_t* local = gpu_buf + i * DATA_SIZE;
            size_t remote_offset = gpu_buf_size / 2 + i * DATA_SIZE;

            hipLaunchKernelGGL(init_data, dim3(16), dim3(256), 0, 0,
                               local, DATA_SIZE, 0x10 + i);

            handles[i] = gda_put(local, DATA_SIZE, 1, remote_offset);
            if (!handles[i]) {
                std::cerr << "Failed to create put " << i << std::endl;
                return;
            }
            gpu_handles[i] = handles[i]->gpu;
        }
        hipDeviceSynchronize();

        // Trigger all operations
        gda_gpu_handle_t* d_handles;
        hipMalloc(&d_handles, NUM_OPS * sizeof(gda_gpu_handle_t));
        hipMemcpy(d_handles, gpu_handles.data(),
                  NUM_OPS * sizeof(gda_gpu_handle_t), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(trigger_ops, dim3(1), dim3(NUM_OPS), 0, 0,
                           d_handles, NUM_OPS);
        hipDeviceSynchronize();

        // Use gda_test_any() to detect completions
        int completed_count = 0;
        std::vector<bool> done(NUM_OPS, false);

        while (completed_count < NUM_OPS) {
            // Build array of pending handles
            std::vector<gda_handle_t*> pending;
            std::vector<int> pending_idx;
            for (int i = 0; i < NUM_OPS; i++) {
                if (!done[i]) {
                    pending.push_back(handles[i]);
                    pending_idx.push_back(i);
                }
            }

            int idx = -1;
            int ret = gda_test_any(pending.data(), pending.size(), &idx);
            if (ret == 1) {
                int orig_idx = pending_idx[idx];
                done[orig_idx] = true;
                completed_count++;
                std::cout << "Rank 0: gda_test_any() found op " << orig_idx
                          << " completed (" << completed_count << "/" << NUM_OPS << ")"
                          << std::endl;
            }
        }

        gda_flush();
        for (auto h : handles) {
            gda_free(h);
        }
        hipFree(d_handles);
    }

    gda_barrier();

    if (rank == 1) {
        int total_errors = 0;
        for (int i = 0; i < NUM_OPS; i++) {
            uint8_t* recv = gpu_buf + gpu_buf_size / 2 + i * DATA_SIZE;
            int* d_errors;
            hipMalloc(&d_errors, sizeof(int));
            hipMemset(d_errors, 0, sizeof(int));

            hipLaunchKernelGGL(verify_data, dim3(16), dim3(256), 0, 0,
                               recv, DATA_SIZE, 0x10 + i, d_errors);
            hipDeviceSynchronize();

            int errors = 0;
            hipMemcpy(&errors, d_errors, sizeof(int), hipMemcpyDeviceToHost);
            hipFree(d_errors);
            total_errors += errors;
        }

        if (total_errors == 0) {
            std::cout << "Rank 1: Test 2 PASSED" << std::endl;
        } else {
            std::cout << "Rank 1: Test 2 FAILED (" << total_errors << " errors)" << std::endl;
        }
    }

    gda_barrier();
}

void test_gda_test_all(int rank, uint8_t* gpu_buf, size_t gpu_buf_size) {
    std::cout << "\n--- Test 3: gda_test_all() batch completion ---" << std::endl;

    if (rank == 0) {
        std::vector<gda_handle_t*> handles(NUM_OPS);
        std::vector<gda_gpu_handle_t> gpu_handles(NUM_OPS);

        // Create multiple put operations
        for (int i = 0; i < NUM_OPS; i++) {
            uint8_t* local = gpu_buf + i * DATA_SIZE;
            size_t remote_offset = gpu_buf_size / 2 + NUM_OPS * DATA_SIZE + i * DATA_SIZE;

            hipLaunchKernelGGL(init_data, dim3(16), dim3(256), 0, 0,
                               local, DATA_SIZE, 0x20 + i);

            handles[i] = gda_put(local, DATA_SIZE, 1, remote_offset);
            if (!handles[i]) {
                std::cerr << "Failed to create put " << i << std::endl;
                return;
            }
            gpu_handles[i] = handles[i]->gpu;
        }
        hipDeviceSynchronize();

        // Trigger all operations
        gda_gpu_handle_t* d_handles;
        hipMalloc(&d_handles, NUM_OPS * sizeof(gda_gpu_handle_t));
        hipMemcpy(d_handles, gpu_handles.data(),
                  NUM_OPS * sizeof(gda_gpu_handle_t), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(trigger_ops, dim3(1), dim3(NUM_OPS), 0, 0,
                           d_handles, NUM_OPS);
        hipDeviceSynchronize();

        // Use gda_test_all() to wait for all completions
        int poll_count = 0;
        while (gda_test_all(handles.data(), NUM_OPS) == 0) {
            poll_count++;
        }
        std::cout << "Rank 0: gda_test_all() returned all completed after "
                  << poll_count << " polls" << std::endl;

        gda_flush();
        for (auto h : handles) {
            gda_free(h);
        }
        hipFree(d_handles);
    }

    gda_barrier();

    if (rank == 1) {
        int total_errors = 0;
        for (int i = 0; i < NUM_OPS; i++) {
            uint8_t* recv = gpu_buf + gpu_buf_size / 2 + NUM_OPS * DATA_SIZE + i * DATA_SIZE;
            int* d_errors;
            hipMalloc(&d_errors, sizeof(int));
            hipMemset(d_errors, 0, sizeof(int));

            hipLaunchKernelGGL(verify_data, dim3(16), dim3(256), 0, 0,
                               recv, DATA_SIZE, 0x20 + i, d_errors);
            hipDeviceSynchronize();

            int errors = 0;
            hipMemcpy(&errors, d_errors, sizeof(int), hipMemcpyDeviceToHost);
            hipFree(d_errors);
            total_errors += errors;
        }

        if (total_errors == 0) {
            std::cout << "Rank 1: Test 3 PASSED" << std::endl;
        } else {
            std::cout << "Rank 1: Test 3 FAILED (" << total_errors << " errors)" << std::endl;
        }
    }

    gda_barrier();
}

int main(int argc, char** argv) {
    std::cout << "=== Test API Example (gda_test/gda_test_any/gda_test_all) ===" << std::endl;

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

    uint8_t* gpu_buf = (uint8_t*)gda_gpu_buf();
    size_t gpu_buf_size = gda_gpu_buf_size();
    if (!gpu_buf) {
        std::cerr << "Rank " << rank << ": No GPU buffer" << std::endl;
        gda_finalize();
        return 1;
    }
    std::cout << "Rank " << rank << ": GPU buffer " << (void*)gpu_buf
              << " (" << gpu_buf_size << " bytes)" << std::endl;

    // Clear buffer
    hipMemset(gpu_buf, 0, gpu_buf_size);
    hipDeviceSynchronize();
    gda_barrier();

    // Run tests
    test_gda_test_single(rank, gpu_buf, gpu_buf_size);
    test_gda_test_any(rank, gpu_buf, gpu_buf_size);
    test_gda_test_all(rank, gpu_buf, gpu_buf_size);

    gda_barrier();
    if (rank == 0) {
        std::cout << "\n*** All tests completed! ***" << std::endl;
    }

#endif

    gda_finalize();
    return 0;
}
