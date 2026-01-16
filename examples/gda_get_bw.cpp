/**
 * gda_get_bw.cpp - GDA Get Bandwidth Benchmark
 * Based on working put_example pattern
 */

#include "gda.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

#ifdef USE_AMDGPU
#include <hip/hip_runtime.h>
#endif

// Same as put_example
constexpr int NUM_OPS = 20;
constexpr int SKIP = 5;
constexpr int ITERATIONS = 20;

const std::vector<size_t> MESSAGE_SIZES = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304
};

#ifdef USE_AMDGPU

// Exact same kernel as put_example
__global__ void trigger_and_wait(gda_gpu_handle_t* handles, int num_ops) {
    int tid = threadIdx.x;
    if (tid < num_ops) {
        gda_gpu_trigger(handles[tid]);
        gda_gpu_wait(handles[tid]);
    }
}

#endif

int main(int argc, char** argv) {
    if (gda_init() != 0) {
        std::cerr << "Failed to init" << std::endl;
        return 1;
    }

    int rank = gda_rank();
    int npes = gda_size();

    if (npes != 2) {
        if (rank == 0) std::cerr << "Need 2 ranks" << std::endl;
        gda_finalize();
        return 1;
    }

#ifndef USE_AMDGPU
    std::cerr << "Need USE_AMDGPU" << std::endl;
    gda_finalize();
    return 1;
#else

    uint8_t* gpu_buf = (uint8_t*)gda_gpu_buf();
    size_t gpu_buf_size = gda_gpu_buf_size();

    gda_gpu_handle_t* d_handles = nullptr;
    hipMalloc(&d_handles, NUM_OPS * sizeof(gda_gpu_handle_t));

    std::vector<gda_handle_t*> handles(NUM_OPS);
    std::vector<gda_gpu_handle_t> host_gpu_handles(NUM_OPS);

    if (rank == 0) {
        std::cout << "# GDA Get Bandwidth Benchmark" << std::endl;
        std::cout << "# Window: " << NUM_OPS << ", Skip: " << SKIP << ", Iters: " << ITERATIONS << std::endl;
        std::cout << std::setw(12) << "Size" << std::setw(20) << "BW (MB/s)" << std::endl;
    }

    gda_barrier();

    for (size_t msg_size : MESSAGE_SIZES) {
        if (msg_size * NUM_OPS > gpu_buf_size / 2) {
            if (rank == 0) {
                std::cout << std::setw(12) << msg_size << std::setw(20) << "SKIP" << std::endl;
            }
            gda_barrier();
            continue;
        }

        double total_time = 0.0;

        for (int iter = 0; iter < SKIP + ITERATIONS; iter++) {
            if (rank == 0) {
                // Create handles - read from rank 1
                for (int i = 0; i < NUM_OPS; i++) {
                    uint8_t* local_ptr = gpu_buf + i * msg_size;
                    size_t remote_offset = gpu_buf_size / 2 + i * msg_size;

                    handles[i] = gda_get(local_ptr, msg_size, 1, remote_offset);
                    host_gpu_handles[i] = handles[i]->gpu;
                }
                hipDeviceSynchronize();

                hipMemcpy(d_handles, host_gpu_handles.data(),
                          NUM_OPS * sizeof(gda_gpu_handle_t), hipMemcpyHostToDevice);

                auto t_start = std::chrono::high_resolution_clock::now();

                hipLaunchKernelGGL(trigger_and_wait, dim3(1), dim3(NUM_OPS),
                                   0, 0, d_handles, NUM_OPS);
                hipDeviceSynchronize();

                for (int i = 0; i < NUM_OPS; i++) {
                    gda_wait(handles[i]);
                }

                auto t_end = std::chrono::high_resolution_clock::now();

                if (iter >= SKIP) {
                    total_time += std::chrono::duration<double>(t_end - t_start).count();
                }

                gda_flush();
                for (int i = 0; i < NUM_OPS; i++) {
                    gda_free(handles[i]);
                }
            }

            gda_barrier();
        }

        if (rank == 0) {
            double bw = (double)(msg_size * NUM_OPS * ITERATIONS) / total_time / 1e6;
            std::cout << std::setw(12) << msg_size
                      << std::setw(20) << std::fixed << std::setprecision(2) << bw << std::endl;
        }

        gda_barrier();
    }

    hipFree(d_handles);
#endif

    gda_finalize();
    return 0;
}
