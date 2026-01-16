/*
 * benchmark_runner.hpp - DWQ benchmark runner with GPU kernel
 */
#pragma once

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include "pmi_session.hpp"
#include "fabric_dwq_context.hpp"
#include "memory_region.hpp"
#include "dwq_work_builder.hpp"

// =============================================================================
// GPU Kernel
// =============================================================================

// GPU kernel to write counter doorbell (trigger NIC operation)
// Polls atomic_result until NIC completes the atomic write
// Records timing inside GPU for accurate latency measurement
__global__ void gpu_write_counter_doorbell(volatile uint64_t* counter_addr,
                                           volatile uint64_t* atomic_result,
                                           uint64_t value,
                                           uint64_t* start_clock,
                                           uint64_t* end_clock) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Record start time (GPU clock cycles)
        uint64_t t_start = clock64();

        // Step 1: Write to trigger counter to initiate RDMA write
        *counter_addr = value;

        // Step 2: Poll atomic_result until NIC writes to it
        do {
            // __threadfence_system();
        } while (*atomic_result < 1);

        // Record end time (GPU clock cycles)
        uint64_t t_end = clock64();
        __threadfence_system();

        // Store timestamps
        if (start_clock) *start_clock = t_start;
        if (end_clock) *end_clock = t_end;
    }
    __syncthreads();
}

// =============================================================================
// Utility Functions
// =============================================================================

inline const char* format_size(size_t size, char* buf) {
    if (size < 1024) {
        snprintf(buf, 32, "%zuB", size);
    } else if (size < 1024 * 1024) {
        snprintf(buf, 32, "%zuKB", size / 1024);
    } else {
        snprintf(buf, 32, "%zuMB", size / (1024 * 1024));
    }
    return buf;
}

inline void bytes_to_hex(const uint8_t* in, size_t len, char* out) {
    static const char* h = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        out[2 * i] = h[(in[i] >> 4) & 0xF];
        out[2 * i + 1] = h[in[i] & 0xF];
    }
    out[2 * len] = '\0';
}

inline int hexval(char c) {
    if ('0' <= c && c <= '9') return c - '0';
    if ('a' <= c && c <= 'f') return c - 'a' + 10;
    if ('A' <= c && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline int hex_to_bytes(const char* in, uint8_t* out, size_t outlen) {
    size_t n = strlen(in);
    if (n % 2 != 0 || outlen < n / 2) return -1;
    for (size_t i = 0; i < n; i += 2) {
        int hi = hexval(in[i]);
        int lo = hexval(in[i + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[i / 2] = (uint8_t)((hi << 4) | lo);
    }
    return (int)(n / 2);
}

// =============================================================================
// Benchmark Configuration
// =============================================================================

constexpr int NUM_ITERATIONS = 20;

constexpr size_t TEST_SIZES[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024,
    2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024
};
constexpr int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);
constexpr size_t MAX_SIZE = 128 * 1024 * 1024;  // 128MB

// =============================================================================
// Benchmark Runner
// =============================================================================

class BenchmarkRunner {
public:
    PmiSession& pmi;
    FabricDwqContext& fabric;
    DwqWorkBuilder& dwq;

    // GPU buffers
    void* d_local_buf;
    void* d_remote_buf;
    uint64_t* atomic_result;
    uint64_t* atomic_operand;
    uint64_t* d_start_clock;
    uint64_t* d_end_clock;

    // Host verification buffer
    uint8_t* h_verify_buf;

    // Memory regions
    MemoryRegion* mr_local;
    MemoryRegion* mr_remote;
    MemoryRegion* mr_atomic;
    MemoryRegion* mr_atomic_operand;

    // Peer RMA info
    uint64_t peer_remote_addr;
    uint64_t peer_remote_key;
    uint64_t remote_addr_for_rma;

    // Statistics
    int total_verifications;
    int total_verification_failures;

    BenchmarkRunner(PmiSession& pmi_, FabricDwqContext& fabric_, DwqWorkBuilder& dwq_)
        : pmi(pmi_), fabric(fabric_), dwq(dwq_),
          d_local_buf(nullptr), d_remote_buf(nullptr),
          atomic_result(nullptr), atomic_operand(nullptr),
          d_start_clock(nullptr), d_end_clock(nullptr),
          h_verify_buf(nullptr),
          mr_local(nullptr), mr_remote(nullptr),
          mr_atomic(nullptr), mr_atomic_operand(nullptr),
          peer_remote_addr(0), peer_remote_key(0), remote_addr_for_rma(0),
          total_verifications(0), total_verification_failures(0)
    {
        allocate_buffers();
        register_memory();
        exchange_addresses();
        exchange_rma_info();
    }

    ~BenchmarkRunner() {
        delete mr_local;
        delete mr_remote;
        delete mr_atomic;
        delete mr_atomic_operand;

        if (d_local_buf) hipFree(d_local_buf);
        if (d_remote_buf) hipFree(d_remote_buf);
        if (atomic_result) hipFree(atomic_result);
        if (atomic_operand) hipFree(atomic_operand);
        if (d_start_clock) hipFree(d_start_clock);
        if (d_end_clock) hipFree(d_end_clock);
        if (h_verify_buf) free(h_verify_buf);
    }

    // No copy/move
    BenchmarkRunner(const BenchmarkRunner&) = delete;
    BenchmarkRunner& operator=(const BenchmarkRunner&) = delete;

    void run() {
        pmi.barrier();
        usleep(100000);  // 100ms for readiness

        if (pmi.rank == 0) {
            printf("%-8s  %12s  %s\n", "Size", "Latency(us)", "Statistics");
            printf("========  ============  ===============================================\n");
            printf("Note: Latency is average of best 10/%d iterations\n", NUM_ITERATIONS);
            printf("      GPU-Direct Async (GDA) with Deferred Work Queue (DWQ)\n\n");
        }

        for (int size_idx = 0; size_idx < NUM_TEST_SIZES; size_idx++) {
            run_size_test(TEST_SIZES[size_idx]);
        }

        // Final verification report
        if (pmi.rank == 1 && total_verification_failures > 0) {
            fprintf(stderr, "\nVerification: %d/%d failed\n",
                    total_verification_failures, total_verifications);
        }
    }

private:
    void check_hip(hipError_t err, const char* msg) {
        if (err != hipSuccess) {
            fprintf(stderr, "Rank %d: %s failed: %s (%d)\n",
                    pmi.rank, msg, hipGetErrorString(err), err);
            exit(1);
        }
    }

    void allocate_buffers() {
        check_hip(hipMalloc(&d_local_buf, MAX_SIZE), "hipMalloc(local)");
        check_hip(hipMalloc(&d_remote_buf, MAX_SIZE), "hipMalloc(remote)");
        check_hip(hipMalloc(&atomic_result, sizeof(uint64_t)), "hipMalloc(atomic_result)");
        check_hip(hipMalloc(&atomic_operand, sizeof(uint64_t)), "hipMalloc(atomic_operand)");
        check_hip(hipMalloc(&d_start_clock, sizeof(uint64_t)), "hipMalloc(start_clock)");
        check_hip(hipMalloc(&d_end_clock, sizeof(uint64_t)), "hipMalloc(end_clock)");

        check_hip(hipMemset(atomic_result, 0, sizeof(uint64_t)), "hipMemset(atomic_result)");
        uint64_t operand_value = 1;
        check_hip(hipMemcpy(atomic_operand, &operand_value, sizeof(uint64_t),
                            hipMemcpyHostToDevice), "hipMemcpy(atomic_operand)");
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

        h_verify_buf = (uint8_t*)malloc(MAX_SIZE);
        if (!h_verify_buf) {
            fprintf(stderr, "Rank %d: malloc(h_verify_buf) failed\n", pmi.rank);
            exit(1);
        }
    }

    void register_memory() {
        mr_local = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                    d_local_buf, MAX_SIZE, true, 0, pmi.rank);
        mr_remote = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                     d_remote_buf, MAX_SIZE, true, 0, pmi.rank);
        mr_atomic = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                     atomic_result, sizeof(uint64_t), true, 0, pmi.rank);
        mr_atomic_operand = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                              atomic_operand, sizeof(uint64_t), true, 0, pmi.rank);
    }

    void exchange_addresses() {
        int peer = (pmi.rank == 0) ? 1 : 0;

        // Encode local address
        char* my_hex = (char*)malloc(2 * fabric.addrlen + 1);
        bytes_to_hex((uint8_t*)fabric.local_addr, fabric.addrlen, my_hex);

        // Exchange via KVS
        char* all_hex = (char*)malloc(pmi.size * (2 * fabric.addrlen + 1));
        char key[PMI2_MAX_KEYLEN];
        snprintf(key, sizeof(key), "addr-%d", pmi.rank);
        pmi.kvs_put(key, my_hex);
        pmi.barrier();

        for (int i = 0; i < pmi.size; i++) {
            snprintf(key, sizeof(key), "addr-%d", i);
            char val[PMI2_MAX_VALLEN];
            pmi.kvs_get(key, val, sizeof(val));
            char* slot = all_hex + i * (2 * fabric.addrlen + 1);
            strncpy(slot, val, 2 * fabric.addrlen);
            slot[2 * fabric.addrlen] = '\0';
        }

        // Decode and insert addresses
        uint8_t* all_bin = (uint8_t*)malloc(pmi.size * fabric.addrlen);
        for (int i = 0; i < pmi.size; i++) {
            const char* hex = all_hex + i * (2 * fabric.addrlen + 1);
            if (hex_to_bytes(hex, all_bin + i * fabric.addrlen, fabric.addrlen) <= 0) {
                fprintf(stderr, "Rank %d: hex_to_bytes failed\n", pmi.rank);
                exit(1);
            }
        }

        // Insert peer address
        void* peer_addr_bin = all_bin + peer * fabric.addrlen;
        int inserted = fi_av_insert(fabric.av, peer_addr_bin, 1, &fabric.peer_addr, 0, NULL);
        if (inserted != 1 || fabric.peer_addr == FI_ADDR_NOTAVAIL) {
            fprintf(stderr, "Rank %d: fi_av_insert(peer) failed\n", pmi.rank);
            exit(1);
        }

        // Insert local address for self-atomics
        inserted = fi_av_insert(fabric.av, fabric.local_addr, 1, &fabric.local_addr_in_av, 0, NULL);
        if (inserted != 1 || fabric.local_addr_in_av == FI_ADDR_NOTAVAIL) {
            fprintf(stderr, "Rank %d: fi_av_insert(local) failed\n", pmi.rank);
            exit(1);
        }

        free(my_hex);
        free(all_hex);
        free(all_bin);
    }

    void exchange_rma_info() {
        int peer = (pmi.rank == 0) ? 1 : 0;

        struct {
            uint64_t addr;
            uint64_t key;
        } my_rma_info, peer_rma_info;

        my_rma_info.addr = (uint64_t)d_remote_buf;
        my_rma_info.key = mr_remote->key;

        char rma_hex[128];
        bytes_to_hex((uint8_t*)&my_rma_info, sizeof(my_rma_info), rma_hex);

        char key_str[PMI2_MAX_KEYLEN];
        snprintf(key_str, sizeof(key_str), "rma-%d", pmi.rank);
        pmi.kvs_put(key_str, rma_hex);
        pmi.barrier();

        snprintf(key_str, sizeof(key_str), "rma-%d", peer);
        char peer_rma_hex[PMI2_MAX_VALLEN];
        pmi.kvs_get(key_str, peer_rma_hex, sizeof(peer_rma_hex));

        if (hex_to_bytes(peer_rma_hex, (uint8_t*)&peer_rma_info, sizeof(peer_rma_info)) <= 0) {
            fprintf(stderr, "Rank %d: hex_to_bytes(rma) failed\n", pmi.rank);
            exit(1);
        }

        peer_remote_addr = peer_rma_info.addr;
        peer_remote_key = peer_rma_info.key;

        // Determine remote address mode
        if (fabric.is_virt_addr_mode()) {
            remote_addr_for_rma = peer_remote_addr;
        } else {
            remote_addr_for_rma = 0;  // Use offset from MR base
        }
    }

    void run_size_test(size_t current_size) {
        pmi.barrier();

        double iteration_times[NUM_ITERATIONS];
        int successful_iterations = 0;

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Reset counters
            fi_cntr_set(fabric.trigger_cntr, 0);
            fi_cntr_set(fabric.completion_cntr, 0);
            fi_cntr_set(fabric.atomic_completion_cntr, 0);

            // Reset atomic_result on rank 0
            if (pmi.rank == 0) {
                uint64_t zero = 0;
                check_hip(hipMemcpy(atomic_result, &zero, sizeof(uint64_t),
                                    hipMemcpyHostToDevice), "reset atomic_result");
                check_hip(hipDeviceSynchronize(), "sync reset");
            }

            // Initialize buffers
            if (pmi.rank == 0) {
                uint8_t pattern = (iter + 0xA0) & 0xFF;
                check_hip(hipMemset(d_local_buf, pattern, current_size), "hipMemset(local)");
                check_hip(hipDeviceSynchronize(), "sync memset");
            } else {
                check_hip(hipMemset(d_remote_buf, 0xFF, current_size), "hipMemset(remote)");
                check_hip(hipDeviceSynchronize(), "sync memset");
            }
            pmi.barrier();

            if (pmi.rank == 0) {
                // Queue RMA write
                dwq.queue_rma_write(
                    fabric.domain, fabric.ep,
                    d_local_buf, mr_local->desc, current_size,
                    fabric.peer_addr, remote_addr_for_rma, peer_remote_key,
                    fabric.trigger_cntr, fabric.completion_cntr, 1);

                // Queue atomic signal
                uint64_t atomic_result_addr = fabric.is_virt_addr_mode()
                    ? (uint64_t)atomic_result : 0;
                dwq.queue_atomic_signal(
                    fabric.domain, fabric.ep,
                    atomic_operand, mr_atomic_operand->desc,
                    atomic_result, mr_atomic->key, atomic_result_addr,
                    fabric.local_addr_in_av,
                    fabric.completion_cntr, fabric.atomic_completion_cntr, 1);

                // Launch kernel and time
                auto t_start = std::chrono::high_resolution_clock::now();
                hipLaunchKernelGGL(gpu_write_counter_doorbell, dim3(1), dim3(1), 0, 0,
                                   fabric.dev_trigger_cntr, atomic_result, 1,
                                   d_start_clock, d_end_clock);
                check_hip(hipDeviceSynchronize(), "kernel sync");
                auto t_end = std::chrono::high_resolution_clock::now();

                double elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    t_end - t_start).count();
                iteration_times[successful_iterations++] = elapsed_us;

            }

            pmi.barrier();

            // Rank 1: Verify data
            if (pmi.rank == 1) {
                check_hip(hipMemcpy(h_verify_buf, d_remote_buf, current_size,
                                    hipMemcpyDeviceToHost), "D2H copy");

                uint8_t expected_pattern = (iter + 0xA0) & 0xFF;
                int errors = 0;
                for (size_t i = 0; i < current_size && errors < 10; i++) {
                    if (h_verify_buf[i] != expected_pattern) errors++;
                }

                total_verifications++;
                if (errors > 0) {
                    total_verification_failures++;
                    fprintf(stderr, "Rank %d: VERIFICATION FAILED - iter=%d, size=%zu, "
                            "expected=0x%02X, errors=%d\n",
                            pmi.rank, iter, current_size, expected_pattern, errors);
                }
            }
        }

        // Print results
        if (pmi.rank == 0 && successful_iterations > 0) {
            // Sort ascending
            for (int i = 0; i < successful_iterations - 1; i++) {
                for (int j = i + 1; j < successful_iterations; j++) {
                    if (iteration_times[j] < iteration_times[i]) {
                        double tmp = iteration_times[i];
                        iteration_times[i] = iteration_times[j];
                        iteration_times[j] = tmp;
                    }
                }
            }

            int samples = (successful_iterations < 10) ? successful_iterations : 10;
            double sum = 0.0;
            for (int i = 0; i < samples; i++) sum += iteration_times[i];
            double avg = sum / samples;

            char size_buf[32];
            printf("%-8s  %12.2f  (best %d/%d: min=%.2f max=%.2f)\n",
                   format_size(current_size, size_buf), avg, samples, successful_iterations,
                   iteration_times[0], iteration_times[samples - 1]);
            fflush(stdout);

            fabric.flush_dwq();
        }
    }
};
