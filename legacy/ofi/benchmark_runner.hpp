/*
 * benchmark_runner.hpp - Concurrent DWQ benchmark with shared trigger counter
 *
 * Key insight: All streams share ONE trigger counter (single MMIO mapping)
 * Each stream has its own completion counter (no MMIO needed) and atomic_result
 * This allows many more concurrent streams than separate counters per stream.
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
// Configuration
// =============================================================================

constexpr int N_STREAMS = 32;      // Number of concurrent transfers
constexpr int NUM_ITERATIONS = 20;

constexpr size_t TEST_SIZES[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024,
    2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024
};
constexpr int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);
constexpr size_t MAX_SIZE = 16 * 1024 * 1024;  // 16MB per stream (for 32 streams)

// =============================================================================
// GPU Kernel - Single trigger, multiple completions
// =============================================================================

// All streams share ONE trigger counter, but each has its own atomic_result
// GPU writes N_STREAMS to trigger all operations at once
__global__ void gpu_shared_trigger_write(
    volatile uint64_t* shared_trigger_addr,  // Single shared trigger counter
    volatile uint64_t** atomic_results,      // Array of per-stream atomic results
    int n_streams,
    uint64_t trigger_value,                  // Value to write (N_STREAMS)
    uint64_t* start_clock,
    uint64_t* end_clock)
{
    int stream_id = threadIdx.x;

    // Thread 0 records start time and triggers ALL operations
    if (stream_id == 0) {
        *start_clock = clock64();
        // Single write triggers all N_STREAMS DWQ operations
        *shared_trigger_addr = trigger_value;
    }
    __syncthreads();

    // Each thread polls its own atomic_result
    if (stream_id < n_streams) {
        while (*atomic_results[stream_id] < 1) {
            // Spin wait
        }
    }

    __syncthreads();
    __threadfence_system();

    // Thread 0 records end time
    if (stream_id == 0) {
        *end_clock = clock64();
    }
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
// Per-stream resources (NO per-stream trigger counter!)
// =============================================================================

struct StreamResources {
    // Completion counters (no MMIO mapping needed - GPU doesn't write to these)
    struct fid_cntr* completion_cntr;
    struct fid_cntr* atomic_completion_cntr;

    // GPU buffers
    void* d_local_buf;
    uint64_t* atomic_result;
    uint64_t* atomic_operand;

    // Memory regions
    MemoryRegion* mr_local;
    MemoryRegion* mr_atomic;
    MemoryRegion* mr_atomic_operand;

    // DWQ work builder
    DwqWorkBuilder* dwq;

    StreamResources() : completion_cntr(nullptr), atomic_completion_cntr(nullptr),
                        d_local_buf(nullptr), atomic_result(nullptr),
                        atomic_operand(nullptr), mr_local(nullptr),
                        mr_atomic(nullptr), mr_atomic_operand(nullptr),
                        dwq(nullptr) {}
};

// =============================================================================
// Benchmark Runner
// =============================================================================

class BenchmarkRunner {
public:
    PmiSession& pmi;
    FabricDwqContext& fabric;

    // Per-stream resources (using shared trigger counter from fabric)
    StreamResources streams[N_STREAMS];

    // Shared remote buffer (destination on peer)
    void* d_remote_buf;
    MemoryRegion* mr_remote;

    // GPU arrays for kernel
    volatile uint64_t** d_atomic_results;
    uint64_t* d_start_clock;
    uint64_t* d_end_clock;

    // Host arrays for setup
    volatile uint64_t* h_atomic_results[N_STREAMS];

    // Host verification buffer
    uint8_t* h_verify_buf;

    // Peer RMA info
    uint64_t peer_remote_addr;
    uint64_t peer_remote_key;
    uint64_t remote_addr_for_rma;

    // Statistics
    int total_verifications;
    int total_verification_failures;

    BenchmarkRunner(PmiSession& pmi_, FabricDwqContext& fabric_)
        : pmi(pmi_), fabric(fabric_),
          d_remote_buf(nullptr), mr_remote(nullptr),
          d_atomic_results(nullptr),
          d_start_clock(nullptr), d_end_clock(nullptr),
          h_verify_buf(nullptr),
          peer_remote_addr(0), peer_remote_key(0), remote_addr_for_rma(0),
          total_verifications(0), total_verification_failures(0)
    {
        allocate_buffers();
        register_memory();
        exchange_addresses();
        exchange_rma_info();
        setup_gpu_arrays();
    }

    ~BenchmarkRunner() {
        // Free GPU arrays
        if (d_atomic_results) (void)hipFree(d_atomic_results);
        if (d_start_clock) (void)hipFree(d_start_clock);
        if (d_end_clock) (void)hipFree(d_end_clock);

        // Free per-stream resources
        for (int i = 0; i < N_STREAMS; i++) {
            delete streams[i].mr_local;
            delete streams[i].mr_atomic;
            delete streams[i].mr_atomic_operand;
            delete streams[i].dwq;
            if (streams[i].d_local_buf) (void)hipFree(streams[i].d_local_buf);
            if (streams[i].atomic_result) (void)hipFree(streams[i].atomic_result);
            if (streams[i].atomic_operand) (void)hipFree(streams[i].atomic_operand);
            if (streams[i].completion_cntr) fi_close(&streams[i].completion_cntr->fid);
            if (streams[i].atomic_completion_cntr) fi_close(&streams[i].atomic_completion_cntr->fid);
        }

        // Free shared resources
        delete mr_remote;
        if (d_remote_buf) (void)hipFree(d_remote_buf);
        if (h_verify_buf) free(h_verify_buf);
    }

    // No copy/move
    BenchmarkRunner(const BenchmarkRunner&) = delete;
    BenchmarkRunner& operator=(const BenchmarkRunner&) = delete;

    void run() {
        pmi.barrier();
        usleep(100000);

        if (pmi.rank == 0) {
            printf("%-8s  %12s  %12s  %s\n", "Size", "Total(us)", "Per-xfer(us)", "Statistics");
            printf("========  ============  ============  =====================================\n");
            printf("Note: %d concurrent streams, shared trigger counter, per-xfer = total/%d\n",
                   N_STREAMS, N_STREAMS);
            printf("      GPU-Direct Async (GDA) with Deferred Work Queue (DWQ)\n\n");
        }

        for (int size_idx = 0; size_idx < NUM_TEST_SIZES; size_idx++) {
            run_size_test(TEST_SIZES[size_idx]);
        }

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

    void check_fi(int ret, const char* msg) {
        if (ret) {
            fprintf(stderr, "Rank %d: %s failed: %s (%d)\n",
                    pmi.rank, msg, fi_strerror(-ret), ret);
            exit(1);
        }
    }

    void allocate_buffers() {
        // Shared remote buffer (destination)
        check_hip(hipMalloc(&d_remote_buf, MAX_SIZE * N_STREAMS), "hipMalloc(remote)");

        // Per-stream buffers and counters
        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;

        for (int i = 0; i < N_STREAMS; i++) {
            check_hip(hipMalloc(&streams[i].d_local_buf, MAX_SIZE), "hipMalloc(local)");
            check_hip(hipMalloc(&streams[i].atomic_result, sizeof(uint64_t)), "hipMalloc(atomic_result)");
            check_hip(hipMalloc(&streams[i].atomic_operand, sizeof(uint64_t)), "hipMalloc(atomic_operand)");

            check_hip(hipMemset(streams[i].atomic_result, 0, sizeof(uint64_t)), "hipMemset(atomic_result)");
            uint64_t operand_value = 1;
            check_hip(hipMemcpy(streams[i].atomic_operand, &operand_value, sizeof(uint64_t),
                                hipMemcpyHostToDevice), "hipMemcpy(atomic_operand)");

            // Create per-stream completion counters (NO MMIO mapping needed!)
            check_fi(fi_cntr_open(fabric.domain, &cntr_attr, &streams[i].completion_cntr, NULL),
                     "fi_cntr_open(completion)");
            check_fi(fi_cntr_open(fabric.domain, &cntr_attr, &streams[i].atomic_completion_cntr, NULL),
                     "fi_cntr_open(atomic_completion)");

            // Create DWQ work builder
            streams[i].dwq = new DwqWorkBuilder(pmi.rank);
        }

        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

        // Timing buffers
        check_hip(hipMalloc(&d_start_clock, sizeof(uint64_t)), "hipMalloc(start_clock)");
        check_hip(hipMalloc(&d_end_clock, sizeof(uint64_t)), "hipMalloc(end_clock)");

        // Host verification buffer
        h_verify_buf = (uint8_t*)malloc(MAX_SIZE * N_STREAMS);
        if (!h_verify_buf) {
            fprintf(stderr, "Rank %d: malloc(h_verify_buf) failed\n", pmi.rank);
            exit(1);
        }
    }

    void register_memory() {
        // Shared remote buffer
        mr_remote = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                     d_remote_buf, MAX_SIZE * N_STREAMS, true, 0, pmi.rank);

        // Per-stream MRs
        for (int i = 0; i < N_STREAMS; i++) {
            streams[i].mr_local = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                                   streams[i].d_local_buf, MAX_SIZE, true, 0, pmi.rank);
            streams[i].mr_atomic = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                                    streams[i].atomic_result, sizeof(uint64_t), true, 0, pmi.rank);
            streams[i].mr_atomic_operand = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                                            streams[i].atomic_operand, sizeof(uint64_t), true, 0, pmi.rank);
        }
    }

    void exchange_addresses() {
        int peer = (pmi.rank == 0) ? 1 : 0;

        char* my_hex = (char*)malloc(2 * fabric.addrlen + 1);
        bytes_to_hex((uint8_t*)fabric.local_addr, fabric.addrlen, my_hex);

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

        uint8_t* all_bin = (uint8_t*)malloc(pmi.size * fabric.addrlen);
        for (int i = 0; i < pmi.size; i++) {
            const char* hex = all_hex + i * (2 * fabric.addrlen + 1);
            if (hex_to_bytes(hex, all_bin + i * fabric.addrlen, fabric.addrlen) <= 0) {
                fprintf(stderr, "Rank %d: hex_to_bytes failed\n", pmi.rank);
                exit(1);
            }
        }

        void* peer_addr_bin = all_bin + peer * fabric.addrlen;
        int inserted = fi_av_insert(fabric.av, peer_addr_bin, 1, &fabric.peer_addr, 0, NULL);
        if (inserted != 1 || fabric.peer_addr == FI_ADDR_NOTAVAIL) {
            fprintf(stderr, "Rank %d: fi_av_insert(peer) failed\n", pmi.rank);
            exit(1);
        }

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

        struct { uint64_t addr; uint64_t key; } my_rma_info, peer_rma_info;

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
        remote_addr_for_rma = fabric.is_virt_addr_mode() ? peer_remote_addr : 0;
    }

    void setup_gpu_arrays() {
        // Build host array of atomic results
        for (int i = 0; i < N_STREAMS; i++) {
            h_atomic_results[i] = streams[i].atomic_result;
        }

        // Allocate and copy to GPU
        check_hip(hipMalloc(&d_atomic_results, N_STREAMS * sizeof(volatile uint64_t*)),
                  "hipMalloc(d_atomic_results)");
        check_hip(hipMemcpy(d_atomic_results, h_atomic_results,
                            N_STREAMS * sizeof(volatile uint64_t*), hipMemcpyHostToDevice),
                  "hipMemcpy(d_atomic_results)");
    }

    void run_size_test(size_t current_size) {
        pmi.barrier();

        double iteration_times[NUM_ITERATIONS];
        int successful_iterations = 0;

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Reset shared trigger counter
            fi_cntr_set(fabric.trigger_cntr, 0);

            // Reset per-stream counters and atomic results
            for (int i = 0; i < N_STREAMS; i++) {
                fi_cntr_set(streams[i].completion_cntr, 0);
                fi_cntr_set(streams[i].atomic_completion_cntr, 0);
            }

            if (pmi.rank == 0) {
                for (int i = 0; i < N_STREAMS; i++) {
                    uint64_t zero = 0;
                    check_hip(hipMemcpy(streams[i].atomic_result, &zero, sizeof(uint64_t),
                                        hipMemcpyHostToDevice), "reset atomic_result");
                }
                check_hip(hipDeviceSynchronize(), "sync reset");
            }

            // Initialize buffers
            if (pmi.rank == 0) {
                for (int i = 0; i < N_STREAMS; i++) {
                    uint8_t pattern = (iter + 0xA0 + i) & 0xFF;
                    check_hip(hipMemset(streams[i].d_local_buf, pattern, current_size),
                              "hipMemset(local)");
                }
                check_hip(hipDeviceSynchronize(), "sync memset");
            } else {
                check_hip(hipMemset(d_remote_buf, 0xFF, current_size * N_STREAMS),
                          "hipMemset(remote)");
                check_hip(hipDeviceSynchronize(), "sync memset");
            }
            pmi.barrier();

            if (pmi.rank == 0) {
                // Queue DWQ operations for all streams
                // ALL use the SHARED trigger counter with threshold = stream_id + 1
                for (int i = 0; i < N_STREAMS; i++) {
                    uint64_t stream_remote_addr = remote_addr_for_rma + (i * MAX_SIZE);
                    if (!fabric.is_virt_addr_mode()) {
                        stream_remote_addr = i * MAX_SIZE;
                    }

                    // Queue RMA write - uses SHARED trigger counter, threshold = i+1
                    streams[i].dwq->queue_rma_write(
                        fabric.domain, fabric.ep,
                        streams[i].d_local_buf, streams[i].mr_local->desc, current_size,
                        fabric.peer_addr, stream_remote_addr, peer_remote_key,
                        fabric.trigger_cntr,           // SHARED trigger counter
                        streams[i].completion_cntr,    // Per-stream completion
                        i + 1);                        // Threshold = stream_id + 1

                    // Queue atomic signal
                    uint64_t atomic_result_addr = fabric.is_virt_addr_mode()
                        ? (uint64_t)streams[i].atomic_result : 0;
                    streams[i].dwq->queue_atomic_signal(
                        fabric.domain, fabric.ep,
                        streams[i].atomic_operand, streams[i].mr_atomic_operand->desc,
                        streams[i].atomic_result, streams[i].mr_atomic->key, atomic_result_addr,
                        fabric.local_addr_in_av,
                        streams[i].completion_cntr,
                        streams[i].atomic_completion_cntr, 1);
                }

                // Launch kernel - writes N_STREAMS to trigger ALL operations at once
                auto t_start = std::chrono::high_resolution_clock::now();
                hipLaunchKernelGGL(gpu_shared_trigger_write, dim3(1), dim3(N_STREAMS), 0, 0,
                                   fabric.dev_trigger_cntr,  // SHARED trigger counter
                                   d_atomic_results, N_STREAMS, N_STREAMS,
                                   d_start_clock, d_end_clock);
                check_hip(hipDeviceSynchronize(), "kernel sync");
                auto t_end = std::chrono::high_resolution_clock::now();

                double elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    t_end - t_start).count();
                iteration_times[successful_iterations++] = elapsed_us;

                // DEBUG: Time each step of completion waiting
                auto t_kernel_done = std::chrono::high_resolution_clock::now();

                // CRITICAL: Wait for completion counters to confirm DWQ ops are done
                // Try driving progress with fi_cq_read
                for (int i = 0; i < N_STREAMS; i++) {
                    while (fi_cntr_read(streams[i].completion_cntr) < 1) {
                        fi_cq_read(fabric.cq, NULL, 0);  // Drive progress
                    }
                }
                auto t_rma_done = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < N_STREAMS; i++) {
                    while (fi_cntr_read(streams[i].atomic_completion_cntr) < 1) {
                        fi_cq_read(fabric.cq, NULL, 0);  // Drive progress
                    }
                }
                auto t_atomic_done = std::chrono::high_resolution_clock::now();

                // Skip flush - completion counters already confirm ops are done
                // fabric.flush_dwq();

                // DEBUG output on first iteration
                // Debug timing (disabled for production)
                // if (iter == 0) {
                //     double kernel_us = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel_done - t_start).count();
                //     double rma_wait_us = std::chrono::duration_cast<std::chrono::microseconds>(t_rma_done - t_kernel_done).count();
                //     double atomic_wait_us = std::chrono::duration_cast<std::chrono::microseconds>(t_atomic_done - t_rma_done).count();
                //     printf("DEBUG size=%zu: kernel=%.0f rma_wait=%.0f atomic_wait=%.0f us\n",
                //            current_size, kernel_us, rma_wait_us, atomic_wait_us);
                // }
            }

            pmi.barrier();

            // Verification
            if (pmi.rank == 1) {
                int total_errors = 0;
                for (int i = 0; i < N_STREAMS; i++) {
                    uint8_t* remote_stream_buf = (uint8_t*)d_remote_buf + (i * MAX_SIZE);
                    uint8_t* verify_buf = h_verify_buf + (i * current_size);
                    check_hip(hipMemcpy(verify_buf, remote_stream_buf, current_size,
                                        hipMemcpyDeviceToHost), "D2H copy");

                    uint8_t expected_pattern = (iter + 0xA0 + i) & 0xFF;
                    int errors = 0;
                    for (size_t j = 0; j < current_size && errors < 10; j++) {
                        if (verify_buf[j] != expected_pattern) errors++;
                    }
                    total_errors += errors;
                }

                total_verifications++;
                if (total_errors > 0) {
                    total_verification_failures++;
                    fprintf(stderr, "Rank %d: VERIFICATION FAILED - iter=%d, size=%zu, errors=%d\n",
                            pmi.rank, iter, current_size, total_errors);
                }
            }
        }

        // Print results
        if (pmi.rank == 0 && successful_iterations > 0) {
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
            double per_xfer = avg / N_STREAMS;

            char size_buf[32];
            printf("%-8s  %12.2f  %12.2f  (min=%.2f max=%.2f)\n",
                   format_size(current_size, size_buf), avg, per_xfer,
                   iteration_times[0], iteration_times[samples - 1]);
            fflush(stdout);
        }
    }
};
