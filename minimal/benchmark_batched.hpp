/*
 * benchmark_batched.hpp - Batched DWQ benchmark using counter pool
 *
 * Supports more concurrent streams than hardware limit by processing
 * in batches of POOL_SIZE, reusing counter slots between batches.
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
#include "counter_pool.hpp"

// =============================================================================
// Configuration
// =============================================================================

constexpr int N_STREAMS = 16;      // Total concurrent transfers (can exceed POOL_SIZE)
constexpr int NUM_ITERATIONS = 20;

constexpr size_t TEST_SIZES[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    128 * 1024, 512 * 1024, 1024 * 1024,
    4 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024
};
constexpr int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);
constexpr size_t MAX_SIZE = 64 * 1024 * 1024;  // 64MB per stream

// =============================================================================
// GPU Kernel - Process streams in batches
// =============================================================================

// Kernel for a single batch - triggers POOL_SIZE operations and waits
__global__ void gpu_batch_trigger_wait(
    volatile uint64_t** trigger_addrs,
    volatile uint64_t** atomic_results,
    int batch_size,          // Actual streams in this batch (may be < POOL_SIZE)
    uint64_t* batch_done)    // Signal to CPU that batch completed
{
    int slot_id = threadIdx.x;
    if (slot_id >= batch_size) return;

    // Trigger this slot's DWQ operation
    *trigger_addrs[slot_id] = 1;

    // Poll for completion
    while (*atomic_results[slot_id] < 1) {
        // Spin wait
    }

    __syncthreads();

    // Thread 0 signals batch completion to CPU
    if (slot_id == 0) {
        __threadfence_system();
        *batch_done = 1;
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
// Batched Benchmark Runner
// =============================================================================

class BatchedBenchmarkRunner {
public:
    PmiSession& pmi;
    FabricDwqContext& fabric;
    CounterPool pool;

    // Per-stream source buffers (on sender)
    void* d_local_bufs[N_STREAMS];
    MemoryRegion* mr_locals[N_STREAMS];

    // Shared remote buffer (destination on receiver)
    void* d_remote_buf;
    MemoryRegion* mr_remote;

    // Batch completion signal (GPU writes, CPU reads)
    volatile uint64_t* h_batch_done;
    uint64_t* d_batch_done;

    // Host verification buffer
    uint8_t* h_verify_buf;

    // Peer RMA info
    uint64_t peer_remote_addr;
    uint64_t peer_remote_key;
    uint64_t remote_addr_for_rma;

    // Statistics
    int total_verifications;
    int total_verification_failures;

    BatchedBenchmarkRunner(PmiSession& pmi_, FabricDwqContext& fabric_)
        : pmi(pmi_), fabric(fabric_), pool(fabric_, pmi_.rank),
          d_remote_buf(nullptr), mr_remote(nullptr),
          h_batch_done(nullptr), d_batch_done(nullptr),
          h_verify_buf(nullptr),
          peer_remote_addr(0), peer_remote_key(0), remote_addr_for_rma(0),
          total_verifications(0), total_verification_failures(0)
    {
        memset(d_local_bufs, 0, sizeof(d_local_bufs));
        memset(mr_locals, 0, sizeof(mr_locals));

        allocate_buffers();
        register_memory();
        exchange_addresses();
        exchange_rma_info();
    }

    ~BatchedBenchmarkRunner() {
        // Free batch signal
        if (h_batch_done) (void)hipHostFree((void*)h_batch_done);

        // Free per-stream resources
        for (int i = 0; i < N_STREAMS; i++) {
            delete mr_locals[i];
            if (d_local_bufs[i]) (void)hipFree(d_local_bufs[i]);
        }

        // Free shared resources
        delete mr_remote;
        if (d_remote_buf) (void)hipFree(d_remote_buf);
        if (h_verify_buf) free(h_verify_buf);
    }

    // No copy/move
    BatchedBenchmarkRunner(const BatchedBenchmarkRunner&) = delete;
    BatchedBenchmarkRunner& operator=(const BatchedBenchmarkRunner&) = delete;

    void run() {
        pmi.barrier();
        usleep(100000);

        if (pmi.rank == 0) {
            int num_batches = (N_STREAMS + POOL_SIZE - 1) / POOL_SIZE;
            printf("%-8s  %12s  %12s  %s\n", "Size", "Total(us)", "Per-xfer(us)", "Statistics");
            printf("========  ============  ============  =====================================\n");
            printf("Note: %d streams in %d batches (pool=%d), per-xfer = total/%d\n",
                   N_STREAMS, num_batches, POOL_SIZE, N_STREAMS);
            printf("      GPU-Direct Async (GDA) with Counter Pool + Batching\n\n");
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
            fprintf(stderr, "Rank %d: %s failed: %s\n",
                    pmi.rank, msg, hipGetErrorString(err));
            exit(1);
        }
    }

    void allocate_buffers() {
        // Shared remote buffer
        check_hip(hipMalloc(&d_remote_buf, MAX_SIZE * N_STREAMS), "hipMalloc(remote)");

        // Per-stream source buffers
        for (int i = 0; i < N_STREAMS; i++) {
            check_hip(hipMalloc(&d_local_bufs[i], MAX_SIZE), "hipMalloc(local)");
        }

        // Batch completion signal (host-pinned for GPU access)
        check_hip(hipHostMalloc((void**)&h_batch_done, sizeof(uint64_t), hipHostMallocMapped),
                  "hipHostMalloc(batch_done)");
        check_hip(hipHostGetDevicePointer((void**)&d_batch_done, (void*)h_batch_done, 0),
                  "hipHostGetDevicePointer(batch_done)");

        // Host verification buffer
        h_verify_buf = (uint8_t*)malloc(MAX_SIZE * N_STREAMS);
        if (!h_verify_buf) {
            fprintf(stderr, "Rank %d: malloc failed\n", pmi.rank);
            exit(1);
        }
    }

    void register_memory() {
        // Shared remote buffer
        mr_remote = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                     d_remote_buf, MAX_SIZE * N_STREAMS, true, 0, pmi.rank);

        // Per-stream source buffers
        for (int i = 0; i < N_STREAMS; i++) {
            mr_locals[i] = new MemoryRegion(fabric.domain, fabric.ep, fabric.cxi_info,
                                            d_local_bufs[i], MAX_SIZE, true, 0, pmi.rank);
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

        struct { uint64_t addr; uint64_t key; } my_info, peer_info;
        my_info.addr = (uint64_t)d_remote_buf;
        my_info.key = mr_remote->key;

        char rma_hex[128];
        bytes_to_hex((uint8_t*)&my_info, sizeof(my_info), rma_hex);

        char key_str[PMI2_MAX_KEYLEN];
        snprintf(key_str, sizeof(key_str), "rma-%d", pmi.rank);
        pmi.kvs_put(key_str, rma_hex);
        pmi.barrier();

        snprintf(key_str, sizeof(key_str), "rma-%d", peer);
        char peer_hex[PMI2_MAX_VALLEN];
        pmi.kvs_get(key_str, peer_hex, sizeof(peer_hex));

        if (hex_to_bytes(peer_hex, (uint8_t*)&peer_info, sizeof(peer_info)) <= 0) {
            fprintf(stderr, "Rank %d: hex_to_bytes(rma) failed\n", pmi.rank);
            exit(1);
        }

        peer_remote_addr = peer_info.addr;
        peer_remote_key = peer_info.key;
        remote_addr_for_rma = fabric.is_virt_addr_mode() ? peer_remote_addr : 0;
    }

    void run_size_test(size_t current_size) {
        pmi.barrier();

        double iteration_times[NUM_ITERATIONS];
        int successful_iterations = 0;

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Initialize buffers
            if (pmi.rank == 0) {
                for (int i = 0; i < N_STREAMS; i++) {
                    uint8_t pattern = (iter + 0xA0 + i) & 0xFF;
                    check_hip(hipMemset(d_local_bufs[i], pattern, current_size), "hipMemset");
                }
                check_hip(hipDeviceSynchronize(), "sync");
            } else {
                check_hip(hipMemset(d_remote_buf, 0xFF, current_size * N_STREAMS), "hipMemset");
                check_hip(hipDeviceSynchronize(), "sync");
            }
            pmi.barrier();

            if (pmi.rank == 0) {
                auto t_start = std::chrono::high_resolution_clock::now();

                // Process streams in batches
                for (int batch_start = 0; batch_start < N_STREAMS; batch_start += POOL_SIZE) {
                    int batch_end = std::min(batch_start + POOL_SIZE, N_STREAMS);
                    int batch_size = batch_end - batch_start;

                    // Reset pool slots and queue operations for this batch
                    pool.reset_all();
                    *h_batch_done = 0;

                    for (int i = 0; i < batch_size; i++) {
                        int stream_id = batch_start + i;
                        uint64_t stream_offset = stream_id * MAX_SIZE;
                        uint64_t remote_addr = fabric.is_virt_addr_mode()
                            ? (remote_addr_for_rma + stream_offset)
                            : stream_offset;

                        pool.queue_operation(
                            i,  // slot_idx
                            stream_id,
                            d_local_bufs[stream_id],
                            mr_locals[stream_id]->desc,
                            current_size,
                            fabric.peer_addr,
                            remote_addr,
                            peer_remote_key);
                    }

                    // Launch kernel to trigger and wait for this batch
                    hipLaunchKernelGGL(gpu_batch_trigger_wait, dim3(1), dim3(batch_size), 0, 0,
                                       pool.get_d_trigger_addrs(),
                                       pool.get_d_atomic_results(),
                                       batch_size,
                                       d_batch_done);

                    // Wait for batch completion (CPU polls GPU signal)
                    while (*h_batch_done == 0) {
                        // Spin wait - could add usleep for power efficiency
                    }
                    check_hip(hipDeviceSynchronize(), "batch sync");
                }

                auto t_end = std::chrono::high_resolution_clock::now();
                double elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    t_end - t_start).count();
                iteration_times[successful_iterations++] = elapsed_us;

                fabric.flush_dwq();
            }

            pmi.barrier();

            // Verification on rank 1
            if (pmi.rank == 1) {
                int total_errors = 0;
                for (int i = 0; i < N_STREAMS; i++) {
                    uint8_t* remote_stream = (uint8_t*)d_remote_buf + (i * MAX_SIZE);
                    uint8_t* verify = h_verify_buf + (i * current_size);
                    check_hip(hipMemcpy(verify, remote_stream, current_size,
                                        hipMemcpyDeviceToHost), "D2H");

                    uint8_t expected = (iter + 0xA0 + i) & 0xFF;
                    int errors = 0;
                    for (size_t j = 0; j < current_size && errors < 10; j++) {
                        if (verify[j] != expected) errors++;
                    }
                    total_errors += errors;
                }

                total_verifications++;
                if (total_errors > 0) {
                    total_verification_failures++;
                    fprintf(stderr, "Rank %d: VERIFY FAILED iter=%d size=%zu errors=%d\n",
                            pmi.rank, iter, current_size, total_errors);
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

            int samples = std::min(successful_iterations, 10);
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
