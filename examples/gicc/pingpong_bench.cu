/**
 * gicc_pingpong_bench.cu - GICC Ping-Pong Benchmark
 *
 * Rewrite of nvidia/gpu_pingpong_bench.cu using the GICC high-level API.
 * All IB/QP/MR boilerplate is handled by gicc::Runtime.
 *
 * Run: mpirun -np 2 ./gicc_pingpong_bench
 */

#include <gicc/gicc.hpp>
#include <gicc/gicc_device.cuh>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <unistd.h>

// Test configurations
constexpr int WARMUP_ITERS = 50;
constexpr int TEST_ITERS = 1000;
constexpr size_t MSG_SIZES[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    128*1024, 256*1024, 512*1024, 1024*1024,
    2*1024*1024, 4*1024*1024, 8*1024*1024, 16*1024*1024
};
constexpr int NUM_SIZES = sizeof(MSG_SIZES) / sizeof(MSG_SIZES[0]);

//==============================================================================
// GPU kernels
//==============================================================================

__global__ void pingpong_kernel(
    gicc::DeviceCtx* ctx,
    uint64_t send_addr,
    uint32_t send_lkey,
    volatile uint64_t* recv_flag,
    uint64_t send_flag_addr,
    uint32_t msg_size,
    int iterations,
    int is_initiator,
    uint64_t* result_cycles,
    uint64_t* min_cycles,
    uint64_t* max_cycles)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t total = 0;
    uint64_t local_min = UINT64_MAX;
    uint64_t local_max = 0;

    uint32_t actual_size = (msg_size < 8) ? 8 : msg_size;
    volatile uint64_t* send_flag = (volatile uint64_t*)send_flag_addr;

    for (int i = 0; i < iterations; i++) {
        uint64_t iter_start = clock64();

        if (is_initiator) {
            *send_flag = i + 1;
            __threadfence_system();
            gicc::put(ctx, send_addr, send_lkey, actual_size);
            while (*recv_flag != (uint64_t)(i + 1)) {}
        } else {
            while (*recv_flag != (uint64_t)(i + 1)) {}
            *send_flag = i + 1;
            __threadfence_system();
            gicc::put(ctx, send_addr, send_lkey, actual_size);
        }

        uint64_t iter_cycles = clock64() - iter_start;
        total += iter_cycles;
        if (iter_cycles < local_min) local_min = iter_cycles;
        if (iter_cycles > local_max) local_max = iter_cycles;
    }

    *result_cycles = total;
    *min_cycles = local_min;
    *max_cycles = local_max;
}

__global__ void warmup_kernel(
    gicc::DeviceCtx* ctx,
    uint64_t send_addr,
    uint32_t send_lkey,
    volatile uint64_t* recv_flag,
    uint64_t send_flag_addr,
    uint32_t msg_size,
    int iterations,
    int is_initiator)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t actual_size = (msg_size < 8) ? 8 : msg_size;
    volatile uint64_t* send_flag = (volatile uint64_t*)send_flag_addr;

    for (int i = 0; i < iterations; i++) {
        if (is_initiator) {
            *send_flag = i + 1;
            __threadfence_system();
            gicc::put(ctx, send_addr, send_lkey, actual_size);
            while (*recv_flag != (uint64_t)(i + 1)) {}
        } else {
            while (*recv_flag != (uint64_t)(i + 1)) {}
            *send_flag = i + 1;
            __threadfence_system();
            gicc::put(ctx, send_addr, send_lkey, actual_size);
        }
    }
}

//==============================================================================
// Helpers
//==============================================================================

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

static double cycles_to_us(uint64_t cycles, double clock_rate_khz) {
    return cycles / (clock_rate_khz / 1000.0);
}

static const char* format_size(size_t size, char* buf) {
    if (size < 1024) snprintf(buf, 32, "%zuB", size);
    else if (size < 1024 * 1024) snprintf(buf, 32, "%zuKB", size / 1024);
    else snprintf(buf, 32, "%zuMB", size / (1024 * 1024));
    return buf;
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    // --- All setup in one object ---
    gicc::Runtime rt;

    if (rt.size() != 2) {
        if (rt.rank() == 0) fprintf(stderr, "Requires exactly 2 ranks\n");
        return 1;
    }

    int peer = 1 - rt.rank();

    // Allocate GPU buffers
    size_t max_size = 16 * 1024 * 1024;
    void* d_send = nullptr;
    void* d_recv = nullptr;
    cuda_check(cudaMalloc(&d_send, max_size), "alloc send");
    cuda_check(cudaMalloc(&d_recv, max_size), "alloc recv");
    cuda_check(cudaMemset(d_send, 0, max_size), "memset send");
    cuda_check(cudaMemset(d_recv, 0, max_size), "memset recv");

    // Register and exchange buffer info
    auto send_buf = rt.register_buffer(d_send, max_size, true);  // index 0
    auto recv_buf = rt.register_buffer(d_recv, max_size, true);  // index 1
    rt.exchange();

    // Prepare device context (target peer's recv_buf = index 1)
    auto* ctx = rt.prepare(peer, recv_buf.index);

    // Timing variables
    uint64_t *d_result, *d_min, *d_max;
    cuda_check(cudaMalloc(&d_result, sizeof(uint64_t)), "alloc result");
    cuda_check(cudaMalloc(&d_min, sizeof(uint64_t)), "alloc min");
    cuda_check(cudaMalloc(&d_max, sizeof(uint64_t)), "alloc max");

    rt.barrier();

    // Header
    if (rt.rank() == 0) {
        printf("=======================================================\n");
        printf("   GICC Ping-Pong Benchmark\n");
        printf("=======================================================\n");
        printf("GPU: %s (%.2f GHz)\n", rt.gpu_name(), rt.clock_rate_khz() / 1e6);
        printf("Warmup: %d, Test: %d iterations\n", WARMUP_ITERS, TEST_ITERS);
        printf("-------------------------------------------------------\n\n");
        fflush(stdout);
    }

    // Warmup
    int is_initiator = (rt.rank() == 0) ? 1 : 0;

    cuda_check(cudaMemset(d_recv, 0, max_size), "clear recv");
    cuda_check(cudaMemset(d_send, 0, max_size), "clear send");
    rt.barrier();

    volatile uint64_t* warmup_recv_flag =
        (volatile uint64_t*)((char*)d_recv + 56);
    uint64_t warmup_send_flag = send_buf.addr + 56;

    warmup_kernel<<<1, 1>>>(
        ctx, send_buf.addr, send_buf.lkey,
        warmup_recv_flag, warmup_send_flag,
        64, WARMUP_ITERS, is_initiator);
    cudaDeviceSynchronize();
    rt.barrier();

    if (rt.rank() == 0) {
        printf("Warmup complete.\n\n");
        fflush(stdout);
    }

    // Main benchmark
    if (rt.rank() == 0) {
        printf("=== Ping-Pong Latency Results ===\n");
        printf("%-8s %12s %12s %12s %12s %12s\n",
               "Size", "RTT (us)", "Half-RTT", "Min RTT", "Max RTT", "BW (Gbps)");
        printf("-----------------------------------------------------------------------\n");
        fflush(stdout);
    }

    for (int sz_idx = 0; sz_idx < NUM_SIZES; sz_idx++) {
        size_t msg_size = MSG_SIZES[sz_idx];

        cuda_check(cudaMemset(d_recv, 0, max_size), "clear recv");
        cuda_check(cudaMemset(d_send, 0, max_size), "clear send");
        rt.barrier();

        size_t flag_offset = (msg_size < 8) ? 0 : (msg_size - 8);
        volatile uint64_t* recv_flag =
            (volatile uint64_t*)((char*)d_recv + flag_offset);
        uint64_t send_flag_addr = send_buf.addr + flag_offset;

        pingpong_kernel<<<1, 1>>>(
            ctx, send_buf.addr, send_buf.lkey,
            recv_flag, send_flag_addr,
            msg_size, TEST_ITERS, is_initiator,
            d_result, d_min, d_max);
        cudaDeviceSynchronize();

        uint64_t total_cycles, min_cycles, max_cycles;
        cudaMemcpy(&total_cycles, d_result, sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&min_cycles, d_min, sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&max_cycles, d_max, sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);

        rt.barrier();

        if (rt.rank() == 0) {
            double avg_rtt = cycles_to_us(total_cycles, rt.clock_rate_khz())
                             / TEST_ITERS;
            double half_rtt = avg_rtt / 2.0;
            double min_rtt = cycles_to_us(min_cycles, rt.clock_rate_khz());
            double max_rtt = cycles_to_us(max_cycles, rt.clock_rate_khz());
            double bw = (msg_size * 2.0 * 8.0) / (avg_rtt * 1000.0);

            char buf[32];
            printf("%-8s %12.3f %12.3f %12.3f %12.3f %12.2f\n",
                   format_size(msg_size, buf),
                   avg_rtt, half_rtt, min_rtt, max_rtt, bw);
            fflush(stdout);
        }

        rt.barrier();
        usleep(10000);
    }

    if (rt.rank() == 0) {
        printf("\n=======================================================\n");
        printf("Benchmark complete!\n");
        printf("=======================================================\n");
        fflush(stdout);
    }

    // Cleanup
    cudaFree(d_result);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_send);
    cudaFree(d_recv);

    return 0;
}
