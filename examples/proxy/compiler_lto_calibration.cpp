/*
 * compiler_lto_calibration.cpp -- compiler-generated calibration workload.
 *
 * This source is disjoint from the frozen compiler_lto_eval test.  It exists
 * only to measure the cost of pass-materialized legal actions over a wider
 * set of compiler fact shapes.  Models consume the emitted LTO dossier and
 * measured action costs, never this source text, and never the frozen test's
 * runtime results or oracle labels.
 */

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/launch.hpp"

constexpr size_t MiB = 1u << 20;
constexpr size_t kBufferBytes = 40u * MiB;

#define DEFINE_SINGLE(NAME, BASE, BYTES)                                    \
__global__ void NAME(gicc::DeviceCtx* ctx, int peer, int buf) {             \
    gicc::put(ctx, peer, buf, BASE, buf, BASE, BYTES);                       \
    gicc::flush(ctx);                                                        \
    gicc::quiet(ctx);                                                        \
}

#define DEFINE_REUSE(NAME, BASE, BYTES, OPS)                                \
__global__ void NAME(gicc::DeviceCtx* ctx, int peer, int buf) {             \
    for (int i = 0; i < OPS; ++i)                                           \
        gicc::put(ctx, peer, buf, BASE, buf, BASE, BYTES);                   \
    gicc::flush(ctx);                                                        \
    gicc::quiet(ctx);                                                        \
}

#define DEFINE_ADJACENT(NAME, BASE, BYTES, OPS)                             \
__global__ void NAME(gicc::DeviceCtx* ctx, int peer, int buf) {             \
    for (int i = 0; i < OPS; ++i) {                                         \
        const size_t off = BASE + static_cast<size_t>(i) * BYTES;            \
        gicc::put(ctx, peer, buf, off, buf, off, BYTES);                     \
    }                                                                        \
    gicc::flush(ctx);                                                        \
    gicc::quiet(ctx);                                                        \
}

#define DEFINE_FAR(NAME, BASE, BYTES, OPS, COMPUTE_ITERS)                   \
__global__ void NAME(gicc::DeviceCtx* ctx, int peer, int buf,              \
                     volatile float* scratch) {                             \
    for (int i = 0; i < OPS; ++i) {                                         \
        const size_t off = BASE + static_cast<size_t>(i) * BYTES;            \
        gicc::put(ctx, peer, buf, off, buf, off, BYTES);                     \
    }                                                                        \
    float acc = scratch[blockIdx.x] + static_cast<float>(threadIdx.x + 1);   \
    for (int i = 0; i < COMPUTE_ITERS; ++i) {                               \
        acc = acc * 1.000001f + 0.5f;                                       \
        acc = acc * 0.999999f - 0.25f;                                      \
        acc = acc + acc * 0.5f;                                             \
        acc = acc * 1.5f - acc * 0.25f;                                    \
    }                                                                        \
    scratch[blockIdx.x] = acc;                                              \
    gicc::flush(ctx);                                                        \
    gicc::quiet(ctx);                                                        \
}

DEFINE_SINGLE(cal_single_512_g1,       0u * MiB,       512u)
DEFINE_SINGLE(cal_single_2k_g4,        2u * MiB,      2048u)
DEFINE_SINGLE(cal_single_16k_g1,       4u * MiB,     16384u)
DEFINE_SINGLE(cal_single_64k_g8,       6u * MiB,     65536u)
DEFINE_SINGLE(cal_single_256k_g4,      8u * MiB,    262144u)
DEFINE_SINGLE(cal_single_512k_g8,     10u * MiB,    524288u)

DEFINE_REUSE(cal_reuse_1k_k8_g1,      12u * MiB,      1024u,  8)
DEFINE_REUSE(cal_reuse_2k_k24_g1,     14u * MiB,      2048u, 24)
DEFINE_REUSE(cal_reuse_8k_k48_g4,     16u * MiB,      8192u, 48)

DEFINE_ADJACENT(cal_adjacent_1k_k6_g1, 18u * MiB,     1024u,  6)
DEFINE_ADJACENT(cal_adjacent_8k_k12_g4, 20u * MiB,    8192u, 12)
DEFINE_ADJACENT(cal_adjacent_32k_k40_g8, 22u * MiB,  32768u, 40)

DEFINE_FAR(cal_far_2k_k12_i64_g4,     24u * MiB,      2048u, 12,   64)
DEFINE_FAR(cal_far_8k_k48_i256_g8,    26u * MiB,      8192u, 48,  256)
DEFINE_FAR(cal_far_16k_k24_i1024_g8,  28u * MiB,     16384u, 24, 1024)

#undef DEFINE_SINGLE
#undef DEFINE_REUSE
#undef DEFINE_ADJACENT
#undef DEFINE_FAR

#define CAL_STATIC_PUT(BASE, BYTES, I)                                      \
    gicc::put(ctx, peer, buf, BASE + (I) * BYTES,                           \
              buf, BASE + (I) * BYTES, BYTES)

__global__ void cal_static2_2k_g2(gicc::DeviceCtx* ctx, int peer, int buf) {
    CAL_STATIC_PUT(30u * MiB, 2048u, 0);
    CAL_STATIC_PUT(30u * MiB, 2048u, 1);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

__global__ void cal_static3_8k_g4(gicc::DeviceCtx* ctx, int peer, int buf) {
    CAL_STATIC_PUT(32u * MiB, 8192u, 0);
    CAL_STATIC_PUT(32u * MiB, 8192u, 1);
    CAL_STATIC_PUT(32u * MiB, 8192u, 2);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

__global__ void cal_static6_16k_g8(gicc::DeviceCtx* ctx, int peer, int buf) {
    CAL_STATIC_PUT(34u * MiB, 16384u, 0);
    CAL_STATIC_PUT(34u * MiB, 16384u, 1);
    CAL_STATIC_PUT(34u * MiB, 16384u, 2);
    CAL_STATIC_PUT(34u * MiB, 16384u, 3);
    CAL_STATIC_PUT(34u * MiB, 16384u, 4);
    CAL_STATIC_PUT(34u * MiB, 16384u, 5);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

#undef CAL_STATIC_PUT

__global__ void fill_bytes(uint8_t* data, size_t n, bool source) {
    for (size_t i = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
         i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        data[i] = source
            ? static_cast<uint8_t>(1 + ((i / 256) * 17 + i) % 251)
            : uint8_t{0};
    }
}

double now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::micro>(
        clock::now().time_since_epoch()).count();
}

struct Scenario {
    const char* name;
    const char* kernel;
    size_t base;
    size_t bytes;
    int grid;
    enum Kind {
        Single512, Single2k, Single16k, Single64k, Single256k, Single512k,
        Reuse1k8, Reuse2k24, Reuse8k48,
        Adjacent1k6, Adjacent8k12, Adjacent32k40,
        Far2k12, Far8k48, Far16k24,
        Static2, Static3, Static6,
    } kind;
};

const Scenario kScenarios[] = {
    {"single-512-g1", "cal_single_512_g1", 0u * MiB, 512u, 1, Scenario::Single512},
    {"single-2k-g4", "cal_single_2k_g4", 2u * MiB, 2048u, 4, Scenario::Single2k},
    {"single-16k-g1", "cal_single_16k_g1", 4u * MiB, 16384u, 1, Scenario::Single16k},
    {"single-64k-g8", "cal_single_64k_g8", 6u * MiB, 65536u, 8, Scenario::Single64k},
    {"single-256k-g4", "cal_single_256k_g4", 8u * MiB, 262144u, 4, Scenario::Single256k},
    {"single-512k-g8", "cal_single_512k_g8", 10u * MiB, 524288u, 8, Scenario::Single512k},
    {"reuse-1k-k8-g1", "cal_reuse_1k_k8_g1", 12u * MiB, 1024u, 1, Scenario::Reuse1k8},
    {"reuse-2k-k24-g1", "cal_reuse_2k_k24_g1", 14u * MiB, 2048u, 1, Scenario::Reuse2k24},
    {"reuse-8k-k48-g4", "cal_reuse_8k_k48_g4", 16u * MiB, 8192u, 4, Scenario::Reuse8k48},
    {"adjacent-1k-k6-g1", "cal_adjacent_1k_k6_g1", 18u * MiB, 6u * 1024u, 1, Scenario::Adjacent1k6},
    {"adjacent-8k-k12-g4", "cal_adjacent_8k_k12_g4", 20u * MiB, 12u * 8192u, 4, Scenario::Adjacent8k12},
    {"adjacent-32k-k40-g8", "cal_adjacent_32k_k40_g8", 22u * MiB, 40u * 32768u, 8, Scenario::Adjacent32k40},
    {"far-2k-k12-i64-g4", "cal_far_2k_k12_i64_g4", 24u * MiB, 12u * 2048u, 4, Scenario::Far2k12},
    {"far-8k-k48-i256-g8", "cal_far_8k_k48_i256_g8", 26u * MiB, 48u * 8192u, 8, Scenario::Far8k48},
    {"far-16k-k24-i1024-g8", "cal_far_16k_k24_i1024_g8", 28u * MiB, 24u * 16384u, 8, Scenario::Far16k24},
    {"static2-2k-g2", "cal_static2_2k_g2", 30u * MiB, 2u * 2048u, 2, Scenario::Static2},
    {"static3-8k-g4", "cal_static3_8k_g4", 32u * MiB, 3u * 8192u, 4, Scenario::Static3},
    {"static6-16k-g8", "cal_static6_16k_g8", 34u * MiB, 6u * 16384u, 8, Scenario::Static6},
};

uint64_t expected_hash(size_t base, size_t bytes) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = base; i < base + bytes; ++i) {
        const uint8_t value = static_cast<uint8_t>(
            1 + ((i / 256) * 17 + i) % 251);
        hash ^= value;
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t check_region(void* buffer, const Scenario& scenario, int rank,
                      int& ok) {
    uint64_t hash = 0;
    if (rank == 1) {
        std::vector<uint8_t> host(scenario.bytes);
        if (hipMemcpy(host.data(), static_cast<uint8_t*>(buffer) + scenario.base,
                      host.size(), hipMemcpyDeviceToHost) != hipSuccess) {
            ok = 0;
        }
        hash = 1469598103934665603ull;
        for (uint8_t value : host) {
            hash ^= value;
            hash *= 1099511628211ull;
        }
        if (hash != expected_hash(scenario.base, scenario.bytes)) ok = 0;
    }
    MPI_Bcast(&ok, 1, MPI_INT, 1, MPI_COMM_WORLD);
    MPI_Bcast(&hash, 1, MPI_UNSIGNED_LONG_LONG, 1, MPI_COMM_WORLD);
    return hash;
}

void launch_scenario(const Scenario& s, gicc::Runtime& runtime, int peer,
                     int buffer, volatile float* scratch) {
    const dim3 block(1);
    switch (s.kind) {
        case Scenario::Single512: gicc::launch<cal_single_512_g1>(runtime, dim3(1), block, peer, buffer); break;
        case Scenario::Single2k: gicc::launch<cal_single_2k_g4>(runtime, dim3(4), block, peer, buffer); break;
        case Scenario::Single16k: gicc::launch<cal_single_16k_g1>(runtime, dim3(1), block, peer, buffer); break;
        case Scenario::Single64k: gicc::launch<cal_single_64k_g8>(runtime, dim3(8), block, peer, buffer); break;
        case Scenario::Single256k: gicc::launch<cal_single_256k_g4>(runtime, dim3(4), block, peer, buffer); break;
        case Scenario::Single512k: gicc::launch<cal_single_512k_g8>(runtime, dim3(8), block, peer, buffer); break;
        case Scenario::Reuse1k8: gicc::launch<cal_reuse_1k_k8_g1>(runtime, dim3(1), block, peer, buffer); break;
        case Scenario::Reuse2k24: gicc::launch<cal_reuse_2k_k24_g1>(runtime, dim3(1), block, peer, buffer); break;
        case Scenario::Reuse8k48: gicc::launch<cal_reuse_8k_k48_g4>(runtime, dim3(4), block, peer, buffer); break;
        case Scenario::Adjacent1k6: gicc::launch<cal_adjacent_1k_k6_g1>(runtime, dim3(1), block, peer, buffer); break;
        case Scenario::Adjacent8k12: gicc::launch<cal_adjacent_8k_k12_g4>(runtime, dim3(4), block, peer, buffer); break;
        case Scenario::Adjacent32k40: gicc::launch<cal_adjacent_32k_k40_g8>(runtime, dim3(8), block, peer, buffer); break;
        case Scenario::Far2k12: gicc::launch<cal_far_2k_k12_i64_g4>(runtime, dim3(4), block, peer, buffer, scratch); break;
        case Scenario::Far8k48: gicc::launch<cal_far_8k_k48_i256_g8>(runtime, dim3(8), block, peer, buffer, scratch); break;
        case Scenario::Far16k24: gicc::launch<cal_far_16k_k24_i1024_g8>(runtime, dim3(8), block, peer, buffer, scratch); break;
        case Scenario::Static2: gicc::launch<cal_static2_2k_g2>(runtime, dim3(2), block, peer, buffer); break;
        case Scenario::Static3: gicc::launch<cal_static3_8k_g4>(runtime, dim3(4), block, peer, buffer); break;
        case Scenario::Static6: gicc::launch<cal_static6_16k_g8>(runtime, dim3(8), block, peer, buffer); break;
    }
}

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}

bool run_scenario(const Scenario& s, gicc::Runtime& runtime, int rank,
                  int peer, int buffer_index, void* device_buffer,
                  volatile float* scratch, int warmup, int runs) {
    auto one = [&]() {
        if (rank == 0)
            launch_scenario(s, runtime, peer, buffer_index, scratch);
        (void)hipDeviceSynchronize();
        runtime.reset();
    };

    runtime.barrier();
    for (int i = 0; i < warmup; ++i) one();
    if (rank == 1) {
        (void)hipMemset(static_cast<uint8_t*>(device_buffer) + s.base, 0,
                        s.bytes);
        (void)hipDeviceSynchronize();
    }
    runtime.barrier();

    const uint64_t staged0 = runtime.staged_ops();
    const uint64_t pushed0 = runtime.proxy_pushes();
    std::vector<double> samples;
    if (rank == 0) samples.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        const double begin = rank == 0 ? now_us() : 0.0;
        one();
        if (rank == 0) samples.push_back(now_us() - begin);
    }
    runtime.barrier();

    int ok = 1;
    const uint64_t hash = check_region(device_buffer, s, rank, ok);
    if (rank == 0) {
        std::printf(
            "COMPILER_LTO_CALIBRATION scenario=%s kernel=%s runs=%d "
            "median_us=%.3f p25_us=%.3f p75_us=%.3f staged=%llu "
            "pushed=%llu hash=%016llx data=%s\n",
            s.name, s.kernel, runs, percentile(samples, 0.50),
            percentile(samples, 0.25), percentile(samples, 0.75),
            static_cast<unsigned long long>(runtime.staged_ops() - staged0),
            static_cast<unsigned long long>(runtime.proxy_pushes() - pushed0),
            static_cast<unsigned long long>(hash), ok ? "OK" : "WRONG");
    }
    return ok != 0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int warmup = 10;
    int runs = 100;
    std::string selected = "all";
    for (int i = 1; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--warmup=", 9)) warmup = std::atoi(argv[i] + 9);
        else if (!std::strncmp(argv[i], "--runs=", 7)) runs = std::atoi(argv[i] + 7);
        else if (!std::strncmp(argv[i], "--scenario=", 11)) selected = argv[i] + 11;
        else if (!std::strcmp(argv[i], "--list")) {
            for (const auto& scenario : kScenarios) std::puts(scenario.name);
            return 0;
        }
    }
    if (warmup < 0 || runs <= 0) return 2;

    gicc::Runtime runtime;
    runtime.enable_host_wait_mode();
    runtime.enable_mixed_dispatch();
    const int rank = runtime.rank();
    if (runtime.size() != 2) {
        if (rank == 0) std::fprintf(stderr, "calibration requires two ranks\n");
        return 2;
    }

    void* device_buffer = nullptr;
    float* scratch = nullptr;
    if (hipMalloc(&device_buffer, kBufferBytes) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&scratch), 8 * sizeof(float)) !=
            hipSuccess) return 3;
    hipLaunchKernelGGL(fill_bytes, dim3(64), dim3(256), 0, 0,
                       static_cast<uint8_t*>(device_buffer), kBufferBytes,
                       rank == 0);
    (void)hipMemset(scratch, 0, 8 * sizeof(float));
    (void)hipDeviceSynchronize();

    auto handle = runtime.register_buffer(device_buffer, kBufferBytes, true);
    runtime.exchange();
    runtime.barrier();

    bool found = false;
    bool all_ok = true;
    for (const auto& scenario : kScenarios) {
        if (selected != "all" && selected != scenario.name) continue;
        found = true;
        all_ok &= run_scenario(scenario, runtime, rank, 1 - rank,
                               handle.index, device_buffer, scratch,
                               warmup, runs);
    }
    if (!found) all_ok = false;
    runtime.barrier();
    (void)hipFree(device_buffer);
    (void)hipFree(scratch);
    return all_ok ? 0 : 4;
}
