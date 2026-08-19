/*
 * compiler_lto_eval.cpp -- frozen heterogeneous workload for evaluating a
 * compiler-level communication decider.
 *
 * This file is benchmark infrastructure, not model input.  Every decision
 * arm is compiled from this exact source; the model sees only the schema-
 * checked dossier emitted by the LTO passes.  Separate kernels make each
 * device operation a real lowering unit without requiring source rewriting
 * or kernel cloning by the model.
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

constexpr size_t kTinyBase      = 0;
constexpr size_t kReuseBase     = 1u << 20;
constexpr size_t kAdjacentBase  = 2u << 20;
constexpr size_t kFarBase       = 4u << 20;
constexpr size_t kLargeBase     = 8u << 20;
constexpr size_t kDynamicBase   = 12u << 20;
constexpr size_t kStatic4Base   = 14u << 20;
constexpr size_t kBufferBytes   = 16u << 20;

constexpr size_t kTinyBytes     = 256;
constexpr size_t kMsgBytes      = 4096;
constexpr size_t kLargeBytes    = 1u << 20;
constexpr int    kReuseOps      = 32;
constexpr int    kAdjacentOps   = 16;
constexpr int    kFarOps        = 64;
constexpr int    kStaticOps     = 4;

__global__ void eval_tiny_single(gicc::DeviceCtx* ctx, int peer, int buf) {
    gicc::put(ctx, peer, buf, kTinyBase, buf, kTinyBase, kTinyBytes);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

// The descriptor is loop invariant: the compiler can prove that one staged
// descriptor may be reused kReuseOps times.
__global__ void eval_reuse_batch(gicc::DeviceCtx* ctx, int peer, int buf) {
    for (int i = 0; i < kReuseOps; ++i)
        gicc::put(ctx, peer, buf, kReuseBase, buf, kReuseBase, kMsgBytes);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

// Consecutive loop iterations cover adjacent regions.  This is a distinct
// compiler fact from descriptor reuse and is a future pass-generated action.
__global__ void eval_adjacent_batch(gicc::DeviceCtx* ctx, int peer, int buf) {
    for (int i = 0; i < kAdjacentOps; ++i) {
        const size_t off = kAdjacentBase + static_cast<size_t>(i) * kMsgBytes;
        gicc::put(ctx, peer, buf, off, buf, off, kMsgBytes);
    }
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

// The arithmetic sits between issue and completion.  Its fixed trip count is
// visible in device IR, unlike a runtime's view of an individual transfer.
__global__ void eval_far_batch(gicc::DeviceCtx* ctx, int peer, int buf,
                               volatile float* scratch) {
    for (int i = 0; i < kFarOps; ++i) {
        const size_t off = kFarBase + static_cast<size_t>(i) * kMsgBytes;
        gicc::put(ctx, peer, buf, off, buf, off, kMsgBytes);
    }

    float acc = scratch[blockIdx.x] + static_cast<float>(threadIdx.x + 1);
    for (int i = 0; i < 512; ++i) {
        acc = acc * 1.000001f + 0.5f;
        acc = acc * 0.999999f - 0.25f;
        acc = acc + acc * 0.5f;
        acc = acc * 1.5f - acc * 0.25f;
    }
    scratch[blockIdx.x] = acc;
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

__global__ void eval_large_single(gicc::DeviceCtx* ctx, int peer, int buf) {
    gicc::put(ctx, peer, buf, kLargeBase, buf, kLargeBase, kLargeBytes);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

// Loading the offset from device memory deliberately makes this descriptor
// unavailable to host trace synthesis.  Proxy is therefore the only legal
// lowering; this is a legality test, not a performance preference.
__global__ void eval_dynamic_offset(gicc::DeviceCtx* ctx, int peer, int buf,
                                    const size_t* dynamic_offset) {
    const size_t off = dynamic_offset[0];
    gicc::put(ctx, peer, buf, off, buf, off, kMsgBytes);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}

// Four static sites share one completion point.  Device lowering can assign
// them to four launched blocks/rings, while trigger staging remains serial.
// Four keeps the exact mixed-action oracle enumerable (2^4 configurations).
#define EVAL_STATIC_PUT(I)                                                   \
    gicc::put(ctx, peer, buf, kStatic4Base + (I) * kMsgBytes,                \
              buf, kStatic4Base + (I) * kMsgBytes, kMsgBytes)
__global__ void eval_static4_parallel(gicc::DeviceCtx* ctx, int peer, int buf) {
    EVAL_STATIC_PUT(0); EVAL_STATIC_PUT(1); EVAL_STATIC_PUT(2);
    EVAL_STATIC_PUT(3);
    gicc::flush(ctx);
    gicc::quiet(ctx);
}
#undef EVAL_STATIC_PUT

__global__ void fill_bytes(uint8_t* data, size_t n, bool source) {
    for (size_t i = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
         i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        data[i] = source
            ? static_cast<uint8_t>(1 + ((i / 256) * 17 + i) % 251)
            : uint8_t{0};
    }
}

__global__ void make_dynamic_offset(size_t* out, const float* seed) {
    if (threadIdx.x == 0)
        out[0] = kDynamicBase + static_cast<size_t>(seed[0] < 0.0f ? 0 : 0);
}

double now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::micro>(
        clock::now().time_since_epoch()).count();
}

struct Scenario {
    const char* name;
    size_t base;
    size_t bytes;
    enum Kind { Tiny, Reuse, Adjacent, Far, Large, Dynamic, Static4 } kind;
};

const Scenario kScenarios[] = {
    {"tiny-k1-grid1",       kTinyBase,     kTinyBytes,               Scenario::Tiny},
    {"reuse-k32-grid1",     kReuseBase,    kMsgBytes,                Scenario::Reuse},
    {"adjacent-k16-grid1",  kAdjacentBase, kAdjacentOps * kMsgBytes, Scenario::Adjacent},
    {"far-k64-grid8",       kFarBase,      kFarOps * kMsgBytes,      Scenario::Far},
    {"large-k1-grid8",      kLargeBase,    kLargeBytes,              Scenario::Large},
    {"dynamic-k1-grid1",    kDynamicBase,  kMsgBytes,                Scenario::Dynamic},
    {"static4-k4-grid4",    kStatic4Base,  kStaticOps * kMsgBytes,   Scenario::Static4},
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

uint64_t check_region(void* device_buffer, const Scenario& scenario, int rank,
                      int& ok) {
    uint64_t hash = 0;
    if (rank == 1) {
        std::vector<uint8_t> host(scenario.bytes);
        if (hipMemcpy(host.data(), static_cast<uint8_t*>(device_buffer) +
                      scenario.base, host.size(), hipMemcpyDeviceToHost) !=
            hipSuccess) {
            ok = 0;
        }
        hash = 1469598103934665603ull;
        for (size_t i = 0; i < host.size(); ++i) {
            hash ^= host[i];
            hash *= 1099511628211ull;
        }
        if (hash != expected_hash(scenario.base, scenario.bytes)) ok = 0;
    }
    MPI_Bcast(&ok, 1, MPI_INT, 1, MPI_COMM_WORLD);
    MPI_Bcast(&hash, 1, MPI_UNSIGNED_LONG_LONG, 1, MPI_COMM_WORLD);
    return hash;
}

void launch_scenario(const Scenario& scenario, gicc::Runtime& runtime,
                     int peer, int buffer, const size_t* dynamic_offset,
                     volatile float* scratch) {
    switch (scenario.kind) {
        case Scenario::Tiny:
            gicc::launch<eval_tiny_single>(runtime, dim3(1), dim3(1),
                                           peer, buffer);
            break;
        case Scenario::Reuse:
            gicc::launch<eval_reuse_batch>(runtime, dim3(1), dim3(1),
                                           peer, buffer);
            break;
        case Scenario::Adjacent:
            gicc::launch<eval_adjacent_batch>(runtime, dim3(1), dim3(1),
                                              peer, buffer);
            break;
        case Scenario::Far:
            gicc::launch<eval_far_batch>(runtime, dim3(8), dim3(1),
                                         peer, buffer, scratch);
            break;
        case Scenario::Large:
            gicc::launch<eval_large_single>(runtime, dim3(8), dim3(1),
                                            peer, buffer);
            break;
        case Scenario::Dynamic:
            gicc::launch<eval_dynamic_offset>(runtime, dim3(1), dim3(1),
                                              peer, buffer, dynamic_offset);
            break;
        case Scenario::Static4:
            gicc::launch<eval_static4_parallel>(runtime, dim3(4), dim3(1),
                                                peer, buffer);
            break;
    }
}

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}

bool run_scenario(const Scenario& scenario, gicc::Runtime& runtime, int rank,
                  int peer, int buffer, void* device_buffer,
                  const size_t* dynamic_offset, volatile float* scratch,
                  int warmup, int runs) {
    auto one = [&]() {
        if (rank == 0)
            launch_scenario(scenario, runtime, peer, buffer, dynamic_offset,
                            scratch);
        (void)hipDeviceSynchronize();
        runtime.reset();
    };

    runtime.barrier();
    for (int i = 0; i < warmup; ++i) one();
    if (rank == 1) {
        (void)hipMemset(static_cast<uint8_t*>(device_buffer) + scenario.base,
                        0, scenario.bytes);
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
    const uint64_t hash = check_region(device_buffer, scenario, rank, ok);
    if (rank == 0) {
        const uint64_t staged = runtime.staged_ops() - staged0;
        const uint64_t pushed = runtime.proxy_pushes() - pushed0;
        std::printf(
            "COMPILER_LTO_EVAL scenario=%s runs=%d median_us=%.3f "
            "p25_us=%.3f p75_us=%.3f staged=%llu pushed=%llu "
            "hash=%016llx data=%s\n",
            scenario.name, runs, percentile(samples, 0.50),
            percentile(samples, 0.25), percentile(samples, 0.75),
            static_cast<unsigned long long>(staged),
            static_cast<unsigned long long>(pushed),
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
    if (warmup < 0 || runs <= 0) {
        std::fprintf(stderr, "warmup must be non-negative and runs positive\n");
        return 2;
    }

    gicc::Runtime runtime;
    runtime.enable_host_wait_mode();
    runtime.enable_mixed_dispatch();
    const int rank = runtime.rank();
    if (runtime.size() != 2) {
        if (rank == 0)
            std::fprintf(stderr, "compiler_lto_eval requires exactly two ranks\n");
        return 2;
    }
    const int peer = 1 - rank;

    void* device_buffer = nullptr;
    size_t* dynamic_offset = nullptr;
    float* seed = nullptr;
    float* scratch = nullptr;
    if (hipMalloc(&device_buffer, kBufferBytes) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&dynamic_offset), sizeof(size_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&seed), sizeof(float)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&scratch), 8 * sizeof(float)) != hipSuccess)
        return 3;

    hipLaunchKernelGGL(fill_bytes, dim3(64), dim3(256), 0, 0,
                       static_cast<uint8_t*>(device_buffer), kBufferBytes,
                       rank == 0);
    (void)hipMemset(seed, 0, sizeof(float));
    (void)hipMemset(scratch, 0, 8 * sizeof(float));
    hipLaunchKernelGGL(make_dynamic_offset, dim3(1), dim3(1), 0, 0,
                       dynamic_offset, seed);
    (void)hipDeviceSynchronize();

    auto handle = runtime.register_buffer(device_buffer, kBufferBytes, true);
    runtime.exchange();
    runtime.barrier();

    bool found = false;
    bool all_ok = true;
    for (const auto& scenario : kScenarios) {
        if (selected != "all" && selected != scenario.name) continue;
        found = true;
        all_ok &= run_scenario(scenario, runtime, rank, peer, handle.index,
                               device_buffer, dynamic_offset, scratch,
                               warmup, runs);
    }
    if (!found) {
        if (rank == 0)
            std::fprintf(stderr, "unknown scenario %s (use --list)\n",
                         selected.c_str());
        all_ok = false;
    }

    runtime.barrier();
    (void)hipFree(device_buffer);
    (void)hipFree(dynamic_offset);
    (void)hipFree(seed);
    (void)hipFree(scratch);
    return all_ok ? 0 : 4;
}
