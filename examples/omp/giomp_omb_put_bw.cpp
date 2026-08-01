// GiOMP adaptation of the OSU Micro-Benchmarks osu_put_bw test.
//
// The measured loop follows OMB 7.5.2's default MPI_Put bandwidth protocol:
// a two-rank run, a 64-operation window over disjoint offsets, one completion
// operation per window, powers-of-two message sizes through 4 MiB, and OMB's
// small/large iteration and warm-up counts.  GiOMP proxy completion and GPU
// trigger completion replace MPI_Win_flush in their respective variants.
//
// Copyright (c) 2003-2025 the Network-Based Computing Laboratory
// (NBCL), The Ohio State University.
//
// Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of The Ohio State University nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gicc/omp.h"
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kDefaultMinMessage = 1;
constexpr size_t kDefaultMaxMessage = 1U << 22;
constexpr size_t kLargeMessage = 8192;
constexpr int kDefaultWindow = 64;
constexpr int kSmallIterations = 100;
constexpr int kSmallWarmup = 10;
constexpr int kLargeIterations = 20;
constexpr int kLargeWarmup = 2;
constexpr int kLatencySmallIterations = 10000;
constexpr int kLatencySmallWarmup = 100;
constexpr int kLatencyLargeIterations = 1000;
constexpr int kLatencyLargeWarmup = 10;

struct Options {
    std::string transport = "proxy";
    size_t min_message = kDefaultMinMessage;
    size_t max_message = kDefaultMaxMessage;
    int window = kDefaultWindow;
    int proxy_lanes = 1;
    bool parallel_proxy = false;
    bool in_kernel_windows = false;
    bool general_producer = false;
    bool latency = false;
    bool quick = false;
};

[[noreturn]] void usage(const char* argv0, const char* error = nullptr) {
    if (error) std::fprintf(stderr, "error: %s\n", error);
    std::fprintf(stderr,
        "usage: %s --transport=proxy|gpu-trigger|mpi-rma "
        "[--window=N] [--min-size=BYTES] [--max-size=BYTES] "
        "[--proxy-lanes=N] [--parallel-proxy] [--in-kernel-windows] "
        "[--general-producer] [--latency] [--quick]\n",
        argv0);
    std::exit(2);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind("--transport=", 0) == 0)
            options.transport = arg.substr(12);
        else if (arg.rfind("--window=", 0) == 0)
            options.window = std::atoi(arg.c_str() + 9);
        else if (arg.rfind("--min-size=", 0) == 0)
            options.min_message = std::strtoull(arg.c_str() + 11, nullptr, 0);
        else if (arg.rfind("--max-size=", 0) == 0)
            options.max_message = std::strtoull(arg.c_str() + 11, nullptr, 0);
        else if (arg.rfind("--proxy-lanes=", 0) == 0)
            options.proxy_lanes = std::atoi(arg.c_str() + 14);
        else if (arg == "--parallel-proxy")
            options.parallel_proxy = true;
        else if (arg == "--in-kernel-windows")
            options.in_kernel_windows = true;
        else if (arg == "--general-producer")
            options.general_producer = true;
        else if (arg == "--latency")
            options.latency = true;
        else if (arg == "--quick")
            options.quick = true;
        else
            usage(argv[0], "unknown option");
    }
    if (options.transport != "proxy" &&
        options.transport != "gpu-trigger" &&
        options.transport != "mpi-rma")
        usage(argv[0], "unsupported transport");
    if (options.window < 1)
        usage(argv[0], "window must be positive");
    if (options.proxy_lanes < 1 || options.proxy_lanes > 32)
        usage(argv[0], "proxy lanes must be in [1,32]");
    if (options.in_kernel_windows && options.transport != "proxy")
        usage(argv[0], "in-kernel windows apply only to the proxy transport");
    if (options.in_kernel_windows && options.parallel_proxy)
        usage(argv[0], "in-kernel windows use a single producer work-item");
    if (options.general_producer && !options.in_kernel_windows)
        usage(argv[0], "general producer applies only to in-kernel windows");
    if (options.min_message < 1 ||
        options.max_message < options.min_message)
        usage(argv[0], "invalid message-size range");
    if (options.latency)
        options.window = 1;
    if (options.quick) {
        options.min_message = 8;
        options.max_message = 1U << 20;
        options.window = std::min(options.window, 8);
    }
    return options;
}

void check_gpu(GpuError status, const char* what, int rank) {
    if (status == GPU_SUCCESS) return;
    std::fprintf(stderr, "rank %d: %s failed: %s\n",
                 rank, what, gpuGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 3);
}

void check_mpi(int status, const char* what, int rank) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(status, message, &length);
    std::fprintf(stderr, "rank %d: %s failed: %.*s\n",
                 rank, what, length, message);
    MPI_Abort(MPI_COMM_WORLD, 4);
}

void require_two_distinct_nodes(int rank, int nranks) {
    if (nranks != 2) {
        if (rank == 0)
            std::fprintf(stderr, "giomp_omb_put_bw requires exactly 2 ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    char local[MPI_MAX_PROCESSOR_NAME] = {};
    int length = 0;
    MPI_Get_processor_name(local, &length);
    char names[2][MPI_MAX_PROCESSOR_NAME] = {};
    MPI_Allgather(local, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
    if (std::strcmp(names[0], names[1]) == 0) {
        if (rank == 0)
            std::fprintf(stderr, "benchmark requires one rank on each of two nodes\n");
        MPI_Abort(MPI_COMM_WORLD, 6);
    }
    if (rank == 0)
        std::printf("nodes=%s,%s (cross-node confirmed)\n", names[0], names[1]);
}

void run_giomp_window(const Options& options, int source_index,
                      int target_index,
                      size_t bytes, int peer) {
    gicc::DeviceCtx* ctx = ompx_prepare();
    if (options.transport == "gpu-trigger") {
        for (int j = 0; j < options.window; ++j) {
            const size_t offset = static_cast<size_t>(j) * bytes;
            ompx_dwq_stage_put(peer, target_index, offset,
                               source_index, offset, bytes);
        }
        ompx_dwq_trigger(ctx);
    } else {
        const int window = options.window;
        const int proxy_lanes = options.proxy_lanes;
        if (options.parallel_proxy) {
            #pragma omp target teams distribute parallel for is_device_ptr(ctx) firstprivate(peer, source_index, target_index, bytes, window, proxy_lanes)
            for (int j = 0; j < window; ++j) {
                const size_t offset = static_cast<size_t>(j) * bytes;
                ompx_put_proxy(ctx, peer, target_index, offset,
                               source_index, offset, bytes, j % proxy_lanes);
            }
        } else {
            #pragma omp target is_device_ptr(ctx) firstprivate(peer, source_index, target_index, bytes, window, proxy_lanes)
            {
                for (int j = 0; j < window; ++j) {
                    const size_t offset = static_cast<size_t>(j) * bytes;
                    ompx_put_proxy(ctx, peer, target_index, offset,
                                   source_index, offset, bytes,
                                   j % proxy_lanes);
                }
            }
        }
    }
    ompx_quiet_host();
}

void run_mpi_rma_window(void* source_buffer, size_t bytes, int window,
                        MPI_Win mpi_window, int rank) {
    auto* base = static_cast<unsigned char*>(source_buffer);
    for (int j = 0; j < window; ++j) {
        const size_t offset = static_cast<size_t>(j) * bytes;
        check_mpi(MPI_Put(base + offset, static_cast<int>(bytes), MPI_BYTE,
                          1, static_cast<MPI_Aint>(offset),
                          static_cast<int>(bytes), MPI_BYTE, mpi_window),
                  "MPI_Put", rank);
    }
    check_mpi(MPI_Win_flush(1, mpi_window), "MPI_Win_flush", rank);
}

void run_proxy_windows_in_kernel(const Options& options, int source_index,
                                 int target_index, size_t bytes, int peer,
                                 int iterations) {
    if (iterations <= 0) return;
    gicc::DeviceCtx* ctx = ompx_prepare();
    const int window = options.window;
    const int proxy_lanes = options.proxy_lanes;
    const bool general_producer = options.general_producer;
    #pragma omp target is_device_ptr(ctx) firstprivate(peer, source_index, target_index, bytes, window, proxy_lanes, iterations, general_producer)
    {
        for (int i = 0; i < iterations; ++i) {
            for (int j = 0; j < window; ++j) {
                const size_t offset = static_cast<size_t>(j) * bytes;
                if (general_producer)
                    ompx_put_proxy(ctx, peer, target_index, offset,
                                   source_index, offset, bytes,
                                   j % proxy_lanes);
                else
                    ompx_put_proxy_single(ctx, peer, target_index, offset,
                                          source_index, offset, bytes,
                                          j % proxy_lanes);
            }
            for (int lane = 0; lane < proxy_lanes; ++lane) {
                if (general_producer)
                    ompx_quiet(ctx, lane);
                else
                    ompx_quiet_single(ctx, lane);
            }
        }
    }
    ompx_quiet_host();
}

int verify_result(int rank, void* target_buffer, size_t bytes, int window,
                  unsigned char expected) {
    int local_errors = 0;
    if (rank == 1) {
        auto* base = static_cast<unsigned char*>(target_buffer);
        for (int j = 0; j < window; ++j) {
            const size_t offset = static_cast<size_t>(j) * bytes;
            unsigned char first = 0;
            unsigned char last = 0;
            if (gpuMemcpy(&first, base + offset, 1,
                          gpuMemcpyDeviceToHost) != GPU_SUCCESS)
                ++local_errors;
            if (gpuMemcpy(&last, base + offset + bytes - 1, 1,
                          gpuMemcpyDeviceToHost) != GPU_SUCCESS)
                ++local_errors;
            if (first != expected || last != expected)
                ++local_errors;
        }
    }
    int errors = 0;
    MPI_Allreduce(&local_errors, &errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return errors;
}

std::vector<size_t> message_sizes(const Options& options) {
    std::vector<size_t> sizes;
    for (size_t bytes = 1; bytes <= options.max_message; bytes *= 2) {
        if (bytes >= options.min_message)
            sizes.push_back(bytes);
        if (bytes > options.max_message / 2)
            break;
    }
    return sizes;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const Options options = parse_options(argc, argv);
    const bool use_mpi_rma = options.transport == "mpi-rma";

    int rank = -1;
    int nranks = 0;
    void* source_buffer = nullptr;
    void* target_buffer = nullptr;
    ompx_buffer giomp_source{};
    ompx_buffer giomp_target{};
    MPI_Win mpi_window = MPI_WIN_NULL;
    const size_t buffer_bytes =
        options.max_message * static_cast<size_t>(options.window);

    if (use_mpi_rma) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        check_gpu(gpuSetDevice(0), "gpuSetDevice", rank);
        omp_set_default_device(0);
    } else {
        ompx_init();
        rank = omp_get_rank_num();
        nranks = omp_get_num_ranks();
        const bool expect_trigger = options.transport == "gpu-trigger";
        if (ompx_dwq_enabled() != expect_trigger) {
            if (rank == 0)
                std::fprintf(stderr,
                    "%s requires GICC_HALO_DWQ=%d before initialization\n",
                    options.transport.c_str(), expect_trigger ? 1 : 0);
            MPI_Abort(MPI_COMM_WORLD, 7);
        }
        giomp_source = ompx_alloc(buffer_bytes);
        giomp_target = ompx_alloc(buffer_bytes);
        source_buffer = giomp_source.ptr;
        target_buffer = giomp_target.ptr;
        ompx_exchange();
    }

    require_two_distinct_nodes(rank, nranks);
    if (rank == 0) {
        std::printf("GiOMP OMB-derived PUT %s benchmark\n",
                    options.latency ? "latency" : "bandwidth");
        std::printf("OMB version=7.5.2 transport=%s window=%d max_bytes=%zu "
                    "proxy_lanes=%d parallel_proxy=%d in_kernel_windows=%d "
                    "general_producer=%d\n",
                    options.transport.c_str(), options.window,
                    options.max_message, options.proxy_lanes,
                    options.parallel_proxy ? 1 : 0,
                    options.in_kernel_windows ? 1 : 0,
                    options.general_producer ? 1 : 0);
        std::printf("# bytes       MB/s       us/message  iterations  warmup  verify\n");
        std::printf("CSV schema: RESULT,transport,size_bytes,window,iterations,"
                    "warmup,MBps,us_per_message,verify\n");
    }

    for (size_t bytes : message_sizes(options)) {
        const int iterations = options.quick ? 5 :
            (options.latency
                ? (bytes > kLargeMessage
                    ? kLatencyLargeIterations : kLatencySmallIterations)
                : (bytes > kLargeMessage
                    ? kLargeIterations : kSmallIterations));
        const int warmup = options.quick ? 2 :
            (options.latency
                ? (bytes > kLargeMessage
                    ? kLatencyLargeWarmup : kLatencySmallWarmup)
                : (bytes > kLargeMessage
                    ? kLargeWarmup : kSmallWarmup));
        const size_t span = bytes * static_cast<size_t>(options.window);
        const unsigned char expected = 0x5a;

        // OMB creates buffers and the RMA window separately for every message
        // size.  Preserve that behavior because window extent can affect the
        // implementation's selected RMA path.
        if (use_mpi_rma) {
            check_gpu(gpuMalloc(&source_buffer, span),
                      "gpuMalloc(source)", rank);
            check_gpu(gpuMalloc(&target_buffer, span),
                      "gpuMalloc(target)", rank);
            check_mpi(MPI_Win_create(target_buffer,
                                     static_cast<MPI_Aint>(span), 1,
                                     MPI_INFO_NULL, MPI_COMM_WORLD,
                                     &mpi_window),
                      "MPI_Win_create", rank);
        }

        check_gpu(gpuMemset(source_buffer, rank == 0 ? expected : 0, span),
                  "gpuMemset(source)", rank);
        check_gpu(gpuMemset(target_buffer, 0, span),
                  "gpuMemset(target)", rank);
        check_gpu(gpuDeviceSynchronize(), "gpuDeviceSynchronize", rank);
        MPI_Barrier(MPI_COMM_WORLD);

        double elapsed = 0.0;
        if (rank == 0) {
            if (use_mpi_rma)
                check_mpi(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, mpi_window),
                          "MPI_Win_lock", rank);
            if (options.in_kernel_windows) {
                run_proxy_windows_in_kernel(
                    options, giomp_source.index, giomp_target.index,
                    bytes, 1, warmup);
                elapsed = -MPI_Wtime();
                run_proxy_windows_in_kernel(
                    options, giomp_source.index, giomp_target.index,
                    bytes, 1, iterations);
            } else {
                for (int i = 0; i < iterations + warmup; ++i) {
                    if (i == warmup)
                        elapsed = -MPI_Wtime();
                    if (use_mpi_rma)
                        run_mpi_rma_window(source_buffer, bytes, options.window,
                                           mpi_window, rank);
                    else
                        run_giomp_window(options, giomp_source.index,
                                         giomp_target.index, bytes, 1);
                }
            }
            elapsed += MPI_Wtime();
            if (use_mpi_rma)
                check_mpi(MPI_Win_unlock(1, mpi_window),
                          "MPI_Win_unlock", rank);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const int errors = verify_result(rank, target_buffer, bytes,
                                         options.window, expected);
        if (rank == 0) {
            const double messages =
                static_cast<double>(iterations) * options.window;
            const double mb_per_second =
                (static_cast<double>(bytes) / 1.0e6) * messages / elapsed;
            const double us_per_message = elapsed * 1.0e6 / messages;
            const char* status = errors == 0 ? "PASS" : "FAIL";
            std::printf("%-10zu %10.2f %16.3f %11d %7d %7s\n",
                        bytes, mb_per_second, us_per_message,
                        iterations, warmup, status);
            std::printf("RESULT,%s,%zu,%d,%d,%d,%.6f,%.6f,%s\n",
                        options.transport.c_str(), bytes, options.window,
                        iterations, warmup, mb_per_second,
                        us_per_message, status);
        }
        if (errors != 0)
            MPI_Abort(MPI_COMM_WORLD, 8);

        if (use_mpi_rma) {
            check_mpi(MPI_Win_free(&mpi_window), "MPI_Win_free", rank);
            check_gpu(gpuFree(source_buffer), "gpuFree(source)", rank);
            check_gpu(gpuFree(target_buffer), "gpuFree(target)", rank);
            source_buffer = nullptr;
            target_buffer = nullptr;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("ALL_CASES_PASS transport=%s\n", options.transport.c_str());

    if (use_mpi_rma) {
        MPI_Finalize();
    } else {
        ompx_free(giomp_source);
        ompx_free(giomp_target);
        ompx_finalize();
    }
    return 0;
}
