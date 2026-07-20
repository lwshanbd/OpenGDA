// giomp_fused_kernel_bench.cpp - minimal A -> communication -> B experiment.
//
// GiOMP executes one OpenMP target region:
//
//     compute A -> GPU-issued PUT/trigger -> compute B -> completion
//
// GPU-aware MPI cannot call MPI from the target region, so its strongest
// conventional form uses two target regions:
//
//     compute A -> return to host -> MPI_Isend/Irecv -> compute B -> Waitall
//
// Compute B is independent of the message and may overlap communication in
// both versions. Separate no-communication controls quantify the cost of one
// target region versus two target regions with exactly the same A/B work.

#include "gicc/omp.h"

#include <hip/hip_runtime_api.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr size_t kMaxMessageBytes = 1024ULL * 1024;
constexpr size_t kSendOffset = 0;
constexpr size_t kRecvOffset = kMaxMessageBytes;
constexpr size_t kScratchOffset = 2 * kMaxMessageBytes;

struct Options {
    std::string transport = "proxy";
    size_t bytes = 4096;
    int work_a = 500;
    int work_b = 2000;
    int threads = 256;
    int iterations = 51;
    int warmup = 10;
};

struct Measurement {
    double us_per_iteration = 0.0;
    uint64_t final_epoch = 0;
};

[[noreturn]] void usage(const char* argv0, const char* error = nullptr) {
    if (error) std::fprintf(stderr, "error: %s\n", error);
    std::fprintf(stderr,
        "usage: %s --transport=proxy|dwq|mpi [--size=BYTES] "
        "[--work-a=N] [--work-b=N] [--threads=N] "
        "[--iterations=N] [--warmup=N] [--quick]\n",
        argv0);
    std::exit(error ? 2 : 0);
}

size_t parse_bytes(const char* text) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 0);
    if (end == text) return 0;
    if (*end == 'K' || *end == 'k') {
        value *= 1024ULL;
        ++end;
    } else if (*end == 'M' || *end == 'm') {
        value *= 1024ULL * 1024ULL;
        ++end;
    }
    return *end == '\0' ? static_cast<size_t>(value) : 0;
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind("--transport=", 0) == 0) o.transport = arg.substr(12);
        else if (arg.rfind("--size=", 0) == 0) o.bytes = parse_bytes(arg.c_str() + 7);
        else if (arg.rfind("--work-a=", 0) == 0) o.work_a = std::atoi(arg.c_str() + 9);
        else if (arg.rfind("--work-b=", 0) == 0) o.work_b = std::atoi(arg.c_str() + 9);
        else if (arg.rfind("--threads=", 0) == 0) o.threads = std::atoi(arg.c_str() + 10);
        else if (arg.rfind("--iterations=", 0) == 0)
            o.iterations = std::atoi(arg.c_str() + 13);
        else if (arg.rfind("--warmup=", 0) == 0) o.warmup = std::atoi(arg.c_str() + 9);
        else if (arg == "--quick") {
            o.iterations = std::min(o.iterations, 9);
            o.warmup = std::min(o.warmup, 3);
        } else if (arg == "--help" || arg == "-h") usage(argv[0]);
        else usage(argv[0], "unknown option");
    }
    if (o.transport != "proxy" && o.transport != "dwq" && o.transport != "mpi")
        usage(argv[0], "--transport must be proxy, dwq, or mpi");
    if (o.bytes == 0 || o.bytes > kMaxMessageBytes)
        usage(argv[0], "--size must be between 1 byte and 1 MiB");
    if (o.work_a < 0 || o.work_b < 0 || o.threads < 1 || o.threads > 1024 ||
        o.iterations < 1 || o.warmup < 0)
        usage(argv[0], "invalid work/thread/iteration count");
    return o;
}

void check_hip(hipError_t status, const char* what, int rank) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "rank %d: %s failed: %s\n",
                 rank, what, hipGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 3);
}

void require_two_distinct_nodes(int rank, int nranks) {
    if (nranks != 2) {
        if (rank == 0) std::fprintf(stderr, "benchmark requires exactly two ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    char local[MPI_MAX_PROCESSOR_NAME] = {};
    char names[2][MPI_MAX_PROCESSOR_NAME] = {};
    int length = 0;
    MPI_Get_processor_name(local, &length);
    MPI_Allgather(local, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
    if (std::strcmp(names[0], names[1]) == 0) {
        if (rank == 0)
            std::fprintf(stderr, "same-node placement is not allowed (%s)\n", names[0]);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    if (rank == 0)
        std::printf("nodes=%s,%s (cross-node confirmed)\n", names[0], names[1]);
}

unsigned char iteration_value(int rank, uint64_t epoch) {
    return static_cast<unsigned char>(1 + ((epoch * 19 + rank * 71) % 251));
}

#pragma omp declare target
inline uint64_t run_work(uint64_t value, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        value *= 2685821657736338717ULL;
    }
    return value;
}
#pragma omp end declare target

void execute_compute_fused(unsigned char* data, size_t bytes,
                           int work_a, int work_b, int threads,
                           uint64_t epoch, unsigned char value) {
    #pragma omp target teams num_teams(1) thread_limit(threads) \
        is_device_ptr(data) firstprivate(bytes, work_a, work_b, epoch, value)
    {
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            auto* scratch = reinterpret_cast<volatile uint64_t*>(data + kScratchOffset);
            uint64_t state = run_work(epoch ^ static_cast<uint64_t>(tid + 1), work_a);
            scratch[tid] = state;
            for (size_t i = static_cast<size_t>(tid); i < bytes;
                 i += static_cast<size_t>(nthreads)) {
                data[kSendOffset + i] = value;
            }
            #pragma omp barrier
            state = scratch[tid];
            scratch[tid] = run_work(state, work_b);
        }
    }
}

void execute_phase_a(unsigned char* data, size_t bytes, int work_a,
                     int threads, uint64_t epoch, unsigned char value) {
    #pragma omp target teams num_teams(1) thread_limit(threads) \
        is_device_ptr(data) firstprivate(bytes, work_a, epoch, value)
    {
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            auto* scratch = reinterpret_cast<volatile uint64_t*>(data + kScratchOffset);
            scratch[tid] = run_work(epoch ^ static_cast<uint64_t>(tid + 1), work_a);
            for (size_t i = static_cast<size_t>(tid); i < bytes;
                 i += static_cast<size_t>(nthreads)) {
                data[kSendOffset + i] = value;
            }
        }
    }
}

void execute_phase_b(unsigned char* data, int work_b, int threads) {
    #pragma omp target teams num_teams(1) thread_limit(threads) \
        is_device_ptr(data) firstprivate(work_b)
    {
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto* scratch = reinterpret_cast<volatile uint64_t*>(data + kScratchOffset);
            scratch[tid] = run_work(scratch[tid], work_b);
        }
    }
}

void execute_proxy_fused(unsigned char* data, gicc::DeviceCtx* ctx,
                         int peer, int buffer_index, size_t bytes,
                         int work_a, int work_b, int threads,
                         uint64_t epoch, unsigned char value) {
    #pragma omp target teams num_teams(1) thread_limit(threads) \
        is_device_ptr(data, ctx) \
        firstprivate(peer, buffer_index, bytes, work_a, work_b, epoch, value)
    {
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            auto* scratch = reinterpret_cast<volatile uint64_t*>(data + kScratchOffset);
            uint64_t state = run_work(epoch ^ static_cast<uint64_t>(tid + 1), work_a);
            scratch[tid] = state;
            for (size_t i = static_cast<size_t>(tid); i < bytes;
                 i += static_cast<size_t>(nthreads)) {
                data[kSendOffset + i] = value;
            }
            #pragma omp barrier
            if (tid == 0) {
                ompx_put_proxy(ctx, peer, buffer_index, kRecvOffset,
                               buffer_index, kSendOffset, bytes);
            }
            #pragma omp barrier
            state = scratch[tid];
            scratch[tid] = run_work(state, work_b);
            #pragma omp barrier
            if (tid == 0) ompx_quiet(ctx);
        }
    }
}

void execute_dwq_fused(unsigned char* data, gicc::DeviceCtx* ctx,
                       size_t bytes, int work_a, int work_b, int threads,
                       uint64_t epoch, unsigned char value) {
    #pragma omp target teams num_teams(1) thread_limit(threads) \
        is_device_ptr(data, ctx) \
        firstprivate(bytes, work_a, work_b, epoch, value)
    {
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            auto* scratch = reinterpret_cast<volatile uint64_t*>(data + kScratchOffset);
            uint64_t state = run_work(epoch ^ static_cast<uint64_t>(tid + 1), work_a);
            scratch[tid] = state;
            for (size_t i = static_cast<size_t>(tid); i < bytes;
                 i += static_cast<size_t>(nthreads)) {
                data[kSendOffset + i] = value;
            }
            #pragma omp barrier
            if (tid == 0) {
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
                *(ctx->trigger_addr_) = ctx->trigger_val_;
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
            }
            #pragma omp barrier
            state = scratch[tid];
            scratch[tid] = run_work(state, work_b);
        }
    }
}

void execute_giomp_full(const Options& options, bool use_dwq,
                        unsigned char* data, int peer, int buffer_index,
                        uint64_t epoch, unsigned char value) {
    gicc::DeviceCtx* ctx = ompx_prepare();
    if (use_dwq) {
        ompx_dwq_stage_put(peer, buffer_index, kRecvOffset,
                           buffer_index, kSendOffset, options.bytes);
        ompx_dwq_arm();
        execute_dwq_fused(data, ctx, options.bytes,
                          options.work_a, options.work_b, options.threads,
                          epoch, value);
    } else {
        execute_proxy_fused(data, ctx, peer, buffer_index, options.bytes,
                            options.work_a, options.work_b, options.threads,
                            epoch, value);
    }
    ompx_quiet_host();
}

void execute_mpi_full(const Options& options, unsigned char* data,
                      int peer, uint64_t epoch, unsigned char value) {
    MPI_Request requests[2];
    const int tag = 100 + static_cast<int>(epoch % 30000);
    MPI_Irecv(data + kRecvOffset, static_cast<int>(options.bytes), MPI_BYTE,
              peer, tag, MPI_COMM_WORLD, &requests[0]);
    execute_phase_a(data, options.bytes, options.work_a,
                    options.threads, epoch, value);
    MPI_Isend(data + kSendOffset, static_cast<int>(options.bytes), MPI_BYTE,
              peer, tag, MPI_COMM_WORLD, &requests[1]);
    execute_phase_b(data, options.work_b, options.threads);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
}

template <typename Execute>
Measurement measure(const Options& options, int rank,
                    uint64_t& next_epoch, Execute&& execute) {
    Measurement result;
    for (int i = 0; i < options.warmup; ++i) {
        const uint64_t epoch = next_epoch++;
        execute(epoch, iteration_value(rank, epoch));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int i = 0; i < options.iterations; ++i) {
        const uint64_t epoch = next_epoch++;
        execute(epoch, iteration_value(rank, epoch));
        result.final_epoch = epoch;
    }
    const double local_elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Bcast(&max_elapsed, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    result.us_per_iteration = max_elapsed * 1.0e6 /
                              static_cast<double>(options.iterations);
    MPI_Barrier(MPI_COMM_WORLD);
    return result;
}

int verify_received(unsigned char* data, size_t bytes,
                    int rank, int peer, uint64_t final_epoch) {
    std::vector<unsigned char> received(bytes);
    check_hip(hipMemcpy(received.data(), data + kRecvOffset, bytes,
                        hipMemcpyDeviceToHost), "hipMemcpy verify", rank);
    const unsigned char expected = iteration_value(peer, final_epoch);
    unsigned long long errors = 0;
    for (unsigned char byte : received) errors += byte != expected;
    unsigned long long total = 0;
    MPI_Allreduce(&errors, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    return total == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const Options options = parse_options(argc, argv);
    const bool use_mpi = options.transport == "mpi";
    const bool use_dwq = options.transport == "dwq";
    const size_t buffer_bytes = kScratchOffset +
        static_cast<size_t>(options.threads) * sizeof(uint64_t);

    int rank = -1;
    int nranks = 0;
    unsigned char* data = nullptr;
    ompx_buffer giomp_buffer{};

    if (use_mpi) {
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        const char* gpu_support = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
        const char* async_progress = std::getenv("MPICH_ASYNC_PROGRESS");
        if (!gpu_support || std::strcmp(gpu_support, "1") != 0 ||
            !async_progress || std::strcmp(async_progress, "1") != 0 ||
            provided < MPI_THREAD_MULTIPLE) {
            if (rank == 0)
                std::fprintf(stderr,
                    "MPI mode requires GPU support, async progress, and THREAD_MULTIPLE\n");
            MPI_Abort(MPI_COMM_WORLD, 6);
        }
        check_hip(hipSetDevice(0), "hipSetDevice", rank);
        omp_set_default_device(0);
        check_hip(hipMalloc(reinterpret_cast<void**>(&data), buffer_bytes),
                  "hipMalloc", rank);
    } else {
        ompx_init();
        rank = omp_get_rank_num();
        nranks = omp_get_num_ranks();
        if (use_dwq != ompx_dwq_enabled()) {
            if (rank == 0)
                std::fprintf(stderr, "%s requires GICC_HALO_DWQ=%d\n",
                             options.transport.c_str(), use_dwq ? 1 : 0);
            MPI_Abort(MPI_COMM_WORLD, 7);
        }
        giomp_buffer = ompx_alloc(buffer_bytes);
        data = static_cast<unsigned char*>(giomp_buffer.ptr);
        ompx_exchange();
    }

    require_two_distinct_nodes(rank, nranks);
    const int peer = rank ^ 1;
    if (!use_mpi && ompx_ipc_reachable(peer, giomp_buffer.index)) {
        if (rank == 0) std::fprintf(stderr, "unexpected IPC-reachable peer\n");
        MPI_Abort(MPI_COMM_WORLD, 8);
    }
    check_hip(hipMemset(data, 0, buffer_bytes), "hipMemset", rank);
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize", rank);

    if (rank == 0) {
        std::printf("giomp_fused_kernel transport=%s size=%zu work_a=%d work_b=%d "
                    "threads=%d iterations=%d warmup=%d IPC=excluded\n",
                    options.transport.c_str(), options.bytes,
                    options.work_a, options.work_b, options.threads,
                    options.iterations, options.warmup);
    }

    uint64_t next_epoch = 1;
    const Measurement one_kernel = measure(options, rank, next_epoch,
        [&](uint64_t epoch, unsigned char value) {
            execute_compute_fused(data, options.bytes,
                                  options.work_a, options.work_b,
                                  options.threads, epoch, value);
        });
    const Measurement two_kernel = measure(options, rank, next_epoch,
        [&](uint64_t epoch, unsigned char value) {
            execute_phase_a(data, options.bytes, options.work_a,
                            options.threads, epoch, value);
            execute_phase_b(data, options.work_b, options.threads);
        });
    const Measurement full = measure(options, rank, next_epoch,
        [&](uint64_t epoch, unsigned char value) {
            if (use_mpi)
                execute_mpi_full(options, data, peer, epoch, value);
            else
                execute_giomp_full(options, use_dwq, data, peer,
                                   giomp_buffer.index, epoch, value);
        });

    const int verify_error = verify_received(data, options.bytes, rank, peer,
                                             full.final_epoch);
    if (rank == 0) {
        const double extra_launch =
            two_kernel.us_per_iteration - one_kernel.us_per_iteration;
        const char* verify = verify_error == 0 ? "PASS" : "FAIL";
        std::printf("one_kernel_us=%.3f two_kernel_us=%.3f "
                    "extra_target_us=%.3f full_us=%.3f %s\n",
                    one_kernel.us_per_iteration, two_kernel.us_per_iteration,
                    extra_launch, full.us_per_iteration, verify);
        std::printf("RESULT,%s,%zu,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%s\n",
                    options.transport.c_str(), options.bytes,
                    options.work_a, options.work_b, options.threads,
                    one_kernel.us_per_iteration, two_kernel.us_per_iteration,
                    extra_launch, full.us_per_iteration, verify);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (use_mpi) {
        check_hip(hipFree(data), "hipFree", rank);
        MPI_Finalize();
    } else {
        ompx_free(giomp_buffer);
        ompx_finalize();
    }
    return verify_error == 0 ? 0 : 9;
}
