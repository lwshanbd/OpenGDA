// giomp_streaming_halo.cpp - GPU-resident halo production/communication overlap.
//
// The benchmark models one iteration of a stencil-like application:
//   1. GPU team 0 produces a halo in device memory.
//   2. The halo is handed to the network.
//   3. Independent GPU work runs while the transfer is in flight.
//   4. Both the work and the transfer complete before the next iteration.
//
// GiOMP proxy and DWQ perform step 2 from inside the producer target region.
// The optimized GPU-aware MPI path preposts the receive, runs a producer target,
// posts MPI_Isend, and overlaps MPI_Waitall with an asynchronous interior target.
// This deliberately compares against an overlap-capable MPI baseline rather than
// a serialized "kernel, MPI, kernel" implementation.

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

constexpr size_t kMaxMessageBytes = 4ULL * 1024 * 1024;
constexpr size_t kSendOffset = 0;
constexpr size_t kRecvOffset = kMaxMessageBytes;
constexpr size_t kReadyOffset = 2 * kMaxMessageBytes;
constexpr size_t kControlBytes = 4096;
constexpr size_t kScratchOffset = kReadyOffset + kControlBytes;

enum class Path : int { None = 0, Proxy = 1, Dwq = 2 };

struct Options {
    std::string transport = "proxy";
    size_t bytes = 64 * 1024;
    int tiles = 1;
    int producer_work = 0;
    int work = 10000;
    int teams = 32;
    int threads = 256;
    int iterations = 31;
    int warmup = 5;
};

struct Measurement {
    double us_per_iteration = 0.0;
    uint64_t final_epoch = 0;
};

[[noreturn]] void usage(const char* argv0, const char* error = nullptr) {
    if (error) std::fprintf(stderr, "error: %s\n", error);
    std::fprintf(stderr,
        "usage: %s --transport=proxy|dwq|mpi [--size=BYTES] [--tiles=N] "
        "[--producer-work=N] [--work=N] "
        "[--teams=N] [--threads=N] [--iterations=N] [--warmup=N] [--quick]\n",
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
        else if (arg.rfind("--tiles=", 0) == 0) o.tiles = std::atoi(arg.c_str() + 8);
        else if (arg.rfind("--producer-work=", 0) == 0)
            o.producer_work = std::atoi(arg.c_str() + 16);
        else if (arg.rfind("--work=", 0) == 0) o.work = std::atoi(arg.c_str() + 7);
        else if (arg.rfind("--teams=", 0) == 0) o.teams = std::atoi(arg.c_str() + 8);
        else if (arg.rfind("--threads=", 0) == 0) o.threads = std::atoi(arg.c_str() + 10);
        else if (arg.rfind("--iterations=", 0) == 0)
            o.iterations = std::atoi(arg.c_str() + 13);
        else if (arg.rfind("--warmup=", 0) == 0) o.warmup = std::atoi(arg.c_str() + 9);
        else if (arg == "--quick") {
            o.iterations = std::min(o.iterations, 7);
            o.warmup = std::min(o.warmup, 2);
        } else if (arg == "--help" || arg == "-h") usage(argv[0]);
        else usage(argv[0], "unknown option");
    }
    if (o.transport != "proxy" && o.transport != "dwq" && o.transport != "mpi")
        usage(argv[0], "--transport must be proxy, dwq, or mpi");
    if (o.bytes == 0 || o.bytes > kMaxMessageBytes)
        usage(argv[0], "--size must be between 1 byte and 4 MiB");
    if (o.work < 0 || o.producer_work < 0 || o.tiles < 1 || o.tiles > 64 ||
        o.teams < o.tiles || o.teams > 64 ||
        o.threads < 1 || o.threads > 1024 ||
        o.iterations < 1 || o.warmup < 0)
        usage(argv[0], "invalid tile/work/team/thread/iteration count");
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
        if (rank == 0)
            std::fprintf(stderr, "streaming halo requires exactly two ranks\n");
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
    return static_cast<unsigned char>(1 + ((epoch * 17 + rank * 67) % 251));
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

void execute_giomp_iteration(Path path, unsigned char* data,
                             int peer, int buffer_index, size_t bytes,
                             int tiles, int producer_work, int work,
                             int teams, int threads,
                             uint64_t epoch, unsigned char value) {
    gicc::DeviceCtx* ctx = ompx_prepare();
    if (path == Path::Dwq) {
        for (int tile = 0; tile < tiles; ++tile) {
            const size_t begin = bytes * static_cast<size_t>(tile) /
                                 static_cast<size_t>(tiles);
            const size_t end = bytes * static_cast<size_t>(tile + 1) /
                               static_cast<size_t>(tiles);
            ompx_dwq_stage_put(peer, buffer_index, kRecvOffset + begin,
                               buffer_index, kSendOffset + begin, end - begin);
        }
        // Arm on the host, but fire the doorbell only after the GPU produces
        // the halo. This avoids the extra target launch in ompx_dwq_trigger().
        ompx_dwq_arm();
    }

    const int path_value = static_cast<int>(path);
    #pragma omp target teams num_teams(teams) thread_limit(threads) \
        is_device_ptr(data, ctx) \
        firstprivate(path_value, peer, buffer_index, bytes, tiles, producer_work, \
                     work, threads, epoch, value)
    {
        const int team = omp_get_team_num();
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int team_threads = omp_get_num_threads();
            auto* ready = reinterpret_cast<uint64_t*>(data + kReadyOffset);
            auto* scratch = reinterpret_cast<volatile uint64_t*>(data + kScratchOffset);

            if (team < tiles) {
                const size_t begin = bytes * static_cast<size_t>(team) /
                                     static_cast<size_t>(tiles);
                const size_t end = bytes * static_cast<size_t>(team + 1) /
                                   static_cast<size_t>(tiles);
                const int delay = static_cast<int>(
                    static_cast<int64_t>(producer_work) * (team + 1) / tiles);
                const uint64_t producer_seed =
                    epoch ^ 0xd1b54a32d192ed03ULL ^
                    (static_cast<uint64_t>(team) << 32) ^
                    static_cast<uint64_t>(tid + 1);
                scratch[static_cast<size_t>(team) * static_cast<size_t>(threads) + tid] =
                    run_work(producer_seed, delay);
                for (size_t i = begin + static_cast<size_t>(tid); i < end;
                     i += static_cast<size_t>(team_threads)) {
                    data[kSendOffset + i] = value;
                }
                #pragma omp barrier
                if (tid == 0) {
                    if (path_value == static_cast<int>(Path::Proxy)) {
                        ompx_put_proxy(ctx, peer, buffer_index, kRecvOffset + begin,
                                       buffer_index, kSendOffset + begin, end - begin);
                    }
                    __atomic_store_n(&ready[team], epoch, __ATOMIC_RELEASE);
                }
            }

            for (int tile = 0; tile < tiles; ++tile) {
                while (__atomic_load_n(&ready[tile], __ATOMIC_ACQUIRE) != epoch) {
#ifdef __AMDGCN__
                    __builtin_amdgcn_s_sleep(1);
#endif
                }
            }

            if (path_value == static_cast<int>(Path::Dwq) && team == 0 && tid == 0) {
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
                *(ctx->trigger_addr_) = ctx->trigger_val_;
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
            }

            uint64_t seed = epoch ^ (static_cast<uint64_t>(team) << 32)
                                      ^ static_cast<uint64_t>(tid + 1);
            scratch[static_cast<size_t>(team) * static_cast<size_t>(threads) + tid] =
                run_work(seed, work);

            #pragma omp barrier
            if (path_value == static_cast<int>(Path::Proxy) && team == 0 && tid == 0)
                ompx_quiet(ctx);
        }
    }
    ompx_quiet_host();
}

void execute_mpi_compute_only(unsigned char* data, size_t bytes,
                              int tiles, int producer_work, int work,
                              int teams, int threads, uint64_t epoch,
                              unsigned char value) {
    #pragma omp target teams num_teams(teams) thread_limit(threads) \
        is_device_ptr(data) firstprivate(bytes, tiles, producer_work, work, \
                                         threads, epoch, value)
    {
        const int team = omp_get_team_num();
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int team_threads = omp_get_num_threads();
            auto* ready = reinterpret_cast<uint64_t*>(data + kReadyOffset);
            auto* scratch = reinterpret_cast<volatile uint64_t*>(data + kScratchOffset);
            if (team < tiles) {
                const size_t begin = bytes * static_cast<size_t>(team) /
                                     static_cast<size_t>(tiles);
                const size_t end = bytes * static_cast<size_t>(team + 1) /
                                   static_cast<size_t>(tiles);
                const int delay = static_cast<int>(
                    static_cast<int64_t>(producer_work) * (team + 1) / tiles);
                const uint64_t producer_seed =
                    epoch ^ 0xd1b54a32d192ed03ULL ^
                    (static_cast<uint64_t>(team) << 32) ^
                    static_cast<uint64_t>(tid + 1);
                scratch[static_cast<size_t>(team) * static_cast<size_t>(threads) + tid] =
                    run_work(producer_seed, delay);
                for (size_t i = begin + static_cast<size_t>(tid); i < end;
                     i += static_cast<size_t>(team_threads)) {
                    data[kSendOffset + i] = value;
                }
                #pragma omp barrier
                if (tid == 0)
                    __atomic_store_n(&ready[team], epoch, __ATOMIC_RELEASE);
            }
            for (int tile = 0; tile < tiles; ++tile) {
                while (__atomic_load_n(&ready[tile], __ATOMIC_ACQUIRE) != epoch) {
#ifdef __AMDGCN__
                    __builtin_amdgcn_s_sleep(1);
#endif
                }
            }
            uint64_t seed = epoch ^ (static_cast<uint64_t>(team) << 32)
                                      ^ static_cast<uint64_t>(tid + 1);
            scratch[static_cast<size_t>(team) * static_cast<size_t>(threads) + tid] =
                run_work(seed, work);
        }
    }
}

void execute_mpi_producer(unsigned char* data, size_t bytes, int tiles,
                          int producer_work, int threads, uint64_t epoch,
                          unsigned char value) {
    #pragma omp target teams num_teams(tiles) thread_limit(threads) \
        is_device_ptr(data) firstprivate(bytes, tiles, producer_work, threads, \
                                         epoch, value)
    {
        const int tile = omp_get_team_num();
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int team_threads = omp_get_num_threads();
            auto* scratch = reinterpret_cast<volatile uint64_t*>(data + kScratchOffset);
            const size_t begin = bytes * static_cast<size_t>(tile) /
                                 static_cast<size_t>(tiles);
            const size_t end = bytes * static_cast<size_t>(tile + 1) /
                               static_cast<size_t>(tiles);
            const int delay = static_cast<int>(
                static_cast<int64_t>(producer_work) * (tile + 1) / tiles);
            const uint64_t producer_seed =
                epoch ^ 0xd1b54a32d192ed03ULL ^
                (static_cast<uint64_t>(tile) << 32) ^
                static_cast<uint64_t>(tid + 1);
            scratch[static_cast<size_t>(tile) * static_cast<size_t>(threads) + tid] =
                run_work(producer_seed, delay);
            for (size_t i = begin + static_cast<size_t>(tid); i < end;
                 i += static_cast<size_t>(team_threads)) {
                data[kSendOffset + i] = value;
            }
        }
    }
}

void execute_mpi_interior(unsigned char* data, int work,
                          int teams, int threads, uint64_t epoch) {
    #pragma omp target teams num_teams(teams) thread_limit(threads) \
        is_device_ptr(data) firstprivate(work, threads, epoch)
    {
        const int team = omp_get_team_num();
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto* scratch = reinterpret_cast<uint64_t*>(data + kScratchOffset);
            uint64_t seed = epoch ^ (static_cast<uint64_t>(team) << 32)
                                      ^ static_cast<uint64_t>(tid + 1);
            scratch[static_cast<size_t>(team) * static_cast<size_t>(threads) + tid] =
                run_work(seed, work);
        }
    }
}

void execute_mpi_iteration(bool communicate, unsigned char* data,
                           int rank, int peer, size_t bytes, int work,
                           int tiles, int producer_work,
                           int teams, int threads, uint64_t epoch,
                           unsigned char value) {
    if (!communicate) {
        execute_mpi_compute_only(data, bytes, tiles, producer_work, work,
                                 teams, threads, epoch, value);
        return;
    }

    MPI_Request requests[128];
    const int tag = 100 + static_cast<int>(epoch % 30000);
    for (int tile = 0; tile < tiles; ++tile) {
        const size_t begin = bytes * static_cast<size_t>(tile) /
                             static_cast<size_t>(tiles);
        const size_t end = bytes * static_cast<size_t>(tile + 1) /
                           static_cast<size_t>(tiles);
        MPI_Irecv(data + kRecvOffset + begin, static_cast<int>(end - begin),
                  MPI_BYTE, peer, tag, MPI_COMM_WORLD, &requests[tile]);
    }
    execute_mpi_producer(data, bytes, tiles, producer_work, threads, epoch, value);
    for (int tile = 0; tile < tiles; ++tile) {
        const size_t begin = bytes * static_cast<size_t>(tile) /
                             static_cast<size_t>(tiles);
        const size_t end = bytes * static_cast<size_t>(tile + 1) /
                           static_cast<size_t>(tiles);
        MPI_Isend(data + kSendOffset + begin, static_cast<int>(end - begin),
                  MPI_BYTE, peer, tag, MPI_COMM_WORLD, &requests[tiles + tile]);
    }

    if (work > 0) {
        // The requests have already been submitted to Slingshot. Execute the
        // independent GPU kernel while the NIC progresses them, then collect
        // any exposed tail with Waitall.
        execute_mpi_interior(data, work, teams, threads, epoch);
    }
    MPI_Waitall(2 * tiles, requests, MPI_STATUSES_IGNORE);
    (void)rank;
}

Measurement measure(const Options& options, bool use_mpi,
                    Path path, bool communicate, int measured_work,
                    unsigned char* data, int rank, int peer, int buffer_index,
                    uint64_t& next_epoch) {
    Measurement result;
    for (int i = 0; i < options.warmup; ++i) {
        const uint64_t epoch = next_epoch++;
        const unsigned char value = iteration_value(rank, epoch);
        if (use_mpi)
            execute_mpi_iteration(communicate, data, rank, peer, options.bytes,
                                  measured_work, options.tiles, options.producer_work,
                                  options.teams, options.threads,
                                  epoch, value);
        else
            execute_giomp_iteration(communicate ? path : Path::None, data,
                                    peer, buffer_index, options.bytes,
                                    options.tiles, options.producer_work, measured_work,
                                    options.teams, options.threads,
                                    epoch, value);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int i = 0; i < options.iterations; ++i) {
        const uint64_t epoch = next_epoch++;
        const unsigned char value = iteration_value(rank, epoch);
        if (use_mpi)
            execute_mpi_iteration(communicate, data, rank, peer, options.bytes,
                                  measured_work, options.tiles, options.producer_work,
                                  options.teams, options.threads,
                                  epoch, value);
        else
            execute_giomp_iteration(communicate ? path : Path::None, data,
                                    peer, buffer_index, options.bytes,
                                    options.tiles, options.producer_work, measured_work,
                                    options.teams, options.threads,
                                    epoch, value);
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
    size_t local_errors = 0;
    for (unsigned char byte : received)
        local_errors += byte != expected;
    unsigned long long local = static_cast<unsigned long long>(local_errors);
    unsigned long long total = 0;
    MPI_Allreduce(&local, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    return total == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const Options options = parse_options(argc, argv);
    const bool use_mpi = options.transport == "mpi";
    const Path path = options.transport == "dwq" ? Path::Dwq : Path::Proxy;
    const size_t buffer_bytes = kScratchOffset +
        static_cast<size_t>(options.teams) * static_cast<size_t>(options.threads) *
        sizeof(uint64_t);

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
                    "MPI mode requires MPICH_GPU_SUPPORT_ENABLED=1, "
                    "MPICH_ASYNC_PROGRESS=1, and MPI_THREAD_MULTIPLE\n");
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
        if ((path == Path::Dwq) != ompx_dwq_enabled()) {
            if (rank == 0)
                std::fprintf(stderr, "%s requires GICC_HALO_DWQ=%d\n",
                             options.transport.c_str(), path == Path::Dwq ? 1 : 0);
            MPI_Abort(MPI_COMM_WORLD, 7);
        }
        giomp_buffer = ompx_alloc(buffer_bytes);
        data = static_cast<unsigned char*>(giomp_buffer.ptr);
        ompx_exchange();
    }

    require_two_distinct_nodes(rank, nranks);
    const int peer = rank ^ 1;
    if (!use_mpi && ompx_ipc_reachable(peer, giomp_buffer.index)) {
        if (rank == 0)
            std::fprintf(stderr, "unexpected IPC-reachable peer\n");
        MPI_Abort(MPI_COMM_WORLD, 8);
    }
    check_hip(hipMemset(data, 0, buffer_bytes), "hipMemset", rank);
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize", rank);

    if (rank == 0) {
        std::printf("giomp_streaming_halo transport=%s size=%zu tiles=%d "
                    "producer_work=%d work=%d teams=%d threads=%d "
                    "iterations=%d warmup=%d IPC=excluded\n",
                    options.transport.c_str(), options.bytes, options.tiles,
                    options.producer_work, options.work,
                    options.teams, options.threads,
                    options.iterations, options.warmup);
    }

    uint64_t next_epoch = 1;
    const Measurement producer = measure(options, use_mpi, path, false, 0,
                                         data, rank, peer, giomp_buffer.index,
                                         next_epoch);
    const Measurement compute = measure(options, use_mpi, path, false, options.work,
                                        data, rank, peer, giomp_buffer.index,
                                        next_epoch);
    const Measurement communication = measure(options, use_mpi, path, true, 0,
                                              data, rank, peer, giomp_buffer.index,
                                              next_epoch);
    const Measurement full = measure(options, use_mpi, path, true, options.work,
                                     data, rank, peer, giomp_buffer.index,
                                     next_epoch);

    const int verify_error = verify_received(data, options.bytes, rank, peer,
                                             full.final_epoch);
    const double independent_compute =
        std::max(0.0, compute.us_per_iteration - producer.us_per_iteration);
    const double communication_cost =
        std::max(0.0, communication.us_per_iteration - producer.us_per_iteration);
    const double ideal = producer.us_per_iteration +
                         std::max(independent_compute, communication_cost);
    const double exposed = full.us_per_iteration - ideal;
    double overlap_efficiency = std::numeric_limits<double>::quiet_NaN();
    const double overlap_denominator =
        std::min(independent_compute, communication_cost);
    if (overlap_denominator > 0.0) {
        overlap_efficiency =
            (independent_compute + communication_cost -
             (full.us_per_iteration - producer.us_per_iteration)) /
            overlap_denominator;
    }

    if (rank == 0) {
        const char* verify = verify_error == 0 ? "PASS" : "FAIL";
        std::printf("producer_us=%.3f compute_only_us=%.3f comm_only_us=%.3f "
                    "full_us=%.3f independent_compute_us=%.3f "
                    "communication_us=%.3f exposed_us=%.3f overlap_eff=%.4f %s\n",
                    producer.us_per_iteration, compute.us_per_iteration,
                    communication.us_per_iteration, full.us_per_iteration,
                    independent_compute, communication_cost, exposed,
                    overlap_efficiency, verify);
        std::printf("RESULT,%s,%zu,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n",
                    options.transport.c_str(), options.bytes, options.tiles,
                    options.producer_work, options.work,
                    options.teams, options.threads,
                    producer.us_per_iteration, compute.us_per_iteration,
                    communication.us_per_iteration, full.us_per_iteration,
                    independent_compute, communication_cost, exposed,
                    overlap_efficiency, verify);
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
