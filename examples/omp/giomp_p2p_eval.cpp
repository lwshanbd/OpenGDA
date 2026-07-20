// giomp_p2p_eval.cpp - paper-oriented cross-node GPU P2P microbenchmark.
//
// Compares the GiOMP public OpenMP API with GPU-aware MPI on exactly two
// nodes, one rank/GPU per node. The benchmark intentionally rejects a
// same-node placement, so GiOMP IPC is never part of these measurements.
//
// Transports:
//   proxy  GiOMP device command -> CPU proxy -> OFI RMA
//   dwq    host-staged OFI deferred work -> GPU MMIO trigger
//   mpi    Cray GPU-aware two-sided MPI baseline
//
// Operations:
//   put    rank 0 sends/writes to rank 1
//   get    GiOMP rank 0 pulls from rank 1. MPI measures the same data
//          direction (rank 1 MPI_Isend/Send -> rank 0 MPI_Irecv/Recv), not
//          MPI_Get; output labels this baseline mpi-two-sided.
//
// Issue patterns:
//   single      each message is completed before the next one
//   concurrent  B messages use disjoint offsets, then complete as one batch

// Build: bash examples/omp/build_giomp_p2p_eval.sh

// Typical runs (one rank per node):
//   HSA_XNACK=1 GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
//     flux run -q pci -N2 -n2 -g1 -o mpibind=off \
//       build_ofi/giomp_p2p_eval --transport=proxy
//   HSA_XNACK=1 GICC_HALO_DWQ=1 \
//     flux run -q pci -N2 -n2 -g1 -o mpibind=off \
//       build_ofi/giomp_p2p_eval --transport=dwq
//   HSA_XNACK=1 MPICH_GPU_SUPPORT_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
//     flux run -q pci -N2 -n2 -g1 -o mpibind=off \
//       build_ofi/giomp_p2p_eval --transport=mpi

#include "gicc/omp.h"

#include <hip/hip_runtime_api.h>
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

constexpr size_t kBufferBytes = 128ULL * 1024 * 1024;
constexpr int kDefaultBatch = 32;
constexpr int kDefaultOuter = 21;
constexpr int kDefaultWarmup = 5;

const size_t kFullSizes[] = {
    8, 64, 256, 1024, 4 * 1024, 16 * 1024, 64 * 1024,
    256 * 1024, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024,
};
const size_t kQuickSizes[] = {8, 4 * 1024, 1024 * 1024};

struct Options {
    std::string transport = "proxy";
    std::string op = "all";
    std::string pattern = "all";
    int batch = kDefaultBatch;
    int outer = kDefaultOuter;
    int warmup = kDefaultWarmup;
    bool quick = false;
    bool profile_phases = false;
    size_t size_override = 0;
};

struct PhaseTimes {
    double prepare = 0.0;
    double enqueue = 0.0;
    double target = 0.0;
    double quiet = 0.0;
};

[[noreturn]] void usage(const char* argv0, const char* error = nullptr) {
    if (error) std::fprintf(stderr, "error: %s\n", error);
    std::fprintf(stderr,
        "usage: %s --transport=proxy|dwq|mpi "
        "[--op=put|get|all] [--pattern=single|in-kernel-single|concurrent|all] "
        "[--batch=N] [--outer=N] [--warmup=N] [--size=BYTES] "
        "[--profile-phases] [--quick]\n",
        argv0);
    std::exit(2);
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.rfind("--transport=", 0) == 0) o.transport = a.substr(12);
        else if (a.rfind("--op=", 0) == 0) o.op = a.substr(5);
        else if (a.rfind("--pattern=", 0) == 0) o.pattern = a.substr(10);
        else if (a.rfind("--batch=", 0) == 0) o.batch = std::atoi(a.c_str() + 8);
        else if (a.rfind("--outer=", 0) == 0) o.outer = std::atoi(a.c_str() + 8);
        else if (a.rfind("--warmup=", 0) == 0) o.warmup = std::atoi(a.c_str() + 9);
        else if (a.rfind("--size=", 0) == 0)
            o.size_override = std::strtoull(a.c_str() + 7, nullptr, 0);
        else if (a == "--profile-phases") o.profile_phases = true;
        else if (a == "--quick") o.quick = true;
        else usage(argv[0], "unknown option");
    }
    if (o.transport != "proxy" && o.transport != "dwq" && o.transport != "mpi")
        usage(argv[0], "--transport must be proxy, dwq, or mpi");
    if (o.op != "put" && o.op != "get" && o.op != "all")
        usage(argv[0], "--op must be put, get, or all");
    if (o.pattern != "single" && o.pattern != "in-kernel-single" &&
        o.pattern != "concurrent" && o.pattern != "all")
        usage(argv[0], "--pattern must be single, in-kernel-single, concurrent, or all");
    if (o.pattern == "in-kernel-single" && o.transport != "proxy")
        usage(argv[0], "in-kernel-single currently applies only to proxy");
    if (o.batch < 1 || o.outer < 1 || o.warmup < 0)
        usage(argv[0], "batch/outer must be positive and warmup nonnegative");
    if (o.size_override > kBufferBytes)
        usage(argv[0], "--size exceeds the registered buffer size");
    if (o.quick) {
        o.outer = std::min(o.outer, 5);
        o.warmup = std::min(o.warmup, 2);
        o.batch = std::min(o.batch, 8);
    }
    return o;
}

void check_hip(hipError_t status, const char* what, int rank) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "rank %d: %s failed: %s\n",
                 rank, what, hipGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 3);
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

const char* display_transport(const std::string& transport) {
    return transport == "mpi" ? "mpi-two-sided" : transport.c_str();
}

void require_two_distinct_nodes(int rank, int nranks) {
    if (nranks != 2) {
        if (rank == 0)
            std::fprintf(stderr, "giomp_p2p_eval requires exactly 2 ranks (got %d)\n",
                         nranks);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    char local[MPI_MAX_PROCESSOR_NAME] = {};
    int name_len = 0;
    MPI_Get_processor_name(local, &name_len);
    char names[2][MPI_MAX_PROCESSOR_NAME] = {};
    MPI_Allgather(local, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
    if (std::strcmp(names[0], names[1]) == 0) {
        if (rank == 0)
            std::fprintf(stderr,
                "giomp_p2p_eval rejects same-node placement (%s); use -N2 -n2\n",
                names[0]);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    if (rank == 0)
        std::printf("nodes=%s,%s (cross-node confirmed)\n", names[0], names[1]);
}

void giomp_issue_one(const std::string& transport, const std::string& op,
                     gicc::DeviceCtx* ctx, int peer, int buf,
                     size_t offset, size_t bytes) {
    if (transport == "dwq") {
        if (op == "put")
            ompx_dwq_stage_put(peer, buf, offset, buf, offset, bytes);
        else
            ompx_dwq_stage_get(peer, buf, offset, buf, offset, bytes);
        ompx_dwq_trigger(ctx);
    } else if (op == "put") {
        #pragma omp target is_device_ptr(ctx) firstprivate(peer, buf, offset, bytes)
        { ompx_put_proxy(ctx, peer, buf, offset, buf, offset, bytes); }
    } else {
        #pragma omp target is_device_ptr(ctx) firstprivate(peer, buf, offset, bytes)
        { ompx_get(ctx, peer, buf, offset, buf, offset, bytes); }
    }
}

double phase_stamp(PhaseTimes* phases) {
    return phases ? MPI_Wtime() : 0.0;
}

void phase_add(PhaseTimes* phases, double& field, double start) {
    if (phases) field += MPI_Wtime() - start;
}

void giomp_window(const std::string& transport, const std::string& op,
                  const std::string& pattern, int rank, int peer, int buf,
                  size_t bytes, int batch, PhaseTimes* phases = nullptr) {
    if (rank != 0) return;

    if (pattern == "in-kernel-single") {
        double stamp = phase_stamp(phases);
        gicc::DeviceCtx* ctx = ompx_prepare();
        if (phases) phase_add(phases, phases->prepare, stamp);
        stamp = phase_stamp(phases);
        if (op == "put") {
            #pragma omp target is_device_ptr(ctx) firstprivate(peer, buf, bytes, batch)
            {
                for (int i = 0; i < batch; ++i) {
                    ompx_put_proxy(ctx, peer, buf, 0, buf, 0, bytes);
                    ompx_quiet(ctx);
                }
            }
        } else {
            #pragma omp target is_device_ptr(ctx) firstprivate(peer, buf, bytes, batch)
            {
                for (int i = 0; i < batch; ++i) {
                    ompx_get(ctx, peer, buf, 0, buf, 0, bytes);
                    ompx_quiet(ctx);
                }
            }
        }
        if (phases) phase_add(phases, phases->target, stamp);
        stamp = phase_stamp(phases);
        ompx_quiet_host();
        if (phases) phase_add(phases, phases->quiet, stamp);
        return;
    }

    if (pattern == "single") {
        for (int i = 0; i < batch; ++i) {
            double stamp = phase_stamp(phases);
            gicc::DeviceCtx* ctx = ompx_prepare();
            if (phases) phase_add(phases, phases->prepare, stamp);
            if (transport == "dwq") {
                stamp = phase_stamp(phases);
                if (op == "put")
                    ompx_dwq_stage_put(peer, buf, 0, buf, 0, bytes);
                else
                    ompx_dwq_stage_get(peer, buf, 0, buf, 0, bytes);
                if (phases) phase_add(phases, phases->enqueue, stamp);
                stamp = phase_stamp(phases);
                ompx_dwq_trigger(ctx);
                if (phases) phase_add(phases, phases->target, stamp);
            } else {
                stamp = phase_stamp(phases);
                giomp_issue_one(transport, op, ctx, peer, buf, 0, bytes);
                if (phases) phase_add(phases, phases->target, stamp);
            }
            stamp = phase_stamp(phases);
            ompx_quiet_host();
            if (phases) phase_add(phases, phases->quiet, stamp);
        }
        return;
    }

    double stamp = phase_stamp(phases);
    gicc::DeviceCtx* ctx = ompx_prepare();
    if (phases) phase_add(phases, phases->prepare, stamp);
    if (transport == "dwq") {
        stamp = phase_stamp(phases);
        for (int i = 0; i < batch; ++i) {
            const size_t offset = static_cast<size_t>(i) * bytes;
            if (op == "put")
                ompx_dwq_stage_put(peer, buf, offset, buf, offset, bytes);
            else
                ompx_dwq_stage_get(peer, buf, offset, buf, offset, bytes);
        }
        if (phases) phase_add(phases, phases->enqueue, stamp);
        stamp = phase_stamp(phases);
        ompx_dwq_trigger(ctx);
        if (phases) phase_add(phases, phases->target, stamp);
    } else if (op == "put") {
        stamp = phase_stamp(phases);
        #pragma omp target is_device_ptr(ctx) firstprivate(peer, buf, bytes, batch)
        {
            for (int i = 0; i < batch; ++i) {
                const size_t offset = static_cast<size_t>(i) * bytes;
                ompx_put_proxy(ctx, peer, buf, offset, buf, offset, bytes);
            }
        }
        if (phases) phase_add(phases, phases->target, stamp);
    } else {
        stamp = phase_stamp(phases);
        #pragma omp target is_device_ptr(ctx) firstprivate(peer, buf, bytes, batch)
        {
            for (int i = 0; i < batch; ++i) {
                const size_t offset = static_cast<size_t>(i) * bytes;
                ompx_get(ctx, peer, buf, offset, buf, offset, bytes);
            }
        }
        if (phases) phase_add(phases, phases->target, stamp);
    }
    stamp = phase_stamp(phases);
    ompx_quiet_host();
    if (phases) phase_add(phases, phases->quiet, stamp);
}

void mpi_window(const std::string& op, const std::string& pattern,
                int rank, void* buffer, size_t bytes, int batch) {
    const int source = op == "put" ? 0 : 1;
    const int dest = 1 - source;
    auto* base = static_cast<unsigned char*>(buffer);

    if (pattern == "single") {
        for (int i = 0; i < batch; ++i) {
            if (rank == source)
                MPI_Send(base, static_cast<int>(bytes), MPI_BYTE,
                         dest, 100, MPI_COMM_WORLD);
            else
                MPI_Recv(base, static_cast<int>(bytes), MPI_BYTE,
                         source, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        return;
    }

    std::vector<MPI_Request> requests(batch);
    for (int i = 0; i < batch; ++i) {
        void* slot = base + static_cast<size_t>(i) * bytes;
        if (rank == source)
            MPI_Isend(slot, static_cast<int>(bytes), MPI_BYTE,
                      dest, 200 + i, MPI_COMM_WORLD, &requests[i]);
        else
            MPI_Irecv(slot, static_cast<int>(bytes), MPI_BYTE,
                      source, 200 + i, MPI_COMM_WORLD, &requests[i]);
    }
    MPI_Waitall(batch, requests.data(), MPI_STATUSES_IGNORE);
}

int verify_destination(const std::string& op, const std::string& pattern,
                       int rank, void* buffer, size_t bytes, int batch,
                       unsigned char expected) {
    const int destination = op == "put" ? 1 : 0;
    int errors = 0;
    if (rank == destination) {
        auto* base = static_cast<unsigned char*>(buffer);
        const int slots = pattern == "concurrent" ? batch : 1;
        for (int i = 0; i < slots; ++i) {
            const size_t offset = static_cast<size_t>(i) * bytes;
            unsigned char first = 0, last = 0;
            if (hipMemcpy(&first, base + offset, 1, hipMemcpyDeviceToHost) != hipSuccess)
                ++errors;
            if (hipMemcpy(&last, base + offset + bytes - 1, 1,
                          hipMemcpyDeviceToHost) != hipSuccess)
                ++errors;
            if (first != expected || last != expected) ++errors;
        }
    }
    int total_errors = 0;
    MPI_Allreduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return total_errors;
}

void run_case(const Options& options, const std::string& op,
              const std::string& pattern, int rank, void* buffer, int buf) {
    std::vector<size_t> sizes;
    if (options.size_override != 0) {
        sizes.push_back(options.size_override);
    } else if (options.quick) {
        sizes.assign(kQuickSizes,
                     kQuickSizes + sizeof(kQuickSizes) / sizeof(kQuickSizes[0]));
    } else {
        sizes.assign(kFullSizes,
                     kFullSizes + sizeof(kFullSizes) / sizeof(kFullSizes[0]));
    }
    const int peer = rank ^ 1;

    if (rank == 0) {
        std::printf("\n=== transport=%s op=%s pattern=%s ===\n",
                    display_transport(options.transport), op.c_str(), pattern.c_str());
        std::printf("%-10s %6s %14s %14s %14s %8s\n",
                    "bytes", "batch", "median_us/msg", "mean_us/msg",
                    "effective_GB/s", "verify");
    }

    for (size_t si = 0; si < sizes.size(); ++si) {
        const size_t bytes = sizes[si];
        int batch = options.batch;
        if (pattern == "concurrent")
            batch = std::min<int>(batch, static_cast<int>(kBufferBytes / bytes));
        const size_t span = pattern == "concurrent"
            ? bytes * static_cast<size_t>(batch) : bytes;
        const int source = op == "put" ? 0 : 1;
        const unsigned char expected = static_cast<unsigned char>(0x51 + si * 13);

        check_hip(hipMemset(buffer, rank == source ? expected : 0, span),
                  "hipMemset", rank);
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize", rank);
        MPI_Barrier(MPI_COMM_WORLD);

        for (int w = 0; w < options.warmup; ++w) {
            if (options.transport == "mpi")
                mpi_window(op, pattern, rank, buffer, bytes, batch);
            else
                giomp_window(options.transport, op, pattern,
                             rank, peer, buf, bytes, batch);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        std::vector<double> samples;
        std::vector<double> phase_prepare, phase_enqueue, phase_target, phase_quiet;
        if (rank == 0) samples.reserve(options.outer);
        for (int outer = 0; outer < options.outer; ++outer) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double start = MPI_Wtime();
            PhaseTimes phases;
            if (options.transport == "mpi")
                mpi_window(op, pattern, rank, buffer, bytes, batch);
            else
                giomp_window(options.transport, op, pattern,
                             rank, peer, buf, bytes, batch,
                             options.profile_phases ? &phases : nullptr);
            const double local_elapsed = MPI_Wtime() - start;
            double elapsed = 0.0;
            MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
                       0, MPI_COMM_WORLD);
            if (rank == 0)
                samples.push_back(elapsed * 1.0e6 / static_cast<double>(batch));
            if (rank == 0 && options.profile_phases && options.transport != "mpi") {
                const double scale = 1.0e6 / static_cast<double>(batch);
                phase_prepare.push_back(phases.prepare * scale);
                phase_enqueue.push_back(phases.enqueue * scale);
                phase_target.push_back(phases.target * scale);
                phase_quiet.push_back(phases.quiet * scale);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const int errors = verify_destination(op, pattern, rank, buffer,
                                              bytes, batch, expected);
        if (rank == 0) {
            double sum = 0.0;
            for (double sample : samples) sum += sample;
            const double mean_us = sum / static_cast<double>(samples.size());
            const double median_us = median(samples);
            const double gb_per_s = static_cast<double>(bytes) / median_us / 1000.0;
            const char* status = errors == 0 ? "PASS" : "FAIL";
            std::printf("%-10zu %6d %14.3f %14.3f %14.3f %8s\n",
                        bytes, batch, median_us, mean_us, gb_per_s, status);
            std::printf("RESULT,%s,%s,%s,%zu,%d,%.6f,%.6f,%.6f,%s\n",
                        display_transport(options.transport), op.c_str(), pattern.c_str(),
                        bytes, batch, median_us, mean_us, gb_per_s, status);
            if (options.profile_phases && options.transport != "mpi") {
                const double p = median(phase_prepare);
                const double e = median(phase_enqueue);
                const double t = median(phase_target);
                const double q = median(phase_quiet);
                std::printf("PROFILE,%s,%s,%s,%zu,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                            display_transport(options.transport), op.c_str(), pattern.c_str(),
                            bytes, batch, p, e, t, q, p + e + t + q, median_us);
            }
            std::fflush(stdout);
        }
        if (errors != 0) MPI_Abort(MPI_COMM_WORLD, 6);
    }
}

void profile_control_overheads(int rank) {
    constexpr int kWarmup = 10;
    constexpr int kSamples = 101;
    if (rank == 0) {
        for (int i = 0; i < kWarmup; ++i) {
            gicc::DeviceCtx* ctx = ompx_prepare();
            #pragma omp target is_device_ptr(ctx)
            {
                volatile uint64_t sink = ctx->trigger_val_;
                (void)sink;
            }
            ompx_quiet_host();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> prepare_us, target_us, quiet_us;
    if (rank == 0) {
        prepare_us.reserve(kSamples);
        target_us.reserve(kSamples);
        quiet_us.reserve(kSamples);
        for (int i = 0; i < kSamples; ++i) {
            double start = MPI_Wtime();
            gicc::DeviceCtx* ctx = ompx_prepare();
            prepare_us.push_back((MPI_Wtime() - start) * 1.0e6);

            start = MPI_Wtime();
            #pragma omp target is_device_ptr(ctx)
            {
                volatile uint64_t sink = ctx->trigger_val_;
                (void)sink;
            }
            target_us.push_back((MPI_Wtime() - start) * 1.0e6);

            start = MPI_Wtime();
            ompx_quiet_host();
            quiet_us.push_back((MPI_Wtime() - start) * 1.0e6);
        }
        std::printf("CONTROL_PROFILE,prepare_us,empty_target_us,empty_quiet_us\n");
        std::printf("CONTROL_PROFILE,%.6f,%.6f,%.6f\n",
                    median(prepare_us), median(target_us), median(quiet_us));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const Options options = parse_options(argc, argv);
    const bool use_mpi_transport = options.transport == "mpi";

    int rank = -1;
    int nranks = 0;
    void* buffer = nullptr;
    ompx_buffer giomp_buffer{};

    if (use_mpi_transport) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        const char* gpu_support = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
        if (!gpu_support || std::strcmp(gpu_support, "1") != 0) {
            if (rank == 0)
                std::fprintf(stderr,
                    "MPI mode requires MPICH_GPU_SUPPORT_ENABLED=1\n");
            MPI_Abort(MPI_COMM_WORLD, 7);
        }
        check_hip(hipSetDevice(0), "hipSetDevice", rank);
        omp_set_default_device(0);
        check_hip(hipMalloc(&buffer, kBufferBytes), "hipMalloc", rank);
    } else {
        ompx_init();
        rank = omp_get_rank_num();
        nranks = omp_get_num_ranks();
        if ((options.transport == "dwq") != ompx_dwq_enabled()) {
            if (rank == 0)
                std::fprintf(stderr,
                    "%s mode requires GICC_HALO_DWQ=%d before ompx_init\n",
                    options.transport.c_str(), options.transport == "dwq" ? 1 : 0);
            MPI_Abort(MPI_COMM_WORLD, 8);
        }
        giomp_buffer = ompx_alloc(kBufferBytes);
        buffer = giomp_buffer.ptr;
        ompx_exchange();
    }

    require_two_distinct_nodes(rank, nranks);
    if (!use_mpi_transport && ompx_ipc_reachable(rank ^ 1, giomp_buffer.index)) {
        if (rank == 0)
            std::fprintf(stderr, "unexpected IPC-reachable peer in cross-node test\n");
        MPI_Abort(MPI_COMM_WORLD, 9);
    }

    if (rank == 0) {
        std::printf("giomp_p2p_eval: transport=%s batch=%d outer=%d warmup=%d "
                    "buffer_bytes=%zu IPC=excluded\n",
                    display_transport(options.transport), options.batch,
                    options.outer, options.warmup, kBufferBytes);
        std::printf("CSV schema: RESULT,transport,op,pattern,size_bytes,batch,"
                    "median_us_msg,mean_us_msg,effective_GBps,verify\n");
    }

    if (options.profile_phases && !use_mpi_transport)
        profile_control_overheads(rank);

    const std::vector<std::string> ops = options.op == "all"
        ? std::vector<std::string>{"put", "get"}
        : std::vector<std::string>{options.op};
    const std::vector<std::string> patterns = options.pattern == "all"
        ? std::vector<std::string>{"single", "concurrent"}
        : std::vector<std::string>{options.pattern};

    for (const std::string& op : ops)
        for (const std::string& pattern : patterns)
            run_case(options, op, pattern, rank, buffer,
                     use_mpi_transport ? -1 : giomp_buffer.index);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("ALL_CASES_PASS transport=%s\n",
                               display_transport(options.transport));

    if (use_mpi_transport) {
        check_hip(hipFree(buffer), "hipFree", rank);
        MPI_Finalize();
    } else {
        ompx_free(giomp_buffer);
        ompx_finalize();
    }
    return 0;
}
