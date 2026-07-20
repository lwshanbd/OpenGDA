// giomp_mm_eval.cpp - fair distributed matrix-multiplication comparison.
//
// One source file is compiled three ways:
//   default                    GiOMP (Proxy or DWQ selected at runtime)
//   -DMM_BACKEND_DIOMP=1       DiOMP
//   -DMM_BACKEND_MPI=1         GPU-aware MPI
//
// Every backend uses the same ring algorithm, OpenMP target compute kernel,
// initialization, timing, and correctness check.  Rank r owns Ns rows of A and
// C and Ns columns of B.  At ring step s it computes C's column block r+s while
// fetching the next B stripe from its right neighbor.
//
// The old DiOMP/MM kernel parallelized (k,j) while accumulating C[i,j], which
// allowed different k iterations to race on the same C element.  This version
// parallelizes the independent (i,j) output elements and performs the k sum in
// one thread, so the result is deterministic and mathematically correct.

#if defined(MM_BACKEND_DIOMP) && defined(MM_BACKEND_MPI)
#error "select exactly one MM backend"
#endif

#if !defined(MM_BACKEND_DIOMP) && !defined(MM_BACKEND_MPI)
#define MM_BACKEND_GIOMP 1
#endif

#if defined(MM_BACKEND_GIOMP)
#include "gicc/omp.h"
#elif defined(MM_BACKEND_DIOMP)
#include <diomp.h>
#endif

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Options {
    int n = 2048;
    int runs = 7;
    int warmup = 2;
    int threads = 512;
    std::string transport = "proxy";
    std::string kernel_style = "loop";
    bool require_one_rank_per_node = false;
};

[[noreturn]] void usage(const char* argv0, const char* error = nullptr) {
    if (error) std::fprintf(stderr, "error: %s\n", error);
    std::fprintf(stderr,
        "usage: %s [--n=N] [--runs=N] [--warmup=N] [--threads=N] "
        "[--transport=proxy|dwq] "
        "[--kernel-style=loop|unified|unified-fenced|master|split] "
        "[--require-one-rank-per-node] [--quick]\n",
        argv0);
    std::exit(error ? 2 : 0);
}

int parse_positive(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    return end != text && *end == '\0' && value > 0 &&
                   value <= std::numeric_limits<int>::max()
               ? static_cast<int>(value)
               : -1;
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind("--n=", 0) == 0) o.n = parse_positive(arg.c_str() + 4);
        else if (arg.rfind("--runs=", 0) == 0)
            o.runs = parse_positive(arg.c_str() + 7);
        else if (arg.rfind("--warmup=", 0) == 0) {
            const char* text = arg.c_str() + 9;
            char* end = nullptr;
            const long value = std::strtol(text, &end, 10);
            o.warmup = end != text && *end == '\0' && value >= 0 &&
                               value <= std::numeric_limits<int>::max()
                           ? static_cast<int>(value)
                           : -1;
        } else if (arg.rfind("--threads=", 0) == 0)
            o.threads = parse_positive(arg.c_str() + 10);
        else if (arg.rfind("--transport=", 0) == 0)
            o.transport = arg.substr(12);
        else if (arg.rfind("--kernel-style=", 0) == 0)
            o.kernel_style = arg.substr(15);
        else if (arg == "--require-one-rank-per-node")
            o.require_one_rank_per_node = true;
        else if (arg == "--quick") {
            o.runs = std::min(o.runs, 3);
            o.warmup = std::min(o.warmup, 1);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            usage(argv[0], "unknown option");
        }
    }
    if (o.n < 1 || o.runs < 1 || o.warmup < 0 || o.threads < 1 ||
        o.threads > 1024)
        usage(argv[0], "invalid numeric option");
    if (o.transport != "proxy" && o.transport != "dwq")
        usage(argv[0], "transport must be proxy or dwq");
    if (o.kernel_style != "loop" && o.kernel_style != "unified" &&
        o.kernel_style != "unified-fenced" &&
        o.kernel_style != "master" && o.kernel_style != "split")
        usage(argv[0],
              "kernel style must be loop, unified, unified-fenced, master, "
              "or split");
    return o;
}

const char* backend_name(const Options& options) {
#if defined(MM_BACKEND_GIOMP)
    return options.transport == "dwq" ? "giomp-dwq" : "giomp-proxy";
#elif defined(MM_BACKEND_DIOMP)
    (void)options;
    return "diomp";
#else
    (void)options;
    return "mpi";
#endif
}

void runtime_init(int argc, char** argv, int& rank, int& nranks) {
#if defined(MM_BACKEND_GIOMP)
    (void)argc;
    (void)argv;
    ompx_init();
    rank = omp_get_rank_num();
    nranks = omp_get_num_ranks();
#elif defined(MM_BACKEND_DIOMP)
    // DiOMP/GASNet bootstraps through PMI and does not initialize MPI.  MPI is
    // used here only for the benchmark's identical placement check, barrier,
    // and max-time/checksum reductions; the matrix data path remains DiOMP.
    MPI_Init(&argc, &argv);
    __init_diomp_target(2);
    rank = omp_get_rank_num();
    nranks = omp_get_num_ranks();
#else
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    omp_set_default_device(0);
#endif
}

void runtime_finalize() {
#if defined(MM_BACKEND_GIOMP)
    ompx_finalize();
#elif defined(MM_BACKEND_DIOMP) || defined(MM_BACKEND_MPI)
    MPI_Finalize();
#endif
    // The public DiOMP API has no matching finalize routine.  This mirrors its
    // shipped MM benchmark; process teardown releases the runtime resources.
}

void check_placement(bool require_one_rank_per_node, int rank) {
    MPI_Comm local = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local);
    int local_size = 0;
    MPI_Comm_size(local, &local_size);
    MPI_Comm_free(&local);

    int min_local = 0;
    int max_local = 0;
    MPI_Allreduce(&local_size, &min_local, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_size, &max_local, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("placement local_ranks_per_node=%d..%d\n",
                    min_local, max_local);
    if (require_one_rank_per_node && (min_local != 1 || max_local != 1)) {
        if (rank == 0)
            std::fprintf(stderr,
                "one rank per node required, observed %d..%d\n",
                min_local, max_local);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
}

inline float a_value(size_t linear_index, int owner) {
    return static_cast<float>((linear_index + static_cast<size_t>(owner)) % 11 + 7);
}

inline float b_value(size_t linear_index, int owner) {
    return static_cast<float>((linear_index + static_cast<size_t>(owner)) % 13 + 5);
}

// Pure compute kernel shared by the DiOMP and MPI builds and by the GiOMP split
// ablation.  Each output work item owns one C(i,j) and performs the k reduction.
void compute_block_plain(float* a, float* current_b, float* c,
                         int n, int ns, int block, int threads) {
    const int column_offset = block * ns;
    #pragma omp target teams distribute parallel for collapse(2) \
        thread_limit(threads) is_device_ptr(a, current_b, c) \
        firstprivate(n, ns, column_offset)
    for (int i = 0; i < ns; ++i) {
        for (int j = 0; j < ns; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
                sum += a[static_cast<size_t>(i) * n + k] *
                       current_b[static_cast<size_t>(k) * ns + j];
            c[static_cast<size_t>(i) * n + column_offset + j] = sum;
        }
    }
}

#if defined(MM_BACKEND_GIOMP)
template <bool UseDwq>
__attribute__((noinline))
void issue_giomp_get(gicc::DeviceCtx* ctx, int right,
                     int current_index, int next_index,
                     size_t stripe_bytes) {
    if constexpr (UseDwq) {
        // This is a GET trigger: the host-staged descriptor is ordered before
        // the target launch and the NIC produces the local destination.  No
        // GPU data needs publishing before the MMIO store, so a pair of
        // seq_cst system fences would be unnecessary and stronger than
        // GiOMP's public DWQ trigger API.
        *(ctx->trigger_addr_) = ctx->trigger_val_;
    } else {
        ompx_get_single(ctx, right, current_index, 0,
                        next_index, 0, stripe_bytes);
    }
}

// Transport-specialized fused implementation.  noinline keeps the rare
// communication control path out of the hot multiply loop, but the compiler
// still emits a different compute kernel for Proxy and DWQ.
template <bool UseDwq>
void compute_block_fused_loop(float* a, float* current_b, float* c,
                              int n, int ns, int block, int threads,
                              gicc::DeviceCtx* ctx, int right,
                              int current_index, int next_index,
                              size_t stripe_bytes) {
    const int column_offset = block * ns;
    #pragma omp target teams distribute parallel for collapse(2) \
        thread_limit(threads) is_device_ptr(a, current_b, c, ctx) \
        firstprivate(n, ns, column_offset, right, current_index, next_index, \
                     stripe_bytes)
    for (int i = 0; i < ns; ++i) {
        for (int j = 0; j < ns; ++j) {
            if (i == 0 && j == 0)
                issue_giomp_get<UseDwq>(ctx, right, current_index,
                                        next_index, stripe_bytes);
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
                sum += a[static_cast<size_t>(i) * n + k] *
                       current_b[static_cast<size_t>(k) * ns + j];
            c[static_cast<size_t>(i) * n + column_offset + j] = sum;
        }
    }
}

// Keep Proxy and DWQ behind one runtime branch so both transports execute the
// exact same generated multiply kernel.  This is an important control for
// compiler/code-generation differences between template specializations.
__attribute__((noinline))
void issue_giomp_get_unified(gicc::DeviceCtx* ctx, int use_dwq,
                             int trigger_fences, int right,
                             int current_index, int next_index,
                             size_t stripe_bytes) {
    if (use_dwq) {
        if (trigger_fences)
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
        *(ctx->trigger_addr_) = ctx->trigger_val_;
        if (trigger_fences)
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
    } else {
        ompx_get_single(ctx, right, current_index, 0,
                        next_index, 0, stripe_bytes);
    }
}

void compute_block_fused_unified(float* a, float* current_b, float* c,
                                 int n, int ns, int block, int threads,
                                 gicc::DeviceCtx* ctx, int use_dwq,
                                 int trigger_fences, int right,
                                 int current_index, int next_index,
                                 size_t stripe_bytes) {
    const int column_offset = block * ns;
    #pragma omp target teams distribute parallel for collapse(2) \
        thread_limit(threads) is_device_ptr(a, current_b, c, ctx) \
        firstprivate(n, ns, column_offset, use_dwq, trigger_fences, right, \
                     current_index, next_index, stripe_bytes)
    for (int i = 0; i < ns; ++i) {
        for (int j = 0; j < ns; ++j) {
            if (i == 0 && j == 0)
                issue_giomp_get_unified(ctx, use_dwq, trigger_fences, right,
                                        current_index, next_index,
                                        stripe_bytes);
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
                sum += a[static_cast<size_t>(i) * n + k] *
                       current_b[static_cast<size_t>(k) * ns + j];
            c[static_cast<size_t>(i) * n + column_offset + j] = sum;
        }
    }
}

// Profiling ablation: the target region still launches one kernel, but team 0
// emits communication before the distribute-parallel worker.  This isolates
// the control path structurally, at the cost of extra teams/distribute runtime
// machinery; it is not the canonical performance configuration.
template <bool UseDwq>
void compute_block_fused_master(float* a, float* current_b, float* c,
                                int n, int ns, int block, int threads,
                                gicc::DeviceCtx* ctx, int right,
                                int current_index, int next_index,
                                size_t stripe_bytes) {
    const int column_offset = block * ns;
    #pragma omp target teams thread_limit(threads) \
        is_device_ptr(a, current_b, c, ctx) \
        firstprivate(n, ns, column_offset, right, current_index, next_index, \
                     stripe_bytes)
    {
        if (omp_get_team_num() == 0)
            issue_giomp_get<UseDwq>(ctx, right, current_index,
                                    next_index, stripe_bytes);

        #pragma omp distribute parallel for collapse(2)
        for (int i = 0; i < ns; ++i) {
            for (int j = 0; j < ns; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k)
                    sum += a[static_cast<size_t>(i) * n + k] *
                           current_b[static_cast<size_t>(k) * ns + j];
                c[static_cast<size_t>(i) * n + column_offset + j] = sum;
            }
        }
    }
}

// Two-kernel ablation.  It keeps the exact pure compute kernel above and shows
// whether any remaining cost comes from fusing communication into that kernel.
template <bool UseDwq>
void issue_giomp_get_split(gicc::DeviceCtx* ctx, int right,
                           int current_index, int next_index,
                           size_t stripe_bytes) {
    #pragma omp target is_device_ptr(ctx) \
        firstprivate(right, current_index, next_index, stripe_bytes)
    {
        issue_giomp_get<UseDwq>(ctx, right, current_index,
                                next_index, stripe_bytes);
    }
}
#endif

struct Verification {
    unsigned long long errors = 0;
    double max_relative_error = 0.0;
    double fingerprint = 0.0;
};

Verification verify(const float* c, int n, int ns, int rank) {
    Verification local;
    constexpr int samples = 32;
    for (int sample = 0; sample < samples; ++sample) {
        const int i = (sample * 37 + 3) % ns;
        const int global_j = (sample * 97 + 5) % n;
        const int owner = global_j / ns;
        const int j = global_j % ns;
        float expected = 0.0f;
        for (int k = 0; k < n; ++k) {
            const float av = a_value(static_cast<size_t>(i) * n + k, rank);
            const float bv = b_value(static_cast<size_t>(k) * ns + j, owner);
            expected += av * bv;
        }
        const float observed = c[static_cast<size_t>(i) * n + global_j];
        const double relative = std::abs(static_cast<double>(observed) - expected) /
                                std::max(1.0, std::abs(static_cast<double>(expected)));
        local.max_relative_error = std::max(local.max_relative_error, relative);
        local.errors += !std::isfinite(observed) || relative > 5.0e-5;
        local.fingerprint += static_cast<double>(observed) * (sample + 1);
    }

    Verification global;
    MPI_Allreduce(&local.errors, &global.errors, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local.max_relative_error, &global.max_relative_error, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local.fingerprint, &global.fingerprint, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const Options options = parse_options(argc, argv);
    int rank = -1;
    int nranks = 0;
    runtime_init(argc, argv, rank, nranks);

    if (nranks < 2 || options.n < nranks || options.n % nranks != 0) {
        if (rank == 0)
            std::fprintf(stderr, "N=%d must be divisible by nranks=%d (nranks >= 2)\n",
                         options.n, nranks);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    check_placement(options.require_one_rank_per_node, rank);

#if defined(MM_BACKEND_GIOMP)
    const bool use_dwq = options.transport == "dwq";
    if (use_dwq != ompx_dwq_enabled()) {
        if (rank == 0)
            std::fprintf(stderr, "%s requires GICC_HALO_DWQ=%d\n",
                         backend_name(options), use_dwq ? 1 : 0);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
#endif

    const int n = options.n;
    const int ns = n / nranks;
    const size_t elements = static_cast<size_t>(n) * ns;
    const size_t stripe_bytes = elements * sizeof(float);
    float* a = new float[elements];
    float* b0 = new float[elements];
    float* b1 = new float[elements];
    float* c = new float[elements];
    for (size_t x = 0; x < elements; ++x) {
        a[x] = a_value(x, rank);
        b0[x] = b_value(x, rank);
    }

    std::vector<double> times;
    times.reserve(options.runs);
    const int total_runs = options.warmup + options.runs;

    if (rank == 0) {
#if defined(MM_BACKEND_GIOMP)
        constexpr int proxy_single_producer = 1;
        const char* selected_kernel_style = options.kernel_style.c_str();
#else
        constexpr int proxy_single_producer = 0;
        const char* selected_kernel_style = "plain";
#endif
        std::printf("mm_eval backend=%s N=%d ranks=%d stripe=%dx%d bytes=%zu "
                    "runs=%d warmup=%d threads=%d kernel_style=%s "
                    "correct_ij_kernel=1 skip_final_rotation=1 "
                    "proxy_single_producer=%d\n",
                    backend_name(options), n, nranks, n, ns, stripe_bytes,
                    options.runs, options.warmup, options.threads,
                    selected_kernel_style, proxy_single_producer);
    }

    #pragma omp target data map(to: a[0:elements], b0[0:elements]) \
        map(alloc: b1[0:elements]) map(from: c[0:elements])
    {
        #pragma omp target data use_device_ptr(a, b0, b1, c)
        {
#if defined(MM_BACKEND_GIOMP)
            const int b0_index = ompx_register(b0, stripe_bytes);
            const int b1_index = ompx_register(b1, stripe_bytes);
            ompx_exchange();
            const int right = (rank + 1) % nranks;
#elif defined(MM_BACKEND_DIOMP)
            const int right = (rank + 1) % nranks;
#else
            const int right = (rank + 1) % nranks;
            const int left = (rank + nranks - 1) % nranks;
#endif

            for (int run = 0; run < total_runs; ++run) {
                #pragma omp target teams distribute parallel for \
                    is_device_ptr(b0, b1, c) firstprivate(elements, rank)
                for (size_t x = 0; x < elements; ++x) {
                    b0[x] = static_cast<float>((x + static_cast<size_t>(rank)) % 13 + 5);
                    b1[x] = 0.0f;
                    c[x] = 0.0f;
                }

                float* current_b = b0;
                float* next_b = b1;
#if defined(MM_BACKEND_GIOMP)
                int current_index = b0_index;
                int next_index = b1_index;
#endif

                MPI_Barrier(MPI_COMM_WORLD);
                const double start = MPI_Wtime();
                for (int step = 0; step < nranks; ++step) {
                    const int block = (rank + step) % nranks;
                    // The final block has no successor to consume another B
                    // stripe.  Avoid rotating one full, unused stripe after
                    // that compute; with two ranks the old loop performed
                    // twice as much communication as the algorithm requires.
                    const bool has_next = step + 1 < nranks;
#if defined(MM_BACKEND_GIOMP)
                    if (has_next) {
                        gicc::DeviceCtx* ctx = ompx_prepare();
                        if (use_dwq) {
                            ompx_dwq_stage_get(right, current_index, 0,
                                               next_index, 0, stripe_bytes);
                            ompx_dwq_arm();
                        }
                        if (options.kernel_style == "loop") {
                            if (use_dwq)
                                compute_block_fused_loop<true>(
                                    a, current_b, c, n, ns, block,
                                    options.threads, ctx, right, current_index,
                                    next_index, stripe_bytes);
                            else
                                compute_block_fused_loop<false>(
                                    a, current_b, c, n, ns, block,
                                    options.threads, ctx, right, current_index,
                                    next_index, stripe_bytes);
                        } else if (options.kernel_style == "unified" ||
                                   options.kernel_style == "unified-fenced") {
                            compute_block_fused_unified(
                                a, current_b, c, n, ns, block, options.threads,
                                ctx, use_dwq,
                                options.kernel_style == "unified-fenced", right,
                                current_index, next_index, stripe_bytes);
                        } else if (options.kernel_style == "master") {
                            if (use_dwq)
                                compute_block_fused_master<true>(
                                    a, current_b, c, n, ns, block,
                                    options.threads, ctx, right, current_index,
                                    next_index, stripe_bytes);
                            else
                                compute_block_fused_master<false>(
                                    a, current_b, c, n, ns, block,
                                    options.threads, ctx, right, current_index,
                                    next_index, stripe_bytes);
                        } else {
                            if (use_dwq)
                                issue_giomp_get_split<true>(
                                    ctx, right, current_index, next_index,
                                    stripe_bytes);
                            else
                                issue_giomp_get_split<false>(
                                    ctx, right, current_index, next_index,
                                    stripe_bytes);
                            compute_block_plain(a, current_b, c, n, ns, block,
                                                options.threads);
                        }
                        ompx_quiet_host();
                        std::swap(current_index, next_index);
                    } else {
                        compute_block_plain(a, current_b, c, n, ns, block,
                                            options.threads);
                    }
#elif defined(MM_BACKEND_DIOMP)
                    if (has_next)
                        ompx_dget(next_b, right, current_b, stripe_bytes, 0, 0);
                    compute_block_plain(a, current_b, c, n, ns, block,
                                        options.threads);
                    if (has_next) diomp_waitALLRMA();
#else
                    MPI_Request requests[2];
                    if (has_next) {
                        MPI_Irecv(next_b, static_cast<int>(elements), MPI_FLOAT,
                                  right, step, MPI_COMM_WORLD, &requests[0]);
                        MPI_Isend(current_b, static_cast<int>(elements), MPI_FLOAT,
                                  left, step, MPI_COMM_WORLD, &requests[1]);
                    }
                    compute_block_plain(a, current_b, c, n, ns, block,
                                        options.threads);
                    if (has_next)
                        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
#endif
                    // A GET completes only the caller's next_b.  Before the
                    // following step, every rank may expose that next_b as a
                    // remote source to its left neighbor, so all ranks must
                    // finish the current rotation first.  Apply the same
                    // synchronization to every backend for identical timing.
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (has_next) std::swap(current_b, next_b);
                }
                const double local_elapsed = MPI_Wtime() - start;
                double elapsed = 0.0;
                MPI_Allreduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE,
                              MPI_MAX, MPI_COMM_WORLD);
                if (run >= options.warmup) times.push_back(elapsed * 1.0e6);
            }
        }
    }

    const Verification verification = verify(c, n, ns, rank);
    int return_code = verification.errors == 0 ? 0 : 6;
    if (rank == 0) {
        std::sort(times.begin(), times.end());
        const double median_us = times[times.size() / 2];
        const double min_us = times.front();
        const double max_us = times.back();
        const double global_flops = 2.0 * static_cast<double>(n) * n * n;
        const double gflops = global_flops / (median_us * 1.0e3);
        const char* status = return_code == 0 ? "PASS" : "FAIL";
        std::printf("median_us=%.3f min_us=%.3f max_us=%.3f global_GFLOP/s=%.3f "
                    "fingerprint=%.9e max_rel_error=%.3e %s\n",
                    median_us, min_us, max_us, gflops,
                    verification.fingerprint, verification.max_relative_error,
                    status);
        std::printf("RESULT,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.9e,%.3e,%s\n",
                    backend_name(options), n, nranks, options.runs,
                    options.warmup, median_us, min_us, max_us, gflops,
                    verification.fingerprint, verification.max_relative_error,
                    status);
    }

    delete[] c;
    delete[] b1;
    delete[] b0;
    delete[] a;
    runtime_finalize();
    return return_code;
}
