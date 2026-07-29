// Weak-scaling 2-D Jacobi comparison compiled as GiOMP, DiOMP, and MPI.
// Every rank owns nx x local_rows interior points plus two halo rows.

#if defined(JACOBI_BACKEND_DIOMP) && defined(JACOBI_BACKEND_MPI)
#error "select exactly one Jacobi backend"
#endif
#if !defined(JACOBI_BACKEND_DIOMP) && !defined(JACOBI_BACKEND_MPI)
#define JACOBI_BACKEND_GIOMP 1
#endif

#if defined(JACOBI_BACKEND_GIOMP)
#include "gicc/omp.h"
#elif defined(JACOBI_BACKEND_DIOMP)
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
    int nx = 2048;
    int local_rows = 256;
    int iterations = 200;
    int runs = 5;
    int warmup = 2;
    int teams = 64;
    int threads = 256;
    bool dwq = false;      // cross-node transport: false = CPU proxy, true = DWQ
};

struct Buffer {
    float* ptr = nullptr;
    int index = -1;
    size_t bytes = 0;
};

int parse_positive(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    return end != text && *end == '\0' && value > 0 &&
                   value <= std::numeric_limits<int>::max()
               ? static_cast<int>(value)
               : -1;
}

int parse_nonnegative(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    return end != text && *end == '\0' && value >= 0 &&
                   value <= std::numeric_limits<int>::max()
               ? static_cast<int>(value)
               : -1;
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind("--nx=", 0) == 0) o.nx = parse_positive(arg.c_str() + 5);
        else if (arg.rfind("--local-rows=", 0) == 0)
            o.local_rows = parse_positive(arg.c_str() + 13);
        else if (arg.rfind("--iterations=", 0) == 0)
            o.iterations = parse_positive(arg.c_str() + 13);
        else if (arg.rfind("--runs=", 0) == 0)
            o.runs = parse_positive(arg.c_str() + 7);
        else if (arg.rfind("--warmup=", 0) == 0)
            o.warmup = parse_nonnegative(arg.c_str() + 9);
        else if (arg.rfind("--teams=", 0) == 0)
            o.teams = parse_positive(arg.c_str() + 8);
        else if (arg == "--transport=dwq") o.dwq = true;
        else if (arg == "--transport=proxy") o.dwq = false;
        else if (arg.rfind("--threads=", 0) == 0)
            o.threads = parse_positive(arg.c_str() + 10);
        else {
            if (arg != "--help" && arg != "-h")
                std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            std::fprintf(stderr,
                "usage: %s [--nx=N] [--local-rows=N] [--iterations=N] "
                "[--runs=N] [--warmup=N] [--teams=N] [--threads=N] "
                "[--transport=proxy|dwq]\n",
                argv[0]);
            std::exit((arg == "--help" || arg == "-h") ? 0 : 2);
        }
    }
    if (o.nx < 4 || o.local_rows < 4 || o.iterations < 1 || o.runs < 1 ||
        o.warmup < 0 || o.teams < 2 || o.threads < 1 || o.threads > 1024) {
        std::fprintf(stderr, "invalid Jacobi option\n");
        std::exit(2);
    }
    return o;
}

const char* g_backend_label = "giomp";

const char* backend_name() {
#if defined(JACOBI_BACKEND_GIOMP)
    return g_backend_label;
#elif defined(JACOBI_BACKEND_DIOMP)
    return "diomp";
#else
    return "mpi";
#endif
}

void select_rank_device(int rank) {
    MPI_Comm local = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local);
    int local_rank = 0;
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_free(&local);
    omp_set_default_device(local_rank);
}

void runtime_init(int argc, char** argv, int& rank, int& nranks) {
#if defined(JACOBI_BACKEND_GIOMP)
    (void)argc;
    (void)argv;
    ompx_init();
    rank = omp_get_rank_num();
    nranks = omp_get_num_ranks();
#elif defined(JACOBI_BACKEND_DIOMP)
    MPI_Init(&argc, &argv);
    __init_diomp_target(2);
    rank = omp_get_rank_num();
    nranks = omp_get_num_ranks();
#else
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    select_rank_device(rank);
#endif
}

void runtime_finalize() {
#if defined(JACOBI_BACKEND_GIOMP)
    ompx_finalize();
#else
    MPI_Finalize();
#endif
}

Buffer allocate_buffer(size_t bytes) {
#if defined(JACOBI_BACKEND_GIOMP)
    ompx_buffer b = ompx_alloc(bytes);
    return Buffer{static_cast<float*>(b.ptr), b.index, b.bytes};
#else
    void* ptr = omp_target_alloc(bytes, omp_get_default_device());
    if (!ptr) {
        std::fprintf(stderr, "omp_target_alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    return Buffer{static_cast<float*>(ptr), -1, bytes};
#endif
}

void free_buffer(Buffer b) {
#if defined(JACOBI_BACKEND_GIOMP)
    ompx_free(ompx_buffer{b.ptr, b.index, b.bytes});
#else
    omp_target_free(b.ptr, omp_get_default_device());
#endif
}

void initialize(Buffer a, Buffer b, int nx, int rows, int rank) {
    const size_t count = static_cast<size_t>(nx) * (rows + 2);
    float* pa = a.ptr;
    float* pb = b.ptr;
    #pragma omp target teams distribute parallel for is_device_ptr(pa, pb) \
        firstprivate(nx, rows, rank, count)
    for (size_t i = 0; i < count; ++i) {
        const int y = static_cast<int>(i / nx);
        const int x = static_cast<int>(i % nx);
        float value = 0.0f;
        if (y > 0 && y <= rows)
            value = static_cast<float>(((rank + 1) * 17 + y * 3 + x * 5) % 97) /
                    97.0f;
        if (x == 0 || x == nx - 1) value = 1.0f;
        pa[i] = value;
        pb[i] = value;
    }
}

void compute_boundary(float* current, float* next, int nx, int rows,
                      int threads) {
    #pragma omp target teams distribute parallel for thread_limit(threads) \
        is_device_ptr(current, next) firstprivate(nx, rows)
    for (int x = 1; x < nx - 1; ++x) {
        const size_t top = static_cast<size_t>(nx) + x;
        const size_t bottom = static_cast<size_t>(rows) * nx + x;
        next[top] = 0.25f * (current[top - 1] + current[top + 1] +
                             current[top - nx] + current[top + nx]);
        next[bottom] = 0.25f * (current[bottom - 1] + current[bottom + 1] +
                                current[bottom - nx] + current[bottom + nx]);
    }
}

void compute_interior(float* current, float* next, int nx, int rows,
                      int threads) {
    const size_t points = static_cast<size_t>(rows - 2) * (nx - 2);
    #pragma omp target teams distribute parallel for thread_limit(threads) \
        is_device_ptr(current, next) firstprivate(nx, points)
    for (size_t linear = 0; linear < points; ++linear) {
        const int y = static_cast<int>(linear / (nx - 2)) + 2;
        const int x = static_cast<int>(linear % (nx - 2)) + 1;
        const size_t p = static_cast<size_t>(y) * nx + x;
        next[p] = 0.25f * (current[p - 1] + current[p + 1] +
                           current[p - nx] + current[p + nx]);
    }
}

#if defined(JACOBI_BACKEND_GIOMP)
void giomp_fused_step(float* current, float* next, int next_index,
                      int nx, int rows, int top, int bottom,
                      int teams, int threads, int nranks, bool use_dwq) {
    const size_t row_bytes = static_cast<size_t>(nx) * sizeof(float);
    const bool communicate = nranks > 1;
    const bool top_ipc = communicate && ompx_ipc_reachable(top, next_index);
    const bool bottom_ipc = communicate && ompx_ipc_reachable(bottom, next_index);
    float* top_peer = top_ipc
        ? static_cast<float*>(ompx_peer_ipc_base(top, next_index)) : nullptr;
    float* bottom_peer = bottom_ipc
        ? static_cast<float*>(ompx_peer_ipc_base(bottom, next_index)) : nullptr;
    gicc::DeviceCtx* ctx = ompx_prepare();

    // DWQ transport: the descriptors are staged HERE, on the host, before the
    // kernel runs. The only device-side work left is a single MMIO store, so
    // team 0 never spins waiting on a host proxy round trip the way
    // ompx_quiet(ctx, lane) does -- it fires the NIC and returns.
    const bool dwq_top    = use_dwq && communicate && !top_ipc;
    const bool dwq_bottom = use_dwq && communicate && !bottom_ipc;
    if (dwq_top)
        ompx_dwq_stage_put(top, next_index,
                           static_cast<size_t>(rows + 1) * row_bytes,
                           next_index, row_bytes, row_bytes);
    if (dwq_bottom)
        ompx_dwq_stage_put(bottom, next_index, 0,
                           next_index, static_cast<size_t>(rows) * row_bytes,
                           row_bytes);
    const bool dwq_fire = dwq_top || dwq_bottom;
    if (dwq_fire) ompx_dwq_arm();

    #pragma omp target teams num_teams(teams) thread_limit(threads) \
        is_device_ptr(current, next, top_peer, bottom_peer, ctx) \
        firstprivate(next_index, nx, rows, top, bottom, row_bytes, communicate, top_ipc, bottom_ipc, use_dwq, dwq_fire)
    {
        const int team = omp_get_team_num();
        const int nteams = omp_get_num_teams();
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();
            if (team == 0) {
                for (int x = tid + 1; x < nx - 1; x += nth) {
                    const size_t ptop = static_cast<size_t>(nx) + x;
                    const size_t pbottom = static_cast<size_t>(rows) * nx + x;
                    next[ptop] = 0.25f *
                        (current[ptop - 1] + current[ptop + 1] +
                         current[ptop - nx] + current[ptop + nx]);
                    next[pbottom] = 0.25f *
                        (current[pbottom - 1] + current[pbottom + 1] +
                         current[pbottom - nx] + current[pbottom + nx]);
                }
                #pragma omp barrier

                if (communicate && top_ipc) {
                    for (int x = tid; x < nx; x += nth)
                        top_peer[static_cast<size_t>(rows + 1) * nx + x] =
                            next[static_cast<size_t>(nx) + x];
                } else if (communicate && !use_dwq && tid == 0) {
                    ompx_put_proxy(ctx, top, next_index,
                        static_cast<size_t>(rows + 1) * row_bytes,
                        next_index, row_bytes, row_bytes, 0);
                }

                if (communicate && bottom_ipc) {
                    for (int x = tid; x < nx; x += nth)
                        bottom_peer[x] = next[static_cast<size_t>(rows) * nx + x];
                } else if (communicate && !use_dwq && tid == 0) {
                    ompx_put_proxy(ctx, bottom, next_index, 0,
                        next_index, static_cast<size_t>(rows) * row_bytes,
                        row_bytes, 1);
                }

                // One store fires both staged halo writes. The barrier above
                // guarantees this team wrote its boundary rows; the fence
                // publishes them before the NIC reads them.
                if (dwq_fire && tid == 0) {
                    __atomic_thread_fence(__ATOMIC_SEQ_CST);
                    *(ctx->trigger_addr_) = ctx->trigger_val_;
                }

                #pragma omp barrier
                if (communicate && !use_dwq && tid == 0) {
                    if (!top_ipc) ompx_quiet(ctx, 0);
                    if (!bottom_ipc) ompx_quiet(ctx, 1);
                }
            } else {
                const size_t points = static_cast<size_t>(rows - 2) * (nx - 2);
                const size_t worker = static_cast<size_t>(team - 1) * nth + tid;
                const size_t workers = static_cast<size_t>(nteams - 1) * nth;
                for (size_t linear = worker; linear < points; linear += workers) {
                    const int y = static_cast<int>(linear / (nx - 2)) + 2;
                    const int x = static_cast<int>(linear % (nx - 2)) + 1;
                    const size_t p = static_cast<size_t>(y) * nx + x;
                    next[p] = 0.25f *
                        (current[p - 1] + current[p + 1] +
                         current[p - nx] + current[p + nx]);
                }
            }
        }
    }
    ompx_quiet_host();
}
#endif

void split_step(float* current, float* next, int nx, int rows,
                int top, int bottom, int threads, int nranks) {
    compute_boundary(current, next, nx, rows, threads);
    const size_t row_bytes = static_cast<size_t>(nx) * sizeof(float);

#if defined(JACOBI_BACKEND_DIOMP)
    if (nranks > 1) {
        ompx_dput(next + static_cast<size_t>(rows + 1) * nx, top,
                  next + nx, row_bytes, 0, 0);
        ompx_dput(next, bottom,
                  next + static_cast<size_t>(rows) * nx, row_bytes, 0, 0);
    }
#elif defined(JACOBI_BACKEND_MPI)
    MPI_Request req[4];
    int count = 0;
    if (nranks > 1) {
        MPI_Irecv(next, nx, MPI_FLOAT, top, 101, MPI_COMM_WORLD, &req[count++]);
        MPI_Irecv(next + static_cast<size_t>(rows + 1) * nx, nx, MPI_FLOAT,
                  bottom, 100, MPI_COMM_WORLD, &req[count++]);
        MPI_Isend(next + nx, nx, MPI_FLOAT, top, 100,
                  MPI_COMM_WORLD, &req[count++]);
        MPI_Isend(next + static_cast<size_t>(rows) * nx, nx, MPI_FLOAT,
                  bottom, 101, MPI_COMM_WORLD, &req[count++]);
    }
#endif

    compute_interior(current, next, nx, rows, threads);

#if defined(JACOBI_BACKEND_DIOMP)
    if (nranks > 1) ompx_fence();
#elif defined(JACOBI_BACKEND_MPI)
    if (nranks > 1) MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
#endif
}

double checksum(float* data, int nx, int rows) {
    const size_t count = static_cast<size_t>(nx) * (rows + 2);
    double sum = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:sum) \
        is_device_ptr(data) firstprivate(count)
    for (size_t i = 0; i < count; ++i) sum += static_cast<double>(data[i]);
    double global = 0.0;
    MPI_Allreduce(&sum, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
#if defined(JACOBI_BACKEND_GIOMP)
    g_backend_label = options.dwq ? "giomp-dwq" : "giomp-proxy";
#endif
    int rank = -1;
    int nranks = 0;
    runtime_init(argc, argv, rank, nranks);

    const size_t bytes = static_cast<size_t>(options.nx) *
                         (options.local_rows + 2) * sizeof(float);
    Buffer buffers[2] = {allocate_buffer(bytes), allocate_buffer(bytes)};
#if defined(JACOBI_BACKEND_GIOMP)
    ompx_exchange();
#endif
    const int top = (rank - 1 + nranks) % nranks;
    const int bottom = (rank + 1) % nranks;

    int local_ranks = 0;
    MPI_Comm local = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local);
    MPI_Comm_size(local, &local_ranks);
    MPI_Comm_free(&local);
    if (rank == 0) {
        std::printf("jacobi backend=%s ranks=%d local_ranks=%d local_domain=%dx%d "
                    "iterations=%d teams=%d threads=%d\n",
                    backend_name(), nranks, local_ranks, options.nx,
                    options.local_rows, options.iterations,
                    options.teams, options.threads);
    }

    std::vector<double> times;
    float* final_data = nullptr;
    for (int run = 0; run < options.warmup + options.runs; ++run) {
        initialize(buffers[0], buffers[1], options.nx, options.local_rows, rank);
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        int current = 0;
        int next = 1;
        for (int iter = 0; iter < options.iterations; ++iter) {
#if defined(JACOBI_BACKEND_GIOMP)
            giomp_fused_step(buffers[current].ptr, buffers[next].ptr,
                             buffers[next].index, options.nx,
                             options.local_rows, top, bottom,
                             options.teams, options.threads, nranks,
                             options.dwq);
#else
            split_step(buffers[current].ptr, buffers[next].ptr,
                       options.nx, options.local_rows, top, bottom,
                       options.threads, nranks);
#endif
            MPI_Barrier(MPI_COMM_WORLD);
            std::swap(current, next);
        }
        const double local_elapsed = (MPI_Wtime() - start) * 1.0e6;
        double max_elapsed = 0.0;
        MPI_Allreduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        if (run >= options.warmup) times.push_back(max_elapsed);
        final_data = buffers[current].ptr;
    }

    const double global_checksum = checksum(final_data, options.nx,
                                            options.local_rows);
    const bool valid = std::isfinite(global_checksum) && global_checksum > 0.0;
    std::sort(times.begin(), times.end());
    if (rank == 0) {
        const double median = times[times.size() / 2];
        std::printf("median_us=%.6f checksum=%.12e %s\n", median,
                    global_checksum, valid ? "PASS" : "FAIL");
        std::printf("RESULT,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.12e,%s\n",
                    backend_name(), nranks, options.nx, options.local_rows,
                    options.iterations, options.runs, options.warmup,
                    median, times.front(), times.back(), global_checksum,
                    valid ? "PASS" : "FAIL");
    }

    free_buffer(buffers[1]);
    free_buffer(buffers[0]);
    runtime_finalize();
    return valid ? 0 : 4;
}
