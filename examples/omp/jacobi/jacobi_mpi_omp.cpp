// NVIDIA multi-gpu-programming-models "mpi" Jacobi solver with its CUDA
// kernels replaced by OpenMP offload: the baseline for jacobi_giomp.cpp.
// The kernels are the same as there, line for line; the halo rows go by
// CUDA-aware MPI_Sendrecv of device pointers, as in the original.
//
//   jacobi_mpi_omp [-niter N] [-nccheck N] [-nx N] [-ny N] [-csv] [-noref]
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
#else
typedef float real;
#define MPI_REAL_TYPE MPI_FLOAT
#endif

constexpr real tol = 1.0e-8;
const real PI = 2.0 * std::asin(1.0);

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    return std::find(begin, end, arg) != end;
}

// Dirichlet boundary on the left and right columns of my_ny rows starting at
// global row `offset`.
static void initialize_boundaries(real* a_new, real* a, real pi, int offset, int nx, int my_ny,
                                  int ny) {
    #pragma omp target teams distribute parallel for is_device_ptr(a, a_new)
    for (int iy = 0; iy < my_ny; iy++) {
        const real y0 = std::sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[iy * nx + 0] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx + 0] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

static void zero(real* a, long n) {
    #pragma omp target teams distribute parallel for is_device_ptr(a)
    for (long i = 0; i < n; i++) a[i] = 0;
}

// One sweep over rows [iy_start, iy_end); returns the squared L2 norm of
// the update.
static real jacobi_sweep(real* a_new, const real* a, int iy_start, int iy_end, int nx) {
    real l2 = 0;
#ifdef NESTED
    // Rows over teams, columns over a team's threads.
    #pragma omp target teams distribute reduction(+ : l2) is_device_ptr(a, a_new)
    for (int iy = iy_start; iy < iy_end; iy++)
        #pragma omp parallel for reduction(+ : l2)
        for (int ix = 1; ix < nx - 1; ix++) {
#else
    #pragma omp target teams distribute parallel for collapse(2) reduction(+ : l2) \
        is_device_ptr(a, a_new)
    for (int iy = iy_start; iy < iy_end; iy++)
        for (int ix = 1; ix < nx - 1; ix++) {
#endif
            const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                         a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
            a_new[iy * nx + ix] = new_val;
            const real residue = new_val - a[iy * nx + ix];
            l2 += residue * residue;
        }
    return l2;
}

// The halo exchange of the original: my first computed row to the top
// neighbour while its last arrives in my bottom halo row, then my last to
// the bottom neighbour while its first arrives in my top halo row.
static void exchange(real* a_new, int iy_start, int iy_end, int nx, int top, int bottom) {
    MPI_Sendrecv(a_new + iy_start * nx, nx, MPI_REAL_TYPE, top, 0, a_new + iy_end * nx, nx,
                 MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a_new + (iy_end - 1) * nx, nx, MPI_REAL_TYPE, bottom, 0, a_new, nx,
                 MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// The whole mesh on this rank's GPU, as the original's single_gpu: the
// reference result and the 1-GPU time.
static double single_gpu(int nx, int ny, int iter_max, real* a_ref_h, int nccheck, bool print) {
    const long n = long(nx) * ny;
    real* a = static_cast<real*>(omp_target_alloc(n * sizeof(real), omp_get_default_device()));
    real* a_new = static_cast<real*>(omp_target_alloc(n * sizeof(real), omp_get_default_device()));
    zero(a, n);
    zero(a_new, n);
    initialize_boundaries(a, a_new, PI, 0, nx, ny, ny);
    const int iy_start = 1, iy_end = ny - 1;
    if (print)
        printf("Single GPU jacobi relaxation: %d iterations on %d x %d mesh with norm check "
               "every %d iterations\n",
               iter_max, ny, nx, nccheck);
    int iter = 0;
    real l2_norm = 1.0;
    const double start = MPI_Wtime();
    while (l2_norm > tol && iter < iter_max) {
        const real l2 = jacobi_sweep(a_new, a, iy_start, iy_end, nx);
        // Periodic boundary: the last computed row to the top halo, the first
        // to the bottom halo.
        #pragma omp target teams distribute parallel for is_device_ptr(a_new)
        for (int ix = 0; ix < nx; ix++) {
            a_new[ix] = a_new[(iy_end - 1) * nx + ix];
            a_new[iy_end * nx + ix] = a_new[iy_start * nx + ix];
        }
        const bool calculate_norm = (iter % nccheck) == 0 || (iter % 100) == 0;
        if (calculate_norm) {
            l2_norm = std::sqrt(l2);
            if (print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }
        std::swap(a_new, a);
        iter++;
    }
    const double stop = MPI_Wtime();
    omp_target_memcpy(a_ref_h, a, n * sizeof(real), 0, 0, omp_get_initial_device(),
                      omp_get_default_device());
    omp_target_free(a, omp_get_default_device());
    omp_target_free(a_new, omp_get_default_device());
    return stop - start;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool csv = get_arg(argv, argv + argc, "-csv");
    const bool noref = get_arg(argv, argv + argc, "-noref");

    std::vector<real> a_ref_h, a_h(size_t(nx) * ny);
    double runtime_serial = 0;
    if (!noref) {
        a_ref_h.resize(size_t(nx) * ny);
        runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h.data(), nccheck, !csv && rank == 0);
    }

    // ny - 2 rows over the ranks, (ny - 2) / size or one more each.
    const int chunk_size_low = (ny - 2) / size;
    const int chunk_size_high = chunk_size_low + 1;
    const int num_ranks_low = size * chunk_size_low + size - (ny - 2);
    auto chunk_of = [&](int r) { return r < num_ranks_low ? chunk_size_low : chunk_size_high; };
    const int chunk_size = chunk_of(rank);

    // The larger chunk on every rank, as in jacobi_giomp.cpp.
    const long n = long(nx) * (chunk_size_high + 2);
    real* a = static_cast<real*>(omp_target_alloc(n * sizeof(real), omp_get_default_device()));
    real* a_new =
        static_cast<real*>(omp_target_alloc(n * sizeof(real), omp_get_default_device()));
    zero(a, n);
    zero(a_new, n);

    int iy_start_global;
    if (rank < num_ranks_low)
        iy_start_global = rank * chunk_size_low + 1;
    else
        iy_start_global =
            num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
    const int iy_end_global = iy_start_global + chunk_size - 1;
    const int iy_start = 1;
    const int iy_end = iy_start + chunk_size;

    initialize_boundaries(a, a_new, PI, iy_start_global - 1, nx, chunk_size + 2, ny);

    const int top = rank > 0 ? rank - 1 : (size - 1);
    const int bottom = (rank + 1) % size;

    // Warm-up: halo exchanges only, as the original's MPI warm-up.
    for (int i = 0; i < 10; ++i) {
        exchange(a_new, iy_start, iy_end, nx, top, bottom);
        std::swap(a_new, a);
    }

    if (!csv && rank == 0)
        printf("Jacobi relaxation: %d iterations on %d x %d mesh with norm check every %d "
               "iterations\n",
               iter_max, ny, nx, nccheck);

    int iter = 0;
    real l2_norm = 1.0;
    double t_step = 0, t_fence = 0, t_norm = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    while (l2_norm > tol && iter < iter_max) {
        const bool calculate_norm = (iter % nccheck) == 0 || (!csv && (iter % 100) == 0);
        const double t0 = MPI_Wtime();
        const real l2 = jacobi_sweep(a_new, a, iy_start, iy_end, nx);
        const double t1 = MPI_Wtime();
        exchange(a_new, iy_start, iy_end, nx, top, bottom);
        const double t2 = MPI_Wtime();
        t_step += t1 - t0;
        t_fence += t2 - t1;
        if (calculate_norm) {
            MPI_Allreduce(&l2, &l2_norm, 1, MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD);
            t_norm += MPI_Wtime() - t2;
            l2_norm = std::sqrt(l2_norm);
            if (!csv && rank == 0 && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }
        std::swap(a_new, a);
        iter++;
    }
    const double stop = MPI_Wtime();
    double tmax[3], tloc[3] = {t_step, t_fence, t_norm};
    MPI_Reduce(tloc, tmax, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0 && !csv)
        printf("phases (max over ranks): step %.4f s, exchange %.4f s, allreduce %.4f s\n",
               tmax[0], tmax[1], tmax[2]);

    omp_target_memcpy(a_h.data() + size_t(iy_start_global) * nx, a,
                      std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(real),
                      0, size_t(nx) * sizeof(real), omp_get_initial_device(),
                      omp_get_default_device());
    int result_correct = 1;
    for (int iy = iy_start_global; !noref && result_correct && iy < iy_end_global; ++iy)
        for (int ix = 1; result_correct && ix < nx - 1; ++ix)
            if (std::fabs(a_ref_h[size_t(iy) * nx + ix] - a_h[size_t(iy) * nx + ix]) > tol) {
                fprintf(stderr, "ERROR on rank %d: a[%d * %d + %d] = %f does not match %f "
                                "(reference)\n",
                        rank, iy, nx, ix, a_h[size_t(iy) * nx + ix], a_ref_h[size_t(iy) * nx + ix]);
                result_correct = 0;
            }
    int global_result_correct = 1;
    MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    result_correct = global_result_correct;
    if (rank == 0 && result_correct) {
        if (csv) {
            printf("mpi_omp, %d, %d, %d, %d, %d, 1, %f, %f\n", nx, ny, iter_max, nccheck, size,
                   stop - start, runtime_serial);
        } else {
            printf("Num GPUs: %d.\n", size);
            printf("%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, efficiency: %8.2f \n",
                   ny, nx, runtime_serial, size, stop - start, runtime_serial / (stop - start),
                   runtime_serial / (size * (stop - start)) * 100);
        }
    }
    if (rank == 0 && !result_correct) printf("RESULT MISMATCH\n");
    omp_target_free(a, omp_get_default_device());
    omp_target_free(a_new, omp_get_default_device());
    MPI_Finalize();
    return result_correct == 1 ? 0 : 1;
}
