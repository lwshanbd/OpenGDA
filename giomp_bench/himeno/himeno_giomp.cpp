// Himeno benchmark (RIKEN, C + MPI version 3.0, himenoBMTxps.c) on GPUs:
// the point-Jacobi solve of the pressure Poisson equation, its 19-point
// kernel and the copy back into P as in the original, offloaded with
// OpenMP target.
//
// Built two ways from this one file:
//   default          GiOMP: the P faces go to the neighbours by ompx_put
//   -DHIMENO_MPI=1   OpenMP target + GPU-aware MPI: the original's
//                    Isend/Irecv of the same faces (device pointers), each
//                    face sent as one contiguous block of jmax * MKMAX floats
//                    like the puts (the original's strided jkvec type is
//                    packed by Open MPI from device memory many times slower)
//
// Departures from the original, all of them needed to run it this way:
//   * The domain is split over the ranks in I only (the original also allows
//     J and K). An I face is one block of memory, which a put can send as is.
//   * The arrays are flat device allocations with the original's index order
//     and padding (MIMAX = MX0 / ND + 3, MJMAX = MY0, MKMAX = MZ0).
//   * The measured loop runs a fixed -niter iterations instead of as many as
//     fit in 60 s.
//   * GiOMP only: the residual's Allreduce comes before the copy instead of
//     after the exchange. gosa depends on the Jacobi sweep alone, so the
//     result is the same; the Allreduce then also tells every rank that its
//     neighbours have finished reading their halo planes, so the puts after
//     the copy cannot overwrite a plane still being read.
//
// Usage: himeno [-size XS|S|M|L|XL] [-niter N]
// Prints the original's report, the time of the measured loop, and an XOR
// of every owned P value's bits (the same for any number of ranks).
#ifndef HIMENO_MPI
#include "gicc/omp.h"
#include "gicc/omp_pipeline.h"
#endif

#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

int rank = 0, nranks = 1;
int dev = 0;

// Global grid (the original's mx, my, mz = MX0 - 1, ...), this rank's part
// and the allocation sizes.
int mx, my, mz;
int imax, jmax, kmax;
int mimax, mjmax, mkmax;
int left = -1, right = -1;      // neighbours in I, -1 at the ends
int imax_left = 0;              // the left neighbour's imax

// Seconds in each phase of the iterations so far: the sweep (with its
// reduction), the residual's Allreduce, the copy (with the puts or the MPI
// exchange) and, for GiOMP, the fence.
double t_sweep = 0, t_reduce = 0, t_copy = 0, t_fence = 0, t_quiet = 0;

float *a0, *a1, *a2, *a3, *b0, *b1, *b2, *c0, *c1, *c2, *bnd, *wrk1, *wrk2, *p;

#define IDX(i, j, k) ((long)mjmax * mkmax * (i) + (long)mkmax * (j) + (k))

double fflop(int _mx, int _my, int _mz) {
    return (double)(_mz - 2) * (double)(_my - 2) * (double)(_mx - 2) * 34.0;
}

double mflops(int nn, double cpu, double flop) { return flop / cpu * 1.e-6 * (double)nn; }

// The original's initmax, for the I split: this rank's imax, and the global
// index of its plane 0 (returned).
int initmax() {
    const int tmp = mx / nranks;
    int mx1[nranks + 1];
    mx1[0] = 0;
    for (int i = 1; i <= nranks; i++) mx1[i] = mx1[i - 1] + tmp + (i <= mx % nranks ? 1 : 0);
    auto width = [&](int r) {
        return mx1[r + 1] - mx1[r] + (r != 0 ? 1 : 0) + (r != nranks - 1 ? 1 : 0);
    };
    imax = width(rank);
    jmax = my;
    kmax = mz;
    if (rank > 0) imax_left = width(rank - 1);
    return rank == 0 ? mx1[rank] : mx1[rank] - 1;
}

float* dev_alloc(long n) {
    return static_cast<float*>(omp_target_alloc(n * sizeof(float), dev));
}

// The original's initmt, on the device.
void initmt(int it) {
    const int ni = imax, nj = jmax, nk = kmax, gx = mx;
    const int sj = mjmax, sk = mkmax;
    float *A0 = a0, *A1 = a1, *A2 = a2, *A3 = a3, *B0 = b0, *B1 = b1, *B2 = b2;
    float *C0 = c0, *C1 = c1, *C2 = c2, *BND = bnd, *W1 = wrk1, *W2 = wrk2, *P = p;
    #pragma omp target teams distribute parallel for collapse(3) \
        is_device_ptr(A0, A1, A2, A3, B0, B1, B2, C0, C1, C2, BND, W1, W2, P)
    for (int i = 0; i < ni; ++i)
        for (int j = 0; j < nj; ++j)
            for (int k = 0; k < nk; ++k) {
                const long x = (long)sj * sk * i + (long)sk * j + k;
                A0[x] = 1.0;
                A1[x] = 1.0;
                A2[x] = 1.0;
                A3[x] = 1.0 / 6.0;
                B0[x] = 0.0;
                B1[x] = 0.0;
                B2[x] = 0.0;
                C0[x] = 1.0;
                C1[x] = 1.0;
                C2[x] = 1.0;
                P[x] = (float)((i + it) * (i + it)) / (float)((gx - 1) * (gx - 1));
                W1[x] = 0.0;
                W2[x] = 0.0;
                BND[x] = 1.0;
            }
}

// The 19-point sweep into wrk2; returns this rank's sum of squared updates.
float sweep() {
    const int ni = imax, nj = jmax, nk = kmax;
    const long si = (long)mjmax * mkmax, sj = mkmax;
    const float *A0 = a0, *A1 = a1, *A2 = a2, *A3 = a3, *B0 = b0, *B1 = b1, *B2 = b2;
    const float *C0 = c0, *C1 = c1, *C2 = c2, *BND = bnd, *W1 = wrk1, *P = p;
    float* W2 = wrk2;
    float wgosa = 0.0;
    #pragma omp target teams distribute parallel for collapse(3) reduction(+ : wgosa) \
        is_device_ptr(A0, A1, A2, A3, B0, B1, B2, C0, C1, C2, BND, W1, W2, P)
    for (int i = 1; i < ni - 1; ++i)
        for (int j = 1; j < nj - 1; ++j)
            for (int k = 1; k < nk - 1; ++k) {
                const long x = si * i + sj * j + k;
                const float s0 =
                    A0[x] * P[x + si] + A1[x] * P[x + sj] + A2[x] * P[x + 1] +
                    B0[x] * (P[x + si + sj] - P[x + si - sj] - P[x - si + sj] + P[x - si - sj]) +
                    B1[x] * (P[x + sj + 1] - P[x - sj + 1] - P[x + sj - 1] + P[x - sj - 1]) +
                    B2[x] * (P[x + si + 1] - P[x - si + 1] - P[x + si - 1] + P[x - si - 1]) +
                    C0[x] * P[x - si] + C1[x] * P[x - sj] + C2[x] * P[x - 1] + W1[x];
                const float ss = (s0 * A3[x] - P[x]) * BND[x];
                wgosa += ss * ss;
                W2[x] = P[x] + 0.8 * ss;
            }
    return wgosa;
}

// P = W2 over the interior.
void copy_back(float* P, const float* W2) {
    const int ni = imax, nj = jmax, nk = kmax;
    const long si = (long)mjmax * mkmax, sj = mkmax;
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(W2, P)
    for (int i = 1; i < ni - 1; ++i)
        for (int j = 1; j < nj - 1; ++j)
            for (int k = 1; k < nk - 1; ++k) {
                const long x = si * i + sj * j + k;
                P[x] = W2[x];
            }
}

#ifndef HIMENO_MPI

// The copy, then plane 1 into the left neighbour's last plane and plane
// imax - 2 into the right neighbour's plane 0. P sits at the same heap
// offset on every rank, so a peer's plane is that offset in my own P.
__attribute__((noinline)) void copy_and_send(float* P, const float* W2, int ni, int ni_left,
                                             long plane, int has_left, int has_right) {
    copy_back(P, W2);
    if (has_left) ompx_put(left, P + (ni_left - 1) * plane, P + plane, plane * sizeof(float));
    if (has_right) ompx_put(right, P, P + (ni - 2) * plane, plane * sizeof(float));
}

float jacobi(int nn) {
    float gosa = 0.0;
    const long plane = (long)mjmax * mkmax;
    for (int n = 0; n < nn; ++n) {
        const double t0 = MPI_Wtime();
        const float wgosa = sweep();
        const double t1 = MPI_Wtime();
        MPI_Allreduce(&wgosa, &gosa, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        const double t2 = MPI_Wtime();
        copy_and_send(p, wrk2, imax, imax_left, plane, left >= 0, right >= 0);
        const double t3 = MPI_Wtime();
        // ompx_fence, timed in its two halves.
        ompx_quiet();
        const double tq = MPI_Wtime();
        ompx_barrier();
        const double t4 = MPI_Wtime();
        t_quiet += tq - t3;
        t_sweep += t1 - t0;
        t_reduce += t2 - t1;
        t_copy += t3 - t2;
        t_fence += t4 - t3;
    }
    return gosa;
}

#else

// The original's sendp1.
void sendp1() {
    MPI_Request req[4];
    int n = 0;
    const int count = jmax * mkmax;
    if (right >= 0) MPI_Irecv(&p[IDX(imax - 1, 0, 0)], count, MPI_FLOAT, right, 1, MPI_COMM_WORLD, &req[n++]);
    if (left >= 0) MPI_Irecv(&p[IDX(0, 0, 0)], count, MPI_FLOAT, left, 2, MPI_COMM_WORLD, &req[n++]);
    if (left >= 0) MPI_Isend(&p[IDX(1, 0, 0)], count, MPI_FLOAT, left, 1, MPI_COMM_WORLD, &req[n++]);
    if (right >= 0) MPI_Isend(&p[IDX(imax - 2, 0, 0)], count, MPI_FLOAT, right, 2, MPI_COMM_WORLD, &req[n++]);
    MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
}

float jacobi(int nn) {
    float gosa = 0.0;
    for (int n = 0; n < nn; ++n) {
        const double t0 = MPI_Wtime();
        const float wgosa = sweep();
        const double t1 = MPI_Wtime();
        copy_back(p, wrk2);
        if (nranks > 1) sendp1();
        const double t2 = MPI_Wtime();
        MPI_Allreduce(&wgosa, &gosa, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        const double t3 = MPI_Wtime();
        t_sweep += t1 - t0;
        t_copy += t2 - t1;
        t_reduce += t3 - t2;
    }
    return gosa;
}

#endif

// XOR of the bits of every P value this rank owns (its halo planes left
// out), combined over the ranks.
unsigned field_xor() {
    const int i0 = rank == 0 ? 0 : 1, i1 = rank == nranks - 1 ? imax : imax - 1;
    const int nj = jmax, nk = kmax;
    const long si = (long)mjmax * mkmax, sj = mkmax;
    const float* P = p;
    unsigned x = 0;
    #pragma omp target teams distribute parallel for collapse(3) reduction(^ : x) is_device_ptr(P)
    for (int i = i0; i < i1; ++i)
        for (int j = 0; j < nj; ++j)
            for (int k = 0; k < nk; ++k) {
                unsigned u;
                __builtin_memcpy(&u, &P[si * i + sj * j + k], sizeof u);
                x ^= u;
            }
    unsigned all = 0;
    MPI_Allreduce(&x, &all, 1, MPI_UNSIGNED, MPI_BXOR, MPI_COMM_WORLD);
    return all;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
#ifndef HIMENO_MPI
    ompx_init();
    rank = ompx_get_rank_num();
    nranks = ompx_get_num_ranks();
#else
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int lr = 0;
    MPI_Comm_rank(local, &lr);
    MPI_Comm_free(&local);
    omp_set_default_device(lr % omp_get_num_devices());
#endif
    dev = omp_get_default_device();

    std::string size = "XL";
    int niter = 1000;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-size")) size = argv[i + 1];
        else if (!strcmp(argv[i], "-niter")) niter = atoi(argv[i + 1]);
    }
    int mx0, my0, mz0;
    if (size == "XS") mx0 = 33, my0 = 33, mz0 = 65;
    else if (size == "S") mx0 = 65, my0 = 65, mz0 = 129;
    else if (size == "M") mx0 = 129, my0 = 129, mz0 = 257;
    else if (size == "L") mx0 = 257, my0 = 257, mz0 = 513;
    else if (size == "XL") mx0 = 513, my0 = 513, mz0 = 1025;
    else {
        if (rank == 0) fprintf(stderr, "unknown size %s\n", size.c_str());
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
    mx = mx0 - 1;
    my = my0 - 1;
    mz = mz0 - 1;
    // paramset.sh's MIMAX, MJMAX, MKMAX for an I-only split.
    mimax = nranks == 1 ? mx0 : mx0 / nranks + 3;
    mjmax = my0;
    mkmax = mz0;
    if (rank > 0) left = rank - 1;
    if (rank < nranks - 1) right = rank + 1;

    const int it = initmax();
    const long n = (long)mimax * mjmax * mkmax;
    a0 = dev_alloc(n); a1 = dev_alloc(n); a2 = dev_alloc(n); a3 = dev_alloc(n);
    b0 = dev_alloc(n); b1 = dev_alloc(n); b2 = dev_alloc(n);
    c0 = dev_alloc(n); c1 = dev_alloc(n); c2 = dev_alloc(n);
    bnd = dev_alloc(n); wrk1 = dev_alloc(n); wrk2 = dev_alloc(n);
#ifndef HIMENO_MPI
    p = static_cast<float*>(ompx_alloc(n * sizeof(float)));
#else
    p = dev_alloc(n);
#endif
    initmt(it);
#ifndef HIMENO_MPI
    ompx_prepare();
#endif

    const double flop = fflop(mz, my, mx);
    if (rank == 0) {
        printf("Sequential version array size\n");
        printf(" mimax = %d mjmax = %d mkmax = %d\n", mx, my, mz);
        printf("Parallel version array size\n");
        printf(" mimax = %d mjmax = %d mkmax = %d\n", mimax, mjmax, mkmax);
        printf("imax = %d jmax = %d kmax =%d\n", imax, jmax, kmax);
        printf("I-decomp = %d J-decomp = %d K-decomp =%d\n", nranks, 1, 1);
        printf(" Start rehearsal measurement process.\n");
        printf(" Measure the performance in %d times.\n\n", 3);
    }

    // Rehearsal, as in the original: three iterations.
    MPI_Barrier(MPI_COMM_WORLD);
    double cpu0 = MPI_Wtime();
    float gosa = jacobi(3);
    double cpu1 = MPI_Wtime() - cpu0, cpu = 0;
    MPI_Allreduce(&cpu1, &cpu, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf(" MFLOPS: %f time(s): %f %e\n\n", mflops(3, cpu, flop), cpu, gosa);

    if (rank == 0) printf(" Now, start the actual measurement process.\n");
    t_sweep = t_reduce = t_copy = t_fence = t_quiet = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    cpu0 = MPI_Wtime();
    gosa = jacobi(niter);
    cpu1 = MPI_Wtime() - cpu0;
    MPI_Allreduce(&cpu1, &cpu, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double tl[5] = {t_sweep, t_reduce, t_copy, t_fence, t_quiet}, tm[5];
    MPI_Reduce(tl, tm, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (getenv("HIMENO_PER_RANK"))
        for (int r = 0; r < nranks; ++r) {
            if (r == rank)
                printf("rank %d: sweep %.4f allreduce %.4f copy %.4f fence %.4f quiet %.4f\n", rank,
                       t_sweep, t_reduce, t_copy, t_fence, t_quiet);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    const unsigned x = field_xor();
    if (rank == 0) {
        printf("cpu : %f sec.\n", cpu);
        printf("Loop executed for %d times\n", niter);
        printf("Gosa : %e \n", gosa);
        printf("MFLOPS measured : %f\n", mflops(niter, cpu, flop));
        printf("Score based on Pentium III 600MHz : %f\n", mflops(niter, cpu, flop) / 82.84);
        printf("phases (max over ranks): sweep %.4f s, allreduce %.4f s, copy+exchange %.4f s, "
               "fence %.4f s (quiet %.4f s)\n", tm[0], tm[1], tm[2], tm[3], tm[4]);
        printf("field_xor = %08x\n", x);
    }

    float* all[] = {a0, a1, a2, a3, b0, b1, b2, c0, c1, c2, bnd, wrk1, wrk2};
    for (float* q : all) omp_target_free(q, dev);
#ifndef HIMENO_MPI
    ompx_finalize();
#else
    omp_target_free(p, dev);
    MPI_Finalize();
#endif
    return 0;
}
