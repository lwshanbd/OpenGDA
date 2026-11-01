// NAS Parallel Benchmarks FT on GPUs: the 3-D FFT PDE solver with the
// MPI version's 1-D (slab) decomposition, its kernels offloaded with OpenMP
// target. The kernels follow NPB-CPP's C++ FT (the FFT is NPB's Stockham
// variant, cfftz/fftz2, one thread per line as in NPB-GPU); the distribution
// follows NPB3.x-MPI's FT with layout_1D.
//
// Built from this one file:
//   default        GiOMP: the transposes move their blocks by ompx_put
//   -DFT_MPI=1     OpenMP target + GPU-aware MPI, the same blocks by
//                  Isend/Irecv (device pointers)
//   -DFT_MPI=2     OpenMP target + GPU-aware MPI by MPI_Alltoall, as in
//                  NPB-MPI, with a pack or unpack kernel around it
//   -DFT_STREAM=1  with any of the above: the inverse z FFT writes each
//                  peer's z planes straight into a send buffer of its own,
//                  ordered by blocks of FT_LB lines, so that its first
//                  blocks are final while the kernel still runs; the
//                  receiver unpacks them. One send per peer.
//
// Layouts, P ranks, rank r (x fastest in both):
//   A   x, y, z  with z in [r * nz/P, (r+1) * nz/P)  -- the x and y FFTs
//   B   x, y, z  with y in [r * ny/P, (r+1) * ny/P)  -- the z FFT
// NPB-MPI keeps z fastest in B (its layout 3); on a GPU x fastest lets
// consecutive threads take consecutive z lines without a gather. A
// transpose sends, for every local z plane, the y rows that belong to the
// peer: nz/P runs of nx * ny/P elements per peer, each landing contiguous.
//
// Usage: ft [-class S|W|A|B|C|D] [-niter N]
// Prints NPB's report: the checksum per iteration, the verification
// against NPB's reference checksums, and the time of the timed section.
#ifndef FT_MPI
#include "gicc/omp.h"
#if FT_STREAM
#include "gicc/omp_pipeline.h"
#endif
#endif
#ifndef FT_STREAM
#define FT_STREAM 0
#endif
#ifndef FT_LB
#define FT_LB 128
#endif
// -DFT_TL=N: N threads per team for the two line FFT kernels (fft_lines and
// fftz_send), whatever the build; the tuner found 1024 for them.
#ifdef FT_TL
#define FT_LINE_TL thread_limit(FT_TL)
#else
#define FT_LINE_TL
#endif
#ifndef FT_ZMAX
#define FT_ZMAX 512   // longest z line the streamed FFT holds per thread
#endif
#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ft_refs.inc"

namespace {

struct dcomplex {
    double re, im;
};
typedef double double2 __attribute__((ext_vector_type(2)));

constexpr double kSeed = 314159265.0, kA = 1220703125.0;
constexpr double kPi = 3.141592653589793238, kAlpha = 1.0e-6;

int rank = 0, nranks = 1, dev = 0;
int nx, ny, nz, niter;
long nzp, nyp;          // local z planes (layout A), local y rows (layout B)
long run;               // elements in one transpose run: nx * ny/P
long ntdivnp;           // local elements
char cls = 'U';

dcomplex *u0, *u1, *u2;   // u0, u1 in layout B; u2 in layout A
dcomplex *w0, *w1;        // FFT scratch
dcomplex *roots;          // fft_init's table, on the device
double* twiddle;          // layout B
#if defined(FT_MPI) && FT_MPI == 2
dcomplex* abuf;           // the Alltoall's packed side
#endif
double* row_seed;         // the generator's state at each local row of A
dcomplex *sbuf, *rbuf;    // FT_STREAM: one slot per peer, sent and received

// Seconds per phase, over the timed section.
double t_evolve = 0, t_fftz = 0, t_xchg = 0, t_quiet = 0, t_ffty = 0, t_fftx = 0, t_chk = 0;

#pragma omp declare target
// NPB's randlc and vranlc: x_{k+1} = a x_k mod 2^46.
inline double randlc(double* x, double a) {
    const double r23 = 1.1920928955078125e-07, r46 = r23 * r23;
    const double t23 = 8.388608e+06, t46 = t23 * t23;
    double t1 = r23 * a;
    const double a1 = (int)t1, a2 = a - t23 * a1;
    t1 = r23 * (*x);
    const double x1 = (int)t1, x2 = (*x) - t23 * x1;
    t1 = a1 * x2 + a2 * x1;
    const double t2 = (int)(r23 * t1);
    const double z = t1 - t23 * t2;
    const double t3 = t23 * z + a2 * x2;
    const double t4 = (int)(r46 * t3);
    *x = t3 - t46 * t4;
    return r46 * (*x);
}
#pragma omp end declare target

void ipow46(double a, int exponent, double* result) {
    *result = 1;
    if (exponent == 0) return;
    double q = a, r = 1;
    int n = exponent;
    while (n > 1) {
        const int n2 = n / 2;
        if (n2 * 2 == n) {
            randlc(&q, q);
            n = n2;
        } else {
            randlc(&r, q);
            n = n - 1;
        }
    }
    randlc(&r, q);
    *result = r;
}

int ilog2(int n) {
    int lg = 0;
    while ((1 << lg) < n) ++lg;
    return lg;
}

dcomplex* dev_alloc(long n) {
    return static_cast<dcomplex*>(omp_target_alloc(n * sizeof(dcomplex), dev));
}
#ifndef FT_MPI
dcomplex* heap_alloc(long n) { return static_cast<dcomplex*>(ompx_alloc(n * sizeof(dcomplex))); }
#else
dcomplex* heap_alloc(long n) { return dev_alloc(n); }
#endif

// N-point FFTs (n = 2^m) of nlines lines in X, Y as scratch, direction is.
// Element e of line L sits at (L % la + (L / la) * sb) + e * es; X and Y
// use the same addressing. One thread per line runs NPB's cfftz: m passes
// of fftz2 alternating between X and Y, then Y copied into X if m is odd.
void fft_lines(int is, int n, long nlines, long la, long sb, long es, dcomplex* X, dcomplex* Y) {
    const int m = ilog2(n);
    const dcomplex* U = roots;
    #pragma omp target teams distribute parallel for FT_LINE_TL is_device_ptr(X, Y, U)
    for (long L = 0; L < nlines; ++L) {
        const long base = L % la + (L / la) * sb;
        dcomplex *x = X, *y = Y;
        for (int l = 1; l <= m; ++l) {
            const int n1 = n / 2, lk = 1 << (l - 1), li = 1 << (m - l), lj = 2 * lk;
            for (int i = 0; i < li; ++i) {
                const int i11 = i * lk, i12 = i11 + n1, i21 = i * lj, i22 = i21 + lk;
                const double ur = U[li + i].re, ui = is >= 1 ? U[li + i].im : -U[li + i].im;
                for (int k = 0; k < lk; ++k) {
                    const dcomplex x11 = x[base + (i11 + k) * es], x21 = x[base + (i12 + k) * es];
                    const double dr = x11.re - x21.re, di = x11.im - x21.im;
                    y[base + (i21 + k) * es] = {x11.re + x21.re, x11.im + x21.im};
                    y[base + (i22 + k) * es] = {ur * dr - ui * di, ur * di + ui * dr};
                }
            }
            dcomplex* t = x;
            x = y;
            y = t;
        }
        if (m & 1)
            for (int e = 0; e < n; ++e) X[base + e * es] = Y[base + e * es];
    }
}

// The x FFT of layout A: x is contiguous, so gather the lines into w0 with
// the line index fastest, transform there, and scatter back.
void fftx(int is, dcomplex* A) {
    const long nl = (long)ny * nzp;
    const int n1 = nx;
    dcomplex *W0 = w0, *W1 = w1;
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(A, W0)
    for (long x = 0; x < n1; ++x)
        for (long L = 0; L < nl; ++L) W0[L + nl * x] = A[x + n1 * L];
    fft_lines(is, nx, nl, nl, 0, nl, w0, w1);
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(A, W0)
    for (long L = 0; L < nl; ++L)
        for (long x = 0; x < n1; ++x) A[x + n1 * L] = W0[L + nl * x];
}

// The y FFT of layout A: lines (x, z), consecutive in x.
void ffty(int is, dcomplex* A) { fft_lines(is, ny, (long)nx * nzp, nx, (long)nx * ny, nx, A, w0); }

// The z FFT of layout B: lines (x, y), consecutive.
void fftz(int is, dcomplex* B) { fft_lines(is, nz, run, run, 0, run, B, w0); }

#if FT_STREAM
#pragma omp declare target
// One pass of fftz2 over a line held by the thread: x -> y.
__attribute__((always_inline)) inline void fftz2_line(int is, int l, int m, int n,
                                                      const dcomplex* U, const dcomplex* x,
                                                      dcomplex* y) {
    const int n1 = n / 2, lk = 1 << (l - 1), li = 1 << (m - l), lj = 2 * lk;
    for (int i = 0; i < li; ++i) {
        const int i11 = i * lk, i12 = i11 + n1, i21 = i * lj, i22 = i21 + lk;
        const double ur = U[li + i].re, ui = is >= 1 ? U[li + i].im : -U[li + i].im;
        for (int k = 0; k < lk; ++k) {
            const double ar = x[i11 + k].re, ai = x[i11 + k].im;
            const double br = x[i12 + k].re, bi = x[i12 + k].im;
            const double dr = ar - br, di = ai - bi;
            y[i21 + k].re = ar + br;
            y[i21 + k].im = ai + bi;
            y[i22 + k].re = ur * dr - ui * di;
            y[i22 + k].im = ur * di + ui * dr;
        }
    }
}
#pragma omp end declare target

// The inverse z FFT of layout B (u1), each line transformed in the
// thread's own arrays (NPB's cfftz), its result going out directly: my own
// z planes into layout A (u2), those of peer (r + 1 + q) % np into slot q of
// S at q * slot + (L / FT_LB) * FT_LB * nzp + zl * FT_LB + L % FT_LB, so
// lines in order fill every slot in order. Slot q of S lands in slot q of
// the peer's R, which therefore holds in its slot q what rank
// (peer - 1 - q) mod np sent.
__attribute__((noinline)) void fftz_send(int is, const dcomplex* B, dcomplex* A, dcomplex* S,
                                         dcomplex* R, int r, int np, long nb, long nzl,
                                         long rn, long a_own, long b_own, long plane) {
    const int n = nz, m = ilog2(n);
    const dcomplex* U = roots;
    const long slot = nb * FT_LB * nzl;
    double2* S2 = reinterpret_cast<double2*>(S);
    #pragma omp target teams distribute parallel for collapse(2) FT_LINE_TL is_device_ptr(B, A, S2, U)
    for (long lb = 0; lb < nb; ++lb)
        for (long li = 0; li < FT_LB; ++li) {
            const long L = lb * FT_LB + li;
            dcomplex x[FT_ZMAX], y[FT_ZMAX];
            // Field by field: a struct copy is a memcpy, which the pass
            // cannot see through.
            for (int e = 0; e < n; ++e) {
                x[e].re = B[L + e * rn].re;
                x[e].im = B[L + e * rn].im;
            }
            for (int l = 1; l <= m; l += 2) {
                fftz2_line(is, l, m, n, U, x, y);
                if (l == m) {
                    for (int e = 0; e < n; ++e) {
                        x[e].re = y[e].re;
                        x[e].im = y[e].im;
                    }
                    break;
                }
                fftz2_line(is, l + 1, m, n, U, y, x);
            }
            for (long zl = 0; zl < nzl; ++zl) {
                A[a_own + L + plane * zl].re = x[b_own + zl].re;
                A[a_own + L + plane * zl].im = x[b_own + zl].im;
            }
            // One 16-byte store per element, so S has one access pattern.
            for (long q = 0; q < np - 1; ++q) {
                const long b_peer = ((r + 1 + q) % np) * nzl;
                for (long zl = 0; zl < nzl; ++zl)
                    S2[q * slot + lb * (FT_LB * nzl) + zl * FT_LB + li] =
                        double2{x[b_peer + zl].re, x[b_peer + zl].im};
            }
        }
#ifndef FT_MPI
    // No %: a division may trap, so the pass cannot compute the peer
    // before the launch.
    for (int q = 0; q < np - 1; ++q) {
        const int p = r + 1 + q < np ? r + 1 + q : r + 1 + q - np;
        ompx_put(p, R + q * slot, S + q * slot, slot * sizeof(dcomplex));
    }
#endif
}

// The peers' z planes, from rbuf into layout A: slot q came from rank
// (r - 1 - q) mod np.
void unpack_stream() {
    const long nb = run / FT_LB, nzl = nzp, plane = (long)nx * ny, slot = run * nzp;
    const long np = nranks, r = rank, ny_p = nyp, n1 = nx;
    dcomplex *A = u2, *R = rbuf;
    #pragma omp target teams distribute parallel for collapse(4) is_device_ptr(A, R)
    for (long q = 0; q < np - 1; ++q)
        for (long zl = 0; zl < nzl; ++zl)
            for (long lb = 0; lb < nb; ++lb)
                for (long li = 0; li < FT_LB; ++li) {
                    const long src = (r - 1 - q + np) % np;
                    A[n1 * src * ny_p + lb * FT_LB + li + plane * zl] =
                        R[q * slot + lb * (FT_LB * nzl) + zl * FT_LB + li];
                }
}

void inverse_z_and_transpose() {
    double t = MPI_Wtime();
    fftz_send(-1, u1, u2, sbuf, rbuf, rank, nranks, run / FT_LB, nzp, run,
              (long)nx * rank * nyp, rank * nzp, (long)nx * ny);
    const double t1 = MPI_Wtime();
    t_fftz += t1 - t;
#ifndef FT_MPI
    ompx_quiet();
    t_quiet += MPI_Wtime() - t1;
    ompx_barrier();
#else
    const long slot = run * nzp, cnt = 2 * slot;
    std::vector<MPI_Request> req(2 * (nranks - 1));
    for (int q = 0; q < nranks - 1; ++q) {
        MPI_Irecv(rbuf + q * slot, (int)cnt, MPI_DOUBLE, (rank - 1 - q + nranks) % nranks, 7,
                  MPI_COMM_WORLD, &req[2 * q]);
        MPI_Isend(sbuf + q * slot, (int)cnt, MPI_DOUBLE, (rank + 1 + q) % nranks, 7,
                  MPI_COMM_WORLD, &req[2 * q + 1]);
    }
    MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);
#endif
    unpack_stream();
    t_xchg += MPI_Wtime() - t1;
}
#endif

// Copy the runs between layout A (run p of local plane zl starts at
// nx * (p * ny/P + ny * zl)) and a block layout (p * nzp * run + zl * run),
// for the peers p in [p0, p1). to_block: A -> block.
void copy_runs(dcomplex* Aa, dcomplex* Bb, long boff, int p0, int p1, bool to_block) {
    const long nz_l = nzp, r_ = run, ny_l = ny, nyp_l = nyp, n1 = nx;
    if (to_block) {
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(Aa, Bb)
        for (long p = p0; p < p1; ++p)
            for (long zl = 0; zl < nz_l; ++zl)
                for (long e = 0; e < r_; ++e)
                    Bb[boff + (p * nz_l + zl) * r_ + e] = Aa[n1 * (p * nyp_l + ny_l * zl) + e];
    } else {
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(Aa, Bb)
        for (long p = p0; p < p1; ++p)
            for (long zl = 0; zl < nz_l; ++zl)
                for (long e = 0; e < r_; ++e)
                    Aa[n1 * (p * nyp_l + ny_l * zl) + e] = Bb[boff + (p * nz_l + zl) * r_ + e];
    }
}

// Layout A -> layout B (the forward transpose, src A, dst B) or B -> A.
// Rank r's run p of A plane zl is peer p's B plane r * nzp + zl.
void transpose(bool a_to_b, dcomplex* src, dcomplex* dst) {
    const double t0 = MPI_Wtime();
    // My own block: A runs r <-> B planes r * nzp + zl.
    if (a_to_b) copy_runs(src, dst, 0, rank, rank + 1, true);
    else copy_runs(dst, src, 0, rank, rank + 1, false);
#ifndef FT_MPI
    // The forward transpose writes B, which a peer may still be reading in
    // its last z FFT; the inverse's target is freed by checksum's Allreduce.
    if (a_to_b) ompx_barrier();
    const size_t bytes = run * sizeof(dcomplex);
    for (int q = 1; q < nranks; ++q) {
        const int p = (rank + q) % nranks;
        for (long zl = 0; zl < nzp; ++zl) {
            const long a_off = nx * (p * nyp + ny * zl);    // my A run for p
            const long b_off = (rank * nzp + zl) * run;     // its B plane
            const long a_dst = nx * (rank * nyp + ny * zl);  // p's A run for me
            const long b_src = (p * nzp + zl) * run;         // my B plane for p
            if (a_to_b) ompx_put(p, dst + b_off, src + a_off, bytes);
            else ompx_put(p, dst + a_dst, src + b_src, bytes);
        }
    }
    const double t1 = MPI_Wtime();
    ompx_quiet();
    t_quiet += MPI_Wtime() - t1;
    ompx_barrier();
#elif FT_MPI == 1
    std::vector<MPI_Request> req;
    req.reserve(2 * nzp * nranks);
    const int cnt = (int)(2 * run);
    for (int q = 1; q < nranks; ++q) {
        const int p = (rank + q) % nranks;
        for (long zl = 0; zl < nzp; ++zl) {
            MPI_Request r;
            if (a_to_b) {
                MPI_Irecv(dst + (p * nzp + zl) * run, cnt, MPI_DOUBLE, p, (int)zl, MPI_COMM_WORLD, &r);
                req.push_back(r);
                MPI_Isend(src + nx * (p * nyp + ny * zl), cnt, MPI_DOUBLE, p, (int)zl, MPI_COMM_WORLD, &r);
            } else {
                MPI_Irecv(dst + nx * (p * nyp + ny * zl), cnt, MPI_DOUBLE, p, (int)zl, MPI_COMM_WORLD, &r);
                req.push_back(r);
                MPI_Isend(src + (p * nzp + zl) * run, cnt, MPI_DOUBLE, p, (int)zl, MPI_COMM_WORLD, &r);
            }
            req.push_back(r);
        }
    }
    MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);
#else
    // NPB-MPI's transpose: pack, Alltoall of nzp * run per peer, unpack.
    const int cnt = (int)(2 * nzp * run);
    if (a_to_b) {
        copy_runs(src, abuf, 0, 0, nranks, true);
        MPI_Alltoall(abuf, cnt, MPI_DOUBLE, dst, cnt, MPI_DOUBLE, MPI_COMM_WORLD);
    } else {
        MPI_Alltoall(src, cnt, MPI_DOUBLE, abuf, cnt, MPI_DOUBLE, MPI_COMM_WORLD);
        copy_runs(dst, abuf, 0, 0, nranks, false);
    }
#endif
    t_xchg += MPI_Wtime() - t0;
}

// Twiddle factors exp(-4 alpha pi^2 (i'^2 + j'^2 + k'^2)), layout B.
void compute_indexmap() {
    const double ap = -4.0 * kAlpha * kPi * kPi;
    const long n1 = nx, nyl = nyp, y0 = rank * nyp, gx = nx, gy = ny, gz = nz;
    double* T = twiddle;
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(T)
    for (long z = 0; z < gz; ++z)
        for (long yl = 0; yl < nyl; ++yl)
            for (long x = 0; x < n1; ++x) {
                const long kk = (z + gz / 2) % gz - gz / 2;
                const long jj = (yl + y0 + gy / 2) % gy - gy / 2;
                const long ii = (x + gx / 2) % gx - gx / 2;
                T[x + n1 * (yl + nyl * z)] = exp(ap * (double)(ii * ii + jj * jj + kk * kk));
            }
}

// NPB's initial conditions into layout A: the generator runs through the
// grid plane by plane, row by row, 2 nx numbers a row. Every local row's
// starting state is computed on the host, then a thread per row draws it.
void compute_initial_conditions(dcomplex* A) {
    double start = kSeed, an_plane, an_row;
    ipow46(kA, 2 * nx * ny, &an_plane);
    ipow46(kA, 2 * nx, &an_row);
    for (long k = 0; k < rank * nzp; ++k) randlc(&start, an_plane);
    std::vector<double> seeds(ny * nzp);
    for (long zl = 0; zl < nzp; ++zl) {
        double x0 = start;
        for (long j = 0; j < ny; ++j) {
            seeds[j + ny * zl] = x0;
            randlc(&x0, an_row);
        }
        randlc(&start, an_plane);
    }
    omp_target_memcpy(row_seed, seeds.data(), seeds.size() * sizeof(double), 0, 0, dev,
                      omp_get_initial_device());
    const long nrows = ny * nzp, n2 = 2L * nx;
    const double* S = row_seed;
    double* D = reinterpret_cast<double*>(A);
    #pragma omp target teams distribute parallel for is_device_ptr(S, D)
    for (long row = 0; row < nrows; ++row) {
        double x = S[row];
        for (long i = 0; i < n2; ++i) D[row * n2 + i] = randlc(&x, kA);
    }
}

void fft_init(int n) {
    std::vector<dcomplex> h(n);
    const int m = ilog2(n);
    h[0] = {(double)m, 0.0};
    int ku = 2, ln = 1;
    for (int j = 1; j <= m; ++j) {
        const double t = kPi / ln;
        for (int i = 0; i <= ln - 1; ++i) h[i + ku - 1] = {cos(i * t), sin(i * t)};
        ku += ln;
        ln *= 2;
    }
    omp_target_memcpy(roots, h.data(), n * sizeof(dcomplex), 0, 0, dev, omp_get_initial_device());
}

void evolve() {
    dcomplex *U0 = u0, *U1 = u1;
    const double* T = twiddle;
    const long n = ntdivnp;
    #pragma omp target teams distribute parallel for is_device_ptr(U0, U1, T)
    for (long i = 0; i < n; ++i) {
        U0[i] = {U0[i].re * T[i], U0[i].im * T[i]};
        U1[i] = U0[i];
    }
}

// Forward: x and y FFTs in A (u1), transpose into B (u0), z FFT there.
void fft_forward() {
    fftx(1, u1);
    ffty(1, u1);
    transpose(true, u1, u0);
    fftz(1, u0);
}

// Inverse, as NPB-MPI's layout_1D: z FFT in B (u1), transpose into A (u2),
// then the y and x FFTs there.
void fft_inverse() {
#if FT_STREAM
    inverse_z_and_transpose();
#else
    const double tz = MPI_Wtime();
    fftz(-1, u1);
    t_fftz += MPI_Wtime() - tz;
    transpose(false, u1, u2);
#endif
    double t = MPI_Wtime();
    ffty(-1, u2);
    const double t1 = MPI_Wtime();
    fftx(-1, u2);
    t_ffty += t1 - t;
    t_fftx += MPI_Wtime() - t1;
}

dcomplex checksum(int it, const dcomplex* A) {
    const long z0 = rank * nzp, z1 = z0 + nzp, gx = nx, gy = ny, gz = nz;
    double cr = 0, ci = 0;
    #pragma omp target teams distribute parallel for reduction(+ : cr, ci) is_device_ptr(A)
    for (long j = 1; j <= 1024; ++j) {
        const long q = j % gx, r = (3 * j) % gy, s = (5 * j) % gz;
        if (s >= z0 && s < z1) {
            const dcomplex v = A[q + gx * (r + gy * (s - z0))];
            cr += v.re;
            ci += v.im;
        }
    }
    double l[2] = {cr, ci}, g[2];
    // An Allreduce, where NPB has a Reduce to rank 0: it also tells a rank
    // that every peer has read u2, so the next transpose may overwrite it.
    MPI_Allreduce(l, g, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double nt = (double)nx * ny * nz;
    dcomplex c = {g[0] / nt, g[1] / nt};
    if (rank == 0 && it > 0) printf(" T =%5d     Checksum =%22.12e%22.12e\n", it, c.re, c.im);
    return c;
}

bool verify(const std::vector<dcomplex>& sums) {
    const double(*ref)[2] = nullptr;
    int nref = 0;
#define REF(c, X, Y, Z, N)                                                       \
    if (nx == X && ny == Y && nz == Z && niter == N) {                           \
        cls = #c[0];                                                             \
        ref = ref_##c;                                                           \
        nref = sizeof(ref_##c) / sizeof(ref_##c[0]);                             \
    }
    REF(S, 64, 64, 64, 6)
    REF(W, 128, 128, 32, 6)
    REF(A, 256, 256, 128, 6)
    REF(B, 512, 256, 256, 20)
    REF(C, 512, 512, 512, 20)
    REF(D, 2048, 1024, 1024, 25)
#undef REF
    if (!ref || nref < niter) return false;
    bool ok = true;
    for (int i = 1; i <= niter; ++i) {
        const double er = (sums[i].re - ref[i - 1][0]) / ref[i - 1][0];
        const double ei = (sums[i].im - ref[i - 1][1]) / ref[i - 1][1];
        if (!(fabs(er) <= 1.0e-12 && fabs(ei) <= 1.0e-12)) ok = false;
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
#ifndef FT_MPI
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
    char c = 'C';
    niter = -1;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-class")) c = argv[i + 1][0];
        else if (!strcmp(argv[i], "-niter")) niter = atoi(argv[i + 1]);
    }
    int nt_default;
    switch (c) {
        case 'S': nx = 64, ny = 64, nz = 64, nt_default = 6; break;
        case 'W': nx = 128, ny = 128, nz = 32, nt_default = 6; break;
        case 'A': nx = 256, ny = 256, nz = 128, nt_default = 6; break;
        case 'B': nx = 512, ny = 256, nz = 256, nt_default = 20; break;
        case 'C': nx = 512, ny = 512, nz = 512, nt_default = 20; break;
        case 'D': nx = 2048, ny = 1024, nz = 1024, nt_default = 25; break;
        default:
            if (rank == 0) fprintf(stderr, "unknown class %c\n", c);
            MPI_Abort(MPI_COMM_WORLD, 2);
            return 2;
    }
    if (niter < 0) niter = nt_default;
    if (ny % nranks || nz % nranks) {
        if (rank == 0) fprintf(stderr, "ny and nz must be multiples of the %d ranks\n", nranks);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
    if (FT_STREAM && (nranks < 2 || ((long)nx * (ny / nranks)) % FT_LB || nz > FT_ZMAX)) {
        if (rank == 0)
            fprintf(stderr, "FT_STREAM needs two ranks or more, nx * ny/P a multiple of %d and nz <= %d\n",
                    FT_LB, FT_ZMAX);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
    nzp = nz / nranks;
    nyp = ny / nranks;
    run = (long)nx * nyp;
    ntdivnp = (long)nx * ny * nzp;

    u0 = heap_alloc(ntdivnp);
    u1 = heap_alloc(ntdivnp);
    u2 = heap_alloc(ntdivnp);
    w0 = dev_alloc(ntdivnp);
    w1 = dev_alloc(ntdivnp);
#if FT_STREAM
    sbuf = heap_alloc(ntdivnp / nranks * (nranks - 1));
    rbuf = heap_alloc(ntdivnp / nranks * (nranks - 1));
#endif
#if defined(FT_MPI) && FT_MPI == 2
    abuf = dev_alloc(ntdivnp);
#endif
    const int maxdim = nx > ny ? (nx > nz ? nx : nz) : (ny > nz ? ny : nz);
    roots = dev_alloc(maxdim);
    twiddle = static_cast<double*>(omp_target_alloc(ntdivnp * sizeof(double), dev));
    row_seed = static_cast<double*>(omp_target_alloc(ny * nzp * sizeof(double), dev));
#ifndef FT_MPI
    ompx_prepare();
#endif
    if (rank == 0) {
        printf("\n\n NAS Parallel Benchmarks FT, OpenMP target%s\n\n",
#ifndef FT_MPI
               " + GiOMP");
#elif FT_MPI == 1
               " + MPI (Isend/Irecv)");
#else
               " + MPI (Alltoall)");
#endif
        printf(" Size                : %4dx%4dx%4d\n", nx, ny, nz);
        printf(" Iterations          :     %7d\n", niter);
        printf(" Number of processes :     %7d\n", nranks);
        printf(" Layout type         :          1D%s\n", FT_STREAM ? ", streamed z FFT" : "");
    }

    // NPB runs the setup and the forward FFT once untimed, to touch all data.
    compute_indexmap();
    compute_initial_conditions(u1);
    fft_init(maxdim);
    fft_forward();
    t_evolve = t_fftz = t_xchg = t_quiet = t_ffty = t_fftx = t_chk = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();
    compute_indexmap();
    compute_initial_conditions(u1);
    fft_init(maxdim);
    fft_forward();
    t_evolve = t_fftz = t_xchg = t_quiet = t_ffty = t_fftx = t_chk = 0;
    const double t_iter = MPI_Wtime();
    std::vector<dcomplex> sums(niter + 1);
    for (int it = 1; it <= niter; ++it) {
        double t = MPI_Wtime();
        evolve();
        t_evolve += MPI_Wtime() - t;
        fft_inverse();
        t = MPI_Wtime();
        sums[it] = checksum(it, u2);
        t_chk += MPI_Wtime() - t;
    }
    const double t_end = MPI_Wtime();
    double loc[2] = {t_end - t_start, t_end - t_iter}, tot[2];
    MPI_Allreduce(loc, tot, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double pl[7] = {t_evolve, t_fftz, t_xchg, t_quiet, t_ffty, t_fftx, t_chk}, pm[7];
    MPI_Reduce(pl, pm, 7, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const bool ok = verify(sums);
        const double ntf = (double)nx * ny * nz;
        const double mflops = 1.0e-6 * ntf *
                              (14.8157 + 7.19641 * log(ntf) + (5.23518 + 7.21113 * log(ntf)) * niter) /
                              tot[0];
        printf(" Result verification %s\n", ok ? "successful" : "failed");
        printf(" class = %c\n\n", cls);
        printf(" FT Benchmark Completed.\n");
        printf(" Class           =                        %c\n", cls);
        printf(" Size            =            %4dx%4dx%4d\n", nx, ny, nz);
        printf(" Iterations      =             %12d\n", niter);
        printf(" Time in seconds =             %12.4f\n", tot[0]);
        printf(" Total processes =             %12d\n", nranks);
        printf(" Mop/s total     =          %15.2f\n", mflops);
        printf(" Verification    =             %12s\n", ok ? "SUCCESSFUL" : "UNSUCCESSFUL");
        printf("iterations (max over ranks): %.4f s; evolve %.4f, fft z %.4f, transpose %.4f "
               "(quiet %.4f), fft y %.4f, fft x %.4f, checksum %.4f\n",
               tot[1], pm[0], pm[1], pm[2], pm[3], pm[4], pm[5], pm[6]);
    }
#ifndef FT_MPI
    ompx_finalize();
#else
    MPI_Finalize();
#endif
    return 0;
}
