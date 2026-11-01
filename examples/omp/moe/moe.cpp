// moe.cpp - one MoE layer with expert parallelism: route, dispatch every
// token to the ranks owning its top-k experts, run the experts, return the
// results and combine them. One source, five ways to move the tokens:
//
//   -DMOE_MPI=1    GPU-aware MPI, bulk: the per-expert counts go to the host
//                  and through MPI_Alltoall, the rows through MPI_Alltoallv,
//                  the experts run once on everything, MPI_Alltoallv back.
//   -DMOE_MPI=2    GPU-aware MPI, overlapped: counts as above, then
//                  Irecv/Isend per peer; the local segment runs first and
//                  each remote segment as soon as it lands (MPI_Waitany),
//                  and its results go back right away.
//   -DMOE_MPI=3    GPU-aware MPI, overlapped as well as MPI allows: a host
//                  thread launches the kernels while the main thread keeps
//                  MPI progressing, and the second expert kernel runs in
//                  MOE_CHUNKS pieces (default 4), each Isent back when done.
//   -DMOE_GIOMP=0  GiOMP, bulk: after the dispatch kernel one target region
//                  puts every segment and its counts (ompx_put_signal); the
//                  experts run once when every signal is in.
//   -DMOE_GIOMP=1  GiOMP, dispatch pipelined in the kernel: each team puts
//                  its rows the moment it has written them, and the last
//                  team to finish sends the counts. Experts as in 0.
//   -DMOE_GIOMP=2  As 1, and the receive side overlaps as in MPI 2: local
//                  segment first, then each source as its signal arrives.
//   -DMOE_GIOMP=4  The code a user would write for the compiler: as 3, but
//                  the return is an ordinary host ompx_put right after the
//                  second expert kernel; the pass may move it into the kernel.
//   -DMOE_GIOMP=3  Bulk dispatch as in 0, receive overlapped as in 2, and
//                  the return pipelined in the second expert kernel: the
//                  last team to finish a row tile puts its rows back, and
//                  the last tile of a segment sends the signal.
//
// Every variant runs the same router, count, scan, dispatch, expert and
// combine kernels; only the transfers differ. Receive buffers are laid out
// by source rank (capacity `cap` rows each) and sorted by expert inside a
// segment, so no rank needs a remote atomic and no size is needed before a
// transfer can land. No token is dropped (cap is twice the mean), and each
// output row depends only on its token, so all variants print the same
// checksum.
#if defined(MOE_MPI) == defined(MOE_GIOMP)
#error "define exactly one of MOE_MPI and MOE_GIOMP"
#endif
#if defined(MOE_GIOMP)
#include "gicc/omp.h"
#if MOE_GIOMP == 4
#include "gicc/omp_pipeline.h"
#endif
#endif
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>
#if defined(MOE_MPI) && MOE_MPI == 3
#include <atomic>
#include <mutex>
#include <thread>
#endif

namespace {

struct Options {
    int tokens = 8192;   // per rank
    int hidden = 2048;
    int ffn = 512;
    int el = 8;          // experts per rank
    int topk = 2;
    int tb = 64;         // tokens per dispatch team
    int iters = 20;
    int warmup = 3;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        int* f = nullptr;
        const char* a = argv[i];
        const char* eq = std::strchr(a, '=');
        if (!eq) { std::fprintf(stderr, "bad argument %s\n", a); std::exit(2); }
        const std::string_view k(a, eq - a);
        if (k == "--tokens") f = &o.tokens;
        else if (k == "--hidden") f = &o.hidden;
        else if (k == "--ffn") f = &o.ffn;
        else if (k == "--experts-per-rank") f = &o.el;
        else if (k == "--topk") f = &o.topk;
        else if (k == "--tb") f = &o.tb;
        else if (k == "--iters") f = &o.iters;
        else if (k == "--warmup") f = &o.warmup;
        if (!f) { std::fprintf(stderr, "unknown argument %s\n", a); std::exit(2); }
        *f = std::atoi(eq + 1);
    }
    return o;
}

struct alignas(16) f4 { float a, b, c, d; };

constexpr int kRT = 16;      // rows per expert tile
constexpr int kCT = 128;     // columns per expert tile (threads)
constexpr int kRetSig = 32;  // return signals use slots 32 + source rank

#pragma omp declare target
inline uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
inline float unif(uint64_t key) {   // [-1, 1)
    return (float)((mix(key) >> 40) * (1.0 / 8388608.0) - 1.0);
}
#pragma omp end declare target

#if defined(MOE_GIOMP) && MOE_GIOMP == 3
#pragma omp declare target
// One column team of a row tile is done. The last one sends the tile's rows;
// the last tile of the segment then sends the signal behind them (same QP).
// Out of line, so the expert kernel's registers do not pay for the put.
__attribute__((noinline)) void return_tile(unsigned* tile_cnt, unsigned* seg_cnt, int ct,
                                           unsigned valid, int peer, float* dst, const float* src,
                                           size_t bytes, float* sig_dst, int sig,
                                           unsigned long long value) {
    if (__atomic_fetch_add(tile_cnt, 1u, __ATOMIC_ACQ_REL) != (unsigned)ct - 1) return;
    *tile_cnt = 0;
    ompx_put(peer, dst, src, bytes);
    if (__atomic_fetch_add(seg_cnt, 1u, __ATOMIC_ACQ_REL) != valid - 1) return;
    *seg_cnt = 0;
    ompx_put_signal(peer, sig_dst, src, 0, sig, value);
}
#pragma omp end declare target
#endif

void* dalloc(size_t bytes, bool sym) {
#if defined(MOE_GIOMP)
    if (sym) return ompx_alloc(bytes);
#endif
    (void)sym;
    void* p = omp_target_alloc(bytes, omp_get_default_device());
    if (!p) { std::fprintf(stderr, "omp_target_alloc(%zu) failed\n", bytes); std::exit(1); }
    return p;
}

void fill(float* p, size_t n, uint64_t seed, float scale) {
    #pragma omp target teams distribute parallel for is_device_ptr(p)
    for (size_t i = 0; i < n; ++i) p[i] = scale * unif(seed * 0x100000000ull + i);
}

struct Layer {
    Options o;
    int me, P, E, K, T, H, F, EL, nteams;
    int64_t cap;
    // Local.
    float *x, *wg, *w1, *w2, *hid, *out;
    float* gate;        // [T*K]
    int *expert, *idx;  // [T*K]: chosen expert, rank inside (team, expert)
    int64_t *cnt, *off; // [nteams*E]
    unsigned* done;     // teams finished, never reset
    unsigned *tcnt, *scnt;  // GiOMP 3: column teams done per row tile, tiles done per segment
    // Symmetric (GiOMP heap) or plain device memory (MPI).
    float *send, *recv, *outr, *ret;   // [P*cap*H]
    int64_t *meta_out, *meta_in;       // [P*EL]: rows per expert, to / from each rank
    int64_t* meta_use;                 // the experts' copy of meta_in: a peer may
                                       // overwrite meta_in for the next step while
                                       // late teams of this step still read it
    // Host time per phase: route+count+scan, dispatch kernel, dispatch
    // transfer, experts, return transfer, combine. Overlapped variants book
    // the overlapped stretch under experts.
    double ph[6] = {0, 0, 0, 0, 0, 0}, last = 0;
    void tick(int i) { const double t = MPI_Wtime(); ph[i] += t - last; last = t; }

    explicit Layer(const Options& opt, int rank, int nranks) : o(opt) {
        me = rank; P = nranks; EL = o.el; E = EL * P; K = o.topk;
        T = o.tokens; H = o.hidden; F = o.ffn;
        nteams = (T + o.tb - 1) / o.tb;
        cap = 2 * ((int64_t)T * K + P - 1) / P;
        const size_t seg = (size_t)P * cap * H * sizeof(float);
        x = (float*)dalloc((size_t)T * H * 4, false);
        wg = (float*)dalloc((size_t)H * E * 4, false);
        w1 = (float*)dalloc((size_t)EL * H * F * 4, false);
        w2 = (float*)dalloc((size_t)EL * F * H * 4, false);
        hid = (float*)dalloc((size_t)P * cap * F * 4, false);
        out = (float*)dalloc((size_t)T * H * 4, false);
        gate = (float*)dalloc((size_t)T * K * 4, false);
        expert = (int*)dalloc((size_t)T * K * 4, false);
        idx = (int*)dalloc((size_t)T * K * 4, false);
        cnt = (int64_t*)dalloc((size_t)nteams * E * 8, false);
        off = (int64_t*)dalloc((size_t)nteams * E * 8, false);
        done = (unsigned*)dalloc(4, false);
        const int ntiles = (int)((cap + kRT - 1) / kRT) + EL;
        tcnt = (unsigned*)dalloc((size_t)P * ntiles * 4, false);
        scnt = (unsigned*)dalloc((size_t)P * 4, false);
        {
            unsigned *tc = tcnt, *sc = scnt;
            const int n1 = P * ntiles, n2 = P;
            #pragma omp target teams distribute parallel for is_device_ptr(tc)
            for (int i = 0; i < n1; ++i) tc[i] = 0;
            #pragma omp target teams distribute parallel for is_device_ptr(sc)
            for (int i = 0; i < n2; ++i) sc[i] = 0;
        }
        send = (float*)dalloc(seg, true);
        recv = (float*)dalloc(seg, true);
        outr = (float*)dalloc(seg, true);
        ret = (float*)dalloc(seg, true);
        meta_out = (int64_t*)dalloc((size_t)P * EL * 8, true);
        meta_in = (int64_t*)dalloc((size_t)P * EL * 8, true);
        meta_use = (int64_t*)dalloc((size_t)P * EL * 8, false);
        fill(x, (size_t)T * H, 1000 + me, 1.0f);
        fill(wg, (size_t)H * E, 7, 1.0f / std::sqrt((float)H));
        fill(w1, (size_t)EL * H * F, 2000 + me, 1.0f / std::sqrt((float)H));
        fill(w2, (size_t)EL * F * H, 3000 + me, 1.0f / std::sqrt((float)F));
        unsigned* d = done;
        #pragma omp target is_device_ptr(d)
        *d = 0;
    }

    // Top-k experts per token, softmax over the chosen scores.
    void route() {
        const float *xx = x, *w = wg;
        float* g = gate;
        int* ex = expert;
        const int T_ = T, H_ = H, E_ = E, K_ = K;
        #pragma omp target teams distribute parallel for is_device_ptr(xx, w, g, ex)
        for (int t = 0; t < T_; ++t) {
            float best[8];
            int bi[8];
            for (int k = 0; k < K_; ++k) { best[k] = -1e30f; bi[k] = -1; }
            for (int e = 0; e < E_; ++e) {
                float s = 0.f;
                for (int h = 0; h < H_; ++h) s += xx[(size_t)t * H_ + h] * w[(size_t)h * E_ + e];
                if (s > best[K_ - 1]) {
                    int k = K_ - 1;
                    while (k > 0 && s > best[k - 1]) { best[k] = best[k - 1]; bi[k] = bi[k - 1]; --k; }
                    best[k] = s; bi[k] = e;
                }
            }
            float z = 0.f;
            for (int k = 0; k < K_; ++k) z += std::exp(best[k] - best[0]);
            for (int k = 0; k < K_; ++k) {
                g[t * K_ + k] = std::exp(best[k] - best[0]) / z;
                ex[t * K_ + k] = bi[k];
            }
        }
    }

    // Per dispatch team and expert: how many rows, and each row's rank in
    // that group (token order, so deterministic).
    void count() {
        const int* ex = expert;
        int* id = idx;
        int64_t* c = cnt;
        const int K_ = K, T_ = T, E_ = E, tb = o.tb;
        #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(ex, id, c)
        for (int b = 0; b < nteams; ++b)
            for (int e = 0; e < E_; ++e) {
                int64_t n = 0;
                const int a1 = std::min((b + 1) * tb, T_) * K_;
                for (int a = b * tb * K_; a < a1; ++a)
                    if (ex[a] == e) id[a] = (int)n++;
                c[(size_t)b * E_ + e] = n;
            }
    }

    // Offsets: a destination's segment holds its experts in order, each
    // expert's rows in dispatch-team order. meta_out[d] = rows per expert
    // of d, and this rank's own share is meta_in[me].
    void scan() {
        const int64_t* c = cnt;
        int64_t *of = off, *mo = meta_out, *mi = meta_in;
        const int E_ = E, EL_ = EL, nt = nteams, me_ = me;
        const int64_t cap_ = cap;
        #pragma omp target teams num_teams(1) thread_limit(1024) is_device_ptr(c, of, mo, mi)
        {
            #pragma omp parallel
            {
                const int nth = omp_get_num_threads(), tid = omp_get_thread_num();
                for (int e = tid; e < E_; e += nth) {
                    int64_t s = 0;
                    for (int b = 0; b < nt; ++b) s += c[(size_t)b * E_ + e];
                    mo[e] = s;
                }
                #pragma omp barrier
                for (int e = tid; e < E_; e += nth) {
                    const int d = e / EL_;
                    int64_t base = 0;
                    for (int q = d * EL_; q < e; ++q) base += mo[q];
                    if (base + mo[e] > cap_) __builtin_trap();   // cap is 2x the mean
                    for (int b = 0; b < nt; ++b) {
                        of[(size_t)b * E_ + e] = base;
                        base += c[(size_t)b * E_ + e];
                    }
                    if (d == me_) mi[me_ * EL_ + e % EL_] = mo[e];
                }
            }
        }
    }

    // Copy each token to its experts' segments: the local one straight into
    // recv, the others into send. With pipelined dispatch each team puts
    // what it wrote as soon as it is done.
    void dispatch(unsigned long long step) {
        const float* xx = x;
        float *sb = send, *rb = recv;
        const int* ex = expert;
        const int* id = idx;
        const int64_t *c = cnt, *of = off;
        int64_t *mo = meta_out, *mi = meta_in;
        unsigned* dn = done;
        const int K_ = K, T_ = T, H_ = H, E_ = E, EL_ = EL, me_ = me, P_ = P, tb = o.tb,
                  nt = nteams;
        const int64_t cap_ = cap;
        (void)mo; (void)mi; (void)dn; (void)c; (void)P_; (void)nt; (void)step;
        #pragma omp target teams num_teams(nt) thread_limit(256) \
            is_device_ptr(xx, sb, rb, ex, id, c, of, mo, mi, dn)
        {
            const int b = omp_get_team_num();
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
                const int a1 = std::min((b + 1) * tb, T_) * K_;
                for (int a = b * tb * K_; a < a1; ++a) {
                    const int e = ex[a], d = e / EL_;
                    const int64_t slot = of[(size_t)b * E_ + e] + id[a];
                    float* dst = (d == me_ ? rb : sb) + ((size_t)d * cap_ + slot) * H_;
                    const f4* s4 = (const f4*)(xx + (size_t)(a / K_) * H_);
                    f4* d4 = (f4*)dst;
                    for (int h = tid; h < H_ / 4; h += nth) d4[h] = s4[h];
                }
#if defined(MOE_GIOMP) && (MOE_GIOMP == 1 || MOE_GIOMP == 2)
                #pragma omp barrier
                if (tid == 0) {
                    for (int e = 0; e < E_; ++e) {
                        const int d = e / EL_;
                        const int64_t n = c[(size_t)b * E_ + e];
                        if (d == me_ || n == 0) continue;
                        const size_t r0 = (size_t)of[(size_t)b * E_ + e];
                        ompx_put(d, rb + ((size_t)me_ * cap_ + r0) * H_,
                                 sb + ((size_t)d * cap_ + r0) * H_, (size_t)n * H_ * 4);
                    }
                    const unsigned prev = __atomic_fetch_add(dn, 1u, __ATOMIC_ACQ_REL);
                    if (prev == (unsigned)(step * nt - 1))
                        for (int d = 0; d < P_; ++d)
                            if (d != me_)
                                ompx_put_signal(d, mi + me_ * EL_, mo + d * EL_, EL_ * 8, me_, step);
                }
#endif
            }
        }
    }

    // Both expert layers for segment s (or all segments if s < 0):
    // out = relu(x W1) W2, tiles of kRT rows of one expert by kCT columns.
    // With rstep != 0 (GiOMP 3) the second kernel also returns the rows.
    int tiles_max() const { return (int)((cap + kRT - 1) / kRT) + EL; }

    void snapshot(int s0, int s1) {
        const int64_t* mi = meta_in;
        int64_t* mu = meta_use;
        const int a = s0 * EL, b = s1 * EL;
        #pragma omp target teams distribute parallel for is_device_ptr(mi, mu)
        for (int i = a; i < b; ++i) mu[i] = mi[i];
    }

    void experts(int s, unsigned long long rstep = 0) {
        const int s0 = s < 0 ? 0 : s, s1 = s < 0 ? P : s + 1;
        snapshot(s0, s1);
        gemm(s0, s1, 0, tiles_max(), recv, w1, hid, H, F, true, 0);
        gemm(s0, s1, 0, tiles_max(), hid, w2, outr, F, H, false, rstep);
    }

    // Tiles [t0, t0 + nt) of segment s, both layers (snapshot first).
    void experts_tiles(int s, int t0, int nt) {
        if (nt <= 0) return;
        gemm(s, s + 1, t0, nt, recv, w1, hid, H, F, true, 0);
        gemm(s, s + 1, t0, nt, hid, w2, outr, F, H, false, 0);
    }

    // Host copy of the kernel's tile map: rows of tile t of a segment whose
    // per-expert counts are n[0..EL); returns the number of valid tiles.
    int tile_rows(const int64_t* n, int t, int64_t* r0, int64_t* nr) const {
        int64_t base = 0, v = 0;
        bool found = false;
        for (int q = 0; q < EL; ++q) {
            const int64_t nt = (n[q] + kRT - 1) / kRT;
            if (!found && t >= v && t < v + nt) {
                *r0 = base + (t - v) * kRT;
                *nr = std::min<int64_t>(kRT, n[q] - (t - v) * kRT);
                found = true;
            }
            v += nt;
            base += n[q];
        }
        return (int)v;
    }

    // Rows [*rb, *re) and tiles [*ta, *tb) of chunk c of nc of a segment.
    void chunk(const int64_t* n, int c, int nc, int* ta, int* tb, int64_t* rb, int64_t* re) const {
        int64_t r0 = 0, nr = 0;
        const int v = tile_rows(n, 0, &r0, &nr);
        *ta = (int)((int64_t)c * v / nc);
        *tb = (int)((int64_t)(c + 1) * v / nc);
        *rb = *re = 0;
        if (*ta == *tb) return;
        tile_rows(n, *ta, &r0, &nr);
        *rb = r0;
        tile_rows(n, *tb - 1, &r0, &nr);
        *re = r0 + nr;
    }

    void gemm(int s0, int s1, int t0, int tiles, const float* in, const float* w, float* o_,
              int din, int dout, bool relu, unsigned long long rstep, int64_t put_rows = -1,
              unsigned long long put_step = 0) {
        const int64_t* mi = meta_use;
        const int EL_ = EL, ct = (dout + kCT - 1) / kCT, me_ = me, tstride = tiles_max();
        const int64_t cap_ = cap;
        const int nteams_ = (s1 - s0) * tiles * ct;
        (void)tstride;
        unsigned *tc = tcnt, *sc = scnt;
        float* rt = ret;
        (void)rstep; (void)tc; (void)sc; (void)rt; (void)me_;
        #pragma omp target teams num_teams(nteams_) thread_limit(kCT) \
            is_device_ptr(in, w, o_, mi, tc, sc, rt)
        {
            const int team = omp_get_team_num();
            const int s = s0 + team / (tiles * ct);
            const int tile = t0 + (team / ct) % tiles, cb = team % ct;
            // Which expert and rows this tile covers.
            int e = -1;
            int64_t row0 = 0, nrows = 0, base = 0, t = tile, valid = 0;
            for (int q = 0; q < EL_; ++q) {
                const int64_t n = mi[s * EL_ + q], nt = (n + kRT - 1) / kRT;
                valid += nt;
                if (e < 0 && t < nt) { e = q; row0 = base + t * kRT; nrows = std::min<int64_t>(kRT, n - t * kRT); }
                if (e < 0) t -= nt;
                base += n;
            }
            #pragma omp parallel num_threads(kCT)
            if (e >= 0) {
                {
                    const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
                    const int j1 = std::min((cb + 1) * kCT, dout);
                    for (int j = cb * kCT + tid; j < j1; j += nth) {
                        float acc[kRT];
                        for (int r = 0; r < kRT; ++r) acc[r] = 0.f;
                        const float* we = w + (size_t)e * din * dout + j;
                        const float* xr = in + ((size_t)s * cap_ + row0) * din;
                        for (int k = 0; k < din; ++k) {
                            const float wk = we[(size_t)k * dout];
                            for (int r = 0; r < kRT; ++r) {
                                const int rr = r < nrows ? r : (int)nrows - 1;
                                acc[r] += xr[(size_t)rr * din + k] * wk;
                            }
                        }
                        float* orow = o_ + ((size_t)s * cap_ + row0) * dout + j;
                        for (int r = 0; r < nrows; ++r)
                            orow[(size_t)r * dout] = relu ? std::max(acc[r], 0.f) : acc[r];
                    }
#if defined(MOE_GIOMP) && MOE_GIOMP == 3
                    // Inside the parallel region, so the kernel stays SPMD.
                    if (rstep && s != me_) {
                        #pragma omp barrier
                        if (tid == 0)
                            return_tile(tc + (size_t)s * tstride + tile, sc + s, ct, (unsigned)valid, s,
                                        rt + ((size_t)me_ * cap_ + row0) * dout,
                                        o_ + ((size_t)s * cap_ + row0) * dout,
                                        (size_t)nrows * dout * 4, rt + (size_t)me_ * cap_ * dout,
                                        kRetSig + me_, rstep);
                    }
#endif
                }
            }
#if defined(MOE_GIOMP) && MOE_GIOMP == 3
            // A segment with no rows still owes its source a signal.
            else if (rstep && s != me_ && valid == 0 && tile == 0 && cb == 0 &&
                     omp_get_thread_num() == 0)
                ompx_put_signal(s, rt + (size_t)me_ * cap_ * dout, o_, 0, kRetSig + me_, rstep);
#endif
        }
#if defined(MOE_GIOMP) && MOE_GIOMP == 4
        // GiOMP 4: return segment s0's rows to its source, as plain code.
        if (put_rows >= 0)
            ompx_put_signal(s0, ret + (size_t)me_ * cap_ * dout, o_ + (size_t)s0 * cap_ * dout,
                            (size_t)put_rows * dout * 4, kRetSig + me_, put_step);
#endif
        (void)put_rows; (void)put_step;
    }

    // out[t] = sum_k gate * expert output; the local expert's rows are read
    // in place, the others from ret (laid out by expert rank).
    void combine() {
        const float *g = gate, *orr = outr, *rt = ret;
        const int* ex = expert;
        const int* id = idx;
        const int64_t* of = off;
        float* ot = out;
        const int K_ = K, T_ = T, H_ = H, E_ = E, EL_ = EL, me_ = me, tb = o.tb;
        const int64_t cap_ = cap;
        #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(g, orr, rt, ex, id, of, ot)
        for (int t = 0; t < T_; ++t)
            for (int h = 0; h < H_; ++h) {
                float s = 0.f;
                for (int k = 0; k < K_; ++k) {
                    const int a = t * K_ + k, e = ex[a], d = e / EL_;
                    const int64_t slot = of[(size_t)(t / tb) * E_ + e] + id[a];
                    const float* src = d == me_ ? orr + ((size_t)me_ * cap_ + slot) * H_
                                                : rt + ((size_t)d * cap_ + slot) * H_;
                    s += g[a] * src[h];
                }
                ot[(size_t)t * H_ + h] = s;
            }
    }

    int64_t rows_from(const std::vector<int64_t>& m, int s) const {
        int64_t n = 0;
        for (int q = 0; q < EL; ++q) n += m[(size_t)s * EL + q];
        return n;
    }

#if defined(MOE_MPI)
    void step(unsigned long long) {
        last = MPI_Wtime();
        route(); count(); scan(); tick(0); dispatch(0); tick(1);
        const int dev = omp_get_default_device(), host = omp_get_initial_device();
        std::vector<int64_t> mo((size_t)P * EL), mi((size_t)P * EL);
        omp_target_memcpy(mo.data(), meta_out, mo.size() * 8, 0, 0, host, dev);
        MPI_Alltoall(mo.data(), EL, MPI_INT64_T, mi.data(), EL, MPI_INT64_T, MPI_COMM_WORLD);
        for (int q = 0; q < EL; ++q) mi[(size_t)me * EL + q] = mo[(size_t)me * EL + q];
        omp_target_memcpy(meta_in, mi.data(), mi.size() * 8, 0, 0, dev, host);
        MPI_Datatype row;
        MPI_Type_contiguous(H, MPI_FLOAT, &row);
        MPI_Type_commit(&row);
#if MOE_MPI == 1
        std::vector<int> sc(P), rc(P), sd(P), rd(P);
        for (int r = 0; r < P; ++r) {
            sc[r] = r == me ? 0 : (int)rows_from(mo, r);
            rc[r] = r == me ? 0 : (int)rows_from(mi, r);
            sd[r] = rd[r] = (int)(r * cap);
        }
        MPI_Alltoallv(send, sc.data(), sd.data(), row, recv, rc.data(), rd.data(), row, MPI_COMM_WORLD);
        tick(2);
        experts(-1);
        tick(3);
        MPI_Alltoallv(outr, rc.data(), rd.data(), row, ret, sc.data(), sd.data(), row, MPI_COMM_WORLD);
        tick(4);
#elif MOE_MPI == 2
        std::vector<MPI_Request> rq(P, MPI_REQUEST_NULL), sq, bq;
        for (int r = 0; r < P; ++r)
            if (r != me)
                MPI_Irecv(recv + (size_t)r * cap * H, (int)rows_from(mi, r), row, r, 0, MPI_COMM_WORLD, &rq[r]);
        for (int r = 0; r < P; ++r)
            if (r != me) {
                sq.emplace_back();
                MPI_Isend(send + (size_t)r * cap * H, (int)rows_from(mo, r), row, r, 0, MPI_COMM_WORLD, &sq.back());
                bq.emplace_back();
                MPI_Irecv(ret + (size_t)r * cap * H, (int)rows_from(mo, r), row, r, 1, MPI_COMM_WORLD, &bq.back());
            }
        experts(me);
        for (int left = P - 1; left > 0; --left) {
            int s;
            MPI_Waitany(P, rq.data(), &s, MPI_STATUS_IGNORE);
            experts(s);
            sq.emplace_back();
            MPI_Isend(outr + (size_t)s * cap * H, (int)rows_from(mi, s), row, s, 1, MPI_COMM_WORLD, &sq.back());
        }
        MPI_Waitall((int)bq.size(), bq.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall((int)sq.size(), sq.data(), MPI_STATUSES_IGNORE);
        tick(3);
#endif
#if MOE_MPI == 3
        const char* ce = std::getenv("MOE_CHUNKS");
        const int nc = ce ? std::max(1, std::atoi(ce)) : 4;
        std::vector<MPI_Request> rq(P, MPI_REQUEST_NULL), sq, bq;
        for (int r = 0; r < P; ++r)
            if (r != me)
                MPI_Irecv(recv + (size_t)r * cap * H, (int)rows_from(mi, r), row, r, 0, MPI_COMM_WORLD, &rq[r]);
        for (int r = 0; r < P; ++r) {
            if (r == me) continue;
            sq.emplace_back();
            MPI_Isend(send + (size_t)r * cap * H, (int)rows_from(mo, r), row, r, 0, MPI_COMM_WORLD, &sq.back());
            for (int c = 0; c < nc; ++c) {
                int ta, tb;
                int64_t rb, re;
                chunk(&mo[(size_t)r * EL], c, nc, &ta, &tb, &rb, &re);
                bq.emplace_back();
                MPI_Irecv(ret + ((size_t)r * cap + rb) * H, (int)(re - rb), row, r, 10 + c, MPI_COMM_WORLD, &bq.back());
            }
        }
        // The worker thread only launches kernels; this thread only calls MPI.
        std::vector<std::atomic<int>> ready(P);
        for (auto& a : ready) a.store(0);
        std::mutex mu;
        std::vector<std::pair<int, int>> finished;   // (segment, chunk)
        std::atomic<int> worker_done{0};
        std::thread worker([&] {
            experts(me);
            for (int left = P - 1; left > 0;)
                for (int s = 0; s < P; ++s) {
                    if (s == me || ready[s].load(std::memory_order_acquire) != 1) continue;
                    ready[s].store(2);
                    --left;
                    snapshot(s, s + 1);
                    for (int c = 0; c < nc; ++c) {
                        int ta, tb;
                        int64_t rb, re;
                        chunk(&mi[(size_t)s * EL], c, nc, &ta, &tb, &rb, &re);
                        experts_tiles(s, ta, tb - ta);
                        std::lock_guard<std::mutex> g(mu);
                        finished.emplace_back(s, c);
                    }
                }
            worker_done.store(1, std::memory_order_release);
        });
        for (int got = 0;;) {
            if (got < P - 1) {
                int s, flag;
                MPI_Testany(P, rq.data(), &s, &flag, MPI_STATUS_IGNORE);
                if (flag && s != MPI_UNDEFINED) { ready[s].store(1, std::memory_order_release); ++got; }
            }
            std::vector<std::pair<int, int>> todo;
            {
                std::lock_guard<std::mutex> g(mu);
                todo.swap(finished);
            }
            for (auto [s, c] : todo) {
                int ta, tb;
                int64_t rb, re;
                chunk(&mi[(size_t)s * EL], c, nc, &ta, &tb, &rb, &re);
                sq.emplace_back();
                MPI_Isend(outr + ((size_t)s * cap + rb) * H, (int)(re - rb), row, s, 10 + c, MPI_COMM_WORLD, &sq.back());
            }
            if (worker_done.load(std::memory_order_acquire) && todo.empty()) {
                std::lock_guard<std::mutex> g(mu);
                if (finished.empty()) break;
            }
            int flag;
            MPI_Testall((int)bq.size(), bq.data(), &flag, MPI_STATUSES_IGNORE);
        }
        worker.join();
        MPI_Waitall((int)bq.size(), bq.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall((int)sq.size(), sq.data(), MPI_STATUSES_IGNORE);
        tick(3);
#endif
        MPI_Type_free(&row);
        combine();
        tick(5);
    }
#else
    // Return segment s's results to rank s (all remote segments if s < 0).
    void give_back(int s, unsigned long long step) {
        const int s0 = s < 0 ? 0 : s, s1 = s < 0 ? P : s + 1;
        const int64_t* mi = meta_in;
        float *orr = outr, *rt = ret;
        const int EL_ = EL, me_ = me, H_ = H;
        const int64_t cap_ = cap;
        #pragma omp target is_device_ptr(mi, orr, rt)
        for (int r = s0; r < s1; ++r) {
            if (r == me_) continue;
            int64_t n = 0;
            for (int q = 0; q < EL_; ++q) n += mi[r * EL_ + q];
            ompx_put_signal(r, rt + (size_t)me_ * cap_ * H_, orr + (size_t)r * cap_ * H_,
                            (size_t)n * H_ * 4, kRetSig + me_, step);
        }
    }

    void step(unsigned long long st) {
        last = MPI_Wtime();
        route(); count(); scan(); tick(0); dispatch(st); tick(1);
#if MOE_GIOMP == 0 || MOE_GIOMP >= 3
        {
            const int64_t* mo = meta_out;
            int64_t* mi = meta_in;
            float *sb = send, *rb = recv;
            const int EL_ = EL, me_ = me, H_ = H, P_ = P;
            const int64_t cap_ = cap;
            #pragma omp target is_device_ptr(mo, mi, sb, rb)
            for (int d = 0; d < P_; ++d) {
                if (d == me_) continue;
                int64_t n = 0;
                for (int q = 0; q < EL_; ++q) n += mo[d * EL_ + q];
                ompx_put(d, rb + (size_t)me_ * cap_ * H_, sb + (size_t)d * cap_ * H_, (size_t)n * H_ * 4);
                ompx_put_signal(d, mi + me_ * EL_, (void*)(mo + d * EL_), EL_ * 8, me_, st);
            }
        }
#endif
#if MOE_GIOMP <= 1
        for (int s = 0; s < P; ++s)
            if (s != me) ompx_signal_wait(s, st);
        tick(2);
        experts(-1);
        tick(3);
        give_back(-1, st);
#else
        if (std::getenv("MOE_NOOVERLAP"))
            for (int s = 0; s < P; ++s)
                if (s != me) ompx_signal_wait(s, st);
        experts(me);
        tick(2);   // overlapped variants: "send" = local experts
        std::vector<char> left(P, 1);
        left[me] = 0;
        for (int n = P - 1; n > 0;) {
            for (int s = 0; s < P; ++s)
                if (left[s] && ompx_signal_read(s) >= st) {
#if MOE_GIOMP == 3
                    experts(s, st);
#elif MOE_GIOMP == 4
                    {
                        int64_t n[64], rows = 0;
                        omp_target_memcpy(n, meta_in + (size_t)s * EL, EL * 8, 0, 0,
                                          omp_get_initial_device(), omp_get_default_device());
                        for (int q = 0; q < EL; ++q) rows += n[q];
                        snapshot(s, s + 1);
                        gemm(s, s + 1, 0, tiles_max(), recv, w1, hid, H, F, true, 0);
                        gemm(s, s + 1, 0, tiles_max(), hid, w2, outr, F, H, false, 0, rows, st);
                    }
#else
                    experts(s);
                    give_back(s, st);
#endif
                    left[s] = 0;
                    --n;
                }
        }
        tick(3);
#endif
        for (int s = 0; s < P; ++s)
            if (s != me) ompx_signal_wait(kRetSig + s, st);
        tick(4);
        combine();
        tick(5);
    }
#endif

    double ret_sum() const {
        const float* rt = ret;
        const size_t n = (size_t)P * cap * H;
        double s = 0;
        #pragma omp target teams distribute parallel for reduction(+ : s) is_device_ptr(rt)
        for (size_t i = 0; i < n; ++i) s += (double)rt[i] * (double)((i % 5) + 1);
        return s;
    }

    // Sum of segment s of buf, rows [0, n).
    double seg_sum(const float* buf, int s, int64_t n) const {
        const size_t a = (size_t)s * cap * H, m = (size_t)n * H;
        double r = 0;
        #pragma omp target teams distribute parallel for reduction(+ : r) is_device_ptr(buf)
        for (size_t i = 0; i < m; ++i) r += (double)buf[a + i] * (double)((i % 5) + 1);
        return r;
    }

    double checksum() {
        const float* ot = out;
        const size_t n = (size_t)T * H;
        double s = 0;
        #pragma omp target teams distribute parallel for reduction(+ : s) is_device_ptr(ot)
        for (size_t i = 0; i < n; ++i) s += (double)ot[i] * (double)((i % 7) + 1);
        return s;
    }
};

}  // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
#if defined(MOE_GIOMP)
    ompx_init();
    const int me = ompx_get_rank_num(), P = ompx_get_num_ranks();
#else
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int me, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
#endif
    if (P > 32) { if (me == 0) std::fprintf(stderr, "at most 32 ranks\n"); std::exit(2); }
    {
        Layer L(o, me, P);
#if defined(MOE_GIOMP)
        ompx_prepare();
#endif
        unsigned long long st = 0;
        for (int i = 0; i < o.warmup; ++i) L.step(++st);
        MPI_Barrier(MPI_COMM_WORLD);
        std::fill(L.ph, L.ph + 6, 0.0);
        std::vector<double> ts;
        for (int i = 0; i < o.iters; ++i) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();
            L.step(++st);
            double dt = MPI_Wtime() - t0, mx;
            MPI_Allreduce(&dt, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            ts.push_back(mx);
        }
        std::sort(ts.begin(), ts.end());
        if (std::getenv("MOE_DEBUG"))
        {
            std::printf("rank %d ret_sum %.10e\n", me, L.ret_sum());
            std::vector<int64_t> mo((size_t)P * L.EL), mi((size_t)P * L.EL);
            omp_target_memcpy(mo.data(), L.meta_out, mo.size() * 8, 0, 0, omp_get_initial_device(), omp_get_default_device());
            omp_target_memcpy(mi.data(), L.meta_in, mi.size() * 8, 0, 0, omp_get_initial_device(), omp_get_default_device());
            for (int q = 0; q < P; ++q)
                std::printf("rank %d seg %d: outr(rows %lld) %.10e  ret(rows %lld) %.10e  recv %.10e\n", me, q,
                            (long long)L.rows_from(mi, q), L.seg_sum(L.outr, q, L.rows_from(mi, q)),
                            (long long)L.rows_from(mo, q), L.seg_sum(L.ret, q, L.rows_from(mo, q)),
                            L.seg_sum(L.recv, q, L.rows_from(mi, q)));
            const int nt = (int)((L.cap + kRT - 1) / kRT) + L.EL;
            std::vector<unsigned> tc((size_t)P * nt), sc(P);
            omp_target_memcpy(tc.data(), L.tcnt, tc.size() * 4, 0, 0, omp_get_initial_device(), omp_get_default_device());
            omp_target_memcpy(sc.data(), L.scnt, sc.size() * 4, 0, 0, omp_get_initial_device(), omp_get_default_device());
            unsigned long long a = 0, nz = 0;
            for (unsigned v : tc) { a += v; nz += v != 0; }
            std::printf("rank %d tcnt sum %llu nonzero %llu scnt %u %u\n", me, a, nz, sc[0], P > 1 ? sc[1] : 0);
        }
        double cs = L.checksum(), tot;
        MPI_Reduce(&cs, &tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (me == 0)
            std::printf("moe P=%d tokens=%d hidden=%d ffn=%d experts=%d topk=%d tb=%d: "
                        "median %.3f ms min %.3f ms checksum %.10e\n",
                        P, o.tokens, o.hidden, o.ffn, o.el * P, o.topk, o.tb,
                        ts[ts.size() / 2] * 1e3, ts[0] * 1e3, tot);
        if (me == 0)
            std::printf("phases ms/step (rank 0): route %.3f dispatch %.3f send %.3f experts %.3f "
                        "return %.3f combine %.3f\n", L.ph[0] * 1e3 / o.iters, L.ph[1] * 1e3 / o.iters,
                        L.ph[2] * 1e3 / o.iters, L.ph[3] * 1e3 / o.iters, L.ph[4] * 1e3 / o.iters,
                        L.ph[5] * 1e3 / o.iters);
    }
#if defined(MOE_GIOMP)
    ompx_finalize();
#else
    MPI_Finalize();
#endif
    return 0;
}
