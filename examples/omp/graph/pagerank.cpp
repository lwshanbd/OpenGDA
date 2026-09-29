// pagerank.cpp - pull PageRank on a distributed R-MAT graph (graph_common.hpp),
// one source for two transports:
//
//   -DGRAPH_BACKEND_MPI    GPU-aware MPI. Each iteration posts its receives,
//                          packs what every other rank needs and sends it
//                          with MPI_Isend from device memory, sums the
//                          neighbours it owns while that travels, waits, and
//                          adds the neighbours it received.
//   -DGRAPH_BACKEND_GIOMP  GiOMP. The pack kernel stores the values a same-node
//                          rank needs straight into that rank's ghost slots
//                          (ompx_peer_ptr), and packs the rest and puts it from
//                          the device, one put per chunk as soon as the chunk
//                          is packed. The same sums overlap it; ompx_fence
//                          then completes the step.
//
// The graph is undirected, so a vertex's in-neighbours are its neighbours and
// its out-degree is its degree. rank'(v) = (1 - d) / n + d * sum over
// neighbours u of rank(u) / deg(u); a vertex with no neighbour gives nothing.
// Every transport runs the same kernels, which sum in a fixed order (the
// neighbours on this rank, the others, then the two sums; a vertex with more
// than kHeavy neighbours by a team, in a fixed tree), so at a given rank
// count the transports agree bit for bit and print the same checksum.
//
// Build: examples/omp/graph/build_graph.sh
// Run  : pagerank_{mpi,giomp} --scale=22 --iters=20 --reps=3
#if defined(GRAPH_BACKEND_MPI) == defined(GRAPH_BACKEND_GIOMP)
#error "define exactly one of GRAPH_BACKEND_MPI and GRAPH_BACKEND_GIOMP"
#endif

#if defined(GRAPH_BACKEND_GIOMP)
#include "gicc/omp.h"
#endif
#include "graph_common.hpp"

#include <cmath>

using namespace graph;

namespace {

constexpr double kDamping = 0.85;

// A vertex with more neighbours than this is summed by a whole team: R-MAT's
// hubs have hundreds of thousands, and one thread each made them the whole
// run time.
constexpr int32_t kHeavy = 256;
constexpr int kTeam = 256;

// What moves each iteration, found once. Ghosts are the vertices other ranks
// own that some neighbour list here names; they are numbered in increasing
// order, which groups them by owner.
struct Plan {
    int64_t nlocal = 0;
    std::vector<int32_t> deg;
    std::vector<int64_t> lrow, rrow;     // neighbours here / elsewhere, per vertex
    std::vector<uint32_t> lcol, rcol;    // local index / ghost index
    std::vector<int64_t> goff;           // ghosts from rank q: [goff[q], goff[q + 1])
    std::vector<int64_t> sdispl;         // sent to rank r: sendidx[sdispl[r] .. sdispl[r + 1])
    std::vector<uint32_t> sendidx;       // local indices, in the receiver's ghost order
    std::vector<int64_t> roff;           // where rank r keeps the ghosts it gets from here
    std::vector<uint32_t> heavy;         // vertices with more than kHeavy neighbours
};

Plan make_plan(const Graph& g, MPI_Comm comm) {
    const int P = g.nranks, me = g.rank;
    Plan p;
    p.nlocal = g.nlocal;
    std::vector<uint32_t> ghosts;
    for (uint32_t c : g.col)
        if (g.owner(c) != me) ghosts.push_back(c);
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    p.goff.resize(P + 1);
    for (int q = 0; q <= P; ++q)
        p.goff[q] = std::lower_bound(ghosts.begin(), ghosts.end(),
                                     static_cast<uint64_t>(q) * g.nlocal) - ghosts.begin();

    p.deg.resize(g.nlocal);
    p.lrow.assign(g.nlocal + 1, 0);
    p.rrow.assign(g.nlocal + 1, 0);
    for (int64_t v = 0; v < g.nlocal; ++v) {
        int64_t here = 0;
        for (int64_t j = g.rowptr[v]; j < g.rowptr[v + 1]; ++j) here += g.owner(g.col[j]) == me;
        p.deg[v] = static_cast<int32_t>(g.rowptr[v + 1] - g.rowptr[v]);
        p.lrow[v + 1] = p.lrow[v] + here;
        p.rrow[v + 1] = p.rrow[v] + p.deg[v] - here;
    }
    for (int64_t v = 0; v < g.nlocal; ++v)
        if (p.deg[v] > kHeavy) p.heavy.push_back(static_cast<uint32_t>(v));
    p.lcol.resize(p.lrow[g.nlocal]);
    p.rcol.resize(p.rrow[g.nlocal]);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t v = 0; v < g.nlocal; ++v) {
        int64_t l = p.lrow[v], r = p.rrow[v];
        for (int64_t j = g.rowptr[v]; j < g.rowptr[v + 1]; ++j) {
            const uint32_t c = g.col[j];
            if (g.owner(c) == me)
                p.lcol[l++] = static_cast<uint32_t>(c - g.lo);
            else
                p.rcol[r++] = static_cast<uint32_t>(
                    std::lower_bound(ghosts.begin(), ghosts.end(), c) - ghosts.begin());
        }
    }

    // Tell each owner which of its vertices this rank needs, in ghost order.
    std::vector<int> need(P), want(P), nd(P + 1, 0), wd(P + 1, 0);
    for (int q = 0; q < P; ++q) need[q] = static_cast<int>(p.goff[q + 1] - p.goff[q]);
    MPI_Alltoall(need.data(), 1, MPI_INT, want.data(), 1, MPI_INT, comm);
    for (int q = 0; q < P; ++q) {
        nd[q + 1] = nd[q] + need[q];
        wd[q + 1] = wd[q] + want[q];
    }
    p.sendidx.resize(wd[P]);
    MPI_Alltoallv(ghosts.data(), need.data(), nd.data(), MPI_UINT32_T, p.sendidx.data(),
                  want.data(), wd.data(), MPI_UINT32_T, comm);
    for (uint32_t& s : p.sendidx) s -= static_cast<uint32_t>(g.lo);
    p.sdispl.assign(wd.begin(), wd.end());

    // A transport that writes into the receiver's ghosts needs their offset
    // there: rank r keeps this rank's values at its goff[me].
    p.roff.resize(P);
    MPI_Alltoall(p.goff.data(), 1, MPI_INT64_T, p.roff.data(), 1, MPI_INT64_T, comm);
    return p;
}

// ---- the three kernels every transport runs ---------------------------------

// What v gives each neighbour.
void contrib(int64_t n, const double* rank, const int32_t* deg, double* x) {
    #pragma omp target teams distribute parallel for is_device_ptr(rank, deg, x)
    for (int64_t v = 0; v < n; ++v) x[v] = deg[v] ? rank[v] / deg[v] : 0.0;
}

// out[v] = the sum of x over v's entries in (row, col): one thread per
// vertex, one team per heavy one. Each sums in a fixed order, the same for
// every transport.
void gather(int64_t n, const int32_t* deg, const int64_t* row, const uint32_t* col,
            const double* x, int64_t nheavy, const uint32_t* heavy, double* out) {
    #pragma omp target teams distribute parallel for is_device_ptr(deg, row, col, x, out)
    for (int64_t v = 0; v < n; ++v) {
        if (deg[v] > kHeavy) continue;
        double s = 0.0;
        for (int64_t j = row[v]; j < row[v + 1]; ++j) s += x[col[j]];
        out[v] = s;
    }
    if (nheavy == 0) return;
    #pragma omp target teams num_teams(nheavy) thread_limit(kTeam) \
            is_device_ptr(heavy, row, col, x, out)
    {
        double part[kTeam];
        const uint32_t v = heavy[omp_get_team_num()];
        #pragma omp parallel num_threads(kTeam)
        {
            const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
            for (int i = tid; i < kTeam; i += nth) part[i] = 0.0;
            #pragma omp barrier
            double s = 0.0;
            for (int64_t j = row[v] + tid; j < row[v + 1]; j += nth) s += x[col[j]];
            part[tid] += s;
            for (int w = kTeam / 2; w > 0; w /= 2) {
                #pragma omp barrier
                for (int i = tid; i < w; i += nth) part[i] += part[i + w];
            }
            #pragma omp barrier
            if (tid == 0) out[v] = part[0];
        }
    }
}

// v's new rank from what its neighbours here (acc) and elsewhere (rem) give.
void finish(int64_t n, const double* acc, const double* rem, double* rank, double base) {
    #pragma omp target teams distribute parallel for is_device_ptr(acc, rem, rank)
    for (int64_t v = 0; v < n; ++v) rank[v] = base + kDamping * (acc[v] + rem[v]);
}

void fill(int64_t n, double* a, double value) {
    #pragma omp target teams distribute parallel for is_device_ptr(a)
    for (int64_t v = 0; v < n; ++v) a[v] = value;
}

// ---- transports -------------------------------------------------------------

struct Dev {
    int32_t* deg;
    int64_t *lrow, *rrow;
    uint32_t *lcol, *rcol, *sendidx, *heavy;
    int64_t nheavy;
    double *rank, *acc, *rem;
};

// The sums over this rank's neighbours, while the exchange is in flight.
void sum_local(int64_t n, const Dev& d, const double* x) {
    gather(n, d.deg, d.lrow, d.lcol, x, d.nheavy, d.heavy, d.acc);
}

// The sums over the received ghosts, then the new ranks.
void sum_remote(int64_t n, const Dev& d, const double* ghost, double base) {
    gather(n, d.deg, d.rrow, d.rcol, ghost, d.nheavy, d.heavy, d.rem);
    finish(n, d.acc, d.rem, d.rank, base);
}

#if defined(GRAPH_BACKEND_MPI)
const char* kBackend = "mpi";

struct Transport {
    const Plan& p;
    MPI_Comm comm;
    int P, me;
    double *x, *ghost, *sendbuf;
    std::vector<MPI_Request> req;

    Transport(const Plan& plan, MPI_Comm c) : p(plan), comm(c) {
        MPI_Comm_rank(comm, &me);
        MPI_Comm_size(comm, &P);
        x = dev_alloc<double>(p.nlocal);
        ghost = dev_alloc<double>(p.goff[P]);
        sendbuf = dev_alloc<double>(p.sendidx.size());
        req.resize(2 * P);
    }
    ~Transport() {
        dev_free(x);
        dev_free(ghost);
        dev_free(sendbuf);
    }

    void step(const Dev& d, double base) {
        int nr = 0;
        for (int q = 0; q < P; ++q) {
            const int64_t c = p.goff[q + 1] - p.goff[q];
            if (c > 0)
                MPI_Irecv(ghost + p.goff[q], static_cast<int>(c), MPI_DOUBLE, q, 7, comm,
                          &req[nr++]);
        }
        contrib(p.nlocal, d.rank, d.deg, x);
        const int64_t ns = static_cast<int64_t>(p.sendidx.size());
        const uint32_t* idx = d.sendidx;
        double* sb = sendbuf;
        const double* xv = x;
        #pragma omp target teams distribute parallel for is_device_ptr(idx, sb, xv)
        for (int64_t k = 0; k < ns; ++k) sb[k] = xv[idx[k]];
        for (int r = 0; r < P; ++r) {
            const int64_t c = p.sdispl[r + 1] - p.sdispl[r];
            if (c > 0)
                MPI_Isend(sendbuf + p.sdispl[r], static_cast<int>(c), MPI_DOUBLE, r, 7, comm,
                          &req[nr++]);
        }
        sum_local(p.nlocal, d, x);
        MPI_Waitall(nr, req.data(), MPI_STATUSES_IGNORE);
        sum_remote(p.nlocal, d, ghost, base);
    }
};
#else
const char* kBackend = "giomp";

// Values put per chunk: one team packs a chunk and puts it at once.
constexpr int64_t kChunk = int64_t{1} << 16;

struct Transport {
    const Plan& p;
    int P, me;
    int64_t maxg;          // ghost slots per parity, the same on every rank
    double* xval;          // [contributions | ghosts, even steps | ghosts, odd steps]
    double* sendbuf;       // packed values for ranks on other nodes
    double** peer;         // per rank: its xval if same-node, else null
    int32_t* item_rank;    // work items: a chunk of what goes to one rank
    int64_t *item_start, *item_len, *sdispl, *roff;
    int nitems = 0;
    int parity = 0;

    Transport(const Plan& plan, MPI_Comm comm) : p(plan) {
        MPI_Comm_rank(comm, &me);
        MPI_Comm_size(comm, &P);
        int64_t g = p.goff[P], s = static_cast<int64_t>(p.sendidx.size()), gs[2] = {g, s}, mx[2];
        MPI_Allreduce(gs, mx, 2, MPI_INT64_T, MPI_MAX, comm);
        maxg = mx[0];
        // Symmetric: every rank allocates the same sizes in the same order.
        xval = static_cast<double*>(ompx_alloc((p.nlocal + 2 * maxg) * sizeof(double)));
        sendbuf = static_cast<double*>(ompx_alloc(std::max<int64_t>(mx[1], 1) * sizeof(double)));
        std::vector<double*> peers(P, nullptr);
        std::vector<int32_t> ir;
        std::vector<int64_t> is, il;
        for (int r = 0; r < P; ++r) {
            if (r == me) continue;
            peers[r] = static_cast<double*>(ompx_peer_ptr(r, xval));
            const int64_t c = p.sdispl[r + 1] - p.sdispl[r];
            for (int64_t o = 0; o < c; o += kChunk) {
                ir.push_back(r);
                is.push_back(o);
                il.push_back(std::min(kChunk, c - o));
            }
        }
        nitems = static_cast<int>(ir.size());
        peer = dev_copy(peers);
        item_rank = dev_copy(ir);
        item_start = dev_copy(is);
        item_len = dev_copy(il);
        sdispl = dev_copy(p.sdispl);
        roff = dev_copy(p.roff);
        ompx_prepare();
    }
    ~Transport() {
        for (void* q : {(void*)peer, (void*)item_rank, (void*)item_start, (void*)item_len,
                        (void*)sdispl, (void*)roff})
            dev_free(q);
        ompx_free(sendbuf);
        ompx_free(xval);
    }

    void step(const Dev& d, double base) {
        double* x = xval;
        contrib(p.nlocal, d.rank, d.deg, x);
        // This step's ghosts go to the half the receiver is not reading: a
        // rank may start the next step's puts while this one still sums.
        const int64_t gbase = p.nlocal + parity * maxg;
        if (nitems > 0) {
            const int32_t* ir = item_rank;
            const int64_t *is = item_start, *il = item_len, *sd = sdispl, *ro = roff;
            double* const* pr = peer;
            const uint32_t* idx = d.sendidx;
            double* sb = sendbuf;
            #pragma omp target teams num_teams(nitems) thread_limit(256) \
                    is_device_ptr(ir, is, il, sd, ro, pr, idx, sb, x) firstprivate(gbase)
            #pragma omp parallel
            {
                const int t = omp_get_team_num();
                const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
                const int r = ir[t];
                const int64_t src = sd[r] + is[t], len = il[t];
                const int64_t dst = gbase + ro[r] + is[t];   // into rank r's xval
                double* pd = pr[r];
                if (pd != nullptr) {
                    for (int64_t i = tid; i < len; i += nth) pd[dst + i] = x[idx[src + i]];
                } else {
                    for (int64_t i = tid; i < len; i += nth) sb[src + i] = x[idx[src + i]];
                    // The NIC reads sb from memory: every thread's stores
                    // reach L2 here, and the put's system fence writes them
                    // back.
#if defined(__AMDGCN__)
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
#else
                    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
                    #pragma omp barrier
                    if (tid == 0) ompx_put(r, x + dst, sb + src, len * sizeof(double), r);
                }
            }
        }
        sum_local(p.nlocal, d, x);
        ompx_fence();
        sum_remote(p.nlocal, d, x + gbase, base);
        parity ^= 1;
    }
};
#endif

}  // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
#if defined(GRAPH_BACKEND_GIOMP)
    ompx_init();
#else
    MPI_Init(&argc, &argv);
#endif
    MPI_Comm comm = MPI_COMM_WORLD;
    int me = 0, P = 1;
    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &P);
#if defined(GRAPH_BACKEND_MPI)
    select_device(comm);
#endif

    double t0 = MPI_Wtime();
    Graph g = build(o, comm);
    Plan p = make_plan(g, comm);
    int64_t nnz = static_cast<int64_t>(g.col.size()), ghosts = p.goff[P],
            sends = static_cast<int64_t>(p.sendidx.size());
    int64_t tot[3], mx[3], loc[3] = {nnz, ghosts, sends};
    MPI_Allreduce(loc, tot, 3, MPI_INT64_T, MPI_SUM, comm);
    MPI_Allreduce(loc, mx, 3, MPI_INT64_T, MPI_MAX, comm);
    const double t_setup = max_time(MPI_Wtime() - t0, comm);
    if (me == 0)
        std::printf("pagerank backend=%s ranks=%d scale=%d edgefactor=%d: %lld directed edges; "
                    "ghosts/rank max %lld, values sent/iter total %lld (max %lld/rank); "
                    "setup %.2f s\n",
                    kBackend, P, o.scale, o.edgefactor, (long long)tot[0], (long long)mx[1],
                    (long long)tot[2], (long long)mx[2], t_setup);

    Dev d;
    d.deg = dev_copy(p.deg);
    d.lrow = dev_copy(p.lrow);
    d.rrow = dev_copy(p.rrow);
    d.lcol = dev_copy(p.lcol);
    d.rcol = dev_copy(p.rcol);
    d.sendidx = dev_copy(p.sendidx);
    d.heavy = dev_copy(p.heavy);
    d.nheavy = static_cast<int64_t>(p.heavy.size());
    d.rank = dev_alloc<double>(p.nlocal);
    d.acc = dev_alloc<double>(p.nlocal);
    d.rem = dev_alloc<double>(p.nlocal);
    std::vector<uint32_t>().swap(g.col);
    const double base = (1.0 - kDamping) / static_cast<double>(g.n);

    std::vector<double> times;
    uint64_t hash = 0;
    double sum = 0;
    {
        Transport tr(p, comm);
        // Untimed: the first steps pay for first touches of peer memory and
        // the transports' own set-up.
        fill(p.nlocal, d.rank, 1.0 / static_cast<double>(g.n));
        for (int it = 0; it < 3; ++it) tr.step(d, base);
        for (int rep = 0; rep < o.reps; ++rep) {
            fill(p.nlocal, d.rank, 1.0 / static_cast<double>(g.n));
            MPI_Barrier(comm);
            const double ts = MPI_Wtime();
            for (int it = 0; it < o.iters; ++it) tr.step(d, base);
            times.push_back(max_time(MPI_Wtime() - ts, comm));
        }
        std::vector<double> r(p.nlocal);
        to_host(r.data(), d.rank, r.size());
        uint64_t h = 0;
        double s = 0;
        for (int64_t v = 0; v < p.nlocal; ++v) {
            uint64_t bits;
            std::memcpy(&bits, &r[v], sizeof(bits));
            h ^= splitmix64(bits ^ static_cast<uint64_t>(g.lo + v));
            s += r[v];
        }
        MPI_Allreduce(&h, &hash, 1, MPI_UINT64_T, MPI_BXOR, comm);
        MPI_Allreduce(&s, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    }
    if (me == 0) {
        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        const size_t k = sorted.size();
        const double med = k % 2 ? sorted[k / 2] : 0.5 * (sorted[k / 2 - 1] + sorted[k / 2]);
        std::printf("pagerank backend=%s ranks=%d scale=%d iters=%d: median %.4f s "
                    "(%.3f ms/iter, %.2f GTEPS), best %.3f ms/iter  reps:",
                    kBackend, P, o.scale, o.iters, med, 1e3 * med / o.iters,
                    static_cast<double>(tot[0]) * o.iters / med * 1e-9,
                    1e3 * sorted[0] / o.iters);
        for (double t : times) std::printf(" %.4f", t);
        std::printf("\npagerank checksum %016llx sum %.12f\n", (unsigned long long)hash, sum);
    }

    for (void* q : {(void*)d.deg, (void*)d.lrow, (void*)d.rrow, (void*)d.lcol, (void*)d.rcol,
                    (void*)d.sendidx, (void*)d.heavy, (void*)d.rank, (void*)d.acc,
                    (void*)d.rem})
        dev_free(q);
#if defined(GRAPH_BACKEND_GIOMP)
    ompx_finalize();
#else
    MPI_Finalize();
#endif
    return 0;
}
