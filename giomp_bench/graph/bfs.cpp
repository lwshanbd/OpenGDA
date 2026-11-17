// bfs.cpp - level-synchronous top-down BFS on a distributed R-MAT graph
// (graph_common.hpp), one source for two transports:
//
//   -DGRAPH_BACKEND_MPI    GPU-aware MPI. Each level the expand kernel buckets
//                          the neighbours other ranks own by owner; the host
//                          copies the bucket sizes back, exchanges them with
//                          MPI_Alltoall and the buckets with MPI_Alltoallv.
//   -DGRAPH_BACKEND_GIOMP  GiOMP. The expand kernel writes a neighbour a
//                          same-node rank owns straight into that rank's inbox
//                          (ompx_peer_ptr); a kernel then puts the other
//                          buckets and every bucket's size from the device,
//                          and ompx_fence completes the level. No size comes
//                          back to the host.
//
// Every sender has its own stretch of each receiver's inbox, sized by the
// edges between them -- no edge is followed twice, so a level cannot send
// more -- so no rank needs a remote atomic. Both transports run the same
// expand and receive kernels. A vertex's level does not depend on which of
// its parents claims it first, so at a given rank count the transports must
// print the same checksum.
//
// Build: examples/omp/graph/build_graph.sh
// Run  : bfs_{mpi,giomp} --scale=22 --roots=8 --reps=3
#if defined(GRAPH_BACKEND_MPI) == defined(GRAPH_BACKEND_GIOMP)
#error "define exactly one of GRAPH_BACKEND_MPI and GRAPH_BACKEND_GIOMP"
#endif

#if defined(GRAPH_BACKEND_GIOMP)
#include "gicc/omp.h"
#endif
#include "graph_common.hpp"

using namespace graph;

namespace {

// Per-level device state, the same for both transports.
struct Dev {
    int64_t nlocal, lo;
    int me, P;
    int64_t* row;
    uint32_t* col;
    int32_t* level;
    // Frontiers, local indices: vertices with at most kHeavy neighbours from
    // the front, the others from the back.
    uint32_t *cur, *next;
    int64_t* ncount;           // [light, heavy] sizes of next
    int64_t* scount;           // per rank: entries bucketed for it this level
    int64_t* sdispl;           // per rank: where its bucket starts in sendbuf
    uint32_t* sendbuf;
    uint32_t** peer;           // per rank: its inbox if same-node, else null
    int64_t* peer_off;         // per rank: this rank's stretch of that inbox
    int64_t* seg_off;          // received entries: P segments of `inbox`
    int64_t* seg_len;
    uint32_t* inbox;
};

// A vertex with more neighbours than this is expanded by a whole team: R-MAT's
// hubs have hundreds of thousands, and one thread each made them the whole
// run time.
constexpr int64_t kHeavy = 256;
constexpr int kTeam = 256;

#pragma omp declare target
// Claims wl for level `lv`; a winner joins the next frontier, at the front if
// light, at the back if heavy.
inline void visit(const int64_t* row, int32_t* level, uint32_t* next, int64_t* ncount,
                  int64_t nlocal, uint32_t wl, int32_t lv) {
    if (__atomic_load_n(&level[wl], __ATOMIC_RELAXED) != -1) return;
    int32_t expected = -1;
    if (!__scoped_atomic_compare_exchange_n(&level[wl], &expected, lv, false, __ATOMIC_RELAXED,
                                            __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE))
        return;
    const bool heavy = row[wl + 1] - row[wl] > kHeavy;
    const int64_t k = __scoped_atomic_fetch_add(&ncount[heavy], int64_t{1}, __ATOMIC_RELAXED,
                                                __MEMORY_SCOPE_DEVICE);
    next[heavy ? nlocal - 1 - k : k] = wl;
}

// Neighbour w of a frontier vertex: claimed if it is here, else bucketed for
// its owner -- stored straight into a same-node owner's inbox when `peer`
// has it.
inline void edge(const Dev& d, uint32_t w, int32_t lv) {
    const int q = static_cast<int>(w / d.nlocal);
    if (q == d.me) {
        visit(d.row, d.level, d.next, d.ncount, d.nlocal, static_cast<uint32_t>(w - d.lo), lv);
        return;
    }
    const int64_t slot = __scoped_atomic_fetch_add(&d.scount[q], int64_t{1}, __ATOMIC_RELAXED,
                                                   __MEMORY_SCOPE_DEVICE);
    uint32_t* pq = d.peer[q];
    if (pq != nullptr) pq[d.peer_off[q] + slot] = w;
    else d.sendbuf[d.sdispl[q] + slot] = w;
}
#pragma omp end declare target

// Level lv - 1 -> lv from this rank's frontier: a thread per light vertex,
// a team per heavy one.
void expand(const Dev& d, int64_t nlight, int64_t nheavy, int32_t lv) {
    const Dev dv = d;   // by value into the kernels: its pointers are device addresses
    if (nlight > 0) {
        #pragma omp target teams distribute parallel for schedule(static, 1) firstprivate(dv, lv)
        for (int64_t i = 0; i < nlight; ++i) {
            const uint32_t u = dv.cur[i];
            for (int64_t j = dv.row[u]; j < dv.row[u + 1]; ++j) edge(dv, dv.col[j], lv);
        }
    }
    if (nheavy > 0) {
        #pragma omp target teams num_teams(nheavy) thread_limit(kTeam) firstprivate(dv, lv)
        #pragma omp parallel
        {
            const uint32_t u = dv.cur[dv.nlocal - 1 - omp_get_team_num()];
            for (int64_t j = dv.row[u] + omp_get_thread_num(); j < dv.row[u + 1];
                 j += omp_get_num_threads())
                edge(dv, dv.col[j], lv);
        }
    }
}

// Claims what arrived: segment s is inbox[seg_off[s] .. + seg_len[s]).
void receive(const Dev& d, int32_t lv) {
    const Dev dv = d;
    constexpr int kTeamsPerSegment = 64;
    #pragma omp target teams num_teams(d.P * kTeamsPerSegment) thread_limit(256) \
            firstprivate(dv, lv)
    #pragma omp parallel
    {
        const int t = omp_get_team_num();
        const int s = t / kTeamsPerSegment;
        const int64_t stride = int64_t{kTeamsPerSegment} * omp_get_num_threads();
        const int64_t first = int64_t{t % kTeamsPerSegment} * omp_get_num_threads() +
                              omp_get_thread_num();
        const uint32_t* seg = dv.inbox + dv.seg_off[s];
        for (int64_t k = first; k < dv.seg_len[s]; k += stride)
            visit(dv.row, dv.level, dv.next, dv.ncount, dv.nlocal,
                  static_cast<uint32_t>(seg[k] - dv.lo), lv);
    }
}

void zero(int64_t n, int64_t* a) {
    #pragma omp target teams distribute parallel for is_device_ptr(a)
    for (int64_t i = 0; i < n; ++i) a[i] = 0;
}

// ---- transports -------------------------------------------------------------

// What the setup found: per rank, the entries this rank may send it and
// receive from it over a whole search.
struct Caps {
    std::vector<int64_t> send, recv;   // P each
};

#if defined(GRAPH_BACKEND_MPI)
const char* kBackend = "mpi";

struct Transport {
    Dev& d;
    MPI_Comm comm;
    std::vector<int64_t> sd;           // send displacements (host)
    std::vector<int> sc, rc, sdi, rdi;
    std::vector<int64_t> hs;

    Transport(Dev& dev, const Caps& caps, MPI_Comm c) : d(dev), comm(c) {
        const int P = d.P;
        sd.assign(P + 1, 0);
        int64_t rtot = 0;
        for (int q = 0; q < P; ++q) {
            sd[q + 1] = sd[q] + caps.send[q];
            rtot += caps.recv[q];
        }
        d.sendbuf = dev_alloc<uint32_t>(sd[P]);
        d.inbox = dev_alloc<uint32_t>(rtot);
        d.sdispl = dev_copy(sd);
        d.peer = dev_copy(std::vector<uint32_t*>(P, nullptr));
        d.peer_off = dev_copy(std::vector<int64_t>(P, 0));
        d.seg_off = dev_alloc<int64_t>(P);
        d.seg_len = dev_alloc<int64_t>(P);
        sc.resize(P); rc.resize(P); sdi.resize(P); rdi.resize(P); hs.resize(P);
    }
    ~Transport() {
        for (void* q : {(void*)d.sendbuf, (void*)d.inbox, (void*)d.sdispl, (void*)d.peer,
                        (void*)d.peer_off, (void*)d.seg_off, (void*)d.seg_len})
            dev_free(q);
    }

    // After expand: every bucket to its owner; `inbox` then holds P segments.
    void exchange() {
        const int P = d.P;
        to_host(hs.data(), d.scount, P);
        for (int q = 0; q < P; ++q) {
            sc[q] = static_cast<int>(hs[q]);
            sdi[q] = static_cast<int>(sd[q]);
        }
        MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, comm);
        std::vector<int64_t> so(P), sl(P);
        int64_t o = 0;
        for (int q = 0; q < P; ++q) {
            rdi[q] = static_cast<int>(o);
            so[q] = o;
            sl[q] = rc[q];
            o += rc[q];
        }
        MPI_Alltoallv(d.sendbuf, sc.data(), sdi.data(), MPI_UINT32_T, d.inbox, rc.data(),
                      rdi.data(), MPI_UINT32_T, comm);
        to_dev(d.seg_off, so.data(), P);
        to_dev(d.seg_len, sl.data(), P);
    }
};
#else
const char* kBackend = "giomp";

struct Transport {
    Dev& d;
    uint32_t* inbox_sym;     // symmetric: [segment from rank 0 | from rank 1 | ...]
    uint32_t* sendbuf_sym;
    int64_t* incount;        // symmetric, per sender: entries it sent this level
    int64_t* scount_sym;     // symmetric: the puts read the sizes from here
    int64_t** peer_count;    // per rank: its incount if same-node, else null
    int64_t* roff;           // per rank: this rank's stretch of its inbox

    Transport(Dev& dev, const Caps& caps, MPI_Comm comm) : d(dev) {
        const int P = d.P, me = d.me;
        std::vector<int64_t> sd(P + 1, 0), in(P + 1, 0);
        for (int q = 0; q < P; ++q) {
            sd[q + 1] = sd[q] + caps.send[q];
            in[q + 1] = in[q] + caps.recv[q];
        }
        // Rank r keeps what this rank sends it at its in[me].
        std::vector<int64_t> ro(P);
        MPI_Alltoall(in.data(), 1, MPI_INT64_T, ro.data(), 1, MPI_INT64_T, comm);
        int64_t sz[2] = {in[P], sd[P]}, mx[2];
        MPI_Allreduce(sz, mx, 2, MPI_INT64_T, MPI_MAX, comm);
        inbox_sym = static_cast<uint32_t*>(ompx_alloc(std::max<int64_t>(mx[0], 1) * 4));
        sendbuf_sym = static_cast<uint32_t*>(ompx_alloc(std::max<int64_t>(mx[1], 1) * 4));
        incount = static_cast<int64_t*>(ompx_alloc(P * sizeof(int64_t)));
        scount_sym = static_cast<int64_t*>(ompx_alloc(P * sizeof(int64_t)));
        std::vector<uint32_t*> pi(P, nullptr);
        std::vector<int64_t*> pc(P, nullptr);
        for (int r = 0; r < P; ++r) {
            if (r == me) continue;
            pi[r] = static_cast<uint32_t*>(ompx_peer_ptr(r, inbox_sym));
            pc[r] = static_cast<int64_t*>(ompx_peer_ptr(r, incount));
        }
        d.inbox = inbox_sym;
        d.sendbuf = sendbuf_sym;
        d.scount = scount_sym;
        d.sdispl = dev_copy(sd);
        d.peer = dev_copy(pi);
        d.peer_off = dev_copy(ro);
        d.seg_off = dev_copy(std::vector<int64_t>(in.begin(), in.end() - 1));
        d.seg_len = incount;
        peer_count = dev_copy(pc);
        roff = d.peer_off;
        ompx_prepare();
    }
    ~Transport() {
        for (void* q : {(void*)d.sdispl, (void*)d.peer, (void*)d.peer_off, (void*)d.seg_off,
                        (void*)peer_count})
            dev_free(q);
        ompx_free(scount_sym);
        ompx_free(incount);
        ompx_free(sendbuf_sym);
        ompx_free(inbox_sym);
    }

    // After expand: same-node buckets are already in place; tell those ranks
    // their sizes, put the other buckets and sizes, and complete the level.
    void exchange() {
        const int P = d.P, me = d.me;
        const int64_t *sc = d.scount, *sd = d.sdispl, *ro = roff;
        int64_t* const* pc = peer_count;
        uint32_t* sb = d.sendbuf;
        uint32_t* ib = d.inbox;
        int64_t* ic = incount;
        #pragma omp target is_device_ptr(sc, sd, ro, pc, sb, ib, ic) firstprivate(P, me)
        {
            for (int r = 0; r < P; ++r) {
                if (r == me) continue;
                if (pc[r] != nullptr) {
                    pc[r][me] = sc[r];
                } else {
                    if (sc[r] > 0)
                        ompx_put(r, ib + ro[r], sb + sd[r], sc[r] * sizeof(uint32_t), r);
                    ompx_put(r, ic + me, sc + r, sizeof(int64_t), r);
                }
            }
        }
        ompx_fence();
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
    Caps caps;
    caps.send.assign(P, 0);
    for (uint32_t c : g.col)
        if (g.owner(c) != me) ++caps.send[g.owner(c)];
    caps.recv.resize(P);
    MPI_Alltoall(caps.send.data(), 1, MPI_INT64_T, caps.recv.data(), 1, MPI_INT64_T, comm);
    int64_t nnz = static_cast<int64_t>(g.col.size()), tot = 0;
    MPI_Allreduce(&nnz, &tot, 1, MPI_INT64_T, MPI_SUM, comm);
    const double t_setup = max_time(MPI_Wtime() - t0, comm);
    if (me == 0)
        std::printf("bfs backend=%s ranks=%d scale=%d edgefactor=%d: %lld directed edges; "
                    "setup %.2f s\n",
                    kBackend, P, o.scale, o.edgefactor, (long long)tot, t_setup);

    Dev d{};
    d.nlocal = g.nlocal;
    d.lo = g.lo;
    d.me = me;
    d.P = P;
    d.row = dev_copy(g.rowptr);
    d.col = dev_copy(g.col);
    d.level = dev_alloc<int32_t>(g.nlocal);
    d.cur = dev_alloc<uint32_t>(g.nlocal);
    d.next = dev_alloc<uint32_t>(g.nlocal);
    d.ncount = dev_alloc<int64_t>(2);
#if defined(GRAPH_BACKEND_MPI)
    d.scount = dev_alloc<int64_t>(P);
#endif

    // Roots: the same on every run, vertices with a neighbour.
    std::vector<int64_t> roots;
    for (uint64_t k = 0; static_cast<int>(roots.size()) < o.roots && k < 1000; ++k) {
        const int64_t v = static_cast<int64_t>(splitmix64(o.seed * 7919 + k) & (g.n - 1));
        int64_t deg = g.owner(v) == me ? g.rowptr[v - g.lo + 1] - g.rowptr[v - g.lo] : 0, all;
        MPI_Allreduce(&deg, &all, 1, MPI_INT64_T, MPI_MAX, comm);
        if (all > 0) roots.push_back(v);
    }

    std::vector<double> times;       // per search, slowest rank
    std::vector<int64_t> teps_edges; // edges in the searched component
    std::vector<uint64_t> checks;
    std::vector<int> depths;
    int unstable = 0;                // searches whose levels differ from rep 0's
    {
        Transport tr(d, caps, comm);
        std::vector<int32_t> lv(g.nlocal);
        // Rep -1 is an untimed search from the first root: it pays for first
        // touches of peer memory and the transports' own set-up.
        for (int rep = -1; rep < o.reps; ++rep) {
            for (size_t ri = 0; ri < roots.size(); ++ri) {
                if (rep < 0 && ri > 0) break;
                const int64_t root = roots[ri];
                std::fill(lv.begin(), lv.end(), -1);
                int64_t fcount[2] = {0, 0};   // light, heavy
                if (g.owner(root) == me) {
                    lv[root - g.lo] = 0;
                    const uint32_t rl = static_cast<uint32_t>(root - g.lo);
                    const bool heavy = g.rowptr[rl + 1] - g.rowptr[rl] > kHeavy;
                    to_dev(d.cur + (heavy ? g.nlocal - 1 : 0), &rl, 1);
                    fcount[heavy] = 1;
                }
                to_dev(d.level, lv.data(), lv.size());
                MPI_Barrier(comm);
                const double ts = MPI_Wtime();
                int32_t depth = 0;
                for (;;) {
                    zero(2, d.ncount);
                    zero(P, d.scount);
                    expand(d, fcount[0], fcount[1], depth + 1);
                    tr.exchange();
                    receive(d, depth + 1);
                    to_host(fcount, d.ncount, 2);
                    int64_t mine = fcount[0] + fcount[1], all = 0;
                    MPI_Allreduce(&mine, &all, 1, MPI_INT64_T, MPI_SUM, comm);
                    if (all == 0) break;
                    std::swap(d.cur, d.next);
                    ++depth;
                }
                const double t = max_time(MPI_Wtime() - ts, comm);
                if (rep >= 0) {
                    // Check every search: the level of every vertex, and the
                    // edges the search covered (each counted from both ends).
                    // A root must give the same levels every time.
                    to_host(lv.data(), d.level, lv.size());
                    uint64_t h = 0;
                    int64_t e = 0;
                    for (int64_t v = 0; v < g.nlocal; ++v) {
                        if (lv[v] < 0) continue;
                        h ^= splitmix64(static_cast<uint64_t>(g.lo + v) << 8 ^
                                        static_cast<uint64_t>(lv[v]));
                        e += g.rowptr[v + 1] - g.rowptr[v];
                    }
                    uint64_t hh = 0;
                    int64_t ee = 0;
                    MPI_Allreduce(&h, &hh, 1, MPI_UINT64_T, MPI_BXOR, comm);
                    MPI_Allreduce(&e, &ee, 1, MPI_INT64_T, MPI_SUM, comm);
                    if (rep == 0) {
                        checks.push_back(hh);
                        teps_edges.push_back(ee / 2);
                        depths.push_back(depth);
                    } else if (hh != checks[ri]) {
                        ++unstable;
                    }
                    times.push_back(t);
                }
            }
        }
    }
    if (me == 0) {
        // Edges covered over time taken, all searches together: a root in a
        // tiny component (a vertex with one neighbour) would dominate
        // Graph500's harmonic mean at these sizes.
        const size_t nr = roots.size();
        double edges = 0, total = 0;
        for (size_t k = 0; k < times.size(); ++k) {
            edges += static_cast<double>(teps_edges[k % nr]);
            total += times[k];
        }
        uint64_t check = 0;
        for (size_t k = 0; k < nr; ++k) check ^= splitmix64(checks[k] + k);
        std::printf("bfs backend=%s ranks=%d scale=%d roots=%zu reps=%d: mean %.4f s/search, "
                    "%.3f GTEPS\n",
                    kBackend, P, o.scale, nr, o.reps, total / times.size(),
                    edges / total * 1e-9);
        std::printf("bfs per root (first rep): ");
        for (size_t k = 0; k < nr; ++k)
            std::printf(" [depth %d, %lld edges, %.4f s]", depths[k], (long long)teps_edges[k],
                        times[k]);
        std::printf("\nbfs checksum %016llx, %d of %zu repeated searches differ\n",
                    (unsigned long long)check, unstable, times.size() - nr);
    }

    for (void* q : {(void*)d.row, (void*)d.col, (void*)d.level, (void*)d.cur, (void*)d.next,
                    (void*)d.ncount})
        dev_free(q);
#if defined(GRAPH_BACKEND_MPI)
    dev_free(d.scount);
#endif
#if defined(GRAPH_BACKEND_GIOMP)
    ompx_finalize();
#else
    MPI_Finalize();
#endif
    return 0;
}
