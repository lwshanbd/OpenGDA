// graph_common.hpp - what the graph examples share, whatever the transport.
//
// The graph is Graph500-style R-MAT (a = 0.57, b = c = 0.19) with 2^scale
// vertices and edgefactor * 2^scale undirected edges, taken as symmetric:
// every edge {u, v} is stored as u -> v and v -> u. Edge e is drawn from a
// counter-based generator seeded with (seed, e), so the graph is the same
// whatever the number of ranks. Vertex numbers are then scrambled by a
// bijection on [0, 2^scale), which spreads R-MAT's hubs (all near vertex 0)
// over the ranks. Self loops and repeated edges are dropped.
//
// The vertices are split in contiguous blocks: rank r owns
// [r * n / P, (r + 1) * n / P). Each rank holds the adjacency of its own
// vertices in CSR form, with neighbours by global number in increasing
// order. Generation and distribution run on the host with MPI; nothing here
// is timed.
#pragma once

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace graph {

struct Options {
    int scale = 22;
    int edgefactor = 16;
    uint64_t seed = 1;
    int iters = 20;   // PageRank iterations
    int roots = 8;    // BFS roots
    int reps = 3;     // timed repetitions
};

// --scale=N --edgefactor=N --seed=N --iters=N --roots=N --reps=N
inline Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto val = [&](const char* key) -> const char* {
            const size_t k = std::strlen(key);
            return std::strncmp(a, key, k) == 0 && a[k] == '=' ? a + k + 1 : nullptr;
        };
        if (const char* v = val("--scale")) o.scale = std::atoi(v);
        else if (const char* v = val("--edgefactor")) o.edgefactor = std::atoi(v);
        else if (const char* v = val("--seed")) o.seed = std::strtoull(v, nullptr, 10);
        else if (const char* v = val("--iters")) o.iters = std::atoi(v);
        else if (const char* v = val("--roots")) o.roots = std::atoi(v);
        else if (const char* v = val("--reps")) o.reps = std::atoi(v);
        else {
            std::fprintf(stderr, "unknown argument %s\n", a);
            std::exit(2);
        }
    }
    return o;
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// A bijection on [0, 2^scale): odd multiplies, an add and xorshifts, each
// invertible modulo 2^scale.
inline uint64_t scramble(uint64_t v, int scale) {
    const uint64_t mask = (1ull << scale) - 1;
    const int s = (scale + 1) / 2;
    v = (v * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull) & mask;
    v ^= v >> s;
    v = (v * 0xBF58476D1CE4E5B9ull) & mask;
    v ^= v >> s;
    return v;
}

// Edge e of the R-MAT graph, before scrambling.
inline void rmat_edge(uint64_t seed, uint64_t e, int scale, uint64_t* u, uint64_t* v) {
    // Thresholds of a, a + b, a + b + c on 32 bits.
    constexpr uint32_t kA = 2448131359u, kAB = 3264175145u, kABC = 4080218931u;
    uint64_t state = splitmix64(seed * 0xD1B54A32D192ED03ull ^ e);
    uint64_t x = 0, y = 0;
    for (int bit = 0; bit < scale; ++bit) {
        if (bit % 2 == 0) state = splitmix64(state);
        const uint32_t r = static_cast<uint32_t>(bit % 2 == 0 ? state : state >> 32);
        const uint64_t bx = r >= kAB, by = (r >= kA && r < kAB) || r >= kABC;
        x = (x << 1) | bx;
        y = (y << 1) | by;
    }
    *u = x;
    *v = y;
}

// This rank's part of the symmetric graph.
struct Graph {
    int scale = 0, nranks = 1, rank = 0;
    int64_t n = 0;        // vertices in the graph
    int64_t nlocal = 0;   // vertices per rank
    int64_t lo = 0;       // first vertex this rank owns
    int64_t edges = 0;    // undirected edges generated (before dropping any)
    std::vector<int64_t> rowptr;   // nlocal + 1
    std::vector<uint32_t> col;     // neighbours, global numbers, ascending per row

    int owner(int64_t v) const { return static_cast<int>(v / nlocal); }
};

inline Graph build(const Options& o, MPI_Comm comm) {
    Graph g;
    MPI_Comm_rank(comm, &g.rank);
    MPI_Comm_size(comm, &g.nranks);
    g.scale = o.scale;
    g.n = int64_t{1} << o.scale;
    if (o.scale > 31 || g.n % g.nranks != 0) {
        if (g.rank == 0)
            std::fprintf(stderr, "scale %d: need scale <= 31 and 2^scale divisible by %d ranks\n",
                         o.scale, g.nranks);
        MPI_Abort(comm, 2);
    }
    g.nlocal = g.n / g.nranks;
    g.lo = g.rank * g.nlocal;
    g.edges = g.n * o.edgefactor;

    // This rank's share of the edges, each sent to the owners of its ends.
    const int P = g.nranks;
    const int64_t e0 = g.edges * g.rank / P, e1 = g.edges * (g.rank + 1) / P;
    std::vector<std::vector<uint64_t>> out(P);
    for (int64_t e = e0; e < e1; ++e) {
        uint64_t u, v;
        rmat_edge(o.seed, static_cast<uint64_t>(e), o.scale, &u, &v);
        u = scramble(u, o.scale);
        v = scramble(v, o.scale);
        if (u == v) continue;
        out[g.owner(static_cast<int64_t>(u))].push_back(u << 32 | v);
        out[g.owner(static_cast<int64_t>(v))].push_back(v << 32 | u);
    }
    std::vector<int> scount(P), rcount(P), sdispl(P + 1, 0), rdispl(P + 1, 0);
    for (int q = 0; q < P; ++q) scount[q] = static_cast<int>(out[q].size());
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, comm);
    for (int q = 0; q < P; ++q) {
        sdispl[q + 1] = sdispl[q] + scount[q];
        rdispl[q + 1] = rdispl[q] + rcount[q];
    }
    std::vector<uint64_t> sbuf(sdispl[P]), rbuf(rdispl[P]);
    for (int q = 0; q < P; ++q) {
        std::copy(out[q].begin(), out[q].end(), sbuf.begin() + sdispl[q]);
        std::vector<uint64_t>().swap(out[q]);
    }
    MPI_Alltoallv(sbuf.data(), scount.data(), sdispl.data(), MPI_UINT64_T, rbuf.data(),
                  rcount.data(), rdispl.data(), MPI_UINT64_T, comm);
    std::vector<uint64_t>().swap(sbuf);

    // CSR by counting sort on the owned end, then each row sorted and its
    // repeats dropped.
    std::vector<int64_t> cnt(g.nlocal + 1, 0);
    for (uint64_t p : rbuf) ++cnt[static_cast<int64_t>(p >> 32) - g.lo + 1];
    for (int64_t i = 0; i < g.nlocal; ++i) cnt[i + 1] += cnt[i];
    std::vector<uint32_t> raw(rbuf.size());
    {
        std::vector<int64_t> next(cnt.begin(), cnt.end() - 1);
        for (uint64_t p : rbuf)
            raw[next[static_cast<int64_t>(p >> 32) - g.lo]++] = static_cast<uint32_t>(p);
    }
    std::vector<uint64_t>().swap(rbuf);
    std::vector<int64_t> keep(g.nlocal + 1, 0);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t i = 0; i < g.nlocal; ++i) {
        uint32_t* b = raw.data() + cnt[i];
        uint32_t* e = raw.data() + cnt[i + 1];
        std::sort(b, e);
        keep[i + 1] = std::unique(b, e) - b;
    }
    g.rowptr.assign(g.nlocal + 1, 0);
    for (int64_t i = 0; i < g.nlocal; ++i) g.rowptr[i + 1] = g.rowptr[i] + keep[i + 1];
    g.col.resize(g.rowptr[g.nlocal]);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t i = 0; i < g.nlocal; ++i)
        std::copy(raw.begin() + cnt[i], raw.begin() + cnt[i] + keep[i + 1],
                  g.col.begin() + g.rowptr[i]);
    return g;
}

// ---- device memory -----------------------------------------------------------

inline int device() { return omp_get_default_device(); }

template <typename T>
T* dev_alloc(size_t count) {
    void* p = omp_target_alloc(std::max<size_t>(count, 1) * sizeof(T), device());
    if (p == nullptr) {
        std::fprintf(stderr, "omp_target_alloc of %zu bytes failed\n", count * sizeof(T));
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    return static_cast<T*>(p);
}

template <typename T>
void to_dev(T* dst, const T* src, size_t count) {
    if (count == 0) return;
    omp_target_memcpy(dst, src, count * sizeof(T), 0, 0, device(), omp_get_initial_device());
}

template <typename T>
void to_host(T* dst, const T* src, size_t count) {
    if (count == 0) return;
    omp_target_memcpy(dst, src, count * sizeof(T), 0, 0, omp_get_initial_device(), device());
}

template <typename T>
T* dev_copy(const std::vector<T>& v) {
    T* p = dev_alloc<T>(v.size());
    to_dev(p, v.data(), v.size());
    return p;
}

inline void dev_free(void* p) {
    if (p != nullptr) omp_target_free(p, device());
}

// The plain-MPI build's device: one per rank on the node, whether the launch
// shows each rank all of the node's GPUs or only its own.
inline void select_device(MPI_Comm comm) {
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int lr = 0;
    MPI_Comm_rank(local, &lr);
    MPI_Comm_free(&local);
    const int nd = omp_get_num_devices();
    if (nd > 0) omp_set_default_device(lr % nd);
}

// Wall time of the slowest rank.
inline double max_time(double t, MPI_Comm comm) {
    double m = 0;
    MPI_Allreduce(&t, &m, 1, MPI_DOUBLE, MPI_MAX, comm);
    return m;
}

}  // namespace graph
