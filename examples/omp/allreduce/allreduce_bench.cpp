// allreduce_bench: the global sum of a distributed dot product, the way a
// conjugate-gradient solver needs it every iteration.
//
//   mpi_host     MPI_Allreduce of a host double, nothing else (MPI's latency)
//   mpi_dot      a target region computes the local dot product (OpenMP
//                reduction) and maps it back; MPI_Allreduce sums it on the
//                host -- what miniFE does
//   mpi_dot_pin  giomp_dot's kernel writes the partial straight to pinned host
//                memory (no map clause); MPI_Allreduce sums it on the host
//   mpi_dot_dev  giomp_dot's kernel leaves the partial on the device, where
//                GPU-aware MPI_Allreduce sums it; then it is copied back
//   giomp_dot    one target region computes the local dot product and sums
//                it across ranks itself: within a node every rank publishes
//                its partial in its own heap and pulls the others' through
//                ompx_peer_ptr; across nodes the rank with the same local
//                index on every other node gets this node's sum by
//                ompx_put_signal. Everyone adds the partials in the same
//                order, so every rank gets the same bits. The sum goes
//                straight to pinned host memory. Every rank adds the call
//                number to its partial, so each call's sum is different and a
//                stale value would fail the check.
//
// Usage: allreduce_bench MODE [N] [ITERS]
//   N      elements per rank in the dot product (default 65536)
//   ITERS  timed repetitions (default 2000), after ITERS/10 untimed
// Prints the mean time per global sum (the slowest rank's), and checks the
// sums against the exact value.
#include "gicc/omp.h"

#include <mpi.h>
#include <omp.h>

// libomptarget's host allocation the device can write (not in ROCm's omp.h).
extern "C" void* llvm_omp_target_alloc_host(size_t size, int device);
extern "C" void omp_target_free(void* ptr, int device);

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kSigSlots = 64;       // GiOMP signal slots
constexpr int kTeams = 440;         // 4 per CU on an MI250X GCD
constexpr int kThreads = 256;

int g_rank = 0, g_size = 1;

// Enough teams for four elements per thread, at most kTeams.
int teams_for(long n) {
    const long t = (n + 4L * kThreads - 1) / (4L * kThreads);
    return (int)std::max(1L, std::min((long)kTeams, t));
}

// Node layout: `local` ranks per node, this rank's index among them and its
// node's index, and every rank's (node, local index).
struct Layout {
    int local = 1, li = 0, nodes = 1, node = 0;
    std::vector<int> rank_of;       // [node * local + li] -> world rank
};

Layout layout() {
    Layout l;
    MPI_Comm lc;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &lc);
    MPI_Comm_rank(lc, &l.li);
    MPI_Comm_size(lc, &l.local);
    int leader = g_rank;
    MPI_Bcast(&leader, 1, MPI_INT, 0, lc);
    MPI_Comm_free(&lc);
    std::vector<int> leaders(g_size), locals(g_size), lis(g_size);
    MPI_Allgather(&leader, 1, MPI_INT, leaders.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&l.local, 1, MPI_INT, locals.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&l.li, 1, MPI_INT, lis.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> ids = leaders;
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    l.nodes = (int)ids.size();
    for (int r = 0; r < g_size; ++r) {
        if (locals[r] != l.local) {
            if (g_rank == 0) std::fprintf(stderr, "every node needs the same number of ranks\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    l.rank_of.assign(l.nodes * l.local, -1);
    for (int r = 0; r < g_size; ++r) {
        const int n = (int)(std::lower_bound(ids.begin(), ids.end(), leaders[r]) - ids.begin());
        l.rank_of[n * l.local + lis[r]] = r;
        if (r == g_rank) l.node = n;
    }
    return l;
}

void select_device_mpi() {
    MPI_Comm lc;
    int li = 0;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &lc);
    MPI_Comm_rank(lc, &li);
    MPI_Comm_free(&lc);
    const int nd = omp_get_num_devices();
    if (nd > 0) omp_set_default_device(li % nd);
}

// x[i] = (rank + 1) * (i + 1) / N, so the global dot product with itself has
// a closed form.
void fill(double* x, long n) {
    const double s = (g_rank + 1.0) / (double)n;
    #pragma omp target teams distribute parallel for is_device_ptr(x)
    for (long i = 0; i < n; ++i) x[i] = s * (double)(i + 1);
}

double exact(long n) {
    double sum_i2 = (double)n * (n + 1) * (2.0 * n + 1) / 6.0;   // sum (i+1)^2
    double sum_r2 = 0;
    for (int r = 0; r < g_size; ++r) sum_r2 += (r + 1.0) * (r + 1.0);
    return sum_r2 * sum_i2 / ((double)n * n);
}

double local_dot(const double* x, long n) {
    double r = 0;
    #pragma omp target teams distribute parallel for reduction(+:r) map(tofrom:r) \
        num_teams(kTeams) is_device_ptr(x)
    for (long i = 0; i < n; ++i) r += x[i] * x[i];
    return r;
}

// ---- GiOMP --------------------------------------------------------------------

struct Giomp {
    Layout l;
    int sbase = 0;                        // signal slots [sbase, sbase + nodes)
    unsigned long long epoch = 0;
    // Symmetric heap.
    double* vals = nullptr;               // [2]: my partial, by parity (peers pull it)
    unsigned long long* flags = nullptr;  // [2]: the call it belongs to
    double* nvals = nullptr;              // [2][nodes]: sums of the other nodes
    double* mysum = nullptr;              // [2]: this node's sum, the put source
    // Device.
    double** peer_vals = nullptr;         // [local]: each same-node rank's vals
    unsigned long long** peer_flags = nullptr;
    int* cpeer = nullptr;                 // [nodes]: my counterpart on each node
    double* team_sum = nullptr;           // [kTeams]
    unsigned* done = nullptr;             // teams finished, over all calls
    double* hres = nullptr;               // pinned host: the global sum
    std::vector<int> h_cpeer;
};

template <typename T>
T* to_device(const std::vector<T>& v) {
    const int d = omp_get_default_device();
    T* p = (T*)omp_target_alloc(std::max<size_t>(v.size(), 1) * sizeof(T), d);
    if (!v.empty())
        omp_target_memcpy(p, v.data(), v.size() * sizeof(T), 0, 0, d, omp_get_initial_device());
    return p;
}

Giomp giomp_setup() {
    Giomp g;
    g.l = layout();
    const Layout& l = g.l;
    if (l.nodes > kSigSlots) {
        if (g_rank == 0) std::fprintf(stderr, "more nodes than signal slots\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    g.sbase = kSigSlots - l.nodes;
    g.vals = (double*)ompx_alloc(2 * sizeof(double));
    g.flags = (unsigned long long*)ompx_alloc(2 * sizeof(unsigned long long));
    g.nvals = (double*)ompx_alloc(2 * l.nodes * sizeof(double));
    g.mysum = (double*)ompx_alloc(2 * sizeof(double));
    std::vector<double*> pv(l.local);
    std::vector<unsigned long long*> pf(l.local);
    for (int j = 0; j < l.local; ++j) {
        const int r = l.rank_of[l.node * l.local + j];
        pv[j] = r == g_rank ? g.vals : (double*)ompx_peer_ptr(r, g.vals);
        pf[j] = r == g_rank ? g.flags : (unsigned long long*)ompx_peer_ptr(r, g.flags);
        if (pv[j] == nullptr || pf[j] == nullptr) {
            std::fprintf(stderr, "rank %d: no IPC mapping of same-node rank %d "
                                 "(needs GICC_HALO_IPC=1 and every GPU visible)\n", g_rank, r);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    g.h_cpeer.resize(l.nodes);
    for (int m = 0; m < l.nodes; ++m) g.h_cpeer[m] = l.rank_of[m * l.local + l.li];
    g.peer_vals = to_device(pv);
    g.peer_flags = to_device(pf);
    g.cpeer = to_device(g.h_cpeer);
    g.team_sum = to_device(std::vector<double>(kTeams, 0.0));
    g.done = to_device(std::vector<unsigned>(1, 0u));
    g.hres = (double*)llvm_omp_target_alloc_host(sizeof(double), omp_get_default_device());
    ompx_prepare();
    MPI_Barrier(MPI_COMM_WORLD);
    return g;
}

#pragma omp declare target
// Same-node exchange through ompx_peer_ptr, inside one kernel, on MI250X's
// coarse-grained heap memory: a GPU's L2 is not told when a peer writes its
// memory, so a rank polling its own memory for a peer's store can read a
// stale line for as long as the kernel runs. Remote memory is cached as
// non-coherent, and a system-scope acquire drops those lines. So every rank
// publishes in its OWN memory (store, then a system-scope fence to write
// the L2 back) and reads the others' through their peer pointers with
// acquire loads: the peers pull.
static inline double as_double(unsigned long long b) {
    double d;
    __builtin_memcpy(&d, &b, sizeof d);
    return d;
}

// This rank's partial `x` of call `e` in, the global sum out. One thread.
static inline double global_sum(double x, unsigned long long e, int local, int li, int nodes,
                                int node, int sbase, double* vals, unsigned long long* flags,
                                double* const* peer_vals, unsigned long long* const* peer_flags,
                                double* nvals, double* mysum, const int* cpeer)
{
    const int p = (int)(e & 1);
    // Publish: my partial, then my flag, each written back.
    vals[p] = x;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(&flags[p], e, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    // Pull every same-node rank's flag in one pass of plain loads (issued
    // back to back, one round trip for all); a system-scope acquire between
    // passes drops the stale lines. Then the partials, in local-index order.
    for (;;) {
        bool all = true;
        for (int j = 0; j < local; ++j) {
            if (j != li && __atomic_load_n(&peer_flags[j][p], __ATOMIC_RELAXED) < e) all = false;
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        if (all) break;
#ifdef __AMDGCN__
        __builtin_amdgcn_s_sleep(1);
#endif
    }
    double node_sum = 0;
    for (int j = 0; j < local; ++j) {
        node_sum += j == li ? x
                  : as_double(__atomic_load_n((unsigned long long*)&peer_vals[j][p], __ATOMIC_RELAXED));
    }
    if (nodes == 1) return node_sum;

    // Across nodes: this node's sum to my counterpart on every other node.
    mysum[p] = node_sum;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    for (int m = 0; m < nodes; ++m) {
        if (m != node) ompx_put_signal(cpeer[m], &nvals[p * nodes + node], &mysum[p],
                                       sizeof(double), sbase + node, e);
    }
    for (int m = 0; m < nodes; ++m) {
        if (m != node) ompx_signal_wait(sbase + m, e);
    }
    double sum = 0;
    for (int m = 0; m < nodes; ++m) sum += m == node ? node_sum : nvals[p * nodes + m];
    return sum;
}
#pragma omp end declare target

// The local dot product left on the device, by the kernel giomp_dot uses
// (a tree per team, the teams' sums added in order by the last team).
void local_dot_device(const double* x, long n, double* team_sum, unsigned* done, double* dpart) {
    const int teams = teams_for(n);
    #pragma omp target teams num_teams(teams) thread_limit(kThreads) \
        is_device_ptr(x, team_sum, done, dpart) firstprivate(n)
    #pragma omp parallel
    {
        static double red[kThreads];
        #pragma omp allocate(red) allocator(omp_pteam_mem_alloc)
        const int t = omp_get_team_num(), nt = omp_get_num_teams();
        const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        double s = 0;
        for (long i = (long)t * nth + tid; i < n; i += (long)nt * nth) s += x[i] * x[i];
        red[tid] = s;
        #pragma omp barrier
        for (int w = nth / 2; w > 0; w /= 2) {
            if (tid < w) red[tid] += red[tid + w];
            #pragma omp barrier
        }
        if (tid == 0) {
            // The team sums stay on this device: device scope. (A system-scope
            // fence here writes back the whole L2 once per team.)
            team_sum[t] = red[0];
            if (__scoped_atomic_add_fetch(done, 1u, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE)
                % (unsigned)nt == 0) {
                double partial = 0;
                for (int k = 0; k < nt; ++k) partial += team_sum[k];
                dpart[0] = partial;
            }
        }
    }
}

double giomp_dot(Giomp& g, const double* x, long n) {
    const Layout& l = g.l;
    const unsigned long long e = ++g.epoch;
    const int p = (int)(e & 1);
    // Under DWQ every staged put_signal holds a signal cell until a quiet.
    if (e % 256 == 0) ompx_quiet();
    // Under DWQ the NIC runs only what the host queued: stage the cross-node
    // puts for the kernel to release. No-op under the CPU proxy.
    for (int m = 0; m < l.nodes; ++m) {
        if (m != l.node)
            ompx_stage_put_signal(g.h_cpeer[m], &g.nvals[p * l.nodes + l.node], &g.mysum[p],
                                  sizeof(double), g.sbase + l.node, e);
    }
    double* out = g.hres;
    const int teams = teams_for(n);
    double* vals = g.vals;
    unsigned long long* flags = g.flags;
    double* nvals = g.nvals;
    double* mysum = g.mysum;
    double* const* peer_vals = g.peer_vals;
    unsigned long long* const* peer_flags = g.peer_flags;
    const int* cpeer = g.cpeer;
    double* team_sum = g.team_sum;
    unsigned* done = g.done;
    const int local = l.local, li = l.li, nodes = l.nodes, node = l.node, sbase = g.sbase;

    #pragma omp target teams num_teams(teams) thread_limit(kThreads) \
        is_device_ptr(out, x, vals, flags, nvals, mysum, peer_vals, peer_flags, cpeer, team_sum, done) \
        firstprivate(n, e, local, li, nodes, node, sbase)
    #pragma omp parallel
    {
        static double red[kThreads];
        #pragma omp allocate(red) allocator(omp_pteam_mem_alloc)
        const int t = omp_get_team_num(), nt = omp_get_num_teams();
        const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        double s = 0;
        for (long i = (long)t * nth + tid; i < n; i += (long)nt * nth) s += x[i] * x[i];
        red[tid] = s;
        #pragma omp barrier
        for (int w = nth / 2; w > 0; w /= 2) {
            if (tid < w) red[tid] += red[tid + w];
            #pragma omp barrier
        }
        if (tid == 0) {
            // Team sums in a fixed order, by the last team to finish.
            // The team sums stay on this device: device scope. (A system-scope
            // fence here writes back the whole L2 once per team.)
            team_sum[t] = red[0];
            if (__scoped_atomic_add_fetch(done, 1u, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE)
                % (unsigned)nt == 0) {
                double partial = 0;
                for (int k = 0; k < nt; ++k) partial += team_sum[k];
                out[0] = global_sum(partial + (double)e, e, local, li, nodes, node, sbase, vals, flags,
                                 peer_vals, peer_flags, nvals, mysum, cpeer);
            }
        }
    }
    return out[0];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s mpi_host|mpi_dot|mpi_dot_pin|mpi_dot_dev|giomp_dot [N] [ITERS]\n",
                     argv[0]);
        return 2;
    }
    const char* mode = argv[1];
    const long n = argc > 2 ? std::atol(argv[2]) : 65536;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 2000;
    const bool giomp = std::strcmp(mode, "giomp_dot") == 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    if (giomp) ompx_init();
    else select_device_mpi();
    const int dev = omp_get_default_device();

    double* x = (double*)omp_target_alloc(n * sizeof(double), dev);
    double* dpart = (double*)omp_target_alloc(sizeof(double), dev);
    fill(x, n);
    Giomp g;
    if (giomp) g = giomp_setup();
    double* team_sum = to_device(std::vector<double>(kTeams, 0.0));
    unsigned* done = to_device(std::vector<unsigned>(1, 0u));
    double* hpart = (double*)llvm_omp_target_alloc_host(sizeof(double), dev);

    auto one = [&]() -> double {
        if (std::strcmp(mode, "mpi_host") == 0) {
            double v = g_rank + 1.0, r = 0;
            MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            return r;
        }
        if (std::strcmp(mode, "mpi_dot") == 0) {
            double v = local_dot(x, n), r = 0;
            MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            return r;
        }
        if (std::strcmp(mode, "mpi_dot_pin") == 0) {
            local_dot_device(x, n, team_sum, done, hpart);
            double v = hpart[0], r = 0;
            MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            return r;
        }
        if (std::strcmp(mode, "mpi_dot_dev") == 0) {
            local_dot_device(x, n, team_sum, done, dpart);
            MPI_Allreduce(MPI_IN_PLACE, dpart, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double r = 0;
            omp_target_memcpy(&r, dpart, sizeof(double), 0, 0, omp_get_initial_device(), dev);
            return r;
        }
        return giomp_dot(g, x, n);
    };

    // Check: the exact sum, and the same bits on every rank. giomp_dot's sum
    // grows by g_size every call (see above); every call is checked.
    const double base = std::strcmp(mode, "mpi_host") == 0
                        ? g_size * (g_size + 1) / 2.0 : exact(n);
    long calls = 0, wrong = 0;
    auto want = [&]() { return giomp ? base + (double)g_size * (double)calls : base; };
    auto checked = [&]() {
        ++calls;
        const double r = one();
        if (std::fabs(r - want()) > 1e-12 * std::fabs(want())) ++wrong;
        return r;
    };
    double got = 0;
    for (int i = 0; i < std::max(1, iters / 10); ++i) got = checked();
    double lo = got, hi = got;
    MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const bool ok = wrong == 0 && lo == hi;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) got = checked();
    double t = (MPI_Wtime() - t0) / iters;
    MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int bad = !ok || wrong != 0;
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (g_rank == 0) {
        std::printf("@@AR mode %s ranks %d n %ld iters %d us %.3f check %s\n", mode, g_size, n,
                    iters, 1e6 * t, bad ? "FAIL" : "ok");
    }
    if (giomp) {
        ompx_quiet();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    omp_target_free(x, dev);
    omp_target_free(dpart, dev);
    omp_target_free(team_sum, dev);
    omp_target_free(done, dev);
    omp_target_free(hpart, dev);
    if (giomp) ompx_finalize();
    MPI_Finalize();
    return bad;
}
