// pipeline_intranode.cpp - does splitting "compute, then send" into blocks pay
// off inside one node, and does it stay bit-exact?
//
// Every rank produces n words and sends them to its right neighbour on the
// ring, all ranks at once, each iteration:
//
//     for i: src[i] = produce(rank, iter, i)      // compute
//     send src -> right neighbour's dst           // communication
//     fence
//
// Same node, so "send" is a store through ompx_peer_ptr (NVLink / xGMI) or a
// copy-engine copy. Variants:
//
//   compute-only / copy-only  compute alone, copy alone: the max(compute,
//                copy) bound pipelining can at best reach
//
//   serial-ce    compute kernel, then host ompx_put (copy engine). The
//                program as a user writes it.
//   serial-kc    compute kernel, then a copy kernel storing through the peer
//                pointer (what the library's put_auto does on the node).
//   compiled     the serial-ce kernel with the send stated inside it as
//                ompx_pipelined_put; the gicc-passes plugin (chunk-lower)
//                proves it splittable and rewrites it: at element grain
//                every word is stored to the peer as it is produced, at block
//                grain each dist_schedule block is sent as it completes.
//                Nothing about the pipelining is written by hand.
//   compute-teams  the compiled kernel's shape without the send, to tell
//                what the shape costs from what sending costs.
//   nobarrier    pipelined without waiting for the block to be complete:
//                threads send words other threads may not have written yet
//                (the copy runs in reverse order so they are other threads'
//                words). Breaks the analysis's "send after the block joins"
//                rule; a control that must fail verification, showing the
//                check can see a torn block.
//
// produce() is integer-only, so every variant must deliver the same bits.
// Verification checks every received word against the value the left
// neighbour should have produced in that very iteration, so a missing,
// stale or torn block is caught.
//
// Build (the pass must run on the device compile, or ompx_pipelined_put
// does not link; GICC_CHUNK_GRAIN=element|block picks the grain):
//   GICC_MODE=chunk-lower GIOMP_BACKEND=cuda \
//   GIOMP_EXTRA_FLAGS="-foffload-lto -fpass-plugin=<build>/libgicc-passes.so" \
//   bash examples/omp/build_giomp_example.sh examples/omp/pipeline_intranode.cpp OUT
// Run  : GICC_HALO_IPC=1 GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
//            srun -N1 -n4 --gpus-per-node=4 --gpu-bind=none ./OUT [options]
//        With GICC_HALO_IPC=0 nothing is IPC-mapped: only serial-ce and
//        compiled run, the latter through its proxy fallback.
// Options: --n WORDS --work ROUNDS | --works R1,R2,... --teams T --iters I
//          --warmup W --verify-iters V --chunks C1,C2,...
// Exit code 1 when a checked variant loses a word or the control does not.

#include "gicc/omp.h"
#include "gicc/omp_pipeline.h"
#include <omp.h>
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#pragma omp declare target
static inline uint32_t produce(int rank, int iter, size_t i, int work) {
    uint32_t x = (uint32_t)rank * 0x9E3779B1u ^ (uint32_t)iter * 0x85EBCA77u ^
                 (uint32_t)i * 0xC2B2AE3Du ^ (uint32_t)(i >> 32);
    for (int k = 0; k < work; ++k) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x += 0x6D2B79F5u;
    }
    return x;
}
#pragma omp end declare target

struct Opts {
    size_t n = size_t(64) << 20;
    int work = 64;
    std::vector<int> works;
    int teams = 528;
    int iters = 20;
    int warmup = 3;
    int verify_iters = 3;
    std::vector<size_t> chunks = {size_t(1) << 14, size_t(1) << 16,
                                  size_t(1) << 18, size_t(1) << 20};
};

static Opts parse(int argc, char** argv) {
    Opts o;
    for (int a = 1; a < argc; a += 2) {
        std::string k = argv[a];
        if (a + 1 == argc) {
            std::fprintf(stderr, "option %s needs a value\n", k.c_str());
            std::exit(2);
        }
        const char* v = argv[a + 1];
        if (k == "--n") o.n = std::strtoull(v, nullptr, 0);
        else if (k == "--work") o.work = std::atoi(v);
        else if (k == "--works") {
            for (char* s = std::strtok(const_cast<char*>(v), ","); s;
                 s = std::strtok(nullptr, ","))
                o.works.push_back(std::atoi(s));
        }
        else if (k == "--teams") o.teams = std::atoi(v);
        else if (k == "--iters") o.iters = std::atoi(v);
        else if (k == "--warmup") o.warmup = std::atoi(v);
        else if (k == "--verify-iters") o.verify_iters = std::atoi(v);
        else if (k == "--chunks") {
            o.chunks.clear();
            for (char* s = std::strtok(const_cast<char*>(v), ","); s;
                 s = std::strtok(nullptr, ","))
                o.chunks.push_back(std::strtoull(s, nullptr, 0));
        } else {
            std::fprintf(stderr, "unknown option %s\n", k.c_str());
            std::exit(2);
        }
    }
    return o;
}

struct Ctx {
    uint32_t* src;
    uint32_t* dst;
    uint32_t* peer_dst;   // right neighbour's dst, through IPC
    int me, left, right, np;
};

enum class Variant { ComputeOnly, ComputeTeams, CopyOnly, SerialCE, SerialKC, Compiled, NoBarrier };

static const char* name(Variant v) {
    switch (v) {
        case Variant::ComputeOnly: return "compute-only";
        case Variant::ComputeTeams: return "compute-teams";
        case Variant::CopyOnly:    return "copy-only";
        case Variant::SerialCE:    return "serial-ce";
        case Variant::SerialKC:    return "serial-kc";
        case Variant::Compiled:    return "compiled";
        case Variant::NoBarrier:   return "nobarrier";
    }
    return "?";
}

static void compute(const Ctx& c, const Opts& o, int it) {
    uint32_t* src = c.src;
    const size_t n = o.n;
    const int me = c.me, work = o.work;
    #pragma omp target teams distribute parallel for num_teams(o.teams) \
            is_device_ptr(src) firstprivate(n, me, it, work)
    for (size_t i = 0; i < n; ++i) src[i] = produce(me, it, i, work);
}

static void copy_kernel(const Ctx& c, const Opts& o) {
    uint32_t* src = c.src;
    uint32_t* pd = c.peer_dst;
    const size_t n = o.n;
    #pragma omp target teams distribute parallel for num_teams(o.teams) \
            is_device_ptr(src, pd) firstprivate(n)
    for (size_t i = 0; i < n; ++i) pd[i] = src[i];
}

// One iteration's compute + send; the caller fences.
static void step(Variant v, const Ctx& c, const Opts& o, int it, size_t chunk) {
    uint32_t* src = c.src;
    uint32_t* pd = c.peer_dst;
    const size_t n = o.n;
    const int me = c.me, work = o.work;
    switch (v) {
    case Variant::ComputeOnly:
        compute(c, o, it);
        break;
    case Variant::ComputeTeams:
        // The compiled variant's kernel shape (teams region around a
        // dist_schedule loop) without the send: separates what the shape
        // costs from what sending costs.
        #pragma omp target teams num_teams(o.teams) is_device_ptr(src) \
                firstprivate(n, me, it, work, chunk)
        {
            #pragma omp distribute parallel for dist_schedule(static, chunk)
            for (size_t i = 0; i < n; ++i) src[i] = produce(me, it, i, work);
        }
        break;
    case Variant::CopyOnly:
        copy_kernel(c, o);
        break;
    case Variant::SerialCE:
        compute(c, o, it);
        ompx_put(c.right, c.dst, c.src, n * sizeof(uint32_t));
        break;
    case Variant::SerialKC:
        compute(c, o, it);
        copy_kernel(c, o);
        break;
    case Variant::Compiled: {
        uint32_t* dst = c.dst;
        const int right = c.right;
        #pragma omp target teams num_teams(o.teams) is_device_ptr(src, dst) \
                firstprivate(n, me, it, work, chunk, right)
        {
            #pragma omp distribute parallel for dist_schedule(static, chunk)
            for (size_t i = 0; i < n; ++i) src[i] = produce(me, it, i, work);
            ompx_pipelined_put(right, dst, src, n * sizeof(uint32_t));
        }
        break;
    }
    case Variant::NoBarrier: {
        const size_t nblk = (n + chunk - 1) / chunk;
        #pragma omp target teams num_teams(o.teams) is_device_ptr(src, pd) \
                firstprivate(n, me, it, work, chunk, nblk)
        {
            #pragma omp distribute dist_schedule(static, 1)
            for (size_t b = 0; b < nblk; ++b) {
                const size_t lo = b * chunk;
                const size_t hi = lo + chunk < n ? lo + chunk : n;
                #pragma omp parallel
                {
                    #pragma omp for nowait
                    for (size_t i = lo; i < hi; ++i) src[i] = produce(me, it, i, work);
                    // BUG on purpose: no barrier, and word j is sent by
                    // the thread that produced word hi-1-(j-lo).
                    #pragma omp for
                    for (size_t i = lo; i < hi; ++i) pd[hi - 1 - (i - lo)] = src[hi - 1 - (i - lo)];
                }
            }
        }
        break;
    }
    }
}

// Words of dst that differ from what the left neighbour produced in `it`.
static unsigned long long check(const Ctx& c, const Opts& o, int it) {
    const uint32_t* dst = c.dst;
    const size_t n = o.n;
    const int left = c.left, work = o.work;
    unsigned long long bad = 0;
    #pragma omp target teams distribute parallel for num_teams(o.teams) \
            reduction(+ : bad) is_device_ptr(dst) firstprivate(n, left, it, work)
    for (size_t i = 0; i < n; ++i) bad += dst[i] != produce(left, it, i, work);
    return bad;
}

static void clear_dst(const Ctx& c, const Opts& o) {
    uint32_t* dst = c.dst;
    const size_t n = o.n;
    #pragma omp target teams distribute parallel for is_device_ptr(dst) firstprivate(n)
    for (size_t i = 0; i < n; ++i) dst[i] = 0xDEADBEEFu;
}

static bool sends(Variant v) { return v != Variant::ComputeOnly && v != Variant::ComputeTeams; }
static bool produces(Variant v) { return v != Variant::CopyOnly; }

// Runs verify_iters checked iterations, then timed iterations. Returns the
// mean iteration time (ms, slowest rank) and the total bad words (all ranks).
static void run(Variant v, const Ctx& c, const Opts& o, size_t chunk,
                double& ms, unsigned long long& bad) {
    bad = 0;
    if (sends(v) && produces(v)) {
        for (int it = 0; it < o.verify_iters; ++it) {
            clear_dst(c, o);
            ompx_barrier();                 // nobody sends into a dst being cleared
            step(v, c, o, 1000 + it, chunk);
            ompx_fence();                   // all sends of this epoch landed
            bad += check(c, o, 1000 + it);
            ompx_barrier();                 // nobody sends into a dst being checked
        }
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
    }
    ms = 0;
    if (v == Variant::NoBarrier) return;   // a correctness control only
    for (int it = 0; it < o.warmup; ++it) { step(v, c, o, it, chunk); ompx_fence(); }
    const double t0 = omp_get_wtime();
    for (int it = 0; it < o.iters; ++it) { step(v, c, o, it, chunk); ompx_fence(); }
    double dt = (omp_get_wtime() - t0) / o.iters * 1e3;
    MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    ms = dt;
}

int main(int argc, char** argv) {
    Opts o = parse(argc, argv);
    ompx_init();
    Ctx c;
    c.me = ompx_get_rank_num();
    c.np = ompx_get_num_ranks();
    c.right = (c.me + 1) % c.np;
    c.left = (c.me + c.np - 1) % c.np;

    const size_t bytes = o.n * sizeof(uint32_t);
    c.src = (uint32_t*)ompx_alloc(bytes);
    c.dst = (uint32_t*)ompx_alloc(bytes);
    c.peer_dst = (uint32_t*)ompx_peer_ptr(c.right, c.dst);
    if (c.np < 2) {
        std::fprintf(stderr, "need at least 2 ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Without an IPC mapping (GICC_HALO_IPC=0, or a peer on another node)
    // only the variants that do not store through the peer pointer run, and
    // the compiled one takes its non-IPC fallback: one proxy put per block.
    int mapped = c.peer_dst != nullptr, all_mapped = 0;
    MPI_Allreduce(&mapped, &all_mapped, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ompx_prepare();   // publishes the device context the block sends use
    ompx_barrier();

    if (o.works.empty()) o.works.push_back(o.work);

    // A variant that must deliver every word FAILs on any bad word; the
    // nobarrier control FAILs if it delivers them all, which would mean the
    // check could not have seen a torn block. Either makes the exit code 1.
    int failures = 0;
    auto line = [&](Variant v, size_t chunk) {
        double ms;
        unsigned long long bad;
        run(v, c, o, chunk, ms, bad);
        const bool checked = sends(v) && produces(v);
        const bool control = v == Variant::NoBarrier;
        const bool ok = !checked || (control ? bad != 0 : bad == 0);
        failures += !ok;
        if (c.me != 0) return;
        char ch[32] = "-";
        if (v == Variant::Compiled || v == Variant::NoBarrier || v == Variant::ComputeTeams)
            std::snprintf(ch, sizeof ch, "%zu", chunk);
        std::string verdict = !checked ? "(not checked)"
                            : "bad=" + std::to_string(bad) + (ok ? " PASS" : " FAIL") +
                                  (control ? " (control: must be > 0)" : "");
        if (ms > 0)
            std::printf("%-13s chunk=%-8s %9.3f ms  %8.1f GB/s  %s\n", name(v), ch, ms,
                        bytes / (ms * 1e6), verdict.c_str());
        else
            std::printf("%-13s chunk=%-8s %9s     %8s       %s\n", name(v), ch, "-", "-",
                        verdict.c_str());
        std::fflush(stdout);
    };

    for (int w : o.works) {
        o.work = w;
        if (c.me == 0)
            std::printf("\nranks=%d n=%zu words (%.1f MB) work=%d teams=%d iters=%d "
                        "verify_iters=%d%s\n", c.np, o.n, bytes / 1e6, o.work, o.teams,
                        o.iters, o.verify_iters,
                        all_mapped ? "" : " (peer not IPC-mapped: fallback path)");
        line(Variant::ComputeOnly, 0);
        for (size_t ch : o.chunks) line(Variant::ComputeTeams, ch);
        if (all_mapped) line(Variant::CopyOnly, 0);
        line(Variant::SerialCE, 0);
        if (all_mapped) line(Variant::SerialKC, 0);
        for (size_t ch : o.chunks) line(Variant::Compiled, ch);
        if (all_mapped) line(Variant::NoBarrier, o.chunks.front());
    }

    MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    ompx_barrier();
    ompx_free(c.dst);
    ompx_free(c.src);
    ompx_finalize();
    return failures ? 1 : 0;
}
