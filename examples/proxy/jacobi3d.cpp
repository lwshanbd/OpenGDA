/*
 * jacobi3d.cpp - a 3D Jacobi stencil whose halo exchange has TWO call sites
 * with naturally different shapes, to test whether the compiler-context
 * decision rule survives contact with a real application loop.
 *
 * Per rank: an (N+2)^3 float field with a one-cell halo, updated by a
 * 7-point stencil. Each timestep exchanges two faces with a ring neighbour:
 *
 *   CONTIG   the z-plane. Contiguous in memory, so it is ONE put of
 *            (N+2)^2 floats. Trip count 1, message large.
 *   STRIDED  the y-face. Its rows are (N+2)^2 floats apart, so without a
 *            packing kernel it is N puts of N floats each -- exactly what a
 *            stencil code that has not packed its faces emits. Trip count N,
 *            messages small.
 *
 * Both call sites are the same operation to the same peer on the same
 * hardware in the same timestep. What differs is trip count and message
 * size, which the compiler can see from the loop bounds and the array
 * layout, and how far the first use is:
 *
 *   --overlap     issue both faces, compute the interior, then wait and
 *                 compute the boundary. Issue-to-first-use distance is the
 *                 interior compute time, which grows as N^3.
 *   (default)     wait immediately. Distance is zero.
 *
 * Enumerates all four per-call-site path assignments and checks every one
 * produces a bit-identical field, so a faster configuration cannot be one
 * that quietly dropped data.
 *
 * Run (>=2 ranks, 1 per node). Do NOT set GICC_SKIP_DWQ_INIT: both paths
 * live in one process, so the trigger BAR has to map.
 *
 *   srun -p pci -N 2 -n 2 --ntasks-per-node=1 -c 8 --gpu-bind=none \
 *     ./jacobi3d --sizes=64,96,128,192 --overlap
 */

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"
#include "gicc/platform/ofi/ofi_runtime.hpp"

enum Path { PROXY = 0, TRIGGER = 1 };

// Absolute reference point: the same two halo faces exchanged with
// GPU-aware MPI. The strided face is sent as N row messages rather than
// packed, so the comparison matches what the GICC path does op for op --
// a packing kernel would change the algorithm, not just the transport.
// Requires MPICH_GPU_SUPPORT_ENABLED=1 and a GTL-linked binary.
static bool g_mpi_mode = false;
// MPI one-sided reference. Two-sided Isend/Irecv gets arrival detection for
// free from message matching, which a one-sided path has to build itself --
// so the like-for-like baseline for GICC's put is MPI_Put over an RMA
// window, where MPI also has to say separately when the data has landed.
static bool g_mpi_rma = false;

// 7-point Jacobi over the interior of an (N+2)^3 field.
//   part 0  every interior point
//   part 1  only points that do NOT touch the halo (safe to compute while
//           the halo is still in flight)
//   part 2  only the boundary shell, i.e. the points part 1 skipped
__global__ void k_jacobi(const float* __restrict__ in, float* __restrict__ out,
                         int N, int part) {
    const int D = N + 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N * N;
    for (; idx < total; idx += blockDim.x * gridDim.x) {
        int x = idx % N,  t = idx / N;
        int y = t % N,    z = t / N;
        bool inner = (x > 0 && x < N-1 && y > 0 && y < N-1 && z > 0 && z < N-1);
        if (part == 1 && !inner) continue;
        if (part == 2 &&  inner) continue;
        int c = ((z + 1) * D + (y + 1)) * D + (x + 1);
        out[c] = (in[c] + in[c - 1] + in[c + 1]
                        + in[c - D] + in[c + D]
                        + in[c - (size_t)D * D] + in[c + (size_t)D * D]) * (1.0f / 7.0f);
    }
}

__global__ void k_init(float* p, size_t n, int rank) {
    for (size_t i = threadIdx.x + (size_t)blockIdx.x * blockDim.x; i < n;
         i += (size_t)blockDim.x * gridDim.x)
        p[i] = (float)((i * 2654435761u + rank * 40503u) % 1024) * 0.001f;
}

// Bit-exactness check across configurations. Sums the RAW BIT PATTERNS as
// integers rather than the floats: a floating-point reduction over blocks
// completes in nondeterministic order, so its rounding differs run to run
// and would flag every configuration as a mismatch.
__global__ void k_checksum(const float* p, size_t n, unsigned long long* out) {
    unsigned long long acc = 0;
    for (size_t i = threadIdx.x + (size_t)blockIdx.x * blockDim.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        unsigned int bits;
        __builtin_memcpy(&bits, &p[i], sizeof(bits));
        acc += (unsigned long long)bits;
    }
    atomicAdd(out, acc);
}

// Proxy issue of the contiguous z-plane: one put.
__global__ void k_px_contig(gicc::DeviceCtx* ctx, int peer, int buf,
                            size_t src_off, size_t dst_off, size_t bytes) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    gicc::put(ctx, peer, buf, dst_off, buf, src_off, bytes);
}

// Proxy issue of the strided y-face: one put per row.
__global__ void k_px_strided(gicc::DeviceCtx* ctx, int peer, int buf,
                             size_t src_off, size_t dst_off, size_t row_bytes,
                             size_t stride_bytes, int rows) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int r = 0; r < rows; ++r)
        gicc::put(ctx, peer, buf,
                  dst_off + (size_t)r * stride_bytes,
                  buf,
                  src_off + (size_t)r * stride_bytes, row_bytes);
}

// Trigger issue: the host staged the descriptors, one flush releases them.
__global__ void k_flush(gicc::DeviceCtx* ctx) { gicc::flush(ctx); }

struct Stats { double median=0, min=0, max=0; int n=0; };
static Stats summarize(std::vector<double> v) {
    Stats s; if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n=(int)v.size(); s.min=v.front(); s.max=v.back();
    s.median = (v.size()%2)? v[v.size()/2] : 0.5*(v[v.size()/2-1]+v[v.size()/2]);
    return s;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    std::vector<int> Ns;
    int samples = 11, warmup = 5, steps = 20;
    bool overlap = false, extra_barrier = false, selftest = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--sizes=",0)==0) {
            std::string c = a.substr(8); size_t p = 0;
            while (p < c.size()) { size_t q = c.find(',', p);
                if (q == std::string::npos) q = c.size();
                Ns.push_back(std::atoi(c.substr(p, q-p).c_str())); p = q+1; }
        }
        else if (a.rfind("--samples=",0)==0) samples = std::atoi(a.c_str()+10);
        else if (a.rfind("--warmup=", 0)==0) warmup  = std::atoi(a.c_str()+9);
        else if (a.rfind("--steps=",  0)==0) steps   = std::atoi(a.c_str()+8);
        else if (a == "--overlap") overlap = true;
        else if (a == "--mpi") g_mpi_mode = true;
        else if (a == "--mpi-rma") { g_mpi_mode = true; g_mpi_rma = true; }
        else if (a == "--extra-barrier") extra_barrier = true;
        else if (a == "--selftest") selftest = true;
        else { fprintf(stderr, "jacobi3d: unknown arg '%s'\n", a.c_str()); return 2; }
    }
    if (Ns.empty()) Ns = {64, 96, 128, 192};

    gicc::Runtime rt;
    rt.enable_host_wait_mode();
    rt.enable_mixed_dispatch();
    const int rank = rt.rank(), nranks = rt.size();
    if (nranks < 2) { if(!rank) fprintf(stderr,"jacobi3d: need >=2 ranks\n"); return 1; }
    const int peer = (rank + 1) % nranks;

    // One registration for the whole sweep: register_buffer/exchange are
    // collective and would otherwise accumulate a window per size.
    int maxN = 0; for (int n : Ns) maxN = std::max(maxN, n);
    const size_t maxFelems = (size_t)(maxN+2) * (maxN+2) * (maxN+2);
    const size_t bufBytes  = 2 * maxFelems * sizeof(float);
    void* d_buf = nullptr;
    if (hipMalloc(&d_buf, bufBytes) != hipSuccess) {
        if (!rank) fprintf(stderr, "jacobi3d: hipMalloc(%zu MB) failed\n",
                           bufBytes >> 20);
        return 3;
    }
    auto bh = rt.register_buffer(d_buf, bufBytes, /*is_device=*/true);
    rt.exchange();

    MPI_Win win = MPI_WIN_NULL;
    if (g_mpi_rma) {
        int err = MPI_Win_create(d_buf, (MPI_Aint)bufBytes, 1,
                                 MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        if (err != MPI_SUCCESS) {
            if (!rank) fprintf(stderr,
                "jacobi3d: MPI_Win_create on device memory failed (%d); "
                "this MPI may not support GPU-buffer RMA\n", err);
            return 4;
        }
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    }
    unsigned long long* d_sum = nullptr;
    (void)hipMalloc((void**)&d_sum, sizeof(unsigned long long));

    if (rank == 0) {
        printf("=== jacobi3d: %d ranks, %d steps/sample, %d samples, "
               "overlap=%s, transport=%s ===\n",
               nranks, steps, samples, overlap ? "on" : "off",
               g_mpi_rma ? "MPI one-sided (MPI_Put + flush)"
                         : (g_mpi_mode ? "GPU-aware MPI (Isend/Irecv)" : "GICC"));
        printf("    window %zu MB registered once for max N=%d\n",
               bufBytes >> 20, maxN);
        printf("%-6s %-10s %10s %10s %10s %10s %9s %9s\n",
               "N", "faces", "px,px", "tr,px", "px,tr", "tr,tr", "best", "gain");
    }

    for (int N : Ns) {
        const int    Dm     = N + 2;
        const size_t plane  = (size_t)Dm * Dm;             // floats in a z-plane
        const size_t felems = plane * Dm;                  // floats per field
        // Two fields (in/out) carved out of the shared window, so both
        // faces' source and destination offsets are expressible.
        float* u0 = (float*)d_buf;
        float* u1 = u0 + felems;

        // CONTIG: rows y=1..N of the z=1 plane -> the peer's z=N+1 halo
        // plane. Still one contiguous put, but it deliberately EXCLUDES the
        // y=0 and y=N+1 rows: those are exactly the cells the strided face
        // below writes, and a full-plane send would read them while the peer
        // was writing them. Real stencil codes avoid the same race by
        // sequencing their dimension exchanges.
        const size_t contig_bytes = (size_t)N * Dm * sizeof(float);
        const size_t contig_src   = (((size_t)1 * Dm) + 1) * Dm * sizeof(float);
        const size_t contig_dst   = (((size_t)(N + 1) * Dm) + 1) * Dm * sizeof(float);
        // STRIDED: the y=1 face -> peer's y=N+1 halo. One put per z-row.
        const int    rows         = N;
        const size_t row_bytes    = (size_t)N * sizeof(float);
        const size_t row_stride   = plane * sizeof(float);
        const size_t strided_src  = (((size_t)1 * Dm + 1) * Dm + 1) * sizeof(float);
        const size_t strided_dst  = (((size_t)1 * Dm + (N + 1)) * Dm + 1) * sizeof(float);

        const int thr = 256, blk = 512;

        int cur = 0;   // which of the two fields is the current input
        auto issue = [&](const int* path) {
            const size_t fo = (size_t)cur * felems * sizeof(float);
            gicc::DeviceCtx* ctx;
            // CONTIG call site
            if (path[0] == TRIGGER) {
                rt.put(bh, peer, bh.index, contig_bytes, fo + contig_src, fo + contig_dst);
                ctx = rt.prepare_slot(0);
                hipLaunchKernelGGL(k_flush, dim3(1), dim3(1), 0, 0, ctx);
            } else {
                ctx = rt.prepare_slot(0);
                hipLaunchKernelGGL(k_px_contig, dim3(1), dim3(1), 0, 0,
                                   ctx, peer, bh.index, fo + contig_src, fo + contig_dst,
                                   contig_bytes);
            }
            // STRIDED call site
            if (path[1] == TRIGGER) {
                for (int r = 0; r < rows; ++r)
                    rt.put(bh, peer, bh.index, row_bytes,
                           fo + strided_src + (size_t)r * row_stride,
                           fo + strided_dst + (size_t)r * row_stride);
                ctx = rt.prepare_slot(1);
                hipLaunchKernelGGL(k_flush, dim3(1), dim3(1), 0, 0, ctx);
            } else {
                ctx = rt.prepare_slot(1);
                hipLaunchKernelGGL(k_px_strided, dim3(1), dim3(1), 0, 0,
                                   ctx, peer, bh.index, fo + strided_src, fo + strided_dst,
                                   row_bytes, row_stride, rows);
            }
        };

        // MPI one-sided: the same faces as MPI_Puts into the peer's window,
        // completed with a flush. Arrival is then a separate problem, the
        // same one GICC has, so the barrier below stands in for it exactly
        // as it does on the GICC path.
        auto mpi_rma_exchange = [&]() {
            const size_t fo = (size_t)cur * felems * sizeof(float);
            char* base = (char*)d_buf;
            MPI_Put(base + fo + contig_src, (int)contig_bytes, MPI_BYTE, peer,
                    (MPI_Aint)(fo + contig_dst), (int)contig_bytes, MPI_BYTE, win);
            for (int i = 0; i < rows; ++i)
                MPI_Put(base + fo + strided_src + (size_t)i * row_stride,
                        (int)row_bytes, MPI_BYTE, peer,
                        (MPI_Aint)(fo + strided_dst + (size_t)i * row_stride),
                        (int)row_bytes, MPI_BYTE, win);
            MPI_Win_flush_all(win);
        };

        // MPI reference: same faces, same message counts, two-sided.
        auto mpi_exchange = [&]() {
            const size_t fo = (size_t)cur * felems * sizeof(float);
            char* base = (char*)d_buf;
            std::vector<MPI_Request> reqs;
            reqs.reserve(2 * (rows + 1));
            const int pred = (rank - 1 + nranks) % nranks;
            MPI_Request r;
            MPI_Irecv(base + fo + contig_dst, contig_bytes, MPI_BYTE, pred,
                      100, MPI_COMM_WORLD, &r); reqs.push_back(r);
            MPI_Isend(base + fo + contig_src, contig_bytes, MPI_BYTE, peer,
                      100, MPI_COMM_WORLD, &r); reqs.push_back(r);
            for (int i = 0; i < rows; ++i) {
                MPI_Irecv(base + fo + strided_dst + (size_t)i * row_stride,
                          row_bytes, MPI_BYTE, pred, 200 + i,
                          MPI_COMM_WORLD, &r); reqs.push_back(r);
                MPI_Isend(base + fo + strided_src + (size_t)i * row_stride,
                          row_bytes, MPI_BYTE, peer, 200 + i,
                          MPI_COMM_WORLD, &r); reqs.push_back(r);
            }
            return reqs;
        };

        auto timestep = [&](const int* path) {
            if (g_mpi_rma) {
                mpi_rma_exchange();
                if (overlap) {
                    hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                       u0, u1, N, 1);
                    (void)hipDeviceSynchronize();
                    rt.barrier();
                    hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                       u0, u1, N, 2);
                } else {
                    rt.barrier();
                    hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                       u0, u1, N, 0);
                }
                (void)hipDeviceSynchronize();
                std::swap(u0, u1);
                cur ^= 1;
                return;
            }
            if (g_mpi_mode) {
                auto reqs = mpi_exchange();
                if (overlap) {
                    hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                       u0, u1, N, 1);
                    (void)hipDeviceSynchronize();
                    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
                    hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                       u0, u1, N, 2);
                } else {
                    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
                    hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                       u0, u1, N, 0);
                }
                (void)hipDeviceSynchronize();
                std::swap(u0, u1);
                cur ^= 1;
                return;
            }
            issue(path);
            if (overlap) {
                // The points that do not touch the halo are updated while the
                // exchange is in flight -- this is the issue-to-first-use
                // distance, and it grows as N^3 while the faces grow as N^2.
                hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                   u0, u1, N, 1);
                (void)hipDeviceSynchronize();
                rt.reset();
                rt.barrier(); if (extra_barrier) rt.barrier();
                hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                   u0, u1, N, 2);
                (void)hipDeviceSynchronize();
            } else {
                (void)hipDeviceSynchronize();
                rt.reset();
                rt.barrier(); if (extra_barrier) rt.barrier();
                hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                   u0, u1, N, 0);
                (void)hipDeviceSynchronize();
            }
            std::swap(u0, u1);
            cur ^= 1;
        };

        auto run_field = [&](const int* path, int nsteps) {
            // Normalize at ENTRY, not just at exit: the timed loops between
            // configurations leave u0/u1 swapped and `cur` at an arbitrary
            // parity, and starting a correctness run from that state makes
            // configurations differ for reasons unrelated to dispatch.
            u0 = (float*)d_buf; u1 = u0 + felems; cur = 0;
            hipLaunchKernelGGL(k_init, dim3(blk), dim3(thr), 0, 0,
                               (float*)d_buf, 2 * felems, rank);
            (void)hipDeviceSynchronize();
            rt.barrier();
            for (int t = 0; t < nsteps; ++t) timestep(path);
            u0 = (float*)d_buf; u1 = u0 + felems; cur = 0;
        };

        auto checksum = [&]() {
            unsigned long long zero = 0;
            (void)hipMemcpy(d_sum, &zero, sizeof(zero), hipMemcpyHostToDevice);
            hipLaunchKernelGGL(k_checksum, dim3(blk), dim3(thr), 0, 0,
                               (float*)d_buf, 2 * felems, d_sum);
            (void)hipDeviceSynchronize();
            unsigned long long h = 0;
            (void)hipMemcpy(&h, d_sum, sizeof(h), hipMemcpyDeviceToHost);
            return h;
        };

        unsigned long long ref_cs = 0;

        if (selftest) {
            // Same configuration, repeated: any variation here is intrinsic
            // nondeterminism, not a difference between dispatch paths.
            for (int m = 0; m < 4; ++m) {
                int pth[2] = { m & 1, (m >> 1) & 1 };
                unsigned long long c1 = 0;
                for (int rep = 0; rep < 3; ++rep) {
                    run_field(pth, 5);
                    unsigned long long c = checksum();
                    if (rep == 0) c1 = c;
                    if (rank == 0)
                        printf("  [selftest] N=%d mask=%d rep=%d cs=%llu %s\n",
                               N, m, rep, c, c == c1 ? "" : "<-- VARIES");
                }
            }
            continue;
        }
        bool ref_set = false; bool all_ok = true;
        double times[4];
        const int nmask = g_mpi_mode ? 1 : 4;
        for (int mask = 0; mask < nmask; ++mask) {
            int path[2] = { (mask >> 0) & 1, (mask >> 1) & 1 };

            // correctness: same field after the same number of steps
            run_field(path, 5);
            unsigned long long cs = checksum();
            if (!ref_set) { ref_cs = cs; ref_set = true; }
            else if (cs != ref_cs) {
                all_ok = false;
                if (!rank) printf("  N=%d mask=%d CHECKSUM MISMATCH %llu vs %llu\n",
                                  N, mask, cs, ref_cs);
            }

            for (int w = 0; w < warmup; ++w) timestep(path);
            std::vector<double> v;
            for (int s = 0; s < samples; ++s) {
                rt.barrier();
                double t0 = MPI_Wtime();
                for (int t = 0; t < steps; ++t) timestep(path);
                double t1 = MPI_Wtime();
                rt.barrier();
                v.push_back((t1 - t0) * 1e6 / steps);
            }
            times[mask] = summarize(std::move(v)).median;
        }
        if (g_mpi_mode) times[1] = times[2] = times[3] = times[0];

        // Cost of the interior kernel alone: the distance the overlap mode buys.
        double comp_us = 0;
        {
            for (int w = 0; w < 3; ++w)
                hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                   u0, u1, N, 1);
            (void)hipDeviceSynchronize();
            double t0 = MPI_Wtime();
            for (int t = 0; t < 10; ++t)
                hipLaunchKernelGGL(k_jacobi, dim3(blk), dim3(thr), 0, 0,
                                   u0, u1, N, 1);
            (void)hipDeviceSynchronize();
            comp_us = (MPI_Wtime() - t0) * 1e6 / 10;
        }

        if (rank == 0) {
            int best = 0;
            for (int m = 1; m < 4; ++m) if (times[m] < times[best]) best = m;
            double glob = std::min(times[0], times[3]);
            char bs[16];
            snprintf(bs, sizeof(bs), "%s,%s",
                     (best & 1) ? "tr" : "px", (best & 2) ? "tr" : "px");
            char faces[24];
            snprintf(faces, sizeof(faces), "%zuKBx1/%dx%zuB",
                     contig_bytes >> 10, rows, row_bytes);
            printf("%-6d %-10s %10.1f %10.1f %10.1f %10.1f %9s %8.2fx%s\n",
                   N, faces, times[0], times[1], times[2], times[3],
                   bs, glob / times[best], all_ok ? "" : "  (CHECKSUM FAIL)");
            printf("CSV,jacobi,%d,%d,%zu,%d,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%llu\n",
                   N, overlap ? 1 : 0, contig_bytes, rows, row_bytes,
                   times[0], times[1], times[2], times[3], comp_us, all_ok ? 1 : 0,
                   (unsigned long long)ref_cs);
        }

    }

    if (win != MPI_WIN_NULL) { MPI_Win_unlock_all(win); MPI_Win_free(&win); }
    rt.barrier();
    (void)hipFree(d_sum);
    (void)hipFree(d_buf);
    return 0;
}
