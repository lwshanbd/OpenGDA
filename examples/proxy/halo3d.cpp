/*
 * halo3d.cpp - a realistic one-sided halo exchange with in-band arrival
 * flags, which is the structure a production GPU-initiated stencil
 * actually uses (see benchmarks/ASF .../src/collective/gicc.cu: "After
 * put+quiet, each sender writes its epoch to the remote's flag via RDMA;
 * the receiver GPU polls its local flags until all expected epochs
 * arrive").
 *
 * Every timestep has TWO communication call sites with opposite shapes:
 *
 *   DATA   F face messages of `bytes` each. Bulk. Its cost is dominated
 *          by wire time, which the proxy's in-kernel pushes pipeline
 *          against, so the proxy's higher per-op issue cost is hidden.
 *
 *   FLAG   F messages of 8 bytes each, one per face, carrying the epoch
 *          that tells the receiver its data has landed. No wire time to
 *          speak of, so nothing hides the per-op issue cost and the
 *          cheaper host staging wins.
 *
 * Both sites issue the same number of operations to the same peers in
 * the same timestep. A runtime sees F puts and F puts. What separates
 * them is message size and what the cost is dominated by -- static
 * facts. The flag site also has to be ORDERED after the data site,
 * which is why it is a separate phase rather than more ops in the same
 * one.
 *
 * jacobi3d fakes arrival with an MPI_Barrier, which a production
 * GPU-initiated halo cannot do; this benchmark signals in band.
 *
 * Enumerates all four (data path, flag path) assignments, verifies every
 * one delivers the right bytes under the right epoch, and compares
 * against GPU-aware MPI with --mpi.
 *
 * Run (>=2 ranks, 1 per node), no GICC_SKIP_DWQ_INIT (the trigger BAR
 * must map):
 *   srun -p pci -N 2 -n 2 --ntasks-per-node=1 -c 8 --gpu-bind=none \
 *     ./halo3d --faces=6 --bytes=65536
 */

#include <mpi.h>

#include <algorithm>
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

static constexpr int    kMaxFaces = 26;
static constexpr size_t kSendBase = 0;
static constexpr size_t kRecvBase = 8u  << 20;
static constexpr size_t kFlagBase = 24u << 20;   // one uint64 per face
static constexpr size_t kBufBytes = 32u << 20;

__device__ __forceinline__ void spin_ticks(long long ticks) {
    if (ticks <= 0) return;
    long long t0 = wall_clock64();
    while ((wall_clock64() - t0) < ticks) { }
}

// Proxy issue of the F bulk face messages.
__global__ void k_px_data(gicc::DeviceCtx* ctx, const int* peers, int faces,
                          int buf, size_t bytes) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int f = 0; f < faces; ++f) {
        size_t so = kSendBase + (size_t)f * bytes;
        size_t doff = kRecvBase + (size_t)f * bytes;
        gicc::put(ctx, peers[f], buf, doff, buf, so, bytes);
    }
    gicc::quiet(ctx);
}

// Proxy issue of the F arrival flags. Ordered after the data by the
// caller: on this path the kernel above already quiesced.
__global__ void k_px_flag(gicc::DeviceCtx* ctx, const int* peers, int faces,
                          int buf, const int* slots) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int f = 0; f < faces; ++f) {
        size_t so = kFlagBase + (size_t)(kMaxFaces + f) * sizeof(uint64_t);
        size_t doff = kFlagBase + (size_t)slots[f] * sizeof(uint64_t);
        gicc::put(ctx, peers[f], buf, doff, buf, so, sizeof(uint64_t));
    }
    gicc::quiet(ctx);
}

__global__ void k_flush(gicc::DeviceCtx* ctx) { gicc::flush(ctx); }

// The structure ASF actually uses on the proxy path: data, an in-kernel
// quiet, then the flags -- all in ONE launch, with no host round trip
// between the two phases. The trigger path cannot express this, because
// its completion wait lives on the host, so this configuration exists
// only for proxy. Without it the comparison forces a two-phase structure
// on a path that does not need one.
__global__ void k_px_fused(gicc::DeviceCtx* ctx, const int* peers,
                           const int* slots, int faces, int buf,
                           size_t bytes) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int f = 0; f < faces; ++f)
        gicc::put(ctx, peers[f], buf, kRecvBase + (size_t)f * bytes,
                  buf, kSendBase + (size_t)f * bytes, bytes);
    gicc::quiet(ctx);                       // data has landed
    for (int f = 0; f < faces; ++f)
        gicc::put(ctx, peers[f], buf,
                  kFlagBase + (size_t)slots[f] * sizeof(uint64_t), buf,
                  kFlagBase + (size_t)(kMaxFaces + f) * sizeof(uint64_t),
                  sizeof(uint64_t));
    gicc::quiet(ctx);
}

// Stamp the outgoing flag payloads with this epoch.
__global__ void k_set_epoch(uint64_t* flags, int faces, uint64_t epoch) {
    if (threadIdx.x != 0) return;
    for (int f = 0; f < faces; ++f) flags[kMaxFaces + f] = epoch;
}

// Receiver: spin until every expected face's flag carries this epoch.
// This is the in-band arrival detection the barrier used to stand in for.
__global__ void k_poll_flags(volatile uint64_t* flags, int faces,
                             uint64_t epoch, long long timeout_ticks,
                             int* stuck_face) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int f = 0; f < faces; ++f) {
        long long t0 = wall_clock64();
        while (flags[f] < epoch) {
            if (wall_clock64() - t0 > timeout_ticks) {
                // Bounded so a missing flag surfaces as a reported face
                // rather than a wedged job.
                if (stuck_face) *stuck_face = f;
                return;
            }
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __threadfence_system();
}

__global__ void k_compute(float* p, size_t n, long long ticks) {
    if (threadIdx.x == 0 && blockIdx.x == 0) spin_ticks(ticks);
    for (size_t i = threadIdx.x + (size_t)blockIdx.x * blockDim.x; i < n;
         i += (size_t)blockDim.x * gridDim.x)
        p[i] = p[i] * 1.0000001f + 1e-8f;
}

// The same compute, but the arrival wait is FUSED into it via
// gicc::wait_signal instead of running as its own kernel. Saves one
// launch per timestep on the critical path, and the block starts
// computing the moment its data lands.
__global__ void k_wait_compute(volatile const unsigned long long* signals,
                               int n, unsigned long long epoch,
                               long long timeout_ticks,
                               int* timed_out,
                               volatile unsigned long long* release,
                               float* p, size_t elems, long long ticks) {
    if (!gicc::wait_signal(signals, n, epoch, timeout_ticks, release)) {
        if (threadIdx.x == 0 && blockIdx.x == 0) *timed_out = 1;
        return;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) spin_ticks(ticks);
    for (size_t i = threadIdx.x + (size_t)blockIdx.x * blockDim.x; i < elems;
         i += (size_t)blockDim.x * gridDim.x)
        p[i] = p[i] * 1.0000001f + 1e-8f;
}

__global__ void k_fill(uint8_t* p, size_t n, uint8_t v) {
    for (size_t i = threadIdx.x + (size_t)blockIdx.x * blockDim.x; i < n;
         i += (size_t)blockDim.x * gridDim.x)
        p[i] = v;
}
__global__ void k_zero64(uint64_t* p, int n) {
    if (threadIdx.x == 0) for (int i = 0; i < n; ++i) p[i] = 0;
}

struct Stats { double median=0, min=0, max=0; int n=0; };
static Stats summarize(std::vector<double> v) {
    Stats s; if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n=(int)v.size(); s.min=v.front(); s.max=v.back();
    s.median=(v.size()%2)?v[v.size()/2]:0.5*(v[v.size()/2-1]+v[v.size()/2]);
    return s;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int faces = 6, samples = 11, warmup = 5, steps = 20, compute_us = 0;
    size_t bytes = 65536;
    bool mpi_mode = false, fused_poll = false;
    int cblocks = 64;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--faces=",0)==0)   faces = atoi(a.c_str()+8);
        else if (a.rfind("--bytes=",0)==0)   bytes = strtoull(a.c_str()+8,nullptr,10);
        else if (a.rfind("--samples=",0)==0) samples = atoi(a.c_str()+10);
        else if (a.rfind("--warmup=",0)==0)  warmup = atoi(a.c_str()+9);
        else if (a.rfind("--steps=",0)==0)   steps = atoi(a.c_str()+8);
        else if (a.rfind("--compute=",0)==0) compute_us = atoi(a.c_str()+10);
        else if (a == "--mpi") mpi_mode = true;
        else if (a == "--fused-poll") fused_poll = true;
        else if (a.rfind("--blocks=",0)==0) cblocks = atoi(a.c_str()+9);
        else { fprintf(stderr,"halo3d: unknown arg '%s'\n",a.c_str()); return 2; }
    }
    if (faces < 1 || faces > kMaxFaces) { fprintf(stderr,"halo3d: faces 1..26\n"); return 2; }

    gicc::Runtime rt;
    rt.enable_host_wait_mode();
    rt.enable_mixed_dispatch();
    const int rank = rt.rank(), nranks = rt.size();
    if (nranks < 2) { if(!rank) fprintf(stderr,"halo3d: need >=2 ranks\n"); return 1; }

    // Face f goes to the f-th distinct successor. With 2 ranks every face
    // lands on the same peer, which changes the peer distribution but not
    // the call-site structure this benchmark is about.
    std::vector<int> h_peers(faces), h_slot(faces);
    for (int f = 0; f < faces; ++f) {
        h_peers[f] = (rank + 1 + (f % (nranks - 1))) % nranks;
        // Slot the receiver will find my flag in: my index among its faces.
        h_slot[f]  = f;
    }
    int* d_peers = nullptr; int* d_slots = nullptr;
    (void)hipMalloc((void**)&d_peers, faces * sizeof(int));
    (void)hipMalloc((void**)&d_slots, faces * sizeof(int));
    (void)hipMemcpy(d_peers, h_peers.data(), faces*sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_slots, h_slot.data(), faces*sizeof(int), hipMemcpyHostToDevice);
    // The flag slot is the face index, which is unambiguous only when a
    // receiver gets at most one message per face index -- true for 2
    // ranks. Generalizing needs the sender identity in the slot, which is
    // bookkeeping, not a change to the call-site structure under test.
    if (nranks > 2 && rank == 0)
        fprintf(stderr, "halo3d: NOTE flag slots assume 2 ranks; "
                        "flag verification is only meaningful there\n");

    void* d_buf = nullptr;
    if (hipMalloc(&d_buf, kBufBytes) != hipSuccess) return 3;
    auto bh = rt.register_buffer(d_buf, kBufBytes, /*is_device=*/true);
    rt.exchange();
    uint64_t* d_flags = (uint64_t*)((char*)d_buf + kFlagBase);
    // The compute workspace must NOT alias the receive region, or the
    // stencil update overwrites the halo that just arrived and every
    // configuration fails verification for a reason unrelated to
    // transport.
    float* d_work = nullptr;
    (void)hipMalloc((void**)&d_work, (size_t)(1 << 16) * sizeof(float));

    int wall_khz = 0;
    (void)hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0);
    const long long ticks = (long long)(compute_us * (wall_khz>0?wall_khz/1000.0:25.0));
    const size_t work_elems = 1 << 16;
    const long long poll_timeout_ticks =
        (long long)((wall_khz > 0 ? wall_khz/1000.0 : 25.0) * 2e6);  // ~2 s
    int* d_stuck = nullptr;
    (void)hipMalloc((void**)&d_stuck, sizeof(int));
    unsigned long long* d_release = nullptr;
    (void)hipMalloc((void**)&d_release, sizeof(unsigned long long));
    (void)hipMemset(d_release, 0, sizeof(unsigned long long));
    (void)hipMemset(d_stuck, 0xFF, sizeof(int));

    (void)rt.prepare(); rt.reset(); rt.barrier();

    uint64_t epoch = 0;

    // One timestep: bulk faces, then the ordered arrival flags, then the
    // receiver waits on the flags and computes.
    bool fused = false, chained = false;
    auto timestep = [&](const int* path) {
        ++epoch;
        hipLaunchKernelGGL(k_set_epoch, dim3(1), dim3(64), 0, 0,
                           d_flags, faces, epoch);

        if (chained) {
            // Both phases staged before a single trigger: the payload waits
            // on the MMIO counter, the flags wait on the payload's
            // COMPLETION counter, so the NIC releases them itself and the
            // host never round-trips between the two.
            for (int f = 0; f < faces; ++f)
                rt.put(bh, h_peers[f], bh.index, bytes,
                       kSendBase + (size_t)f*bytes,
                       kRecvBase + (size_t)f*bytes);
            const uint64_t after_data = rt.staged_ops();
            for (int f = 0; f < faces; ++f)
                rt.put_after(bh, h_peers[f], bh.index, sizeof(uint64_t),
                             kFlagBase + (size_t)(kMaxFaces+f)*sizeof(uint64_t),
                             kFlagBase + (size_t)h_slot[f]*sizeof(uint64_t),
                             after_data);
            gicc::DeviceCtx* c = rt.prepare_slot(0);
            hipLaunchKernelGGL(k_flush, dim3(1), dim3(1), 0, 0, c);
            // No device sync: nothing was pushed to the proxy ring, and the
            // completion counter cannot advance until the kernel fired the
            // trigger, so the wait below already subsumes it.
            rt.reset_dwq();
            if (!fused_poll)
                hipLaunchKernelGGL(k_poll_flags, dim3(1), dim3(1), 0, 0,
                                   (volatile uint64_t*)d_flags, faces, epoch,
                                   poll_timeout_ticks, d_stuck);
        } else if (fused) {
            gicc::DeviceCtx* c = rt.prepare_slot(0);
            hipLaunchKernelGGL(k_px_fused, dim3(1), dim3(1), 0, 0,
                               c, d_peers, d_slots, faces, bh.index, bytes);
            (void)hipDeviceSynchronize();
            rt.reset();
            hipLaunchKernelGGL(k_poll_flags, dim3(1), dim3(1), 0, 0,
                               (volatile uint64_t*)d_flags, faces, epoch,
                               poll_timeout_ticks, d_stuck);
        } else if (mpi_mode) {
            std::vector<MPI_Request> rq;
            for (int f = 0; f < faces; ++f) {
                MPI_Request r;
                MPI_Irecv((char*)d_buf + kRecvBase + (size_t)f*bytes, bytes,
                          MPI_BYTE, MPI_ANY_SOURCE, 100+f, MPI_COMM_WORLD, &r);
                rq.push_back(r);
                MPI_Isend((char*)d_buf + kSendBase + (size_t)f*bytes, bytes,
                          MPI_BYTE, h_peers[f], 100+f, MPI_COMM_WORLD, &r);
                rq.push_back(r);
            }
            MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
            // Two-sided completion IS the arrival signal; no flag needed.
        } else {
            // ---- DATA phase ----
            if (path[0] == TRIGGER) {
                for (int f = 0; f < faces; ++f)
                    rt.put(bh, h_peers[f], bh.index, bytes,
                           kSendBase + (size_t)f*bytes,
                           kRecvBase + (size_t)f*bytes);
                gicc::DeviceCtx* c = rt.prepare_slot(0);
                hipLaunchKernelGGL(k_flush, dim3(1), dim3(1), 0, 0, c);
            } else {
                gicc::DeviceCtx* c = rt.prepare_slot(0);
                hipLaunchKernelGGL(k_px_data, dim3(1), dim3(1), 0, 0,
                                   c, d_peers, faces, bh.index, bytes);
            }
            (void)hipDeviceSynchronize();
            rt.reset();          // data has landed at the peers

            // ---- FLAG phase (must follow the data) ----
            if (path[1] == TRIGGER) {
                for (int f = 0; f < faces; ++f)
                    rt.put(bh, h_peers[f], bh.index, sizeof(uint64_t),
                           kFlagBase + (size_t)(kMaxFaces+f)*sizeof(uint64_t),
                           kFlagBase + (size_t)h_slot[f]*sizeof(uint64_t));
                gicc::DeviceCtx* c = rt.prepare_slot(1);
                hipLaunchKernelGGL(k_flush, dim3(1), dim3(1), 0, 0, c);
            } else {
                gicc::DeviceCtx* c = rt.prepare_slot(1);
                hipLaunchKernelGGL(k_px_flag, dim3(1), dim3(1), 0, 0,
                                   c, d_peers, faces, bh.index, d_slots);
            }
            (void)hipDeviceSynchronize();
            rt.reset();

            // ---- receiver waits in band, then computes ----
            if (!fused_poll)
                hipLaunchKernelGGL(k_poll_flags, dim3(1), dim3(1), 0, 0,
                                   (volatile uint64_t*)d_flags, faces, epoch,
                                   poll_timeout_ticks, d_stuck);
        }
        if (fused_poll && !mpi_mode) {
            hipLaunchKernelGGL(k_wait_compute, dim3(cblocks), dim3(256), 0, 0,
                               (volatile const unsigned long long*)d_flags,
                               faces, (unsigned long long)epoch,
                               poll_timeout_ticks, d_stuck, d_release,
                               d_work, work_elems, ticks);
        } else {
            hipLaunchKernelGGL(k_compute, dim3(cblocks), dim3(256), 0, 0,
                               d_work, work_elems, ticks);
        }
        (void)hipDeviceSynchronize();
    };

    auto reset_state = [&]() {
        epoch = 0;
        hipLaunchKernelGGL(k_fill, dim3(64), dim3(256), 0, 0,
                           (uint8_t*)d_buf + kSendBase,
                           (size_t)faces*bytes, (uint8_t)(0xB0 + rank));
        hipLaunchKernelGGL(k_fill, dim3(64), dim3(256), 0, 0,
                           (uint8_t*)d_buf + kRecvBase,
                           (size_t)faces*bytes, (uint8_t)0);
        hipLaunchKernelGGL(k_zero64, dim3(1), dim3(64), 0, 0, d_flags, 2*kMaxFaces);
        (void)hipMemset(d_release, 0, sizeof(unsigned long long));
        (void)hipDeviceSynchronize();
        rt.barrier();
    };

    // A configuration is correct only if the bytes arrived AND the flag
    // that announced them carries the right epoch.
    auto verify = [&](const int* path) {
        // Clear the stuck marker, or a later configuration inherits an
        // earlier one's failure.
        (void)hipMemset(d_stuck, 0xFF, sizeof(int));
        reset_state();
        for (int t = 0; t < 3; ++t) timestep(path);
        rt.barrier();
        int ok = 1;
        int stuck = -1;
        (void)hipMemcpy(&stuck, d_stuck, sizeof(int), hipMemcpyDeviceToHost);
        if (stuck >= 0) {
            if (fused_poll)
                printf("[verify] rank %d: fused wait timed out\n", rank);
            else
                printf("[verify] rank %d: flag for face %d never arrived\n",
                       rank, stuck);
            ok = 0;
        }
        std::vector<uint8_t> h(bytes);
        std::vector<uint64_t> hf(kMaxFaces);
        (void)hipMemcpy(hf.data(), d_flags, kMaxFaces*sizeof(uint64_t),
                        hipMemcpyDeviceToHost);
        for (int f = 0; f < faces && ok; ++f) {
            if (!mpi_mode && hf[f] != epoch) {
                printf("[verify] rank %d face %d flag=%llu want=%llu\n", rank, f,
                       (unsigned long long)hf[f], (unsigned long long)epoch);
                ok = 0; break;
            }
            (void)hipMemcpy(h.data(), (uint8_t*)d_buf + kRecvBase + (size_t)f*bytes,
                            bytes, hipMemcpyDeviceToHost);
            for (size_t i = 0; i < bytes; i += 512)
                if ((h[i] & 0xF0) != 0xB0) {
                    printf("[verify] rank %d face %d data 0x%02x\n", rank, f, h[i]);
                    ok = 0; break;
                }
        }
        int all_ok = 0;
        MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return all_ok != 0;
    };

    auto measure = [&](const int* path) {
        reset_state();
        for (int w = 0; w < warmup; ++w) timestep(path);
        std::vector<double> v;
        for (int s = 0; s < samples; ++s) {
            rt.barrier();
            double t0 = MPI_Wtime();
            for (int t = 0; t < steps; ++t) timestep(path);
            double t1 = MPI_Wtime();
            v.push_back((t1-t0)*1e6/steps);
        }
        return summarize(std::move(v));
    };

    if (rank == 0) {
        printf("=== halo3d: %d ranks, %d faces x %zuB + %d flags x 8B, "
               "compute=%dus, transport=%s ===\n",
               nranks, faces, bytes, faces, compute_us,
               mpi_mode ? "GPU-aware MPI"
                        : (fused_poll ? "GICC (wait fused into compute)"
                                      : "GICC (separate poll kernel)"));
        printf("%-16s %14s %10s\n", "data,flag", "us/timestep", "verify");
    }

    double t[6]; bool okall = true;
    const int nmask = mpi_mode ? 1 : 6;
    for (int mask = 0; mask < nmask; ++mask) {
        // mask 4 is the fused proxy structure and mask 5 is the chained
        // trigger, both of which avoid the two-phase host round trip; the
        // four below are (data path, flag path).
        fused   = (mask == 4);
        chained = (mask == 5);
        int path[2] = { mask & 1, (mask >> 1) & 1 };
        bool ok = verify(path);
        okall &= ok;
        t[mask] = measure(path).median;
        if (rank == 0) {
            char lbl[32];
            if (chained) snprintf(lbl, sizeof(lbl), "trig chained");
            else if (fused) snprintf(lbl, sizeof(lbl), "prox,prox (fused)");
            else snprintf(lbl, sizeof(lbl), "%s,%s",
                          path[0]?"trig":"prox", path[1]?"trig":"prox");
            printf("%-16s %14.2f %10s\n", mpi_mode ? "mpi" : lbl, t[mask],
                   ok ? "pass" : "FAIL");
        }
    }
    if (mpi_mode) { t[1]=t[2]=t[3]=t[4]=t[5]=t[0]; }

    if (rank == 0) {
        int best = 0;
        for (int m = 1; m < 6; ++m) if (t[m] < t[best]) best = m;
        // A single global policy may also use the fused proxy structure,
        // so it is a candidate for "best global" too.
        double glob = std::min(std::min(std::min(t[0], t[3]), t[4]), t[5]);
        printf("\n  best global (one path everywhere): %8.2f us\n", glob);
        printf("  best overall                     : %8.2f us  (%s)\n",
               t[best], best == 5 ? "trigger, NIC-chained flags"
                        : best == 4 ? "prox,prox fused"
                                    : ((best&1)?"trig,":"prox,"));
        printf("  per-call-site gain               : %8.3fx (%.1f%%)\n",
               glob/t[best], (glob/t[best]-1)*100);
        printf("CSV,halo3d,%d,%zu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d\n",
               faces, bytes, compute_us, t[0], t[1], t[2], t[3], t[4], t[5],
               mpi_mode?1:0, okall?1:0);
    }

    rt.barrier();
    (void)hipFree(d_buf); (void)hipFree(d_peers); (void)hipFree(d_slots); (void)hipFree(d_stuck); (void)hipFree(d_work); (void)hipFree(d_release);
    return 0;
}
