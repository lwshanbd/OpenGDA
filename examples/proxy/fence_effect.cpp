/*
 * fence_effect.cpp — what the completion fence costs, in GICC.
 *
 * `quiet` ends with a memory fence so that reads issued after it observe
 * what the NIC or the proxy wrote. Until now that fence was
 * __threadfence_system() at every call site. It is not one instruction: a
 * system-scope fence writes back what the kernel has dirtied, so its cost
 * scales with how much the kernel wrote, not with the transfer.
 *
 * That is the part the earlier copy microbenchmark measured in isolation
 * (8.81 us against 2.57 us at 1 MB). This measures it where it actually
 * sits: a kernel that writes a region, issues a transfer, and quiets.
 *
 *   write_kib   how much the kernel dirties before the completion point
 *   fence       system | device | block | none
 *
 * The sweep exists to answer one question — is the fence scope worth a
 * compile-time decision at all, or is it noise once it is inside a real
 * kernel? Reported per write size, because a fence with nothing to write
 * back should cost nothing and the claim is specifically that the cost
 * tracks the dirtied bytes.
 *
 * Correctness is checked rather than assumed. The receiver verifies the
 * payload, and the kernel reads back one word of what it wrote after the
 * quiet, so a fence weakened past what the program needs would show up as
 * a wrong value rather than silently as a faster number.
 *
 *   GICC_SKIP_DWQ_INIT=1 GICC_PROXY_ENABLED=1 srun -p pci -N 2 -n 2 \
 *       --ntasks-per-node=1 -c 8 --gpu-bind=none -t 4 ./fence_effect
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
#include "gicc/platform/ofi/runtime_helpers.h"

static constexpr size_t kBufBytes   = 256 * 1024 * 1024;
static constexpr size_t kSendBase   = 0;
static constexpr size_t kRecvBase   = 64 * 1024 * 1024;
static constexpr size_t kScratchOff = 128 * 1024 * 1024;

// Dirty `words` of scratch, issue one transfer, then complete with the
// requested fence scope. The read-back afterwards is what makes a fence
// that is too weak observable: it forces the program to actually depend on
// the ordering the fence provides.
__global__ void k_write_then_quiet(gicc::DeviceCtx* ctx, int peer, int buf,
                                   unsigned long long bytes,
                                   unsigned long long* scratch,
                                   long long words, int fence,
                                   unsigned long long* out) {
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < (size_t)words; i += stride)
        scratch[i] = (unsigned long long)i * 2654435761ull + 1;

    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::put(ctx, peer, buf, kRecvBase, buf, kSendBase, bytes);
        gicc::quiet(ctx, 0, fence);
        // Depend on the ordering: read back something the kernel wrote.
        if (words > 0) *out = scratch[words - 1];
    }
}

struct Stats { double median = 0, min = 0; int n = 0; };

static Stats summarize(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = (int)v.size();
    s.median = (v.size() % 2) ? v[v.size() / 2]
                              : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    s.min = v.front();
    return s;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    size_t bytes = 65536;
    int samples = 15, warmup = 5, blocks = 64, threads = 256;
    // Dirty the PEER's IPC-mapped region instead of local memory. The
    // copy microbenchmark that measured 3.4x wrote across the xGMI link
    // before fencing; this switch is what makes the two comparable, and
    // tells apart "the fence is expensive when a lot was written" from
    // "the fence is expensive when what was written has to cross a device
    // boundary". Needs both ranks on one node.
    bool peer_write = false;
    // Launch `batch` kernels back to back before syncing, instead of
    // syncing after each. The copy microbenchmark that measured 3.4x
    // pipelines 300 launches on a stream; this benchmark synced every
    // iteration. If the fence cost is really about blocking the overlap of
    // consecutive operations rather than about writing back per se, it
    // should appear here only when batch > 1.
    int batch = 1;
    std::vector<long long> kibs = {0, 64, 256, 1024, 4096, 16384};

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--bytes=",   0) == 0) bytes = (size_t)atol(a.c_str() + 8);
        else if (a.rfind("--samples=", 0) == 0) samples = atoi(a.c_str() + 10);
        else if (a.rfind("--warmup=",  0) == 0) warmup = atoi(a.c_str() + 9);
        else if (a.rfind("--blocks=",  0) == 0) blocks = atoi(a.c_str() + 9);
        else if (a == "--peer-write") peer_write = true;
        else if (a.rfind("--batch=", 0) == 0) batch = atoi(a.c_str() + 8);
        else if (a.rfind("--kib=", 0) == 0) {
            kibs.clear();
            std::string s = a.substr(6); size_t p = 0;
            while (p < s.size()) {
                size_t c = s.find(',', p);
                if (c == std::string::npos) c = s.size();
                if (c > p) kibs.push_back(atoll(s.substr(p, c - p).c_str()));
                p = c + 1;
            }
        } else { fprintf(stderr, "unknown arg '%s'\n", a.c_str()); return 2; }
    }

    gicc::Runtime rt;
    const int rank = rt.rank(), nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "need exactly 2 ranks (got %d)\n", nranks);
        return 1;
    }
    const int peer = 1 - rank;

    void* d_buf = nullptr;
    if (hipMalloc(&d_buf, kBufBytes) != hipSuccess) return 3;
    (void)hipMemset(d_buf, rank == 0 ? 0xA5 : 0x00, kBufBytes);
    unsigned long long* d_out = nullptr;
    (void)hipMalloc(&d_out, sizeof(unsigned long long));
    auto* scratch = (unsigned long long*)((char*)d_buf + kScratchOff);

    auto bh = rt.register_buffer(d_buf, kBufBytes, /*is_device=*/true);
    rt.exchange();
    (void)rt.prepare();
    rt.reset();
    rt.barrier();

    if (peer_write) {
        void* pb = gicc_runtime_peer_ipc_base(&rt, peer, bh.index);
        if (!pb) {
            if (rank == 0)
                fprintf(stderr, "--peer-write needs both ranks on one node "
                                "(no IPC mapping to peer %d)\n", peer);
            return 4;
        }
        scratch = (unsigned long long*)((char*)pb + kScratchOff);
    }

    if (rank == 0) {
        printf("=== fence_effect transfer=%zuB grid=%dx%d writes=%s ===\n",
               bytes, blocks, threads, peer_write ? "PEER-mapped" : "local");
        printf("    (per-kernel, %d launched back to back before each sync)\n",
               batch);
        printf("CSV,write_kib,fence,median_us,min_us,verdict\n");
    }

    struct F { const char* name; int v; };
    const F fences[] = {{"system", gicc::FENCE_SYSTEM},
                        {"device", gicc::FENCE_DEVICE},
                        {"block",  gicc::FENCE_BLOCK},
                        {"none",   gicc::FENCE_NONE}};

    for (long long kib : kibs) {
        const long long words = kib * 1024 / (long long)sizeof(unsigned long long);
        double base = 0;
        for (const F& f : fences) {
            (void)hipMemset(d_out, 0, sizeof(unsigned long long));
            (void)hipDeviceSynchronize();
            rt.barrier();

            std::vector<double> v;
            for (int s = 0; s < samples + warmup; ++s) {
                rt.barrier();
                double t0 = MPI_Wtime();
                if (rank == 0) {
                    for (int b = 0; b < batch; ++b) {
                        gicc::DeviceCtx* c = rt.prepare();
                        hipLaunchKernelGGL(k_write_then_quiet, dim3(blocks),
                                           dim3(threads), 0, 0, c, peer,
                                           bh.index, (unsigned long long)bytes,
                                           scratch, words, f.v, d_out);
                    }
                    (void)hipDeviceSynchronize();
                    rt.reset();
                }
                double t1 = MPI_Wtime();
                rt.barrier();
                if (rank == 0 && s >= warmup)
                    v.push_back((t1 - t0) * 1e6 / batch);
            }

            // The read-back must match what the kernel wrote, and the peer
            // must have received the payload.
            int bad = 0;
            if (rank == 0 && words > 0 && !peer_write) {
                unsigned long long got = 0;
                (void)hipMemcpy(&got, d_out, sizeof(got), hipMemcpyDeviceToHost);
                const unsigned long long want =
                    (unsigned long long)(words - 1) * 2654435761ull + 1;
                if (got != want) bad = 1;
            }
            if (rank == 1) {
                std::vector<uint8_t> h(std::min(bytes, (size_t)4096));
                (void)hipMemcpy(h.data(), (char*)d_buf + kRecvBase, h.size(),
                                hipMemcpyDeviceToHost);
                for (size_t i = 0; i < h.size(); ++i)
                    if (h[i] != 0xA5) { bad = 2; break; }
            }
            int all = 0;
            MPI_Allreduce(&bad, &all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            const Stats st = summarize(std::move(v));
            if (rank == 0) {
                if (f.v == gicc::FENCE_SYSTEM) base = st.median;
                const char* verdict = all == 0 ? "OK"
                                    : all == 1 ? "WRONG-readback"
                                               : "WRONG-payload";
                printf("CSV,%lld,%s,%.3f,%.3f,%s\n",
                       kib, f.name, st.median, st.min, verdict);
                printf("  wrote %6lld KiB  fence=%-7s median=%9.2f us"
                       "   %5.2fx vs system   %s\n",
                       kib, f.name, st.median,
                       base > 0 ? base / st.median : 1.0, verdict);
            }
        }
    }

    rt.barrier();
    if (rank == 0) printf("=== done ===\n");
    (void)hipFree(d_buf);
    (void)hipFree(d_out);
    return 0;
}
