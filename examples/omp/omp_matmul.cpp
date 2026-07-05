// omp_matmul.cpp - GiOMP distributed matrix multiply (GICC's npes-step ring
// algorithm, expressed in the OpenMP-offload form).
//
// This is the GICC ring matmul (examples/ofi/mm_minimal.cpp) ported to the GiOMP
// path: each rank owns horizontal stripes of A and C and a vertical stripe of B.
// Over npes steps the B stripes rotate around the ring; at each step a rank does
//   ompx_put         (send my B stripe to my left neighbor, from an omp target region)
//   omp target       (local block matmul, identical loop to the DiOMP baseline)
//   ompx_quiet_host  (wait for the in-flight RDMA to land)
//   swap(Bs, Bn)
// Put-to-left is equivalent to DiOMP's get-from-right, so the data flow and the
// per-step compute are identical to DiOMP's diomp_mm.cpp; only the transport
// (ompx_put vs ompx_dget) differs.
//
// Build: clang++ $(gicc-omp-config --cflags) omp_matmul.cpp $(gicc-omp-config --libs)
// Run  : HSA_XNACK=1 GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
//        flux run -N <nodes> -n <ranks> -g1 -o mpibind=off ./omp_matmul [N]

#include "gicc/omp.h"
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    ompx_init();

    const int mype = omp_get_rank_num();
    const int npes = omp_get_num_ranks();
    if (npes < 2) { if (mype == 0) fprintf(stderr, "need >=2 ranks\n");
                    ompx_finalize(); return 1; }

    int N = (argc > 1) ? atoi(argv[1]) : 30240;
    N = (N / npes) * npes;                 // divisible by npes
    const int Ns = N / npes;               // stripe width
    const size_t stripe = (size_t)N * Ns * sizeof(float);
    const int left = (npes + mype - 1) % npes;

    if (mype == 0)
        printf("GiOMP matmul: N=%d npes=%d stripe=%dx%d (%zu B)\n",
               N, npes, N, Ns, stripe);

    // Allocate all four device buffers via ompx_alloc (IPC-capable, auto-registered).
    ompx_buffer As = ompx_alloc(stripe), Cs = ompx_alloc(stripe);
    ompx_buffer Bs = ompx_alloc(stripe), Bn = ompx_alloc(stripe);
    float* dAs = (float*)As.ptr; float* dCs = (float*)Cs.ptr;
    float* dBs = (float*)Bs.ptr; float* dBn = (float*)Bn.ptr;
    int idxBs = Bs.index, idxBn = Bn.index;

    // Publish the RMA address book.
    ompx_exchange();
    gicc::DeviceCtx* d_ctx = ompx_prepare();

    // Fill dAs once before the run loop (same closed-form as the old hAs init).
    #pragma omp target teams distribute parallel for is_device_ptr(dAs) firstprivate(N, Ns, mype)
    for (size_t i = 0; i < (size_t)N * Ns; ++i)
        dAs[i] = (float)((i + mype) % 11 + 7);

    const int TOTAL = 10, WARMUP = 2;
    std::vector<double> times;

    for (int run = 0; run < TOTAL + WARMUP; ++run) {
        // Reset dCs=0 and dBs to the initial B at the start of each run.
        #pragma omp target teams distribute parallel for is_device_ptr(dBs) firstprivate(N, Ns, mype)
        for (size_t i = 0; i < (size_t)N * Ns; ++i)
            dBs[i] = (float)((i + mype) % 13 + 5);
        #pragma omp target teams distribute parallel for is_device_ptr(dCs) firstprivate(N, Ns)
        for (size_t i = 0; i < (size_t)N * Ns; ++i)
            dCs[i] = 0.f;

        float* curBs = dBs; float* curBn = dBn; int iBs = idxBs, iBn = idxBn;

        ompx_barrier();
        double t0 = omp_get_wtime();

        for (int s = 0; s < npes; ++s) {
            const int block_num = (mype + s) % npes;
            d_ctx = ompx_prepare();
            // (1) send my current B stripe to my left neighbor (-> its Bn).
            //     ompx_put picks IPC (same-node xGMI) vs proxy (cross-node) itself.
            #pragma omp target is_device_ptr(d_ctx) firstprivate(left, iBn, iBs, stripe)
            { ompx_put(d_ctx, left, iBn, 0, iBs, 0, stripe); }

            // (2) local block matmul, identical loop to DiOMP's diomp_mm.cpp.
            #pragma omp target is_device_ptr(dAs, curBs, dCs) firstprivate(N, Ns, block_num)
            #pragma omp teams distribute parallel for collapse(2)
            for (int k = 0; k < N; ++k) {
                for (int j = 0; j < Ns; ++j) {
                    const float b_kj = curBs[k * Ns + j];
                    float* Cb = dCs + (size_t)block_num * Ns;
                    for (int i = 0; i < Ns; ++i)
                        Cb[(size_t)i * N + j] += dAs[(size_t)i * N + k] * b_kj;
                }
            }

            // (3) wait for the RDMA to land, then rotate buffers.
            ompx_quiet_host();
            std::swap(curBs, curBn); std::swap(iBs, iBn);
            ompx_barrier();
        }

        double t1 = omp_get_wtime();
        if (run >= WARMUP) times.push_back((t1 - t0) * 1e6);  // us
    }

    // Correctness fingerprint: sum of this rank's C stripe (depends on every B
    // block having rotated through, so a wrong transfer corrupts it).
    double csum = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:csum) is_device_ptr(dCs)
    for (size_t x = 0; x < (size_t)N * Ns; ++x) csum += (double)dCs[x];

    if (mype == 0) {
        std::sort(times.begin(), times.end());
        double med = times[times.size() / 2];
        printf("GiOMP matmul (ring, ompx_put): N=%d npes=%d  median=%.1f us  (%d runs)  rank0_Csum=%.6e\n",
               N, npes, med, (int)times.size(), csum);
    }

    ompx_free(As); ompx_free(Cs); ompx_free(Bs); ompx_free(Bn);
    ompx_finalize();
    return 0;
}
