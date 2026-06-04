/*
 * coll_ll_test.cpp - LL (low-latency) packet ring all-reduce: correctness +
 * latency vs MPI, small-message regime. Proxy build only (LL needs device-side
 * put/quiet). The LL packet embeds the arrival flag in the data line, so there
 * is no separate per-segment flag round-trip — the trick NCCL/UCCL use that
 * GICC's ring_allreduce_pipe lacked. See ring_allreduce_ll in coll_common.hpp.
 *
 * Build: ./examples/proxy/build_collectives.sh  (emits coll_ll_test_proxy)
 * Run:   MPICH_GPU_SUPPORT_ENABLED=1 PMI_MAX_KVS_ENTRIES=512 FI_MR_CACHE_MAX_COUNT=0 \
 *        srun -p pdebug -N 2 -n 8 --ntasks-per-node=4 --cpus-per-task=8 \
 *            --gpu-bind=none -t 2 ./build_ofi/coll_ll_test_proxy [iters] [warmup]
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>

#include "coll_common.hpp"

int main(int argc, char** argv) {
    gicc::Runtime rt;
    const int rank = rt.rank();
    const int N    = rt.size();
    setvbuf(stdout, nullptr, _IONBF, 0);

    const int chunks[] = {256, 1024, 4096, 16384};
    const int n_sizes  = (int)(sizeof(chunks) / sizeof(chunks[0]));
    const int MAXCHUNK = 16384;
    const int iters    = (argc > 1) ? std::atoi(argv[1]) : 30;
    const int warmup   = (argc > 2) ? std::atoi(argv[2]) : 5;

    const size_t ar_bytes  = (size_t)N * MAXCHUNK * sizeof(float);
    const size_t ps_lines  = (size_t)MAXCHUNK / 2;
    const size_t pr_lines  = (size_t)2 * (N - 1) * (MAXCHUNK / 2);

    float *d_data = nullptr, *d_mpi_in = nullptr, *d_mpi_out = nullptr;
    gicc_coll::LLPkt *d_ps = nullptr, *d_pr = nullptr;
    if (hipMalloc(&d_data, ar_bytes) != hipSuccess ||
        hipMalloc(&d_mpi_in, ar_bytes) != hipSuccess ||
        hipMalloc(&d_mpi_out, ar_bytes) != hipSuccess ||
        hipMalloc(&d_ps, ps_lines * sizeof(gicc_coll::LLPkt)) != hipSuccess ||
        hipMalloc(&d_pr, pr_lines * sizeof(gicc_coll::LLPkt)) != hipSuccess) {
        fprintf(stderr, "rank %d: hipMalloc failed\n", rank); return 2;
    }
    (void)hipMemset(d_mpi_in, 1, ar_bytes);
    (void)hipMemset(d_pr, 0, pr_lines * sizeof(gicc_coll::LLPkt));  // stale flags = 0
    (void)hipDeviceSynchronize();

    gicc::Buffer data_buf = rt.register_buffer(d_data, ar_bytes, true);
    gicc::Buffer ps_buf   = rt.register_buffer(d_ps, ps_lines * sizeof(gicc_coll::LLPkt), true);
    gicc::Buffer pr_buf   = rt.register_buffer(d_pr, pr_lines * sizeof(gicc_coll::LLPkt), true);
    rt.exchange();
    rt.barrier();

    if (rank == 0) {
        printf("\n=== GICC LL-packet all-reduce vs MPI (%d ranks) ===\n", N);
        printf("%-12s %12s | %14s %14s %8s\n",
               "collective", "bytes", "GICC (us)", "MPI (us)", "speedup");
        printf("------------------------------------------------------------------\n");
    }

    unsigned int flag_base = 1000;   // monotonic, never reset

    // correctness per size (fresh init)
    for (int si = 0; si < n_sizes; ++si) {
        const int chunk = chunks[si];
        const int count = N * chunk;
        std::vector<float> hv(count);
        for (int i = 0; i < count; ++i) hv[i] = (float)((rank + 1) + (i % 7));
        (void)hipMemcpy(d_data, hv.data(), (size_t)count * sizeof(float),
                        hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        gicc_coll::ring_allreduce_ll(rt, data_buf, d_data, ps_buf, d_ps,
                                     pr_buf, d_pr, chunk, flag_base);
        flag_base += 2 * N;
        (void)hipMemcpy(hv.data(), d_data, (size_t)count * sizeof(float),
                        hipMemcpyDeviceToHost);
        (void)hipDeviceSynchronize();
        int errs = 0;
        for (int i = 0; i < count; ++i) {
            float want = (float)((double)N * (N + 1) / 2.0 + (double)N * (i % 7));
            if (hv[i] != want) ++errs;
        }
        int all_errs = 0;
        MPI_Reduce(&errs, &all_errs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("[LL correctness @%zu B: %s, %d errors]\n",
                   (size_t)count * sizeof(float),
                   all_errs == 0 ? "PASS" : "FAIL", all_errs);
    }

    // timing per size
    for (int si = 0; si < n_sizes; ++si) {
        const int chunk = chunks[si];
        const size_t arr_bytes = (size_t)N * chunk * sizeof(float);
        for (int w = 0; w < warmup; ++w) {
            gicc_coll::ring_allreduce_ll(rt, data_buf, d_data, ps_buf, d_ps,
                                         pr_buf, d_pr, chunk, flag_base);
            flag_base += 2 * N;
        }
        rt.barrier();
        double t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it) {
            gicc_coll::ring_allreduce_ll(rt, data_buf, d_data, ps_buf, d_ps,
                                         pr_buf, d_pr, chunk, flag_base);
            flag_base += 2 * N;
        }
        double gicc_us = (MPI_Wtime() - t0) / iters * 1e6;

        const int count = N * chunk;
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it)
            MPI_Allreduce(d_mpi_in, d_mpi_out, count, MPI_FLOAT, MPI_SUM,
                          MPI_COMM_WORLD);
        double mpi_us = (MPI_Wtime() - t0) / iters * 1e6;

        double gmax, mmax;
        MPI_Reduce(&gicc_us, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mpi_us,  &mmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("%-12s %12zu | %14.2f %14.2f %7.2fx\n",
                   "ar-ll", arr_bytes, gmax, mmax, mmax / gmax);
    }

    rt.barrier();
    (void)hipFree(d_data); (void)hipFree(d_mpi_in); (void)hipFree(d_mpi_out);
    (void)hipFree(d_ps); (void)hipFree(d_pr);
    return 0;
}
