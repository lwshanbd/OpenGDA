/*
 * coll_hier_test.cpp - hierarchical all-reduce (2 nodes): correctness + perf
 * vs MPI. Intra-node phases over xGMI, the single inter-node phase over the
 * proxy with ALL ranks crossing simultaneously (so all NICs are used and only
 * count/P crosses per rank). Proxy build. Requires N == 2 * ranks_per_node.
 *
 * Run: MPICH_GPU_SUPPORT_ENABLED=1 ... srun -N 2 -n <2P> --ntasks-per-node=<P> \
 *        ./build_ofi/coll_hier_test_proxy [iters] [warmup]
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include "coll_common.hpp"

int main(int argc, char** argv) {
    gicc::Runtime rt;
    const int rank = rt.rank();
    const int N    = rt.size();
    const int P    = rt.boot().local_size();   // ranks per node
    setvbuf(stdout, nullptr, _IONBF, 0);

    if (N != 2 * P) {
        if (rank == 0)
            fprintf(stderr, "hier test needs exactly 2 nodes (N=%d, P=%d)\n", N, P);
        return 1;
    }

    const int chunks[] = {1024, 16384, 262144, 1048576};
    const int n_sizes  = (int)(sizeof(chunks) / sizeof(chunks[0]));
    const int MAXCHUNK = 1048576;
    const int iters    = (argc > 1) ? std::atoi(argv[1]) : 10;
    const int warmup   = (argc > 2) ? std::atoi(argv[2]) : 2;

    const size_t cnt_max = (size_t)N * MAXCHUNK;          // max total elements
    float *d_data = nullptr, *d_recv = nullptr, *d_one = nullptr;
    float *d_mpi_in = nullptr, *d_mpi_out = nullptr;
    unsigned int *h_flag = nullptr, *d_flag = nullptr;
    if (hipMalloc(&d_data, cnt_max * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_recv, cnt_max * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_one, sizeof(float)) != hipSuccess ||
        hipMalloc(&d_mpi_in, cnt_max * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_mpi_out, cnt_max * sizeof(float)) != hipSuccess) {
        fprintf(stderr, "rank %d: hipMalloc failed\n", rank); return 2;
    }
    (void)hipHostMalloc((void**)&h_flag, (size_t)2 * N * sizeof(unsigned int),
                        hipHostMallocMapped);
    (void)hipHostGetDevicePointer((void**)&d_flag, h_flag, 0);
    (void)hipMemset(d_one, 1, sizeof(float));
    (void)hipMemset(d_mpi_in, 1, cnt_max * sizeof(float));
    (void)hipDeviceSynchronize();

    gicc::Buffer data_buf = rt.register_buffer(d_data, cnt_max * sizeof(float), true);
    gicc::Buffer recv_buf = rt.register_buffer(d_recv, cnt_max * sizeof(float), true);
    gicc::Buffer flag_buf = rt.register_buffer(d_flag, (size_t)2 * N * sizeof(unsigned int), true);
    gicc::Buffer one_buf  = rt.register_buffer(d_one, sizeof(float), true);
    rt.exchange();
    rt.barrier();

    if (rank == 0) {
        printf("\n=== GICC hierarchical all-reduce vs MPI (%d ranks, %d/node) ===\n", N, P);
        printf("%-10s %12s | %14s %14s %8s\n",
               "collective", "bytes", "GICC (us)", "MPI (us)", "speedup");
        printf("------------------------------------------------------------------\n");
    }

    // correctness per size
    for (int si = 0; si < n_sizes; ++si) {
        const int count = N * chunks[si];      // divisible by P (N=2P)
        std::vector<float> hv(count);
        for (int i = 0; i < count; ++i) hv[i] = (float)((rank + 1) + (i % 7));
        (void)hipMemcpy(d_data, hv.data(), (size_t)count * sizeof(float), hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        gicc_coll::ring_allreduce_hier(rt, data_buf, d_data, recv_buf, d_recv,
                                       flag_buf, d_flag, one_buf, count, P);
        (void)hipMemcpy(hv.data(), d_data, (size_t)count * sizeof(float), hipMemcpyDeviceToHost);
        (void)hipDeviceSynchronize();
        int errs = 0;
        for (int i = 0; i < count; ++i) {
            float want = (float)((double)N * (N + 1) / 2.0 + (double)N * (i % 7));
            if (hv[i] != want) ++errs;
        }
        int all_errs = 0;
        MPI_Reduce(&errs, &all_errs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("[hier correctness @%zu B: %s, %d errors]\n",
                   (size_t)count * sizeof(float), all_errs == 0 ? "PASS" : "FAIL", all_errs);
    }

    // timing
    for (int si = 0; si < n_sizes; ++si) {
        const int count = N * chunks[si];
        const size_t bytes = (size_t)count * sizeof(float);
        for (int w = 0; w < warmup; ++w)
            gicc_coll::ring_allreduce_hier(rt, data_buf, d_data, recv_buf, d_recv,
                                           flag_buf, d_flag, one_buf, count, P);
        rt.barrier();
        double t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it)
            gicc_coll::ring_allreduce_hier(rt, data_buf, d_data, recv_buf, d_recv,
                                           flag_buf, d_flag, one_buf, count, P);
        double gicc_us = (MPI_Wtime() - t0) / iters * 1e6;

        for (int w = 0; w < warmup; ++w)   // warm up MPI too (fair comparison)
            MPI_Allreduce(d_mpi_in, d_mpi_out, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it)
            MPI_Allreduce(d_mpi_in, d_mpi_out, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        double mpi_us = (MPI_Wtime() - t0) / iters * 1e6;

        double gmax, mmax;
        MPI_Reduce(&gicc_us, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mpi_us,  &mmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("%-10s %12zu | %14.2f %14.2f %7.2fx\n",
                   "ar-hier", bytes, gmax, mmax, mmax / gmax);
    }

    // ---- DIRECT hierarchical: correctness + timing ----
    for (int si = 0; si < n_sizes; ++si) {
        const int count = N * chunks[si];
        std::vector<float> hv(count);
        for (int i = 0; i < count; ++i) hv[i] = (float)((rank + 1) + (i % 7));
        (void)hipMemcpy(d_data, hv.data(), (size_t)count * sizeof(float), hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        gicc_coll::ring_allreduce_hier_direct(rt, data_buf, d_data, recv_buf, d_recv,
                                              flag_buf, d_flag, one_buf, count, P);
        (void)hipMemcpy(hv.data(), d_data, (size_t)count * sizeof(float), hipMemcpyDeviceToHost);
        (void)hipDeviceSynchronize();
        int errs = 0;
        for (int i = 0; i < count; ++i) {
            float want = (float)((double)N * (N + 1) / 2.0 + (double)N * (i % 7));
            if (hv[i] != want) ++errs;
        }
        int all_errs = 0;
        MPI_Reduce(&errs, &all_errs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("[hdir correctness @%zu B: %s, %d errors]\n",
                   (size_t)count * sizeof(float), all_errs == 0 ? "PASS" : "FAIL", all_errs);
    }
    for (int si = 0; si < n_sizes; ++si) {
        const int count = N * chunks[si];
        const size_t bytes = (size_t)count * sizeof(float);
        for (int w = 0; w < warmup; ++w)
            gicc_coll::ring_allreduce_hier_direct(rt, data_buf, d_data, recv_buf, d_recv,
                                                  flag_buf, d_flag, one_buf, count, P);
        rt.barrier();
        double t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it)
            gicc_coll::ring_allreduce_hier_direct(rt, data_buf, d_data, recv_buf, d_recv,
                                                  flag_buf, d_flag, one_buf, count, P);
        double gicc_us = (MPI_Wtime() - t0) / iters * 1e6;
        for (int w = 0; w < warmup; ++w)
            MPI_Allreduce(d_mpi_in, d_mpi_out, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it)
            MPI_Allreduce(d_mpi_in, d_mpi_out, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        double mpi_us = (MPI_Wtime() - t0) / iters * 1e6;
        double gmax, mmax;
        MPI_Reduce(&gicc_us, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mpi_us,  &mmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("%-10s %12zu | %14.2f %14.2f %7.2fx\n",
                   "ar-hdir", bytes, gmax, mmax, mmax / gmax);
    }

    rt.barrier();
    (void)hipFree(d_data); (void)hipFree(d_recv); (void)hipFree(d_one);
    (void)hipFree(d_mpi_in); (void)hipFree(d_mpi_out); (void)hipHostFree(h_flag);
    return 0;
}
