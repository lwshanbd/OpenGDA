/*
 * coll_dtree_test.cpp - flat double binary tree all-reduce: correctness + perf
 * vs MPI. The tree spans ALL N ranks (block layout); each edge auto-routes
 * over xGMI (same-node) or the proxy (cross-node). Works for any N (power of 2
 * is the well-behaved case). Proxy build.
 *
 * Run: MPICH_GPU_SUPPORT_ENABLED=1 ... srun -N 2 -n 16 --ntasks-per-node=8 \
 *        ./build_ofi/coll_dtree_test_proxy [iters] [warmup]
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include "coll_common.hpp"

int main(int argc, char** argv) {
    gicc::Runtime rt;
    // Disable the same-node IPC fast path: these tree collectives mix same-node
    // operations (xGMI data + the flag put, which routes a same-node gicc::put
    // through the IPC fast path = SDMA hipMemcpyAsync) with concurrent
    // cross-node proxy RMA. On AMD+CXI that SDMA-engine contention intermittently
    // deadlocks (see Runtime::set_ipc_fastpath). Every other collective example
    // (allreduce_ring, alltoall, coll_bench) sets this; the tree tests need it
    // too. This removes the nondeterministic hang in the sustained timing loops.
    rt.set_ipc_fastpath(false);
    const int rank = rt.rank();
    const int N    = rt.size();
    const int P    = rt.boot().local_size();   // ranks per node (banner only)
    setvbuf(stdout, nullptr, _IONBF, 0);

    // total bytes = N*chunk*4. At N=16: bytes = chunk*64 -> 16KB..256MB.
    const int chunks[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304};
    const int n_sizes  = (int)(sizeof(chunks) / sizeof(chunks[0]));
    const int MAXCHUNK = 4194304;
    const int iters    = (argc > 1) ? std::atoi(argv[1]) : 10;
    const int warmup   = (argc > 2) ? std::atoi(argv[2]) : 2;
    const bool skip_flat = (std::getenv("DTREE_SKIP_FLAT") != nullptr);
    const bool skip_hier = (std::getenv("DTREE_SKIP_HIER") != nullptr);
    // Hierarchical timing is opt-in: a single hierarchical call is reliable
    // (correctness passes every size, every run) but sustained back-to-back
    // timing intermittently deadlocks in the cooperative RS+tree kernel's
    // proxy path (runtime-level, nondeterministic in size). At K=2 the
    // inter-node tree is depth-1 anyway, so it has no perf advantage over the
    // fused `hier` collective; the double tree only earns its keep at K>=4.
    const bool hier_time = (std::getenv("DTREE_HIER_TIME") != nullptr);
    // Pipelined flat tree: stream each half in S chunks so tree levels overlap
    // (NCCL's technique). S via DTREE_PIPE_CHUNKS (default 8). Run with DTREE_PIPE.
    const bool do_pipe = (std::getenv("DTREE_PIPE") != nullptr);
    const int  pipe_S  = (std::getenv("DTREE_PIPE_CHUNKS"))
                         ? std::atoi(std::getenv("DTREE_PIPE_CHUNKS")) : 8;

    const size_t cnt_max = (size_t)N * MAXCHUNK;
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
    const int NFLAG = 512;   // flat needs 6; pipelined needs 6*S; round up generously
    (void)hipHostMalloc((void**)&h_flag, (size_t)NFLAG * sizeof(unsigned int),
                        hipHostMallocMapped);
    (void)hipHostGetDevicePointer((void**)&d_flag, h_flag, 0);
    (void)hipMemset(d_one, 1, sizeof(float));
    (void)hipMemset(d_mpi_in, 1, cnt_max * sizeof(float));
    (void)hipMemset(d_flag, 0, (size_t)NFLAG * sizeof(unsigned int));   // one-time
    (void)hipDeviceSynchronize();

    gicc::Buffer data_buf = rt.register_buffer(d_data, cnt_max * sizeof(float), true);
    gicc::Buffer recv_buf = rt.register_buffer(d_recv, cnt_max * sizeof(float), true);
    gicc::Buffer flag_buf = rt.register_buffer(d_flag, (size_t)NFLAG * sizeof(unsigned int), true);
    gicc::Buffer one_buf  = rt.register_buffer(d_one, sizeof(float), true);
    rt.exchange();
    rt.barrier();   // all flag buffers zeroed everywhere before any signal

    if (rank == 0) {
        printf("\n=== GICC double binary tree all-reduce vs MPI (%d ranks, %d/node) ===\n", N, P);
        printf("%-10s %12s | %14s %14s %8s\n",
               "collective", "bytes", "GICC (us)", "MPI (us)", "speedup");
        printf("------------------------------------------------------------------\n");
    }

    // correctness per size
    for (int si = 0; si < n_sizes && !skip_flat; ++si) {
        const int count = N * chunks[si];          // always even (N even)
        std::vector<float> hv(count);
        for (int i = 0; i < count; ++i) hv[i] = (float)((rank + 1) + (i % 7));
        (void)hipMemcpy(d_data, hv.data(), (size_t)count * sizeof(float), hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        rt.barrier();
        gicc_coll::allreduce_double_tree(rt, data_buf, d_data, recv_buf, d_recv,
                                         flag_buf, d_flag, one_buf, count);
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
            printf("[dtree correctness @%zu B: %s, %d errors]\n",
                   (size_t)count * sizeof(float), all_errs == 0 ? "PASS" : "FAIL", all_errs);
    }

    // timing
    for (int si = 0; si < n_sizes && !skip_flat; ++si) {
        const int count = N * chunks[si];
        const size_t bytes = (size_t)count * sizeof(float);
        for (int w = 0; w < warmup; ++w)
            gicc_coll::allreduce_double_tree(rt, data_buf, d_data, recv_buf, d_recv,
                                             flag_buf, d_flag, one_buf, count);
        rt.barrier();
        double t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it) {
            gicc_coll::allreduce_double_tree(rt, data_buf, d_data, recv_buf, d_recv,
                                             flag_buf, d_flag, one_buf, count);
            rt.barrier();   // standalone completion (no overlap of consecutive calls)
        }
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
                   "ar-dtree", bytes, gmax, mmax, mmax / gmax);
    }

    // ---- PIPELINED flat double tree (NCCL-style chunked streaming) ----
    if (do_pipe) {
        (void)hipMemset(d_flag, 0, (size_t)NFLAG * sizeof(unsigned int));
        (void)hipDeviceSynchronize();
        rt.barrier();
        for (int si = 0; si < n_sizes; ++si) {
            const int count = N * chunks[si];
            std::vector<float> hv(count);
            for (int i = 0; i < count; ++i) hv[i] = (float)((rank + 1) + (i % 7));
            (void)hipMemcpy(d_data, hv.data(), (size_t)count * sizeof(float), hipMemcpyHostToDevice);
            (void)hipDeviceSynchronize();
            rt.barrier();
            gicc_coll::allreduce_double_tree_pipe(rt, data_buf, d_data, recv_buf, d_recv,
                                                  flag_buf, d_flag, one_buf, count, pipe_S);
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
                printf("[dpipe(S=%d) correctness @%zu B: %s, %d errors]\n",
                       pipe_S, (size_t)count * sizeof(float), all_errs == 0 ? "PASS" : "FAIL", all_errs);
        }
        for (int si = 0; si < n_sizes; ++si) {
            const int count = N * chunks[si];
            const size_t bytes = (size_t)count * sizeof(float);
            for (int w = 0; w < warmup; ++w)
                gicc_coll::allreduce_double_tree_pipe(rt, data_buf, d_data, recv_buf, d_recv,
                                                      flag_buf, d_flag, one_buf, count, pipe_S);
            rt.barrier();
            double t0 = MPI_Wtime();
            for (int it = 0; it < iters; ++it) {
                gicc_coll::allreduce_double_tree_pipe(rt, data_buf, d_data, recv_buf, d_recv,
                                                      flag_buf, d_flag, one_buf, count, pipe_S);
                rt.barrier();
            }
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
                       "ar-dpipe", bytes, gmax, mmax, mmax / gmax);
        }
    }

    // ---- HIERARCHICAL double tree (intra xGMI + inter-node tree) ----
    (void)hipMemset(d_flag, 0, (size_t)NFLAG * sizeof(unsigned int));
    (void)hipDeviceSynchronize();
    rt.barrier();
    for (int si = 0; si < n_sizes && !skip_hier; ++si) {
        const int count = N * chunks[si];
        std::vector<float> hv(count);
        for (int i = 0; i < count; ++i) hv[i] = (float)((rank + 1) + (i % 7));
        (void)hipMemcpy(d_data, hv.data(), (size_t)count * sizeof(float), hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        rt.barrier();
        gicc_coll::allreduce_double_tree_hier(rt, data_buf, d_data, recv_buf, d_recv,
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
            printf("[dtreeh correctness @%zu B: %s, %d errors]\n",
                   (size_t)count * sizeof(float), all_errs == 0 ? "PASS" : "FAIL", all_errs);
    }
    for (int si = 0; si < n_sizes && !skip_hier && hier_time; ++si) {
        const int count = N * chunks[si];
        const size_t bytes = (size_t)count * sizeof(float);
        for (int w = 0; w < warmup; ++w)
            gicc_coll::allreduce_double_tree_hier(rt, data_buf, d_data, recv_buf, d_recv,
                                                  flag_buf, d_flag, one_buf, count, P);
        rt.barrier();
        double t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it) {
            gicc_coll::allreduce_double_tree_hier(rt, data_buf, d_data, recv_buf, d_recv,
                                                  flag_buf, d_flag, one_buf, count, P);
            rt.barrier();
        }
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
                   "ar-dtreeh", bytes, gmax, mmax, mmax / gmax);
    }

    rt.barrier();
    (void)hipFree(d_data); (void)hipFree(d_recv); (void)hipFree(d_one);
    (void)hipFree(d_mpi_in); (void)hipFree(d_mpi_out); (void)hipHostFree(h_flag);
    return 0;
}
