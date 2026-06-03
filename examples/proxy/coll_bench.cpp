/*
 * coll_bench.cpp - latency benchmark for the GICC ring all-reduce and
 * all-to-all collectives vs the native MPI equivalents (MPI_Allreduce /
 * MPI_Alltoall) on the SAME device buffers.
 *
 * One source, two transports (proxy / DWQ) via -DGICC_CPU_PROXY. The MPI
 * baseline is identical in both builds; only the GICC path changes.
 *
 * MPI baseline uses GPU-aware MPI, so the run MUST set
 * MPICH_GPU_SUPPORT_ENABLED=1 (Cray MPICH GTL) or the device-pointer MPI
 * calls are invalid.
 *
 * Build:  ./examples/proxy/build_collectives.sh   (emits coll_bench_{proxy,dwq})
 * Run:    MPICH_GPU_SUPPORT_ENABLED=1 PMI_MAX_KVS_ENTRIES=512 \
 *         FI_MR_CACHE_MAX_COUNT=0 srun -p pdebug -N 2 -n 8 \
 *             --ntasks-per-node=4 --cpus-per-task=8 --gpu-bind=none -t 2 \
 *             ./build_ofi/coll_bench_proxy        # or _dwq
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
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered so progress is visible

#ifndef GICC_CPU_PROXY
    rt.enable_host_wait_mode();
    // Same-node via DWQ/NIC too (avoid IPC+DWQ SDMA deadlock; see
    // Runtime::set_ipc_fastpath).
    rt.set_ipc_fastpath(false);
#endif

    // Chunk = floats/ints exchanged with one neighbour per ring step (and the
    // per-peer block in all-to-all). Total all-reduce array = N * chunk.
    const int chunks[] = {256, 1024, 4096, 16384};
    const int n_sizes  = (int)(sizeof(chunks) / sizeof(chunks[0]));
    const int MAXCHUNK = 16384;
    const int iters    = (argc > 1) ? std::atoi(argv[1]) : 30;
    const int warmup   = (argc > 2) ? std::atoi(argv[2]) : 5;

    const size_t ar_bytes = (size_t)N * MAXCHUNK * sizeof(float);
    const size_t aa_bytes = (size_t)N * MAXCHUNK * sizeof(int);

    // All-reduce buffers: GICC (data + 1-chunk scratch) and MPI (in/out).
    float *d_ar = nullptr, *d_ar_scratch = nullptr;
    float *d_mpi_in = nullptr, *d_mpi_out = nullptr;
    // All-to-all buffers (shared by GICC + MPI).
    int *d_send = nullptr, *d_recv = nullptr;
    if (hipMalloc(&d_ar, ar_bytes)               != hipSuccess ||
        hipMalloc(&d_ar_scratch, (size_t)MAXCHUNK * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_mpi_in, ar_bytes)           != hipSuccess ||
        hipMalloc(&d_mpi_out, ar_bytes)          != hipSuccess ||
        hipMalloc(&d_send, aa_bytes)             != hipSuccess ||
        hipMalloc(&d_recv, aa_bytes)             != hipSuccess) {
        fprintf(stderr, "rank %d: hipMalloc failed\n", rank);
        return 2;
    }
    (void)hipMemset(d_ar, 1, ar_bytes);
    (void)hipMemset(d_mpi_in, 1, ar_bytes);
    (void)hipMemset(d_send, 1, aa_bytes);
    (void)hipDeviceSynchronize();

    gicc::Buffer ar_buf      = rt.register_buffer(d_ar, ar_bytes, true);
    gicc::Buffer ar_scratch  = rt.register_buffer(d_ar_scratch,
                                                  (size_t)MAXCHUNK * sizeof(float), true);
    gicc::Buffer send_buf    = rt.register_buffer(d_send, aa_bytes, true);
    gicc::Buffer recv_buf    = rt.register_buffer(d_recv, aa_bytes, true);

#ifdef GICC_CPU_PROXY
    // Fused proxy all-reduce buffers: in-place data, (N-1)-chunk recv staging,
    // host-pinned flags (PCIe-coherent for the GPU spin), 4-byte nonzero src.
    const size_t fr_bytes = (size_t)(N - 1) * MAXCHUNK * sizeof(float);
    float *d_fz = nullptr, *d_fz_recv = nullptr;
    unsigned int *h_flag = nullptr, *d_flag = nullptr, *d_one = nullptr;
    (void)hipMalloc(&d_fz, ar_bytes);
    (void)hipMalloc(&d_fz_recv, fr_bytes);
    (void)hipMalloc(&d_one, sizeof(unsigned int));
    (void)hipHostMalloc((void**)&h_flag, (size_t)2 * N * sizeof(unsigned int),
                        hipHostMallocMapped);
    (void)hipHostGetDevicePointer((void**)&d_flag, h_flag, 0);
    (void)hipMemset(d_one, 1, sizeof(unsigned int));   // 0x01010101 (nonzero)
    (void)hipDeviceSynchronize();
    gicc::Buffer fz_buf   = rt.register_buffer(d_fz, ar_bytes, true);
    gicc::Buffer fz_recv  = rt.register_buffer(d_fz_recv, fr_bytes, true);
    gicc::Buffer flag_buf = rt.register_buffer(d_flag,
                                (size_t)2 * N * sizeof(unsigned int), true);
    gicc::Buffer one_buf  = rt.register_buffer(d_one, sizeof(unsigned int), true);
#endif

    rt.exchange();
    rt.barrier();

    if (rank == 0) {
        printf("\n=== GICC [%s] vs MPI  (%d ranks) ===\n",
               gicc_coll::transport_name(), N);
        printf("%-10s %12s | %14s %14s %8s\n",
               "collective", "bytes", "GICC (us)", "MPI (us)", "speedup");
        printf("--------------------------------------------------------------"
               "----\n");
    }

    // ---------------- ALL-REDUCE ----------------
    for (int si = 0; si < n_sizes; ++si) {
        const int chunk = chunks[si];
        const size_t arr_bytes = (size_t)N * chunk * sizeof(float);

        for (int w = 0; w < warmup; ++w)
            gicc_coll::ring_allreduce(rt, ar_buf, d_ar, ar_scratch,
                                      d_ar_scratch, chunk);
        rt.barrier();
        double t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it)
            gicc_coll::ring_allreduce(rt, ar_buf, d_ar, ar_scratch,
                                      d_ar_scratch, chunk);
        double gicc_us = (MPI_Wtime() - t0) / iters * 1e6;

        const int count = N * chunk;
        for (int w = 0; w < warmup; ++w)
            MPI_Allreduce(d_mpi_in, d_mpi_out, count, MPI_FLOAT, MPI_SUM,
                          MPI_COMM_WORLD);
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
            printf("%-10s %12zu | %14.2f %14.2f %7.2fx\n",
                   "allreduce", arr_bytes, gmax, mmax, mmax / gmax);
    }

#ifdef GICC_CPU_PROXY
    // ---------------- FUSED ALL-REDUCE (proxy, on-device ring) ----------------
    for (int si = 0; si < n_sizes; ++si) {
        // Fresh-init correctness at each size: data[i] = (rank+1)+(i%7).
        const int chunk = chunks[si];
        const int count = N * chunk;
        std::vector<float> hv(count);
        for (int i = 0; i < count; ++i) hv[i] = (float)((rank + 1) + (i % 7));
        (void)hipMemcpy(d_fz, hv.data(), (size_t)count * sizeof(float),
                        hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        gicc_coll::ring_allreduce_fused(rt, fz_buf, d_fz, fz_recv, d_fz_recv,
                                        flag_buf, d_flag, one_buf, chunk);
        (void)hipMemcpy(hv.data(), d_fz, (size_t)count * sizeof(float),
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
            printf("[fused allreduce correctness @%zu B: %s, %d total errors]\n",
                   (size_t)count * sizeof(float),
                   all_errs == 0 ? "PASS" : "FAIL", all_errs);
    }
    for (int si = 0; si < n_sizes; ++si) {
        const int chunk = chunks[si];
        const size_t arr_bytes = (size_t)N * chunk * sizeof(float);

        for (int w = 0; w < warmup; ++w)
            gicc_coll::ring_allreduce_fused(rt, fz_buf, d_fz, fz_recv, d_fz_recv,
                                            flag_buf, d_flag, one_buf, chunk);
        rt.barrier();
        double t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it)
            gicc_coll::ring_allreduce_fused(rt, fz_buf, d_fz, fz_recv, d_fz_recv,
                                            flag_buf, d_flag, one_buf, chunk);
        double gicc_us = (MPI_Wtime() - t0) / iters * 1e6;

        const int count = N * chunk;
        for (int w = 0; w < warmup; ++w)
            MPI_Allreduce(d_mpi_in, d_mpi_out, count, MPI_FLOAT, MPI_SUM,
                          MPI_COMM_WORLD);
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
            printf("%-10s %12zu | %14.2f %14.2f %7.2fx\n",
                   "ar-fused", arr_bytes, gmax, mmax, mmax / gmax);
    }
#endif

    // ---------------- ALL-TO-ALL ----------------
    for (int si = 0; si < n_sizes; ++si) {
        const int chunk = chunks[si];
        const size_t total_bytes = (size_t)N * chunk * sizeof(int);

        for (int w = 0; w < warmup; ++w)
            gicc_coll::alltoall_run(rt, send_buf, d_send, recv_buf, d_recv, chunk);
        rt.barrier();
        double t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it)
            gicc_coll::alltoall_run(rt, send_buf, d_send, recv_buf, d_recv, chunk);
        double gicc_us = (MPI_Wtime() - t0) / iters * 1e6;

        for (int w = 0; w < warmup; ++w)
            MPI_Alltoall(d_send, chunk, MPI_INT, d_recv, chunk, MPI_INT,
                         MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it)
            MPI_Alltoall(d_send, chunk, MPI_INT, d_recv, chunk, MPI_INT,
                         MPI_COMM_WORLD);
        double mpi_us = (MPI_Wtime() - t0) / iters * 1e6;

        double gmax, mmax;
        MPI_Reduce(&gicc_us, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mpi_us,  &mmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("%-10s %12zu | %14.2f %14.2f %7.2fx\n",
                   "alltoall", total_bytes, gmax, mmax, mmax / gmax);
    }

    rt.barrier();
    (void)hipFree(d_ar); (void)hipFree(d_ar_scratch);
    (void)hipFree(d_mpi_in); (void)hipFree(d_mpi_out);
    (void)hipFree(d_send); (void)hipFree(d_recv);
#ifdef GICC_CPU_PROXY
    (void)hipFree(d_fz); (void)hipFree(d_fz_recv); (void)hipFree(d_one);
    (void)hipHostFree(h_flag);
#endif
    return 0;
}
