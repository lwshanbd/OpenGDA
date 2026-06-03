/*
 * allreduce_ring.cpp - ring all-reduce (sum) over N ranks, built on the
 * GICC put primitive. ONE source, two transports (proxy / DWQ) selected at
 * compile time via -DGICC_CPU_PROXY (see coll_common.hpp).
 *
 * Algorithm (Baidu-style ring all-reduce):
 *   The per-rank array of `count` floats is split into N equal chunks.
 *   next = (rank+1)%N, prev = (rank-1+N)%N.
 *
 *   Reduce-scatter (N-1 steps): in step s, send chunk (rank-s) to `next`'s
 *   recv buffer, then add the chunk `prev` wrote into MY recv buffer onto
 *   my own chunk (rank-1-s). After N-1 steps, rank r holds the complete
 *   sum of chunk (r+1)%N.
 *
 *   All-gather (N-1 steps): in step s, write my finalised chunk
 *   (rank+1-s) directly into `next`'s data buffer at the same slot. After
 *   N-1 steps every rank has every finalised chunk.
 *
 * A barrier separates every step so a single recv buffer is safe (the add
 * consumes it before the upstream neighbour overwrites it next step).
 *
 * Verification: data[i] = (rank+1) + (i % 7). After sum-allreduce every
 * rank must hold expected[i] = N*(N+1)/2 + N*(i%7), exact in float for the
 * rank counts used here.
 *
 * Build:  ./examples/proxy/build_collectives.sh
 * Run:    srun -p pdebug -t 2 -N <nodes> -n <ranks> --cpus-per-task=8 \
 *             ./build_ofi/allreduce_ring_proxy   # or _dwq
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "coll_common.hpp"

int main(int argc, char** argv) {
    gicc::Runtime rt;
    const int rank   = rt.rank();
    const int N      = rt.size();

#ifndef GICC_CPU_PROXY
    // DWQ path: host pre-stages writes into the deferred work queue and the
    // kernel only fires the trigger. Proxy path must NOT enable this (it
    // would disable the device-side ring push).
    rt.enable_host_wait_mode();
    // Route same-node peers through DWQ/NIC too: mixing same-node IPC copies
    // with concurrent cross-node DWQ writes deadlocks the GPU SDMA engine on
    // AMD+CXI (see Runtime::set_ipc_fastpath).
    rt.set_ipc_fastpath(false);
#endif

    gicc_coll::start_watchdog(rank);   // no-op unless -DGICC_COLL_DEBUG

    // count must be divisible by N for equal chunks. Default 1024 elems/rank
    // chunk; override with argv[1] = elems-per-chunk.
    int elems_per_chunk = (argc > 1) ? std::atoi(argv[1]) : 1024;
    if (elems_per_chunk < 1) elems_per_chunk = 1024;
    const int    count       = N * elems_per_chunk;
    const size_t chunk_bytes = (size_t)elems_per_chunk * sizeof(float);
    const size_t data_bytes  = (size_t)count * sizeof(float);

    // Host init pattern + expected result.
    std::vector<float> h_data(count);
    for (int i = 0; i < count; ++i)
        h_data[i] = (float)((rank + 1) + (i % 7));

    float* d_data = nullptr;   // the array being reduced (in place)
    float* d_recv = nullptr;   // staging slot for one incoming chunk
    if (hipMalloc(&d_data, data_bytes)  != hipSuccess ||
        hipMalloc(&d_recv, chunk_bytes) != hipSuccess) {
        fprintf(stderr, "rank %d: hipMalloc failed\n", rank);
        return 2;
    }
    (void)hipMemcpy(d_data, h_data.data(), data_bytes, hipMemcpyHostToDevice);
    (void)hipMemset(d_recv, 0, chunk_bytes);
    (void)hipDeviceSynchronize();

    gicc::Buffer data_buf = rt.register_buffer(d_data, data_bytes,  true);
    gicc::Buffer recv_buf = rt.register_buffer(d_recv, chunk_bytes, true);
    rt.exchange();
    rt.barrier();

    if (rank == 0) {
        printf("=== ring all-reduce [%s] : %d ranks, %d elems "
               "(%d/chunk) ===\n",
               gicc_coll::transport_name(), N, count, elems_per_chunk);
    }

    gicc_coll::ring_allreduce(rt, data_buf, d_data, recv_buf, d_recv,
                              elems_per_chunk);

    // ---- Verify ----
    (void)hipMemcpy(h_data.data(), d_data, data_bytes, hipMemcpyDeviceToHost);
    (void)hipDeviceSynchronize();

    int    errors = 0;
    double err_at = -1;
    for (int i = 0; i < count; ++i) {
        const float want = (float)((double)N * (N + 1) / 2.0 + (double)N * (i % 7));
        if (h_data[i] != want) {
            if (errors < 4) {
                fprintf(stderr,
                    "  rank %d: mismatch @%d got=%.1f want=%.1f\n",
                    rank, i, h_data[i], want);
            }
            if (err_at < 0) err_at = i;
            ++errors;
        }
    }
    printf("rank %d allreduce_ring: %s (%d errors over %d elems)\n",
           rank, errors == 0 ? "PASS" : "FAIL", errors, count);

    rt.barrier();
    (void)hipFree(d_data);
    (void)hipFree(d_recv);
    return errors == 0 ? 0 : 4;
}
