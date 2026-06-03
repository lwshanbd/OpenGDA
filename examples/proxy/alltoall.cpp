/*
 * alltoall.cpp - all-to-all over N ranks, built on the GICC put primitive.
 * ONE source, two transports (proxy / DWQ) selected at compile time via
 * -DGICC_CPU_PROXY (see coll_common.hpp).
 *
 * Layout: every rank has a send buffer and a recv buffer of N chunks each.
 *   send[j] is destined for rank j; it lands in rank j's recv[rank].
 *
 * It is a single communication round (no inter-step dependency): each rank
 * issues N-1 RMA writes (one per remote peer) plus a local copy for its own
 * chunk, then one completion + barrier.
 *
 *   Proxy: one kernel issues all N-1 puts, then a single quiet().
 *   DWQ:   host stages all N-1 puts via rt.put(), the kernel fires one
 *          trigger; rt.reset() waits.
 *
 * Note: the DWQ host-staged path consumes one slot per put from the
 * POOL_SIZE=32 batch, so this demo supports up to 33 ranks per round.
 *
 * Verification: send[j][k] = rank*1000 + j*10 + (k%10). After the exchange
 * rank r's recv[i][k] must equal i*1000 + r*10 + (k%10) (the chunk rank i
 * sent to r).
 *
 * Build:  ./examples/proxy/build_collectives.sh
 * Run:    srun -p pdebug -t 2 -N <nodes> -n <ranks> --cpus-per-task=8 \
 *             ./build_ofi/alltoall_proxy   # or _dwq
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "coll_common.hpp"

int main(int argc, char** argv) {
    gicc::Runtime rt;
    const int rank = rt.rank();
    const int N    = rt.size();

#ifndef GICC_CPU_PROXY
    rt.enable_host_wait_mode();
    // Same-node via DWQ/NIC too (avoid IPC+DWQ SDMA deadlock; see
    // Runtime::set_ipc_fastpath).
    rt.set_ipc_fastpath(false);
    if (N - 1 > gicc::Runtime::POOL_SIZE) {
        if (rank == 0)
            fprintf(stderr, "alltoall DWQ: %d peers exceeds POOL_SIZE=%d\n",
                    N - 1, gicc::Runtime::POOL_SIZE);
        return 1;
    }
#endif

    int elems_per_chunk = (argc > 1) ? std::atoi(argv[1]) : 1024;
    if (elems_per_chunk < 1) elems_per_chunk = 1024;
    const size_t chunk_bytes = (size_t)elems_per_chunk * sizeof(int);
    const int    total_elems = N * elems_per_chunk;
    const size_t buf_bytes   = (size_t)total_elems * sizeof(int);

    // send[j][k] = rank*1000 + j*10 + (k%10).
    std::vector<int> h_send(total_elems);
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < elems_per_chunk; ++k)
            h_send[(size_t)j * elems_per_chunk + k] =
                rank * 1000 + j * 10 + (k % 10);

    int* d_send = nullptr;
    int* d_recv = nullptr;
    if (hipMalloc(&d_send, buf_bytes) != hipSuccess ||
        hipMalloc(&d_recv, buf_bytes) != hipSuccess) {
        fprintf(stderr, "rank %d: hipMalloc failed\n", rank);
        return 2;
    }
    (void)hipMemcpy(d_send, h_send.data(), buf_bytes, hipMemcpyHostToDevice);
    (void)hipMemset(d_recv, 0, buf_bytes);
    (void)hipDeviceSynchronize();

    gicc::Buffer send_buf = rt.register_buffer(d_send, buf_bytes, true);
    gicc::Buffer recv_buf = rt.register_buffer(d_recv, buf_bytes, true);
    rt.exchange();
    rt.barrier();

    if (rank == 0) {
        printf("=== all-to-all [%s] : %d ranks, %d elems/chunk ===\n",
               gicc_coll::transport_name(), N, elems_per_chunk);
    }

    gicc_coll::alltoall_run(rt, send_buf, d_send, recv_buf, d_recv,
                            elems_per_chunk);
    (void)chunk_bytes;

    // ---- Verify ----
    std::vector<int> h_recv(total_elems);
    (void)hipMemcpy(h_recv.data(), d_recv, buf_bytes, hipMemcpyDeviceToHost);
    (void)hipDeviceSynchronize();

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < elems_per_chunk; ++k) {
            const int want = i * 1000 + rank * 10 + (k % 10);
            const int got  = h_recv[(size_t)i * elems_per_chunk + k];
            if (got != want) {
                if (errors < 4) {
                    fprintf(stderr,
                        "  rank %d: recv[%d][%d] got=%d want=%d\n",
                        rank, i, k, got, want);
                }
                ++errors;
            }
        }
    }
    printf("rank %d alltoall: %s (%d errors over %d elems)\n",
           rank, errors == 0 ? "PASS" : "FAIL", errors, total_elems);

    rt.barrier();
    (void)hipFree(d_send);
    (void)hipFree(d_recv);
    return errors == 0 ? 0 : 4;
}
