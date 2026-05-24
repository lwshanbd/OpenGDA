/**
 * asf_pattern_bench.cpp — microbench that reproduces the ASF DWQ +
 * trigger pattern WITHOUT the LTO Pass and without any ASF code.
 *
 * What ASF does each step:
 *   1. rt.reset() to drain prev step's NIC completions
 *   2. For each cross-node peer: rt.put_no_db(data) + rt.put_no_db(flag)
 *   3. rt.prepare() to refresh trigger_val
 *   4. Launch a kernel that writes flag_buf[27] = epoch + MMIO trigger
 *   5. hipDeviceSynchronize
 *   6. (next step) MPI_Barrier
 *
 * What ASF observes: receivers' flag_buf[ghost_dirs] don't advance past
 * step 2's value — i.e. step 3+ NIC writes never land at peers.
 *
 * This bench reproduces that pattern with:
 *   - Variable rank count (mpirun -n N)
 *   - Each rank picks all OTHER ranks as peers (worst case: full N-1 fanout
 *     forces multi-peer per trigger)
 *   - 10 iterations of pre-stage + trigger
 *   - After each iter, verify flag_buf[peer_slot] == current epoch
 *
 * If THIS bench hangs/fails at iter 3, confirms libfabric/CXI DWQ pattern
 * issue with multi-peer trigger.
 *
 * If THIS bench works, ASF integration has a different bug.
 *
 * Build: see asf_pattern_bench_build.sh — plain hipcc, no LTO pass.
 * Run:
 *   FI_MR_CACHE_MAX_COUNT=0 srun -p pci -N <nodes> -n <ranks> \
 *     --ntasks-per-node=1 ./asf_pattern_bench [niter]
 */

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>

#include "gicc/gicc.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d: %s at %s:%d\n", \
                (int)err, hipGetErrorString(err), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)

constexpr size_t kDataBytes = 4096;   // small data per put
constexpr int    kFlagSlots = 64;     // flag_buf size in uint64_t entries
// flag_buf[my_src_slot] is the source we read for outgoing flags.
// flag_buf[other_slot] are inbound slots peers write to us.
constexpr int    kSrcSlot   = 63;     // our outgoing source

// Tiny kernel: writes flag_buf[kSrcSlot] = epoch, then MMIO trigger.
// Single block, single thread — mirrors what the ASF last-block does.
__global__ void trigger_kernel(uint64_t* flag_buf, uint64_t epoch,
                                gicc::DeviceCtx* ctx)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    flag_buf[kSrcSlot] = epoch;
    __threadfence_system();
    if (ctx->trigger_addr_ != nullptr) {
        *ctx->trigger_addr_ = ctx->trigger_val_;
        __threadfence_system();
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    int niter = (argc >= 2) ? std::atoi(argv[1]) : 10;
    if (niter < 1) niter = 1;

    gicc::Runtime rt;
    rt.enable_host_wait_mode();
    int rank   = rt.rank();
    int nranks = rt.size();

    if (rank == 0) {
        printf("=== asf_pattern_bench: %d ranks, %d iters ===\n", nranks, niter);
        printf("Each rank pre-stages (data+flag) puts to ALL %d other ranks,\n"
               "kernel writes flag_buf[%d]=epoch + MMIO trigger.\n",
               nranks - 1, kSrcSlot);
    }
    if (nranks < 2) {
        if (rank == 0) fprintf(stderr, "need >=2 ranks\n");
        return 1;
    }

    // --- Buffer setup ---
    // data_buf: each peer writes into a different slot of MY data_buf.
    // flag_buf: each peer writes its epoch into MY flag_buf[their_rank_id].
    size_t data_total = kDataBytes * nranks;
    void* d_data = nullptr;
    HIP_CHECK(hipMalloc(&d_data, data_total));
    HIP_CHECK(hipMemset(d_data, 0, data_total));

    void* d_flag = nullptr;
    HIP_CHECK(hipExtMallocWithFlags(&d_flag, kFlagSlots * sizeof(uint64_t),
                                     hipDeviceMallocUncached));
    HIP_CHECK(hipMemset(d_flag, 0, kFlagSlots * sizeof(uint64_t)));
    uint64_t* flag_buf = static_cast<uint64_t*>(d_flag);

    HIP_CHECK(hipDeviceSynchronize());

    auto bh_data = rt.register_buffer(d_data, data_total, true);
    auto bh_flag = rt.register_buffer(d_flag, kFlagSlots * sizeof(uint64_t),
                                       true);
    rt.exchange();
    rt.barrier();

    if (rank == 0) {
        printf("buffers registered: data idx=%d (%zu B), flag idx=%d (%zu B)\n",
               bh_data.index, data_total, bh_flag.index,
               kFlagSlots * sizeof(uint64_t));
    }

    // --- Iteration loop ---
    std::vector<uint64_t> hflag(kFlagSlots);
    int failures = 0;
    for (int it = 1; it <= niter; ++it) {
        // Bump epoch to current iter.
        uint64_t epoch = (uint64_t)it;

        // Drain previous iter's NIC.
        if (it > 1) rt.reset();

        // Pre-stage: data + flag put to EACH other rank.  The flag put
        // writes our flag_buf[kSrcSlot] (== epoch after kernel runs) into
        // peer's flag_buf[my_rank].  After this iter, peer's
        // flag_buf[my_rank] should equal `epoch`.
        const char* _minimal = std::getenv("ASF_PATTERN_BENCH_MINIMAL");
        bool minimal = (_minimal && _minimal[0] == '1');
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == rank) continue;
            if (!minimal) {
                // Data put: write into peer's data_buf at rank-keyed offset.
                size_t data_dst_off = (size_t)rank * kDataBytes;
                rt.put_no_db(bh_data, peer, bh_data.index, kDataBytes,
                             /*src_off=*/0, data_dst_off);
            }
            // Flag put: peer's flag_buf[my_rank] gets our flag_buf[kSrcSlot].
            size_t flag_src_off = (size_t)kSrcSlot * sizeof(uint64_t);
            size_t flag_dst_off = (size_t)rank * sizeof(uint64_t);
            rt.put_no_db(bh_flag, peer, bh_flag.index, sizeof(uint64_t),
                         flag_src_off, flag_dst_off);
        }

        // Write the source flag slot from host (h2d memcpy).
        HIP_CHECK(hipMemcpy(&flag_buf[kSrcSlot], &epoch, sizeof(uint64_t),
                             hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        // Refresh trigger_val_ to current mono_total_ops_.  This also
        // writes the host-pinned DeviceCtx, so reading trigger_val_
        // host-side is well-defined.
        gicc::DeviceCtx* ctx = rt.prepare();

        // Trigger semantic switch:
        //   ASF_PATTERN_BENCH_HOST_TRIGGER=1 → host writes the MMIO trigger
        //   ASF_PATTERN_BENCH_NO_TRIGGER=1   → DON'T trigger at all
        //                                       (diagnostic: if rt.reset()
        //                                        still returns immediately,
        //                                        the completion counter is
        //                                        bumped at queue-time not
        //                                        delivery-time -> bug)
        //   default                          → kernel writes MMIO
        const char* _host_trig = std::getenv("ASF_PATTERN_BENCH_HOST_TRIGGER");
        const char* _no_trig   = std::getenv("ASF_PATTERN_BENCH_NO_TRIGGER");
        if (_no_trig && _no_trig[0] == '1') {
            // Skip the trigger entirely.
            (void)ctx;
        } else if (_host_trig && _host_trig[0] == '1') {
            volatile uint64_t* host_trig =
                (volatile uint64_t*)rt.fabric().fabric->trigger_mmio_addr;
            *host_trig = ctx->trigger_val_;
            __sync_synchronize();
        } else {
            trigger_kernel<<<1, 1>>>(flag_buf, epoch, ctx);
            HIP_CHECK(hipDeviceSynchronize());
        }

        // Read trigger counter value as libfabric sees it.  If this
        // doesn't increment monotonically across iters, the MMIO write
        // isn't actually reaching the NIC counter and DWQ descriptors
        // can't fire.
        uint64_t trig_val_after = fi_cntr_read(rt.fabric().fabric->trigger_cntr);
        uint64_t comp_val_before = fi_cntr_read(/*shared_completion_cntr*/
            rt.fabric().fabric->completion_cntr);
        if (rank == 0) {
            printf("iter %d: trigger_cntr=%llu  completion_cntr=%llu  "
                   "expected_trigger=%llu\n",
                   it, (unsigned long long)trig_val_after,
                   (unsigned long long)comp_val_before,
                   (unsigned long long)ctx->trigger_val_);
        }

        // Drain THIS iter's NIC.
        rt.reset();

        // Explicit libfabric progress polling — many fi_cq_read calls to
        // make sure the receive-side has actually processed incoming
        // RMA writes.  ASF_PATTERN_BENCH_POLL_ITERS=N (default 0).
        const char* _poll_env = std::getenv("ASF_PATTERN_BENCH_POLL_ITERS");
        int poll_iters = _poll_env ? std::atoi(_poll_env) : 0;
        for (int p = 0; p < poll_iters; ++p) {
            fi_cq_read(rt.fabric().fabric->cq, nullptr, 0);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // HYPOTHESIS TEST: insert a delay between rt.reset() (which only
        // confirms TRANSMIT-complete in CXI's default semantic) and the
        // barrier/verify.  If this makes the bench pass, the bug is that
        // libfabric DWQ RMA writes complete with TRANSMIT-COMPLETE rather
        // than DELIVERY-COMPLETE, so peer's NIC hasn't actually committed
        // our write to peer's memory when rt.reset() returns.
        // Tunable via env: ASF_PATTERN_BENCH_DELAY_US=1000 (default 0).
        const char* _delay_env = std::getenv("ASF_PATTERN_BENCH_DELAY_US");
        int delay_us = _delay_env ? std::atoi(_delay_env) : 0;
        if (delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        }

        // Sync everyone so peers' writes have had time to arrive.
        MPI_Barrier(MPI_COMM_WORLD);

        // Read back our flag_buf and verify each peer's slot == epoch.
        HIP_CHECK(hipMemcpy(hflag.data(), d_flag,
                             kFlagSlots * sizeof(uint64_t),
                             hipMemcpyDeviceToHost));

        int local_fail = 0;
        if (rank == 0) {
            printf("iter %d: rank 0 flag_buf={", it);
            for (int p = 0; p < nranks; ++p) {
                printf("%llu%s", (unsigned long long)hflag[p],
                       p == nranks - 1 ? "" : ",");
            }
            printf("}\n");
        }
        for (int p = 0; p < nranks; ++p) {
            if (p == rank) continue;
            if (hflag[p] != epoch) {
                ++local_fail;
                if (local_fail <= 4) {
                    fprintf(stderr, "[r%d iter=%d] flag[peer=%d]=%llu "
                            "EXPECTED %llu\n",
                            rank, it, p, (unsigned long long)hflag[p],
                            (unsigned long long)epoch);
                }
            }
        }
        if (local_fail > 0) ++failures;

        // Aggregate failures across ranks for a clean summary.
        int global_fail = 0;
        MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        if (rank == 0) {
            printf("iter %d: global_failures_in_iter=%d\n", it, global_fail);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int total_global_fail = 0;
    MPI_Allreduce(&failures, &total_global_fail, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (rank == 0) {
        printf("=== DONE: %d ranks × %d iters, ranks with any failure: %d ===\n",
               nranks, niter, total_global_fail);
    }

    rt.barrier();
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_flag));
    return total_global_fail == 0 ? 0 : 1;
}
