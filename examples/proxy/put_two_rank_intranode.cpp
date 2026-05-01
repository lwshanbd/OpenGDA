/*
 * put_two_rank_intranode.cpp - L2 two-rank CPU proxy test.
 *
 * Two MPI ranks on the same node (the gicc bootstrap will figure out
 * locality automatically). Each rank registers one device buffer of
 * BUF_BYTES. Rank 0 writes a deterministic byte pattern into its src half
 * via host->device copy, then issues one gicc::put_no_db that targets
 * rank 1's buffer (offset 0, BUF_BYTES bytes). Rank 1 only calls
 * rt.reset() (which drains the proxy ring and CQ; for an inbound write
 * we get the libfabric-side completion of the local fi_recv equivalent
 * via the target-side counter). After a barrier, rank 1 host-copies its
 * buffer back and asserts byte i == (i & 0xFF).
 *
 * Run (intranode, two ranks):
 *   GICC_PROXY_ENABLED=1 srun -n 2 ./examples/proxy/put_two_rank_intranode
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

__global__ void put_kernel(gicc::DeviceCtx* ctx,
                           int dst_rank, int dst_buf_idx, int src_buf_idx,
                           size_t bytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::put_no_db(ctx,
                        dst_rank,
                        dst_buf_idx, /*dst_offset=*/0,
                        src_buf_idx, /*src_offset=*/0,
                        bytes);
    }
}

int main(int /*argc*/, char** /*argv*/) {
    gicc::Runtime rt;
    const int rank   = rt.rank();
    const int nranks = rt.size();
    if (nranks != 2) {
        if (rank == 0) {
            fprintf(stderr,
                "put_two_rank_intranode: need exactly 2 ranks (got %d)\n",
                nranks);
        }
        return 1;
    }

    constexpr size_t BUF_BYTES = 8192;

    void* d_buf = nullptr;
    if (gpuMalloc(&d_buf, BUF_BYTES) != GPU_SUCCESS) {
        fprintf(stderr, "rank %d: gpuMalloc failed\n", rank);
        return 2;
    }
    (void)gpuMemset(d_buf, 0, BUF_BYTES);

    if (rank == 0) {
        // Pre-fill our local buffer with a deterministic pattern so rank 1
        // can verify byte-by-byte after the proxy write.
        auto* h_pattern = static_cast<uint8_t*>(std::malloc(BUF_BYTES));
        for (size_t i = 0; i < BUF_BYTES; ++i) h_pattern[i] = (uint8_t)(i & 0xFF);
        (void)gpuMemcpy(d_buf, h_pattern, BUF_BYTES, gpuMemcpyHostToDevice);
        std::free(h_pattern);
    }

    auto bh = rt.register_buffer(d_buf, BUF_BYTES, /*is_device=*/true);
    rt.exchange();

    // Make sure rank 1's initial zero-fill is visible before rank 0 fires
    // its put — otherwise the readback could race with the kernel-side
    // gpuMemset above.
    rt.barrier();

    if (rank == 0) {
        gicc::DeviceCtx* d_ctx = rt.prepare();
        gpuLaunchKernel(put_kernel, dim3(1), dim3(1), 0, 0,
                        d_ctx, /*dst_rank=*/1, /*dst_buf=*/bh.index,
                        /*src_buf=*/bh.index, BUF_BYTES);
        if (gpuDeviceSynchronize() != GPU_SUCCESS) {
            fprintf(stderr, "rank 0: kernel sync failed\n");
            return 3;
        }
        rt.reset();   // drains the proxy ring + CQ.
    } else {
        // Rank 1 still calls prepare()+reset() so its proxy worker is
        // started (the inbound write is target-side, no submission needed,
        // but draining keeps the API symmetric).
        (void)rt.prepare();
        rt.reset();
    }

    // Synchronize so rank 1 only reads after rank 0's submission has
    // landed (the proxy ring drain + libfabric CQ both fire on rank 0; the
    // peer-side write completes when the NIC ack returns).
    rt.barrier();

    int rc = 0;
    if (rank == 1) {
        auto* h_check = static_cast<uint8_t*>(std::malloc(BUF_BYTES));
        (void)gpuMemcpy(h_check, d_buf, BUF_BYTES, gpuMemcpyDeviceToHost);
        int errors = 0;
        for (size_t i = 0; i < BUF_BYTES; ++i) {
            if (h_check[i] != (uint8_t)(i & 0xFF)) {
                if (errors < 4) {
                    fprintf(stderr,
                        "  rank 1: mismatch @%zu got=0x%02x want=0x%02x\n",
                        i, h_check[i], (unsigned)(i & 0xFF));
                }
                ++errors;
            }
        }
        printf("put_two_rank_intranode: %s (%d errors over %zu bytes)\n",
               errors == 0 ? "PASS" : "FAIL", errors, BUF_BYTES);
        std::free(h_check);
        rc = (errors == 0) ? 0 : 4;
    }

    rt.barrier();
    (void)gpuFree(d_buf);
    return rc;
}
