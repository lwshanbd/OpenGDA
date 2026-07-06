// e_dwq_single.cpp - OpenMP-target single-put over the DWQ transport, on the
// public GiOMP API. rank 0 issues one DWQ put to rank 1 from inside an omp
// target region using the ompx_dwq_* markers; the compiler-synthesized host
// trace pre-stages the DWQ descriptor (gicc_runtime_dwq_enqueue) and the device
// flush lowers to the lead-thread MMIO trigger that fires it. rank 1 verifies
// the bytes landed. Cross-node (DWQ is NIC-only) -- run 2 ranks on 2 nodes.
//
// The DWQ markers need GIOMP_ENABLE_DWQ + the LTO pass (a different build from
// the proxy path): use examples/omp/build_omp_dwq.sh (2-pass, ROCm 6.4.0 clang).
// Run: HSA_XNACK=1 GICC_HALO_DWQ=1 flux run -N2 -n2 -g1 -o mpibind=off ./omp_dwq
#include "gicc/omp.h"          // ompx_dwq_put / ompx_dwq_flush (GIOMP_ENABLE_DWQ)
#include <omp.h>
#include <cstdio>
#include <cstddef>

int main() {
    const size_t bytes = 65536;
    ompx_init();

    int my = omp_get_rank_num();
    int nr = omp_get_num_ranks();
    if (nr != 2) {
        if (my == 0) fprintf(stderr, "need 2 ranks\n");
        ompx_finalize();
        return 1;
    }
    int peer = my ^ 1;
    ompx_buffer b = ompx_alloc(bytes);
    ompx_exchange();
    int bidx = b.index;
    unsigned char* d = static_cast<unsigned char*>(b.ptr);

    // Sender stamps 0xAB; receiver stays 0x00 (on-device fill -- no host helper).
    unsigned char fillv = (my == 0) ? (unsigned char)0xAB : (unsigned char)0x00;
    #pragma omp target teams distribute parallel for is_device_ptr(d) firstprivate(fillv, bytes)
    for (size_t i = 0; i < bytes; ++i) d[i] = fillv;
    // Barrier so the receiver's 0x00 fill is DONE before the sender's DWQ write
    // lands -- otherwise the async cross-rank order can let the fill overwrite
    // the delivered 0xAB. (The old host-side fill_buffer was synchronous at init.)
    ompx_barrier();

    gicc::DeviceCtx* d_ctx = ompx_prepare();

    if (my == 0) {
        #pragma omp target is_device_ptr(d_ctx) firstprivate(peer, bidx, bytes)
        {
            ompx_dwq_put(d_ctx, peer, bidx, /*dst_off=*/0,
                                bidx, /*src_off=*/0, bytes);
            ompx_dwq_flush(d_ctx);
        }
        ompx_quiet_host();   // host polls DWQ completion counter
    } else {
        ompx_quiet_host();
    }
    ompx_barrier();

    int rc = 0;
    if (my == 1) {
        int bad = 0;
        #pragma omp target teams distribute parallel for reduction(+:bad) is_device_ptr(d) firstprivate(bytes)
        for (size_t i = 0; i < bytes; ++i)
            if (d[i] != (unsigned char)0xAB) bad++;
        printf("dwq recv: bad_bytes=%d/%zu : %s\n", bad, bytes,
               bad == 0 ? "PASS" : "FAIL");
        rc = (bad == 0) ? 0 : 2;
    }
    ompx_barrier();
    ompx_free(b);
    ompx_finalize();
    return rc;
}
