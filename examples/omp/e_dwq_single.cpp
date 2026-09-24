// e_dwq_single.cpp - OpenMP-target single-put over the DWQ transport, on the
// public GiOMP API. rank 0 issues one DWQ put to rank 1 from inside an omp
// target region using the ompx_dwq_*_dev markers; the compiler-synthesized host
// trace pre-stages the DWQ descriptor (gicc_runtime_dwq_enqueue) and the device
// flush lowers to the lead-thread MMIO trigger that fires it. rank 1 verifies
// the bytes landed. Cross-node (DWQ is NIC-only) -- run 2 ranks on 2 nodes.
//
// The DWQ markers need GIOMP_ENABLE_DWQ + the LTO pass (a different build from
// the proxy path): use examples/omp/build_omp_dwq.sh (2-pass, ROCm 6.4.0 clang).
// Run: HSA_XNACK=1 GICC_HALO_DWQ=1 flux run -N2 -n2 -g1 -o mpibind=off ./omp_dwq
#include "gicc/omp.h"
#include "gicc/omp_compiler.h"          // ompx_dwq_put_dev / ompx_dwq_flush_dev (GIOMP_ENABLE_DWQ)
#include <omp.h>
#include <cstdio>
#include <cstddef>

int main() {
    const size_t bytes = 65536;
    ompx_init();

    int my = ompx_get_rank_num();
    int nr = ompx_get_num_ranks();
    if (nr != 2) {
        if (my == 0) fprintf(stderr, "need 2 ranks\n");
        ompx_finalize();
        return 1;
    }
    int peer = my ^ 1;
    unsigned char* d = static_cast<unsigned char*>(ompx_alloc(bytes));

    // Sender stamps 0xAB; receiver stays 0x00 (on-device fill -- no host helper).
    unsigned char fillv = (my == 0) ? (unsigned char)0xAB : (unsigned char)0x00;
    #pragma omp target teams distribute parallel for is_device_ptr(d) firstprivate(fillv, bytes)
    for (size_t i = 0; i < bytes; ++i) d[i] = fillv;
    // Barrier so the receiver's 0x00 fill is DONE before the sender's DWQ write
    // lands -- otherwise the async cross-rank order can let the fill overwrite
    // the delivered 0xAB. (The old host-side fill_buffer was synchronous at init.)
    ompx_barrier();

    ompx_ctx* d_ctx = ompx_prepare();

    // The markers name the heap registration plus a heap-relative offset, as
    // host scalars: the marker arguments must stay host-knowable for the LTO
    // pass, so they may only be built from kernel formals and constants.
    const int    heap_buf  = ompx_heap_index();
    const size_t heap_base = ompx_heap_offset_of(d);

    if (my == 0) {
        #pragma omp target is_device_ptr(d_ctx) firstprivate(peer, heap_buf, heap_base, bytes)
        {
            ompx_dwq_put_dev(d_ctx, peer, heap_buf, /*dst_off=*/heap_base,
                                          heap_buf, /*src_off=*/heap_base, bytes);
            ompx_dwq_flush_dev(d_ctx);
        }
    }
    ompx_fence();   // host polls DWQ completion counter, then rendezvous

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
    ompx_free(d);
    ompx_finalize();
    return rc;
}
