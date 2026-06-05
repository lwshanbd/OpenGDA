// e_dwq_single.cpp - OpenMP-target single-put end-to-end test over the DWQ
// transport (Phase-2 M4). rank 0 issues one put to rank 1 from inside an omp
// target region using the gicc::omp_dwq markers; the compiler-synthesized host
// trace pre-stages the DWQ descriptor (gicc_runtime_dwq_enqueue) and the
// device flush lowers to the lead-thread MMIO trigger that fires it. rank 1
// verifies the bytes landed. Cross-node (NIC) per srun layout.
#include <omp.h>
#include <cstdio>
#include <cstddef>

#include "examples/omp/gicc_omp_dwq.hpp"
#include "examples/omp/gicc_omp_bridge.hpp"

int main() {
    const size_t bytes = 65536;
    gicc_omp_bridge::init(bytes);

    int my = gicc_omp_bridge::rank();
    int nr = gicc_omp_bridge::nranks();
    if (nr != 2) {
        if (my == 0) fprintf(stderr, "need 2 ranks\n");
        gicc_omp_bridge::finalize();
        return 1;
    }
    int peer = my ^ 1;
    int bidx = gicc_omp_bridge::buf_index();

    // Sender stamps 0xAB; receiver stays 0x00.
    gicc_omp_bridge::fill_buffer(my == 0 ? 0xAB : 0x00, bytes);

    gicc::DeviceCtx* d_ctx = gicc_omp_bridge::prepare();

    if (my == 0) {
        #pragma omp target is_device_ptr(d_ctx)
        {
            gicc::omp_dwq::put(d_ctx, peer, bidx, /*dst_off=*/0,
                                      bidx, /*src_off=*/0, bytes);
            gicc::omp_dwq::flush(d_ctx);
        }
        gicc_omp_bridge::reset();   // host polls DWQ completion counter
    } else {
        gicc_omp_bridge::reset();
    }
    gicc_omp_bridge::barrier();

    int rc = 0;
    if (my == 1) {
        size_t bad = gicc_omp_bridge::count_mismatches(0xAB, bytes);
        printf("dwq recv: bad_bytes=%zu/%zu : %s\n", bad, bytes,
               bad == 0 ? "PASS" : "FAIL");
        rc = (bad == 0) ? 0 : 2;
    }
    gicc_omp_bridge::barrier();
    gicc_omp_bridge::finalize();
    return rc;
}
