/*
 * device_ctx.hpp - GPU-accessible DeviceCtx struct, shared by the HIP device
 * API (ofi_device.cuh) and the OpenMP-target port (gicc_omp_device.hpp).
 *
 * Kept free of HIP/CUDA intrinsics so it parses under a plain -fopenmp TU.
 * This is the SINGLE definition of gicc::DeviceCtx — do not duplicate it.
 */
#pragma once

#include <cstdint>

namespace gicc {

//==============================================================================
// DeviceCtx — GPU-accessible context for the libfabric/CXI backend.
// Shrunk per spec §7.3: only the trigger MMIO addr + threshold value
// survive into the LTO-only world. Everything else (IPC table, local
// buf table, command ring) was Pattern C state, deleted in Phase 6.
//==============================================================================
struct DeviceCtx {
    volatile uint64_t* trigger_addr_;       // MMIO trigger counter
    uint64_t           trigger_val_;        // value to write to trigger_addr_
#ifdef GICC_CPU_PROXY
    // Single-ring back-compat pointer (== proxy_rings_arr[0]). Used as a
    // fallback when proxy_rings_arr has not been set up.
    void*              proxy_ring;          // gicc::proxy::ProxyRing*
    // Multi-ring fan-out: device-mapped array of N ring pointers, plus N.
    // put/get/quiet pick proxy_rings_arr[lane % num_proxy_rings] to spread
    // submissions across independent proxy threads — N separate workers,
    // each owning its own ring.
    void**             proxy_rings_arr;
    int                num_proxy_rings;
#endif
    // Locality-aware collectives: flat table of peer IPC-mapped buffer base
    // pointers, indexed [peer * ipc_n_bufs + buf_idx]. Entry is non-null only
    // when `peer` is same-node AND that buffer was IPC-mapped, so a kernel can
    // write a same-node peer's buffer directly over xGMI (no NIC). Null entry
    // => not local/mapped => fall back to put/proxy. Set by prepare().
    void**             peer_ipc_base = nullptr;
    int                ipc_n_bufs    = 0;

    // Symmetric heap (GiOMP ompx_* API): base of this rank's heap and the
    // address-book index it was registered under. A device-side put converts an
    // address to an offset with (addr - heap_base), so the kernel needs no
    // buffer handle. Set by Runtime::set_symmetric_heap(); zero when the
    // application does not use the ompx_* allocator.
    void*              heap_base     = nullptr;
    int                heap_buf      = -1;

    // Signal slots (ompx_put_signal / ompx_signal_wait): the device alias of
    // this rank's inbox, its address-book index, and -- under DWQ only -- one
    // trigger doorbell per slot. A doorbell is null until the host first
    // stages a put_signal on that slot; sig_trigger itself is null under the
    // CPU proxy, which is how a device put_signal picks its transport.
    uint64_t*           sig_base     = nullptr;
    int                 sig_buf      = -1;
    volatile uint64_t** sig_trigger  = nullptr;
};

} // namespace gicc
