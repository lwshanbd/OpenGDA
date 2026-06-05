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
};

} // namespace gicc
