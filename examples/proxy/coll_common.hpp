/*
 * coll_common.hpp - shared device kernels + host driver for the GICC
 * collective examples (allreduce_ring.cpp, alltoall.cpp).
 *
 * The whole point of these examples is that ONE source compiles to BOTH
 * transports, selected at compile time exactly like the rest of GICC:
 *
 *   - CPU proxy  (-DGICC_CPU_PROXY): the device kernel pushes each RMA
 *     write into the proxy ring; gicc::quiet() spins until the CPU worker
 *     acks the libfabric CQE. Host-side rt.reset() drains the ring.
 *
 *   - DWQ / GPU-trigger (no -DGICC_CPU_PROXY + rt.enable_host_wait_mode()):
 *     the host pre-stages every RMA write via rt.put() into the CXI
 *     deferred work queue; the kernel's lead thread fires them all with a
 *     single MMIO trigger store (gicc::flush). Host-side rt.reset() spins
 *     on the shared completion counter.
 *
 * Both transports share the same completion model at the call site:
 *   issue puts -> hipDeviceSynchronize() -> rt.reset() -> rt.barrier().
 * After the barrier every rank's RMA writes have landed remotely, so a
 * neighbour may safely read what was written into its buffer. This is the
 * same proven pattern minimod and ASF use.
 */
#pragma once

#include <hip/hip_runtime.h>

#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

namespace gicc_coll {

#ifdef GICC_CPU_PROXY
// Proxy mode: lead thread pushes one RMA write into the proxy ring, then
// quiet() blocks until the worker reports the CQE (remote write complete).
__global__ void proxy_put_kernel(gicc::DeviceCtx* ctx, int peer,
                                 int dst_buf, size_t dst_off,
                                 int src_buf, size_t src_off, size_t bytes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::put(ctx, peer, dst_buf, dst_off, src_buf, src_off, bytes);
        gicc::quiet(ctx);
    }
}
#endif

// DWQ mode: lead thread writes the trigger MMIO once, firing every RMA
// write the host pre-staged via rt.put() before the launch. Compiled in
// both builds (gicc::flush exists in both); only launched on the DWQ path.
__global__ void dwq_flush_kernel(gicc::DeviceCtx* ctx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gicc::flush(ctx);
    }
}

// add_kernel - elementwise data[dst_off + i] += recv[i] for the ring
// reduce-scatter phase. Plain HIP kernel, no GICC involvement.
__global__ void add_kernel(float* data, const float* recv,
                           size_t dst_off, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[dst_off + i] += recv[i];
}

//----------------------------------------------------------------------------
// put_one - issue a single RMA write (this rank -> peer), wait for it to
// complete remotely, and barrier so the result is visible to everyone.
//
// Identical call site for both transports; the #ifdef only changes HOW the
// write is launched. After return: my write has landed on `peer`, and
// (because every rank ran the symmetric step) my own buffer holds whatever
// my upstream neighbour wrote into it.
//----------------------------------------------------------------------------
inline void put_one(gicc::Runtime& rt, int peer,
                    const gicc::Buffer& dst_buf, size_t dst_off,
                    const gicc::Buffer& src_buf, size_t src_off,
                    size_t bytes) {
#ifdef GICC_CPU_PROXY
    gicc::DeviceCtx* d = rt.prepare();
    hipLaunchKernelGGL(proxy_put_kernel, dim3(1), dim3(1), 0, 0,
                       d, peer, dst_buf.index, dst_off,
                       src_buf.index, src_off, bytes);
    (void)hipDeviceSynchronize();
    rt.reset();
#else
    rt.put(src_buf, peer, dst_buf.index, bytes, src_off, dst_off);
    gicc::DeviceCtx* d = rt.prepare();
    hipLaunchKernelGGL(dwq_flush_kernel, dim3(1), dim3(1), 0, 0, d);
    (void)hipDeviceSynchronize();
    rt.reset();
#endif
    rt.barrier();
}

// transport_name - for banner printing.
inline const char* transport_name() {
#ifdef GICC_CPU_PROXY
    return "CPU-PROXY";
#else
    return "DWQ (GPU-trigger)";
#endif
}

} // namespace gicc_coll
