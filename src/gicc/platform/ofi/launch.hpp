/**
 * launch.hpp - gicc::launch wrapper for OFI backend.
 *
 * Thin wrapper: prepare() snapshots trigger_addr_ / trigger_val_ into the
 * DeviceCtx, then hipLaunchKernelGGL fires the user kernel. The LTO host
 * pass identifies this wrapper via the GICC_LAUNCH_SITE annotation
 * (clang::annotate("gicc.launch_site")) and synthesizes a trace function
 * call BEFORE the launch site that pre-stages every put_no_db / get_no_db
 * via the gicc_runtime_* C ABI.
 *
 * The kernel is passed as a NON-TYPE TEMPLATE PARAMETER (constant
 * expression) so the LTO host pass can recover the kernel's mangled
 * name from the wrapper's mangled NTTP and look up its trace template.
 */
#pragma once

#include <hip/hip_runtime.h>

#include "gicc/launch.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"

namespace gicc {

template <auto Kernel, typename... Args>
GICC_LAUNCH_SITE
inline void launch(Runtime& rt,
                   dim3 grid, dim3 block,
                   Args... args)
{
    DeviceCtx* ctx = rt.prepare();
    hipLaunchKernelGGL(Kernel, grid, block, 0, 0, ctx, args...);
}

template <auto Kernel, typename... Args>
GICC_LAUNCH_SITE
inline void launch(Runtime& rt,
                   dim3 grid, dim3 block,
                   size_t shmem_bytes, hipStream_t stream,
                   Args... args)
{
    DeviceCtx* ctx = rt.prepare();
    hipLaunchKernelGGL(Kernel, grid, block, shmem_bytes, stream, ctx, args...);
}

} // namespace gicc
