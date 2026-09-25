/**
 * gicc.hpp - GICC (GPU-Initiated Communication and Coordination) Host API
 *
 *   gicc::Runtime rt;
 *   auto buf = rt.register_buffer(d_ptr, size, true);   // device memory
 *   rt.exchange();                                   // collective
 *   gicc::launch<kernel>(rt, grid, block, args...);  // kernel(ctx, args...)
 *
 * Device side: gicc/gicc_device.cuh.
 */
#pragma once

#include "gicc_types.hpp"

namespace gicc {

//==============================================================================
// Runtime class declaration (platform-neutral interface)
//
// The Runtime class is defined by the platform-specific header below.
// All platforms must provide these methods:
//
//   Runtime()                          - Initialize (Bootstrap, GPU, network)
//   ~Runtime()                         - Cleanup
//   Buffer register_buffer(ptr, size, is_device)
//   void exchange()                    - Collective buffer info exchange
//   RemoteBufferInfo remote_buffer(rank, buf_index)
//   DeviceCtx* prepare()               - Context for kernels / target regions
//   void reset()                       - Complete outstanding transfers
//   void barrier()                     - Global barrier (via Bootstrap)
//   int rank(), size(), gpu_id()
//   Bootstrap& boot()                  - Access the Bootstrap for collectives
//   double clock_rate_khz()
//   const char* gpu_name()
//==============================================================================

} // namespace gicc

//==============================================================================
// Include platform-specific Runtime implementation
//==============================================================================

#if defined(GICC_PLATFORM_MLX5)
#include "platform/mlx5/mlx5_runtime.hpp"
#elif defined(GICC_PLATFORM_OFI)
#include "platform/ofi/ofi_runtime.hpp"
#else
#error "No GICC platform defined. Define GICC_PLATFORM_MLX5 or GICC_PLATFORM_OFI."
#endif

//==============================================================================
// gicc::launch — backend-agnostic kernel launch wrapper
//==============================================================================
#include "launch.hpp"
