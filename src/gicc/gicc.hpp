/**
 * gicc.hpp - GICC (GPU-Initiated Communication and Coordination) Host API
 *
 * Platform-neutral public API for GPU-triggered RDMA operations.
 * The actual implementation is selected at compile time via platform macros.
 *
 * Usage:
 *   gicc::Runtime rt;
 *   auto send_buf = rt.register_buffer(d_send, size, true);
 *   auto recv_buf = rt.register_buffer(d_recv, size, true);
 *   rt.exchange();
 *   auto* ctx = rt.prepare(peer_rank, recv_buf.index);
 *   my_kernel<<<1,1>>>(ctx, send_buf.addr, send_buf.lkey, ...);
 *   cudaDeviceSynchronize();
 *   rt.reset();
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
//   Runtime()                          - Initialize (MPI, GPU, network)
//   ~Runtime()                         - Cleanup
//   Buffer register_buffer(ptr, size, is_device)
//   void exchange()                    - Collective buffer info exchange
//   RemoteBufferInfo remote_buffer(rank, buf_index)
//   DeviceCtx* prepare(peer_rank, remote_buf_index)
//   void reset()                       - Free GPU DeviceCtxs
//   void barrier()                     - MPI barrier
//   int rank(), size(), gpu_id()
//   double clock_rate_khz()
//   const char* gpu_name()
//==============================================================================

} // namespace gicc

//==============================================================================
// Include platform-specific Runtime implementation
//==============================================================================

#if defined(GICC_PLATFORM_MLX5)
#include "platform/mlx5/mlx5_runtime.hpp"
#elif defined(GICC_PLATFORM_CXI)
#include "platform/cxi/cxi_runtime.hpp"
#else
#error "No GICC platform defined. Define GICC_PLATFORM_MLX5 or GICC_PLATFORM_CXI."
#endif
