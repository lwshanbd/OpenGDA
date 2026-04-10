/**
 * gicc.hpp - GICC (GPU-Initiated Communication and Coordination) Host API
 *
 * Simplified NVSHMEM-style API:
 *
 *   gicc::init(MPI_COMM_WORLD);
 *   void* send = gicc::malloc(size);              // collective
 *   void* recv = gicc::malloc(size);              // collective
 *   auto* ctx  = gicc::context();                 // GPU-accessible context
 *   void* peer_recv = gicc::remote_ptr(recv, 1);  // remote address on PE 1
 *   my_kernel<<<...>>>(ctx, peer_recv, send, size, 1);
 *   gicc::finalize();
 *
 * Device-side (in kernel):
 *   gicc::put(ctx, dst, src, size, peer);         // one-line RDMA
 *   gicc::flush(ctx, peer);
 *   gicc::quiet(ctx, peer);
 *
 * Legacy Runtime API is still available for advanced use.
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
#elif defined(GICC_PLATFORM_OFI)
#include "platform/ofi/ofi_runtime.hpp"
#else
#error "No GICC platform defined. Define GICC_PLATFORM_MLX5 or GICC_PLATFORM_OFI."
#endif

//==============================================================================
// Simplified NVSHMEM-style API — currently mlx5-only (depends on CUDA)
//==============================================================================

#if defined(GICC_PLATFORM_MLX5)
#include "platform/mlx5/gicc_api.hpp"
#endif
