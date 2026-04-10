/**
 * gicc.hpp - GICC (GPU-Initiated Communication and Coordination) host API
 *
 * Unified umbrella header. Selects the platform-specific Runtime via
 * compile-time dispatch:
 *
 *   -DGICC_PLATFORM_MLX5  -> InfiniBand (NVIDIA + Mellanox), GPU-issued WQEs
 *   -DGICC_PLATFORM_CXI   -> HPE Slingshot (AMD + libfabric), DWQ trigger MMIO
 *
 * Both backends expose the same gicc::Runtime / gicc::Buffer API:
 *
 *   gicc::Runtime rt;                                  // collective
 *   auto src = rt.register_buffer(d_src, n, true);
 *   auto dst = rt.register_buffer(d_dst, n, true);
 *   rt.exchange();                                     // collective
 *
 *   // Queue an RDMA write WITHOUT triggering the NIC.
 *   //  - mlx5: __device__ gicc::put_no_db(ctx, dst, src, n, peer)
 *   //  - cxi : host    rt.put_no_db(src, peer, dst.index, n)
 *   //
 *   // In a kernel:
 *   //   gicc::flush(ctx);   // ring doorbell (mlx5) or write trigger MMIO (cxi)
 *   //   gicc::quiet(ctx);   // wait for completion
 *
 * The single asymmetry is WHERE put_no_db lives: device-side on mlx5
 * (because the GPU can build WQEs), host-side on cxi (because libfabric
 * deferred work must be queued from CPU before the GPU triggers it).
 * The flush/quiet contract — and every other line of user code — is identical.
 */
#pragma once

#include "gicc_types.hpp"

namespace gicc {

//==============================================================================
// gicc::Runtime — defined by the platform header below.
// Required interface (both platforms must provide):
//
//   Runtime(MPI_Comm comm = MPI_COMM_WORLD);
//   ~Runtime();
//   Buffer register_buffer(void* buf, size_t size, bool is_device);
//   void   exchange();
//   RemoteBufferInfo remote_buffer(int rank, int buf_index) const;
//   DeviceCtx* prepare(int peer_rank, int remote_buf_index);
//   void   reset();
//   void   barrier();
//   int    rank() const;
//   int    size() const;
//   int    gpu_id() const;
//
// CXI additionally provides host-side queueing:
//
//   void   put_no_db(const Buffer& src, int dest_rank, int dest_buf_index,
//                    size_t size, size_t src_offset = 0, size_t dst_offset = 0);
//==============================================================================

} // namespace gicc

#if defined(GICC_PLATFORM_MLX5)
#include "platform/mlx5/mlx5_runtime.hpp"
#elif defined(GICC_PLATFORM_CXI)
#include "platform/cxi/cxi_runtime.hpp"
#else
#error "No GICC platform defined. Define GICC_PLATFORM_MLX5 or GICC_PLATFORM_CXI."
#endif
