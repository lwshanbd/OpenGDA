/*
 * gicc_omp_device.hpp - the gicc::omp device API on InfiniBand.
 *
 * Same names and arguments as the libfabric version
 * (src/gicc/platform/ofi/gicc_omp_device.hpp), so gicc/omp.h and the
 * applications built on it compile unchanged. The transport differs: there
 * the target region pushes a descriptor for a CPU proxy or rings a
 * pre-staged DWQ; here the GPU thread itself writes the WQE and rings the
 * NIC doorbell (gda_device.hpp). `lane` picks one of the per-peer QPs, so
 * transfers on different lanes never wait on each other's doorbells.
 *
 * Usage from the application:
 *   #pragma omp target is_device_ptr(ctx)
 *   { gicc::omp::put(ctx, peer, dbuf, doff, sbuf, soff, bytes); }
 */
#pragma once

#include "gicc/platform/mlx5/gda_device.hpp"   // declare-target when -fopenmp
#include <cstddef>
#include <cstdint>

#pragma omp declare target

namespace gicc {
namespace omp {

using Ctx = gicc::mlx5::GdaCtx;

inline void put(Ctx* ctx, int target_rank,
                int dst_buf, size_t dst_offset,
                int src_buf, size_t src_offset,
                size_t size, int lane = 0) {
    if (!ctx) return;
    gicc::mlx5::gda::put(ctx, target_rank, dst_buf, dst_offset,
                         src_buf, src_offset, size, lane);
}

// Payload, then an 8-byte signal carrying `value` at byte `sig_offset` of the
// peer's signal inbox, both behind one doorbell on one QP.
inline void put_signal(Ctx* ctx, int target_rank,
                       int dst_buf, size_t dst_offset,
                       int src_buf, size_t src_offset, size_t size,
                       size_t sig_offset, uint64_t value, int lane = 0) {
    if (!ctx) return;
    gicc::mlx5::gda::put_signal(ctx, target_rank, dst_buf, dst_offset,
                                src_buf, src_offset, size, sig_offset, value, lane);
}

inline void get(Ctx* ctx, int source_rank,
                int src_buf, size_t src_offset,
                int dst_buf, size_t dst_offset,
                size_t size, int lane = 0) {
    if (!ctx) return;
    gicc::mlx5::gda::get(ctx, source_rank, src_buf, src_offset,
                         dst_buf, dst_offset, size, lane);
}

// Slot reservation is already a single atomic add, so the single-issuer form
// has nothing to skip.
inline void get_single(Ctx* ctx, int source_rank,
                       int src_buf, size_t src_offset,
                       int dst_buf, size_t dst_offset,
                       size_t size, int lane = 0) {
    get(ctx, source_rank, src_buf, src_offset, dst_buf, dst_offset, size, lane);
}

inline void quiet(Ctx* ctx, int lane = 0) {
    if (!ctx) return;
    gicc::mlx5::gda::quiet(ctx, lane);
}

}  // namespace omp
}  // namespace gicc

#pragma omp end declare target
