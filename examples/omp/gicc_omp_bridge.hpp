/*
 * gicc_omp_bridge.hpp - thin interface between an OpenMP-target application TU
 * and the HIP-compiled GICC runtime. The OpenMP TU calls these; the
 * implementation (gicc_omp_bridge_hip.cpp) is compiled with -x hip and owns a
 * static gicc::Runtime. Only the HIP-free DeviceCtx is exposed across the
 * boundary, so this header is safe to include in a -fopenmp TU.
 */
#pragma once
#include <cstddef>
#include "gicc/platform/ofi/device_ctx.hpp"   // gicc::DeviceCtx (HIP-free)

namespace gicc_omp_bridge {

// Init MPI (if needed) + gicc::Runtime, register ONE device buffer of `bytes`,
// and exchange the RMA address book. Call once at start.
void init(size_t bytes);

int  rank();
int  nranks();
int  buf_index();

// Per-iteration: returns the DEVICE pointer to the DeviceCtx (pass to the
// omp target region via is_device_ptr).
gicc::DeviceCtx* prepare();

// Drain the proxy ring + completion (host-side completion of all puts).
void reset();

// MPI_Barrier(MPI_COMM_WORLD).
void barrier();

// Fill the entire registered buffer with `value` (host -> device).
void fill_buffer(unsigned char value, size_t bytes);

// Count bytes in the registered buffer that differ from `expected` (device -> host).
size_t count_mismatches(unsigned char expected, size_t bytes);

// Free buffer + MPI_Finalize (if we initialized it).
void finalize();

}  // namespace gicc_omp_bridge
