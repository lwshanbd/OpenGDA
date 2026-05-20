/*
 * gicc_context_types.hpp - POD layout of GiccContext + buffer entry types.
 *
 * Pure-data slice of gicc_context.cuh so the runtime header (and any
 * host-only TU that routes through it) can describe the type without
 * pulling in the __device__ helpers (gicc::put / get / flush / quiet)
 * whose bodies reference CUDA atomics + PTX intrinsics.
 *
 * The full device-side API still lives in gicc_context.cuh and is
 * included by gicc/gicc_device.cuh for user kernels compiled by nvcc.
 */
#pragma once

#include <cstdint>

#if defined(GICC_PLATFORM_MLX5)
#include "gicc/platform/mlx5/device_state_opt.hpp"
namespace gicc { using RawDeviceCtx = gicc::mlx5::DeviceStateOpt; }
#endif

namespace gicc {

enum {
    GICC_MAX_BUFS  = 16,
    GICC_MAX_PEERS = 64,
};

struct BufEntry {
    uint64_t addr;
    uint64_t size;
    uint32_t lkey;
    uint32_t rkey;
};

struct PeerBufEntry {
    uint64_t addr;
    uint32_t rkey;
};

struct GiccContext {
    int my_rank;
    int num_peers;

    RawDeviceCtx* peer_ctxs[GICC_MAX_PEERS];

    BufEntry      local_bufs[GICC_MAX_BUFS];
    int           num_local_bufs;

    PeerBufEntry  remote_bufs[GICC_MAX_PEERS][GICC_MAX_BUFS];
};

} // namespace gicc
