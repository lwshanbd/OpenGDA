/**
 * launch.hpp - top-level gicc::launch dispatcher.
 *
 * Selects the per-backend implementation. Both backends expose the same
 * gicc::launch<kernel>(rt, grid, block, peer, dst_buf_idx, args...) form
 * (kernel is a NON-TYPE TEMPLATE PARAMETER, not a runtime argument).
 */
#pragma once

// Marker for the LTO host pass: every gicc::launch wrapper carries this
// attribute so GICCHostDiscovery can identify launch instantiation
// sites in IR via @llvm.global.annotations.
#if defined(__clang__)
  #define GICC_LAUNCH_SITE [[clang::annotate("gicc.launch_site")]]
#else
  #define GICC_LAUNCH_SITE
#endif

#if defined(GICC_PLATFORM_MLX5)
  #include "platform/mlx5/launch.hpp"
#elif defined(GICC_PLATFORM_OFI)
  #include "platform/ofi/launch.hpp"
#else
  #error "No GICC platform defined."
#endif
