/**
 * launch.hpp - top-level gicc::launch dispatcher.
 *
 * Selects the per-backend implementation. Both backends expose the same
 * gicc::launch<kernel>(rt, grid, block, peer, dst_buf_idx, args...) form
 * (kernel is a NON-TYPE TEMPLATE PARAMETER, not a runtime argument).
 */
#pragma once

#if defined(GICC_PLATFORM_MLX5)
  #include "platform/mlx5/launch.hpp"
#elif defined(GICC_PLATFORM_OFI)
  #include "platform/ofi/launch.hpp"
#else
  #error "No GICC platform defined."
#endif
