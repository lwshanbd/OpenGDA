/**
 * gicc_build_config.hpp — Build-time tunables for the GICC-Pilot selector.
 *
 * Defines the macros the selector drives at compile time:
 *   GICC_SLOT_DEPTH     : rotating signal-slot depth for the dissemination
 *                         barrier (default 2 = HPDC'26 shipped value).
 *   GICC_POOL_SIZE      : source-buffer pool depth for active-message
 *                         staging (default 16 = HPDC'26 shipped value).
 *   GICC_CHANNEL_MAP    : proxy-thread channel-mapping strategy.
 *                         0 = static_graph_aware, 1 = modular_hash
 *                         (default 0, only consulted on the ofi_proxy path).
 *   GICC_CAP_PRUNE      : enable build-time CXI NIC-cap feasibility pruning
 *                         (default 1). Set to 0 to build the cap-OFF variant
 *                         used by experiment block B2 (R048/R050/R052/R054).
 *
 * These default to the HPDC'26 static configuration so that a stock build
 * reproduces the parent paper's behaviour. The M2 selector (LLVM pass + host
 * dispatcher) overrides them per translation unit via policy_<platform>.json.
 */
#pragma once

#ifndef GICC_SLOT_DEPTH
#define GICC_SLOT_DEPTH 2
#endif

#ifndef GICC_POOL_SIZE
#define GICC_POOL_SIZE 16
#endif

#ifndef GICC_CHANNEL_MAP
#define GICC_CHANNEL_MAP 0
#endif

#ifndef GICC_CAP_PRUNE
#define GICC_CAP_PRUNE 1
#endif
