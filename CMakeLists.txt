cmake_minimum_required(VERSION 3.21)

# ============================================================================
# Backend selection — set BEFORE project() so the language list is correct.
# ============================================================================
set(GICC_BACKEND "mlx5" CACHE STRING "GICC backend: mlx5 (NVIDIA+IB) or ofi (AMD+libfabric)")
set_property(CACHE GICC_BACKEND PROPERTY STRINGS mlx5 ofi)

if(GICC_BACKEND STREQUAL "ofi")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "HIP architectures")
    project(GICC VERSION 1.0.0 LANGUAGES CXX HIP)
elseif(GICC_BACKEND STREQUAL "mlx5")
    project(GICC VERSION 1.0.0 LANGUAGES CXX CUDA)
else()
    message(FATAL_ERROR "GICC_BACKEND must be 'mlx5' or 'ofi' (got '${GICC_BACKEND}')")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES    "Build GICC examples"    ON)

# ============================================================================
# M2 selector build-time tunables (see refine-logs/M2_IMPLEMENTATION_PLAN.md)
# These default to the HPDC'26 shipped values so that a stock build
# reproduces the parent paper. The selector (LLVM pass + host dispatcher)
# overrides them per translation unit at LTO time.
# ============================================================================
set(GICC_SLOT_DEPTH  "2"  CACHE STRING "Rotating signal-slot depth for the OFI dissemination barrier")
set(GICC_POOL_SIZE   "16" CACHE STRING "Source-buffer pool depth for active-message staging")
set(GICC_CHANNEL_MAP "0"  CACHE STRING "Proxy channel-mapping strategy (0=static_graph_aware, 1=modular_hash)")
option(GICC_CAP_PRUNE "Enable compile-time NIC-cap feasibility pruning (disable for B2 cap-OFF builds)" ON)

add_compile_definitions(
    GICC_SLOT_DEPTH=${GICC_SLOT_DEPTH}
    GICC_POOL_SIZE=${GICC_POOL_SIZE}
    GICC_CHANNEL_MAP=${GICC_CHANNEL_MAP})

if(GICC_CAP_PRUNE)
    add_compile_definitions(GICC_CAP_PRUNE=1)
else()
    add_compile_definitions(GICC_CAP_PRUNE=0)
endif()

# ============================================================================
# Subdirectories
# ============================================================================
if(GICC_BACKEND STREQUAL "mlx5")
    add_subdirectory(src)
    if(BUILD_EXAMPLES)
        add_subdirectory(examples/gicc)
    endif()
elseif(GICC_BACKEND STREQUAL "ofi")
    if(BUILD_EXAMPLES)
        add_subdirectory(examples/ofi)
    endif()
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "GICC Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version:       ${PROJECT_VERSION}")
message(STATUS "Backend:       ${GICC_BACKEND}")
message(STATUS "Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler:  ${CMAKE_CXX_COMPILER}")
message(STATUS "Examples:      ${BUILD_EXAMPLES}")
message(STATUS "SLOT_DEPTH:    ${GICC_SLOT_DEPTH}")
message(STATUS "POOL_SIZE:     ${GICC_POOL_SIZE}")
message(STATUS "CHANNEL_MAP:   ${GICC_CHANNEL_MAP}")
message(STATUS "CAP_PRUNE:     ${GICC_CAP_PRUNE}")
message(STATUS "========================================")
message(STATUS "")
