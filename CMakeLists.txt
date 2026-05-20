cmake_minimum_required(VERSION 3.21)

# ============================================================================
# Backend selection — set BEFORE project() so the language list is correct.
# ============================================================================
set(GICC_BACKEND "mlx5" CACHE STRING "GICC backend: mlx5 (NVIDIA+IB) or ofi (AMD+libfabric)")
set_property(CACHE GICC_BACKEND PROPERTY STRINGS mlx5 ofi)

# ============================================================================
# GPU API selection — chooses between HIP and CUDA for GPU-side code paths
# inside the OFI backend (Fabric / MemoryRegion / GpuDeviceContext).
# Default is hip to preserve the historical Tioga build behaviour.
# ============================================================================
set(GICC_GPU_API "hip" CACHE STRING "GPU API: hip | cuda")
set_property(CACHE GICC_GPU_API PROPERTY STRINGS hip cuda)

if(GICC_BACKEND STREQUAL "ofi")
    if(GICC_GPU_API STREQUAL "cuda")
        set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures")
        project(GICC VERSION 1.0.0 LANGUAGES CXX CUDA)
        add_compile_definitions(GICC_GPU_CUDA=1)
    elseif(GICC_GPU_API STREQUAL "hip")
        set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "HIP architectures")
        project(GICC VERSION 1.0.0 LANGUAGES CXX HIP)
        add_compile_definitions(GICC_GPU_HIP=1)
    else()
        message(FATAL_ERROR "GICC_GPU_API must be 'hip' or 'cuda' (got '${GICC_GPU_API}')")
    endif()
elseif(GICC_BACKEND STREQUAL "mlx5")
    project(GICC VERSION 1.0.0 LANGUAGES CXX CUDA)
    add_compile_definitions(GICC_GPU_CUDA=1)
else()
    message(FATAL_ERROR "GICC_BACKEND must be 'mlx5' or 'ofi' (got '${GICC_BACKEND}')")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES    "Build GICC examples"    ON)

# ============================================================================
# CPU Proxy path (Task 4+) — header-only ring + tests live under
# src/gicc/platform/ofi/proxy/ and tests/proxy/. The full proxy thread,
# libfabric integration, and runtime wiring land in later tasks.
# ============================================================================
option(GICC_ENABLE_CPU_PROXY "Build CPU Proxy path under platform/ofi/" OFF)

if(GICC_ENABLE_CPU_PROXY)
    add_compile_definitions(GICC_CPU_PROXY=1)
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
        # examples/ofi is HIP-only (Tioga). Skip under the CUDA build path
        # so the proxy CUDA examples can configure on Delta.
        if(GICC_GPU_API STREQUAL "hip")
            add_subdirectory(examples/ofi)
        endif()
    endif()
endif()

if(GICC_ENABLE_CPU_PROXY)
    # tests/proxy and examples/proxy currently target the OFI/HIP proxy path
    # (Tioga). Skip them under the MLX5/CUDA backend so the MLX5 CPU-proxy
    # build doesn't try to drag in HIP. A future refactor should split the
    # truly backend-agnostic tests (e.g. the D2H ring microtest) out from
    # the OFI-specific ones.
    if(GICC_BACKEND STREQUAL "ofi")
        add_subdirectory(tests/proxy)
        if(BUILD_EXAMPLES)
            add_subdirectory(examples/proxy)
        endif()
    endif()
endif()

# ============================================================================
# LTO pass plugin (separate build because of Clang ABI requirements;
# also buildable standalone via tools/gicc-passes/CMakeLists.txt).
# ============================================================================
option(GICC_BUILD_PASSES "Build the LTO pass plugin (libgicc-passes.so)" OFF)
if(GICC_BUILD_PASSES)
    add_subdirectory(tools/gicc-passes)
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "GICC Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version:      ${PROJECT_VERSION}")
message(STATUS "Backend:      ${GICC_BACKEND}")
message(STATUS "GPU API:      ${GICC_GPU_API}")
message(STATUS "Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Examples:     ${BUILD_EXAMPLES}")
message(STATUS "========================================")
message(STATUS "")
