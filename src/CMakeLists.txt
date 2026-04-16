cmake_minimum_required(VERSION 3.18)
project(gicc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ===========================================================================
# Dependencies
# ===========================================================================

# Independent build knobs. Each has a fixed default; no cross-option inference.
set(GICC_BACKEND   "ofi" CACHE STRING "Network backend: mlx5 | ofi")
set(GICC_BOOTSTRAP "mpi" CACHE STRING "Bootstrap backend: mpi | pmi2")
set_property(CACHE GICC_BACKEND   PROPERTY STRINGS "mlx5" "ofi")
set_property(CACHE GICC_BOOTSTRAP PROPERTY STRINGS "mpi"  "pmi2")

find_package(CUDAToolkit REQUIRED)

find_library(IBVERBS_LIBRARY ibverbs)
if(NOT IBVERBS_LIBRARY)
    message(FATAL_ERROR "libibverbs not found")
endif()

find_library(MLX5_LIBRARY mlx5)
if(NOT MLX5_LIBRARY)
    message(FATAL_ERROR "libmlx5 not found")
endif()

# ===========================================================================
# GICC Shared Library
# ===========================================================================

add_library(gicc SHARED
    gicc_lib.cu
)

# Include directories: src/ is the root so includes work as <gicc/...>
target_include_directories(gicc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(gicc PUBLIC GICC_PLATFORM_MLX5)

target_link_libraries(gicc
    PUBLIC
        ${IBVERBS_LIBRARY}
        ${MLX5_LIBRARY}
        CUDA::cudart
)

# ===========================================================================
# Bootstrap backend
# ===========================================================================

if(GICC_BOOTSTRAP STREQUAL "mpi")
    find_package(MPI REQUIRED)
    target_compile_definitions(gicc PUBLIC GICC_BOOTSTRAP_MPI=1)
    target_link_libraries     (gicc PUBLIC MPI::MPI_CXX)
elseif(GICC_BOOTSTRAP STREQUAL "pmi2")
    find_path   (PMI2_INCLUDE_DIR pmi2.h
                 PATHS ${PMI_ROOT} /opt/cray/pe/pmi/6.1.15
                 PATH_SUFFIXES include)
    find_library(PMI2_LIBRARY pmi2
                 PATHS ${PMI_ROOT} /opt/cray/pe/pmi/6.1.15
                 PATH_SUFFIXES lib lib64)
    if(NOT PMI2_INCLUDE_DIR OR NOT PMI2_LIBRARY)
        message(FATAL_ERROR "PMI2 not found (set PMI_ROOT=/path/to/pmi)")
    endif()
    target_compile_definitions(gicc PUBLIC GICC_BOOTSTRAP_PMI2=1)
    target_include_directories(gicc PUBLIC ${PMI2_INCLUDE_DIR})
    target_link_libraries     (gicc PUBLIC ${PMI2_LIBRARY})
else()
    message(FATAL_ERROR
        "GICC_BOOTSTRAP must be 'mpi' or 'pmi2' (got '${GICC_BOOTSTRAP}').")
endif()

target_compile_options(gicc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        -Xcompiler -Wall
        --expt-relaxed-constexpr
    >
)

set_target_properties(gicc PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
)

# ===========================================================================
# Header-only interface target (for users who only need headers)
# ===========================================================================

add_library(gicc_headers INTERFACE)
target_include_directories(gicc_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_compile_definitions(gicc_headers INTERFACE GICC_PLATFORM_MLX5)

# ===========================================================================
# Install
# ===========================================================================

install(TARGETS gicc gicc_headers
    EXPORT gicc-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

install(DIRECTORY gicc/
    DESTINATION include/gicc
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.cuh"
        PATTERN "*.h"
)

install(EXPORT gicc-targets
    FILE gicc-config.cmake
    NAMESPACE gicc::
    DESTINATION lib/cmake/gicc
)

# ===========================================================================
# Summary
# ===========================================================================

message(STATUS "")
message(STATUS "=== GICC Library Build Configuration ===")
message(STATUS "Backend:    ${GICC_BACKEND}")
message(STATUS "Bootstrap:  ${GICC_BOOTSTRAP}")
message(STATUS "CUDA:       ${CUDAToolkit_VERSION}")
message(STATUS "CUDA arch:  ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "ibverbs:    ${IBVERBS_LIBRARY}")
message(STATUS "mlx5:       ${MLX5_LIBRARY}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  gicc          - Shared library (libgicc.so)")
message(STATUS "  gicc_headers  - Header-only interface")
message(STATUS "")
