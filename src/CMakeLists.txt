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

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_library(IBVERBS_LIBRARY ibverbs)
if(NOT IBVERBS_LIBRARY)
    message(FATAL_ERROR "libibverbs not found")
endif()

find_library(MLX5_LIBRARY mlx5)
if(NOT MLX5_LIBRARY)
    message(FATAL_ERROR "libmlx5 not found")
endif()

find_library(PMIX_LIBRARY pmix HINTS $ENV{MPI_ROOT}/lib)

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
        ${MPI_CXX_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(gicc PUBLIC GICC_PLATFORM_MLX5)

target_link_libraries(gicc
    PUBLIC
        ${MPI_CXX_LIBRARIES}
        ${IBVERBS_LIBRARY}
        ${MLX5_LIBRARY}
        CUDA::cudart
)
if(PMIX_LIBRARY)
    target_link_libraries(gicc PUBLIC ${PMIX_LIBRARY})
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
message(STATUS "Platform:   MLX5 / InfiniBand")
message(STATUS "CUDA:       ${CUDAToolkit_VERSION}")
message(STATUS "CUDA arch:  ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "ibverbs:    ${IBVERBS_LIBRARY}")
message(STATUS "mlx5:       ${MLX5_LIBRARY}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  gicc          - Shared library (libgicc.so)")
message(STATUS "  gicc_headers  - Header-only interface")
message(STATUS "")
