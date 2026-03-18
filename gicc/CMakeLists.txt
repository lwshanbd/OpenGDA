cmake_minimum_required(VERSION 3.18)
project(gicc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures" FORCE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ibverbs
find_library(IBVERBS_LIBRARY ibverbs)
if(NOT IBVERBS_LIBRARY)
    message(FATAL_ERROR "libibverbs not found")
endif()

# mlx5
find_library(MLX5_LIBRARY mlx5)
if(NOT MLX5_LIBRARY)
    message(FATAL_ERROR "libmlx5 not found")
endif()

# pmix (required by OpenMPI 5.x)
find_library(PMIX_LIBRARY pmix
    HINTS $ENV{MPI_ROOT}/lib)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}                 # gicc headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../nvidia       # platform backend headers
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Platform macro: MLX5 backend
add_definitions(-DGICC_PLATFORM_MLX5)

# Common libraries
set(GICC_LIBS
    ${MPI_CXX_LIBRARIES}
    ${IBVERBS_LIBRARY}
    ${MLX5_LIBRARY}
    CUDA::cudart
)
if(PMIX_LIBRARY)
    list(APPEND GICC_LIBS ${PMIX_LIBRARY})
endif()

# CUDA flags
set(GICC_CUDA_FLAGS
    -lineinfo
    -Xcompiler -Wall
    --expt-relaxed-constexpr
)

# ===========================================================================
# Benchmarks
# ===========================================================================

# Ping-pong benchmark (rewrite of nvidia/gpu_pingpong_bench.cu using GICC API)
add_executable(gicc_pingpong_bench gicc_pingpong_bench.cu)
target_link_libraries(gicc_pingpong_bench ${GICC_LIBS})
target_compile_options(gicc_pingpong_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${GICC_CUDA_FLAGS}>)
set_target_properties(gicc_pingpong_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Matrix multiplication (rewrite of nvidia/mm_gda_nv.cu)
add_executable(gicc_mm gicc_mm.cu)
target_link_libraries(gicc_mm ${GICC_LIBS})
target_compile_options(gicc_mm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${GICC_CUDA_FLAGS}>)
set_target_properties(gicc_mm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Jacobi solver (rewrite of nvgda/jacobi_gpu_init.cu)
add_executable(gicc_jacobi gicc_jacobi.cu)
target_link_libraries(gicc_jacobi ${GICC_LIBS})
target_compile_options(gicc_jacobi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${GICC_CUDA_FLAGS}>)
set_target_properties(gicc_jacobi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# ===========================================================================
# Summary
# ===========================================================================

message(STATUS "")
message(STATUS "=== GICC Build Configuration ===")
message(STATUS "Platform: MLX5 / InfiniBand")
message(STATUS "CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "ibverbs: ${IBVERBS_LIBRARY}")
message(STATUS "mlx5: ${MLX5_LIBRARY}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  gicc_pingpong_bench - Ping-pong latency/bandwidth")
message(STATUS "  gicc_mm             - Distributed matrix multiplication")
message(STATUS "  gicc_jacobi         - Jacobi solver with fused RDMA")
message(STATUS "")
