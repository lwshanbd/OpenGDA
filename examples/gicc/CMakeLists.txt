cmake_minimum_required(VERSION 3.18)
project(gicc_examples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures")

# ===========================================================================
# Find GICC library (either from parent build or installed)
# ===========================================================================

if(NOT TARGET gicc)
    find_package(gicc REQUIRED)
endif()

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Common CUDA flags for examples
set(EXAMPLE_CUDA_FLAGS
    -lineinfo
    -Xcompiler -Wall
    --expt-relaxed-constexpr
)

# ===========================================================================
# Helper function for adding examples
# ===========================================================================

function(add_gicc_example target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE gicc)
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${EXAMPLE_CUDA_FLAGS}>)
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()

# ===========================================================================
# Examples
# ===========================================================================

add_gicc_example(gicc_pingpong_bench pingpong_bench.cu)
add_gicc_example(gicc_mm             mm.cu)
add_gicc_example(gicc_jacobi         jacobi.cu)

# ===========================================================================
# Summary
# ===========================================================================

message(STATUS "")
message(STATUS "=== GICC Examples ===")
message(STATUS "  gicc_pingpong_bench - Ping-pong latency/bandwidth")
message(STATUS "  gicc_mm             - Distributed matrix multiplication")
message(STATUS "  gicc_jacobi         - Jacobi solver with fused RDMA")
message(STATUS "")
