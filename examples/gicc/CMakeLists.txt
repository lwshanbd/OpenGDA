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

# ===========================================================================
# Examples
# ===========================================================================

# ib_test: GPU-initiated put / put_nbi / put_signal / get from CUDA kernels
# and host put / get, checked word for word, plus a device ping-pong.
add_executable(gicc_ib_test ib_test.cu)
target_link_libraries(gicc_ib_test PRIVATE gicc MPI::MPI_CXX)
target_compile_options(gicc_ib_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)

message(STATUS "")
message(STATUS "=== GICC Examples ===")
message(STATUS "  gicc_ib_test - GPU-initiated InfiniBand transfers, checked word for word")
message(STATUS "")
