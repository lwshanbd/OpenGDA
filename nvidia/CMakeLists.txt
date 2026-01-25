# CMakeLists.txt - OpenGDA NVIDIA + InfiniBand implementation
#
# Build:
#   mkdir build && cd build
#   cmake ..
#   make -j
#
# Run (2 nodes):
#   mpirun -np 2 -hostfile hosts ./gda_pingpong

cmake_minimum_required(VERSION 3.18)
project(opengda_nvidia LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Force CUDA architecture to sm_90 for GPU-triggered RDMA optimizations
# (relaxed/release memory orderings and L1::no_allocate cache hints)
set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures" FORCE)

# Enable separable compilation for device code
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find ibverbs
find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h
    HINTS
        /usr/include
        /usr/local/include
        $ENV{IBVERBS_ROOT}/include
)

find_library(IBVERBS_LIBRARY ibverbs
    HINTS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        $ENV{IBVERBS_ROOT}/lib
        $ENV{IBVERBS_ROOT}/lib64
)

if(NOT IBVERBS_INCLUDE_DIR OR NOT IBVERBS_LIBRARY)
    message(FATAL_ERROR "Could not find libibverbs. Install rdma-core or set IBVERBS_ROOT.")
endif()

message(STATUS "Found ibverbs: ${IBVERBS_LIBRARY}")
message(STATUS "ibverbs include: ${IBVERBS_INCLUDE_DIR}")

# Common include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${IBVERBS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Find pmix (required by OpenMPI 5.x)
find_library(PMIX_LIBRARY pmix
    HINTS
        /hrtc/apps/mpi/openmpi/aarch64/rocky9/5.0.5/gcc/12.2.0/cuda.12.4/lib
        $ENV{MPI_ROOT}/lib
        $ENV{PMIX_ROOT}/lib
)

if(PMIX_LIBRARY)
    message(STATUS "Found pmix: ${PMIX_LIBRARY}")
endif()

# Common libraries
set(COMMON_LIBS
    ${MPI_CXX_LIBRARIES}
    ${IBVERBS_LIBRARY}
    CUDA::cudart
)

if(PMIX_LIBRARY)
    list(APPEND COMMON_LIBS ${PMIX_LIBRARY})
endif()

# Common CUDA flags
set(CUDA_NVCC_FLAGS
    -lineinfo
    -Xcompiler -Wall
    --expt-relaxed-constexpr
)

# ============================================================================
# Targets
# ============================================================================

# Ping-pong latency test
add_executable(gda_pingpong pingpong.cu)
target_link_libraries(gda_pingpong ${COMMON_LIBS})
target_compile_options(gda_pingpong PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
set_target_properties(gda_pingpong PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# AM test
add_executable(gda_am_test am_test.cu)
target_link_libraries(gda_am_test ${COMMON_LIBS})
target_compile_options(gda_am_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
set_target_properties(gda_am_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Find mlx5 direct verbs library
find_library(MLX5_LIBRARY mlx5
    HINTS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
)

if(MLX5_LIBRARY)
    message(STATUS "Found mlx5: ${MLX5_LIBRARY}")

    # GPU-triggered ping-pong test (uses MLX5 DevX)
    add_executable(gpu_pingpong gpu_pingpong.cu)
    target_link_libraries(gpu_pingpong ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(gpu_pingpong PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(gpu_pingpong PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # GPU-triggered optimized benchmark (compares original vs optimized)
    add_executable(gpu_benchmark_opt gpu_benchmark_opt.cu)
    target_link_libraries(gpu_benchmark_opt ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(gpu_benchmark_opt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(gpu_benchmark_opt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # GPU-triggered benchmark using DevX API (correct approach)
    add_executable(gpu_benchmark_devx gpu_benchmark_devx.cu)
    target_link_libraries(gpu_benchmark_devx ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(gpu_benchmark_devx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(gpu_benchmark_devx PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
else()
    message(STATUS "mlx5 library not found - GPU-triggered tests will not be built")
endif()

# ============================================================================
# Installation (optional)
# ============================================================================

install(TARGETS gda_pingpong gda_am_test
    RUNTIME DESTINATION bin
)

install(FILES
    cuda_device_context.hpp
    mpi_bootstrap.hpp
    ibv_context.hpp
    memory_region.hpp
    gda_types.hpp
    gda_comm.hpp
    gda_am.hpp
    DESTINATION include/opengda
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== OpenGDA NVIDIA Build Configuration ===")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "MPI: ${MPI_CXX_COMPILER}")
message(STATUS "ibverbs: ${IBVERBS_LIBRARY}")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  gda_pingpong      - RDMA ping-pong latency/bandwidth test")
message(STATUS "  gda_am_test       - Active Message test")
if(MLX5_LIBRARY)
message(STATUS "  gpu_pingpong      - GPU-triggered RDMA test")
message(STATUS "  gpu_benchmark_opt - GPU-triggered optimization benchmark")
message(STATUS "  gpu_benchmark_devx- GPU-triggered benchmark using DevX API (recommended)")
endif()
message(STATUS "")
