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

    # Comprehensive ping-pong benchmark
    add_executable(gpu_pingpong_bench gpu_pingpong_bench.cu)
    target_link_libraries(gpu_pingpong_bench ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(gpu_pingpong_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(gpu_pingpong_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # MPI ping-pong benchmark for comparison
    add_executable(mpi_pingpong_bench mpi_pingpong_bench.cu)
    target_link_libraries(mpi_pingpong_bench ${COMMON_LIBS})
    target_compile_options(mpi_pingpong_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(mpi_pingpong_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # GPU-triggered concurrent RDMA benchmark (similar to minimal/)
    add_executable(gpu_concurrent_bench gpu_concurrent_bench.cu)
    target_link_libraries(gpu_concurrent_bench ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(gpu_concurrent_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(gpu_concurrent_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # MPI concurrent transfers benchmark for comparison
    add_executable(mpi_concurrent_bench mpi_concurrent_bench.cu)
    target_link_libraries(mpi_concurrent_bench ${COMMON_LIBS})
    target_compile_options(mpi_concurrent_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(mpi_concurrent_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # GPU-triggered concurrent benchmark v2 (optimized with globaltimer)
    add_executable(gpu_concurrent_bench_v2 gpu_concurrent_bench_v2.cu)
    target_link_libraries(gpu_concurrent_bench_v2 ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(gpu_concurrent_bench_v2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(gpu_concurrent_bench_v2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # nvshmem-style benchmark (compares original vs nvshmem optimizations)
    add_executable(gpu_bench_nvshmem gpu_bench_nvshmem.cu)
    target_link_libraries(gpu_bench_nvshmem ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(gpu_bench_nvshmem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(gpu_bench_nvshmem PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # GPU put bandwidth benchmark (thread/warp/block scope)
    add_executable(gpu_put_bw gpu_put_bw.cu)
    target_link_libraries(gpu_put_bw ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(gpu_put_bw PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(gpu_put_bw PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # GPU put latency benchmark (matches nvshmem shmem_put_latency)
    add_executable(gpu_put_latency gpu_put_latency.cu)
    target_link_libraries(gpu_put_latency ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(gpu_put_latency PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(gpu_put_latency PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # NVIB AM ping-pong latency benchmark (matches minimal/am_pingpong)
    add_executable(nvib_am_pingpong nvib_am_pingpong.cu)
    target_link_libraries(nvib_am_pingpong ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(nvib_am_pingpong PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(nvib_am_pingpong PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # NVIB AM ping-pong ReqRep mode (lightweight reply)
    add_executable(nvib_am_pingpong_reqrep nvib_am_pingpong_reqrep.cu)
    target_link_libraries(nvib_am_pingpong_reqrep ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(nvib_am_pingpong_reqrep PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(nvib_am_pingpong_reqrep PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Distributed Matrix Multiplication with GPU-initiated RDMA
    add_executable(mm_gda_nv mm_gda_nv.cu)
    target_link_libraries(mm_gda_nv ${COMMON_LIBS} ${MLX5_LIBRARY})
    target_compile_options(mm_gda_nv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(mm_gda_nv PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
else()
    message(STATUS "mlx5 library not found - GPU-triggered tests will not be built")
endif()

# ============================================================================
# CUDA-aware MPI Matrix Multiplication (no MLX5 dependency)
# ============================================================================

# Distributed Matrix Multiplication with CUDA-aware MPI
add_executable(mm_mpi_cuda mm_mpi_cuda.cu)
target_link_libraries(mm_mpi_cuda ${COMMON_LIBS})
target_compile_options(mm_mpi_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
set_target_properties(mm_mpi_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ============================================================================
# NVSHMEM Matrix Multiplication (optional - requires NVSHMEM)
# ============================================================================

# Find NVSHMEM
find_path(NVSHMEM_INCLUDE_DIR nvshmem.h
    HINTS
        $ENV{NVSHMEM_HOME}/include
        /usr/local/nvshmem/include
        /opt/nvidia/hpc_sdk/Linux_*/24.*/comm_libs/nvshmem/include
)

find_library(NVSHMEM_LIBRARY nvshmem
    HINTS
        $ENV{NVSHMEM_HOME}/lib
        /usr/local/nvshmem/lib
        /opt/nvidia/hpc_sdk/Linux_*/24.*/comm_libs/nvshmem/lib
)

if(NVSHMEM_INCLUDE_DIR AND NVSHMEM_LIBRARY)
    message(STATUS "Found NVSHMEM: ${NVSHMEM_LIBRARY}")
    message(STATUS "NVSHMEM include: ${NVSHMEM_INCLUDE_DIR}")

    # Distributed Matrix Multiplication with NVSHMEM
    add_executable(mm_nvshmem mm_nvshmem.cu)
    target_include_directories(mm_nvshmem PRIVATE ${NVSHMEM_INCLUDE_DIR})
    target_link_libraries(mm_nvshmem ${NVSHMEM_LIBRARY} ${COMMON_LIBS} CUDA::cuda_driver)
    target_compile_options(mm_nvshmem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(mm_nvshmem PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
else()
    message(STATUS "NVSHMEM not found - mm_nvshmem will not be built")
    message(STATUS "  Set NVSHMEM_HOME environment variable to enable")
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
message(STATUS "  mm_mpi_cuda       - Matrix multiplication with CUDA-aware MPI")
if(MLX5_LIBRARY)
message(STATUS "  gpu_pingpong      - GPU-triggered RDMA test")
message(STATUS "  gpu_benchmark_opt - GPU-triggered optimization benchmark")
message(STATUS "  gpu_benchmark_devx- GPU-triggered benchmark using DevX API (recommended)")
message(STATUS "  mm_gda_nv         - Matrix multiplication with GPU-initiated RDMA")
endif()
if(NVSHMEM_LIBRARY)
message(STATUS "  mm_nvshmem        - Matrix multiplication with NVSHMEM")
endif()
message(STATUS "")
