# examples/proxy/CMakeLists.txt - Task 7 L1/L2 examples for the CPU Proxy.
# Mirrors tests/proxy/CMakeLists.txt's libfabric / hwloc / MPI / hip /
# CUDA discovery; gated on GICC_ENABLE_CPU_PROXY.

if(NOT GICC_ENABLE_CPU_PROXY)
    return()
endif()

# ---------------------------------------------------------------------------
# Source bundles (one of each per example, plus the proxy implementation).
# ---------------------------------------------------------------------------
set(PROXY_LIBFABRIC_SRC
    ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/proxy/proxy_libfabric.cpp)
set(PROXY_THREAD_SRC
    ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/proxy/proxy_thread.cpp)

# libfabric (Cray on Tioga, CXI 1.22 on Delta).
if(NOT DEFINED LIBFABRIC_ROOT)
    set(LIBFABRIC_ROOT "/opt/cray/libfabric/1.22.0" CACHE PATH "libfabric path")
endif()
find_path(PROXY_EX_LIBFABRIC_INCLUDE_DIR rdma/fabric.h
    PATHS ${LIBFABRIC_ROOT} /opt/cray/libfabric/2.1
    PATH_SUFFIXES include)
find_library(PROXY_EX_LIBFABRIC_LIBRARY fabric
    PATHS ${LIBFABRIC_ROOT} /opt/cray/libfabric/2.1
    PATH_SUFFIXES lib lib64)
if(NOT PROXY_EX_LIBFABRIC_INCLUDE_DIR OR NOT PROXY_EX_LIBFABRIC_LIBRARY)
    message(WARNING "examples/proxy: libfabric not found, skipping")
    return()
endif()

# hwloc.
find_path(PROXY_EX_HWLOC_INCLUDE_DIR hwloc.h)
find_library(PROXY_EX_HWLOC_LIBRARY hwloc)
if(NOT PROXY_EX_HWLOC_INCLUDE_DIR OR NOT PROXY_EX_HWLOC_LIBRARY)
    message(WARNING "examples/proxy: hwloc not found, skipping")
    return()
endif()

# Bootstrap (mirror tests/proxy/CMakeLists.txt).
set(GICC_BOOTSTRAP "mpi" CACHE STRING "Bootstrap backend: mpi | pmi2")
if(GICC_BOOTSTRAP STREQUAL "mpi")
    find_package(MPI REQUIRED)
    set(PROXY_EX_BOOTSTRAP_DEFINE GICC_BOOTSTRAP_MPI=1)
    set(PROXY_EX_BOOTSTRAP_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
    if(NOT PROXY_EX_BOOTSTRAP_INCLUDES)
        find_path(_PROXY_EX_MPI_H mpi.h
            PATHS /opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/include
                  /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include
                  /opt/cray/pe/mpich/8.1.32/ofi/cray/18.0/include
            NO_DEFAULT_PATH)
        if(_PROXY_EX_MPI_H)
            set(PROXY_EX_BOOTSTRAP_INCLUDES ${_PROXY_EX_MPI_H})
        endif()
    endif()
    set(PROXY_EX_BOOTSTRAP_LIBS MPI::MPI_CXX)
    find_library(_PROXY_EX_MPI_LIB
        NAMES mpi_gnu_112 mpi
        PATHS /opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib
              /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib
        NO_DEFAULT_PATH)
    if(_PROXY_EX_MPI_LIB)
        list(APPEND PROXY_EX_BOOTSTRAP_LIBS ${_PROXY_EX_MPI_LIB})
    endif()
elseif(GICC_BOOTSTRAP STREQUAL "pmi2")
    set(PMI_ROOT "/opt/cray/pe/pmi/6.1.15" CACHE PATH "Cray PMI path")
    find_path(PMI2_INCLUDE_DIR pmi2.h
        PATHS ${PMI_ROOT} PATH_SUFFIXES include)
    find_library(PMI2_LIBRARY pmi2
        PATHS ${PMI_ROOT} PATH_SUFFIXES lib lib64)
    if(NOT PMI2_INCLUDE_DIR OR NOT PMI2_LIBRARY)
        message(FATAL_ERROR "PMI2 not found (set PMI_ROOT)")
    endif()
    set(PROXY_EX_BOOTSTRAP_DEFINE GICC_BOOTSTRAP_PMI2=1)
    set(PROXY_EX_BOOTSTRAP_INCLUDES ${PMI2_INCLUDE_DIR})
    set(PROXY_EX_BOOTSTRAP_LIBS     ${PMI2_LIBRARY})
endif()

# ---------------------------------------------------------------------------
# Helper: shared common settings for both examples.
# ---------------------------------------------------------------------------
function(_proxy_example_setup tgt)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/internal
        ${PROXY_EX_LIBFABRIC_INCLUDE_DIR}
        ${PROXY_EX_HWLOC_INCLUDE_DIR}
        ${PROXY_EX_BOOTSTRAP_INCLUDES})
    target_compile_definitions(${tgt} PRIVATE
        GICC_PLATFORM_OFI
        ${PROXY_EX_BOOTSTRAP_DEFINE})
    target_compile_features(${tgt} PRIVATE cxx_std_17)
    set_target_properties(${tgt} PROPERTIES HIP_STANDARD 17)
    # Use -flto so the comparison vs bench_pingpong_lto isolates the
    # LTO pass's trace-synthesis effect from the cross-TU inlining
    # effect that the LTO bench gets for free via -flto.
    #
    # CUDA builds must opt out: nvcc embeds device code as a `fatbinData`
    # object, and GNU LTO re-emits it per translation unit, so the link
    # fails with "symbol `fatbinData' is already defined". The flag only
    # ever mattered for the HIP bench comparison anyway.
    if(GICC_GPU_API STREQUAL "cuda")
        target_compile_options(${tgt} PRIVATE -O3)
    else()
        target_compile_options(${tgt} PRIVATE -O3 -flto)
        target_link_options(${tgt} PRIVATE -flto)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# put_loopback (L1) - single-rank.
# ---------------------------------------------------------------------------
if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(put_loopback.cu
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE CUDA)
    add_executable(put_loopback
        put_loopback.cu
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(put_loopback PRIVATE
        CUDA::cudart
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(put_loopback.hip.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE HIP)
    add_executable(put_loopback
        put_loopback.hip.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
    find_package(hip REQUIRED)
    target_link_libraries(put_loopback PRIVATE
        hip::device
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
endif()
_proxy_example_setup(put_loopback)

# ---------------------------------------------------------------------------
# put_two_rank_intranode (L2) - 2 MPI ranks.
# ---------------------------------------------------------------------------
if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(put_two_rank_intranode.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE CUDA)
    add_executable(put_two_rank_intranode
        put_two_rank_intranode.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(put_two_rank_intranode PRIVATE
        CUDA::cudart
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(put_two_rank_intranode.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE HIP)
    add_executable(put_two_rank_intranode
        put_two_rank_intranode.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(put_two_rank_intranode PRIVATE
        hip::device
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
endif()
_proxy_example_setup(put_two_rank_intranode)

# ---------------------------------------------------------------------------
# put_mixed_dispatch (L4) - 2 MPI ranks, two put_no_db sites in one kernel
# routed to different dispatches via hint_mixed.json. Exercises the Task 8
# hint-aware device-lowering pass; see file header for the build / run flow.
# ---------------------------------------------------------------------------
if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(put_mixed_dispatch.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE CUDA)
    add_executable(put_mixed_dispatch
        put_mixed_dispatch.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(put_mixed_dispatch PRIVATE
        CUDA::cudart
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(put_mixed_dispatch.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE HIP)
    add_executable(put_mixed_dispatch
        put_mixed_dispatch.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(put_mixed_dispatch PRIVATE
        hip::device
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
endif()
_proxy_example_setup(put_mixed_dispatch)

# ---------------------------------------------------------------------------
# quiet_pingpong (L5) - 2 MPI ranks. Each rank's kernel does
# put_no_db(peer) -> quiet(); the device-side quiet() body (Task 9) pushes
# a QUIET cmd and spins until the proxy's handle_quiet acks past the
# slot. An MPI barrier between gpuDeviceSynchronize() and the readback
# provides the cross-rank ordering each rank's quiet() cannot guarantee
# on its own. See file header for scope-honesty caveats.
# ---------------------------------------------------------------------------
if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(quiet_pingpong.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE CUDA)
    add_executable(quiet_pingpong
        quiet_pingpong.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(quiet_pingpong PRIVATE
        CUDA::cudart
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(quiet_pingpong.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE HIP)
    add_executable(quiet_pingpong
        quiet_pingpong.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(quiet_pingpong PRIVATE
        hip::device
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
endif()
_proxy_example_setup(quiet_pingpong)

# ---------------------------------------------------------------------------
# get_pingpong - 2 MPI ranks. Each rank's kernel does
# get_no_db(peer) -> quiet(); the device-side quiet() body acts as the
# read-ordering fence (proxy advances ring tail only after the libfabric
# CQE, kernel __threadfence_system() invalidates GPU L2 for the landing
# range). Smoke-tests the CPU-Proxy GET path end-to-end (device push +
# fi_read + ack + L2 invalidate). See file header for caveats.
# ---------------------------------------------------------------------------
if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(get_pingpong.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE CUDA)
    add_executable(get_pingpong
        get_pingpong.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(get_pingpong PRIVATE
        CUDA::cudart
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(get_pingpong.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE HIP)
    add_executable(get_pingpong
        get_pingpong.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(get_pingpong PRIVATE
        hip::device
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
endif()
_proxy_example_setup(get_pingpong)

# ---------------------------------------------------------------------------
# get_stress - 2 MPI ranks. Back-to-back large GETs into a shared landing
# region, checks the tail bytes immediately after each quiet(). Designed
# to expose the PCIe-ordering race window between fi_read CQE and the
# NIC's posted writes actually landing in GPU HBM.
# ---------------------------------------------------------------------------
if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(get_stress.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE CUDA)
    add_executable(get_stress
        get_stress.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(get_stress PRIVATE
        CUDA::cudart
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(get_stress.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE HIP)
    add_executable(get_stress
        get_stress.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(get_stress PRIVATE
        hip::device
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
endif()
_proxy_example_setup(get_stress)

# ---------------------------------------------------------------------------
# get_pingpong_dwq - 2 MPI ranks. DWQ path GET smoke test.
# Uses rt.get_no_db() to queue a deferred fi_read + chained atomic_signal;
# kernel issues a single flush() that fires the trigger MMIO; host waits
# on libfabric counters via rt.wait/rt.reset. Verifies via SM-driven copy
# (avoids the hipMemcpy-D2H stale-L2 pathology on small buffers).
# ---------------------------------------------------------------------------
if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(get_pingpong_dwq.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE CUDA)
    add_executable(get_pingpong_dwq
        get_pingpong_dwq.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(get_pingpong_dwq PRIVATE
        CUDA::cudart
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(get_pingpong_dwq.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
                                ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/runtime_helpers.cpp
        PROPERTIES LANGUAGE HIP)
    add_executable(get_pingpong_dwq
        get_pingpong_dwq.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC}
        ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/runtime_helpers.cpp)
    target_link_libraries(get_pingpong_dwq PRIVATE
        hip::device
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
endif()
_proxy_example_setup(get_pingpong_dwq)

# ---------------------------------------------------------------------------
# get_stress_dwq - back-to-back DWQ GETs, full window verification.
# ---------------------------------------------------------------------------
if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(get_stress_dwq.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE CUDA)
    add_executable(get_stress_dwq
        get_stress_dwq.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(get_stress_dwq PRIVATE
        CUDA::cudart
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(get_stress_dwq.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
                                ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/runtime_helpers.cpp
        PROPERTIES LANGUAGE HIP)
    add_executable(get_stress_dwq
        get_stress_dwq.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC}
        ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/runtime_helpers.cpp)
    target_link_libraries(get_stress_dwq PRIVATE
        hip::device
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
endif()
_proxy_example_setup(get_stress_dwq)

# ---------------------------------------------------------------------------
# jacobi_bench / mm_bench - application-shaped benchmarks that compare the
# DWQ (GPU-trigger) transport against Cray MPI on the same compute kernels.
# Host-side Runtime API only, so they need no LTO pass.
# ---------------------------------------------------------------------------
foreach(_app jacobi_bench mm_bench)
    if(GICC_GPU_API STREQUAL "cuda")
        set_source_files_properties(${_app}.cpp
                                    ${PROXY_LIBFABRIC_SRC}
                                    ${PROXY_THREAD_SRC}
            PROPERTIES LANGUAGE CUDA)
        add_executable(${_app}
            ${_app}.cpp
            ${PROXY_LIBFABRIC_SRC}
            ${PROXY_THREAD_SRC})
        target_link_libraries(${_app} PRIVATE
            CUDA::cudart
            ${PROXY_EX_LIBFABRIC_LIBRARY}
            ${PROXY_EX_HWLOC_LIBRARY}
            ${PROXY_EX_BOOTSTRAP_LIBS}
            pthread)
    elseif(GICC_GPU_API STREQUAL "hip")
        set_source_files_properties(${_app}.cpp
                                    ${PROXY_LIBFABRIC_SRC}
                                    ${PROXY_THREAD_SRC}
                                    ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/runtime_helpers.cpp
            PROPERTIES LANGUAGE HIP)
        add_executable(${_app}
            ${_app}.cpp
            ${PROXY_LIBFABRIC_SRC}
            ${PROXY_THREAD_SRC}
            ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/runtime_helpers.cpp)
        target_link_libraries(${_app} PRIVATE
            hip::device
            ${PROXY_EX_LIBFABRIC_LIBRARY}
            ${PROXY_EX_HWLOC_LIBRARY}
            ${PROXY_EX_BOOTSTRAP_LIBS}
            pthread)
    endif()
    _proxy_example_setup(${_app})
endforeach()

# ---------------------------------------------------------------------------
# bench_pingpong - latency benchmark comparing CPU Proxy and Cray MPI for
# unidirectional GPU-buffer-to-GPU-buffer transfers across a size sweep.
# Selects mode at runtime via --mode={mpi,proxy}.
# ---------------------------------------------------------------------------
if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(bench_pingpong.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
                                ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/runtime_helpers.cpp
        PROPERTIES LANGUAGE CUDA)
    add_executable(bench_pingpong
        bench_pingpong.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC}
        ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/runtime_helpers.cpp)
    target_link_libraries(bench_pingpong PRIVATE
        CUDA::cudart
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
    # Cray MPICH GPU-aware link: requires the GTL library when
    # MPICH_GPU_SUPPORT_ENABLED=1 (enables sending GPU pointers via MPI).
    if(EXISTS "/opt/cray/pe/lib64/libmpi_gtl_cuda.so")
        target_link_libraries(bench_pingpong PRIVATE
            "/opt/cray/pe/lib64/libmpi_gtl_cuda.so")
    endif()
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(bench_pingpong.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
                                ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/runtime_helpers.cpp
        PROPERTIES LANGUAGE HIP)
    add_executable(bench_pingpong
        bench_pingpong.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC}
        ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/runtime_helpers.cpp)
    target_link_libraries(bench_pingpong PRIVATE
        hip::device
        ${PROXY_EX_LIBFABRIC_LIBRARY}
        ${PROXY_EX_HWLOC_LIBRARY}
        ${PROXY_EX_BOOTSTRAP_LIBS}
        pthread)
endif()
_proxy_example_setup(bench_pingpong)
