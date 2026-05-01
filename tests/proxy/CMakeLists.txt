# tests/proxy/CMakeLists.txt - microtests for the CPU Proxy ring buffer.
# Gated on GICC_ENABLE_CPU_PROXY (set by the top-level CMakeLists.txt).

if(NOT GICC_ENABLE_CPU_PROXY)
    return()
endif()

# ---------------------------------------------------------------------------
# test_d2h_ring (Task 4) - header-only ring producer/consumer microtest.
# ---------------------------------------------------------------------------
if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(test_d2h_ring.cu PROPERTIES LANGUAGE CUDA)
    add_executable(test_d2h_ring test_d2h_ring.cu)
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(test_d2h_ring.hip.cpp PROPERTIES LANGUAGE HIP)
    add_executable(test_d2h_ring test_d2h_ring.hip.cpp)
else()
    message(FATAL_ERROR
        "tests/proxy: unknown GICC_GPU_API '${GICC_GPU_API}'")
endif()

target_include_directories(test_d2h_ring PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
# test_proxy_libfabric_loopback (Task 5) - exercises ProxyLibfabric end to end.
#
# There is no shared `gicc` library target under the OFI backend (each example
# is built standalone), so we compile proxy_libfabric.cpp directly into the
# test executable. The test transitively pulls in ofi_runtime.hpp, which is
# still HIP-only as of Task 5; the CUDA build target is therefore expected to
# fail at compile time of ofi_runtime.hpp until Task 7 lands the
# Runtime GPU-API split. The HIP build is the one that should work today.
# ---------------------------------------------------------------------------
set(PROXY_LIBFABRIC_SRC
    ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/proxy/proxy_libfabric.cpp)

# libfabric (Cray on Tioga, CXI 1.22 on Delta).
if(NOT DEFINED LIBFABRIC_ROOT)
    set(LIBFABRIC_ROOT "/opt/cray/libfabric/1.22.0" CACHE PATH "libfabric path")
endif()
find_path(PROXY_LIBFABRIC_INCLUDE_DIR rdma/fabric.h
    PATHS ${LIBFABRIC_ROOT} /opt/cray/libfabric/2.1
    PATH_SUFFIXES include)
find_library(PROXY_LIBFABRIC_LIBRARY fabric
    PATHS ${LIBFABRIC_ROOT} /opt/cray/libfabric/2.1
    PATH_SUFFIXES lib lib64)
if(NOT PROXY_LIBFABRIC_INCLUDE_DIR OR NOT PROXY_LIBFABRIC_LIBRARY)
    message(WARNING "tests/proxy: libfabric not found, skipping "
                    "test_proxy_libfabric_loopback")
    return()
endif()

# hwloc (pulled in by Fabric -> DeviceAffinityDetector).
find_path(PROXY_HWLOC_INCLUDE_DIR hwloc.h)
find_library(PROXY_HWLOC_LIBRARY hwloc)
if(NOT PROXY_HWLOC_INCLUDE_DIR OR NOT PROXY_HWLOC_LIBRARY)
    message(WARNING "tests/proxy: hwloc not found, skipping "
                    "test_proxy_libfabric_loopback")
    return()
endif()

# Bootstrap (mirrors examples/ofi/CMakeLists.txt logic).
set(GICC_BOOTSTRAP "mpi" CACHE STRING "Bootstrap backend: mpi | pmi2")
if(GICC_BOOTSTRAP STREQUAL "mpi")
    find_package(MPI REQUIRED)
    set(PROXY_BOOTSTRAP_DEFINE GICC_BOOTSTRAP_MPI=1)
    # Cray's `CC` wrapper resolves mpi.h implicitly; CMake's find_package
    # then leaves MPI_CXX_INCLUDE_DIRS empty. Probe common Cray locations
    # so plain clang++ / nvcc invocations also see mpi.h.
    set(PROXY_BOOTSTRAP_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
    if(NOT PROXY_BOOTSTRAP_INCLUDES)
        find_path(_PROXY_MPI_H mpi.h
            PATHS /opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/include
                  /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include
                  /opt/cray/pe/mpich/8.1.32/ofi/cray/18.0/include
            NO_DEFAULT_PATH)
        if(_PROXY_MPI_H)
            set(PROXY_BOOTSTRAP_INCLUDES ${_PROXY_MPI_H})
        endif()
    endif()
    set(PROXY_BOOTSTRAP_LIBS     MPI::MPI_CXX)
    # Cray's MPI::MPI_CXX target carries no library file when CC is the wrapper.
    # Probe the canonical Cray location and link it explicitly when needed.
    find_library(_PROXY_MPI_LIB
        NAMES mpi_gnu_112 mpi
        PATHS /opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib
              /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib
        NO_DEFAULT_PATH)
    if(_PROXY_MPI_LIB)
        list(APPEND PROXY_BOOTSTRAP_LIBS ${_PROXY_MPI_LIB})
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
    set(PROXY_BOOTSTRAP_DEFINE GICC_BOOTSTRAP_PMI2=1)
    set(PROXY_BOOTSTRAP_INCLUDES ${PMI2_INCLUDE_DIR})
    set(PROXY_BOOTSTRAP_LIBS     ${PMI2_LIBRARY})
endif()

# Proxy thread sources are now linked into the loopback test as well: with
# Task 7 wiring the ProxyThread into Runtime, every Runtime destructor needs
# the ProxyThread dtor symbol available, even if the test never starts a
# proxy worker.
set(PROXY_THREAD_SRC_FOR_LOOPBACK
    ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/proxy/proxy_thread.cpp)

if(GICC_GPU_API STREQUAL "hip")
    # Tioga path: build the .cpp's as HIP so hipMalloc/hipHostMalloc resolve.
    set_source_files_properties(test_proxy_libfabric_loopback.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC_FOR_LOOPBACK}
        PROPERTIES LANGUAGE HIP)
    add_executable(test_proxy_libfabric_loopback
        test_proxy_libfabric_loopback.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC_FOR_LOOPBACK})
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
    find_package(hip REQUIRED)
    target_link_libraries(test_proxy_libfabric_loopback PRIVATE
        hip::device
        ${PROXY_LIBFABRIC_LIBRARY}
        ${PROXY_HWLOC_LIBRARY}
        ${PROXY_BOOTSTRAP_LIBS}
        pthread)
elseif(GICC_GPU_API STREQUAL "cuda")
    # Delta path: ofi_runtime.hpp transitively pulls device-side bits
    # (e.g. fabric.hpp's gpuLaunchKernel(gda_trigger_kernel,...) and
    # ofi_device.cuh's threadIdx/blockIdx in flush()). Those require nvcc
    # parsing, so compile the test sources as CUDA.
    set_source_files_properties(test_proxy_libfabric_loopback.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC_FOR_LOOPBACK}
        PROPERTIES LANGUAGE CUDA)
    add_executable(test_proxy_libfabric_loopback
        test_proxy_libfabric_loopback.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC_FOR_LOOPBACK})
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(test_proxy_libfabric_loopback PRIVATE
        CUDA::cudart
        ${PROXY_LIBFABRIC_LIBRARY}
        ${PROXY_HWLOC_LIBRARY}
        ${PROXY_BOOTSTRAP_LIBS}
        pthread)
endif()

target_include_directories(test_proxy_libfabric_loopback PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/internal
    ${PROXY_LIBFABRIC_INCLUDE_DIR}
    ${PROXY_HWLOC_INCLUDE_DIR}
    ${PROXY_BOOTSTRAP_INCLUDES})
target_compile_definitions(test_proxy_libfabric_loopback PRIVATE
    GICC_PLATFORM_OFI
    ${PROXY_BOOTSTRAP_DEFINE})

# ---------------------------------------------------------------------------
# test_proxy_thread (Task 6) - exercises the ProxyThread worker end to end.
#
# Reuses the same libfabric / hwloc / MPI / hip discovery from above. Like
# the loopback test, this transitively pulls ofi_runtime.hpp and is HIP-only
# until Task 7 splits the GPU API out of Runtime.
# ---------------------------------------------------------------------------
set(PROXY_THREAD_SRC
    ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/proxy/proxy_thread.cpp)

if(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(test_proxy_thread.cpp
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE HIP)
    add_executable(test_proxy_thread
        test_proxy_thread.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(test_proxy_thread PRIVATE
        hip::device
        ${PROXY_LIBFABRIC_LIBRARY}
        ${PROXY_HWLOC_LIBRARY}
        ${PROXY_BOOTSTRAP_LIBS}
        pthread)
elseif(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(test_proxy_thread.cpp
                                ${PROXY_LIBFABRIC_SRC}
                                ${PROXY_THREAD_SRC}
        PROPERTIES LANGUAGE CUDA)
    add_executable(test_proxy_thread
        test_proxy_thread.cpp
        ${PROXY_LIBFABRIC_SRC}
        ${PROXY_THREAD_SRC})
    target_link_libraries(test_proxy_thread PRIVATE
        CUDA::cudart
        ${PROXY_LIBFABRIC_LIBRARY}
        ${PROXY_HWLOC_LIBRARY}
        ${PROXY_BOOTSTRAP_LIBS}
        pthread)
endif()

target_include_directories(test_proxy_thread PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/internal
    ${PROXY_LIBFABRIC_INCLUDE_DIR}
    ${PROXY_HWLOC_INCLUDE_DIR}
    ${PROXY_BOOTSTRAP_INCLUDES})
target_compile_definitions(test_proxy_thread PRIVATE
    GICC_PLATFORM_OFI
    ${PROXY_BOOTSTRAP_DEFINE})
