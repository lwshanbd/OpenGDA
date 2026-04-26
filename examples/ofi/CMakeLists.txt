# ============================================================================
# GICC OFI backend examples (libfabric / CXI / HIP)
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_HIP_FLAGS_RELEASE "-O3")

# ROCm/HIP
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
find_package(hip REQUIRED)

# Cray libfabric (with CXI extensions)
set(LIBFABRIC_ROOT "/opt/cray/libfabric/2.1" CACHE PATH "Cray libfabric path")
find_path(LIBFABRIC_INCLUDE_DIR rdma/fi_cxi_ext.h
    PATHS ${LIBFABRIC_ROOT} PATH_SUFFIXES include NO_DEFAULT_PATH)
find_library(LIBFABRIC_LIBRARY fabric
    PATHS ${LIBFABRIC_ROOT} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
if(NOT LIBFABRIC_INCLUDE_DIR OR NOT LIBFABRIC_LIBRARY)
    message(FATAL_ERROR "Cray libfabric not found at ${LIBFABRIC_ROOT}")
endif()

# Bootstrap backend (mpi | pmi2)
set(GICC_BOOTSTRAP "mpi" CACHE STRING "Bootstrap backend: mpi | pmi2")
set_property(CACHE GICC_BOOTSTRAP PROPERTY STRINGS "mpi" "pmi2")

if(GICC_BOOTSTRAP STREQUAL "mpi")
    find_package(MPI REQUIRED)
    set(GICC_BOOTSTRAP_DEFINE GICC_BOOTSTRAP_MPI=1)
    set(GICC_BOOTSTRAP_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
    set(GICC_BOOTSTRAP_LIBS     MPI::MPI_CXX)
elseif(GICC_BOOTSTRAP STREQUAL "pmi2")
    set(PMI_ROOT "/opt/cray/pe/pmi/6.1.15" CACHE PATH "Cray PMI path")
    find_path(PMI2_INCLUDE_DIR pmi2.h
        PATHS ${PMI_ROOT} /opt/cray/pe/pmi/6.0.17 /opt/cray/pe/pmi/6.0.13
        PATH_SUFFIXES include)
    find_library(PMI2_LIBRARY pmi2
        PATHS ${PMI_ROOT} /opt/cray/pe/pmi/6.0.17 /opt/cray/pe/pmi/6.0.13
        PATH_SUFFIXES lib lib64)
    if(NOT PMI2_INCLUDE_DIR OR NOT PMI2_LIBRARY)
        message(FATAL_ERROR "PMI2 not found (set PMI_ROOT=/path/to/pmi)")
    endif()
    set(GICC_BOOTSTRAP_DEFINE GICC_BOOTSTRAP_PMI2=1)
    set(GICC_BOOTSTRAP_INCLUDES ${PMI2_INCLUDE_DIR})
    set(GICC_BOOTSTRAP_LIBS     ${PMI2_LIBRARY})
else()
    message(FATAL_ERROR
        "GICC_BOOTSTRAP must be 'mpi' or 'pmi2' (got '${GICC_BOOTSTRAP}').")
endif()

# hwloc (GPU-NIC affinity)
find_path(HWLOC_INCLUDE_DIR hwloc.h)
find_library(HWLOC_LIBRARY hwloc)
if(NOT HWLOC_INCLUDE_DIR OR NOT HWLOC_LIBRARY)
    message(FATAL_ERROR "hwloc not found")
endif()

set(GICC_OFI_INCLUDES
    ${CMAKE_SOURCE_DIR}/src                                  # for "gicc/..."
    ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/internal       # for sibling-relative includes inside internal/
    ${LIBFABRIC_INCLUDE_DIR}
    ${HWLOC_INCLUDE_DIR}
    ${GICC_BOOTSTRAP_INCLUDES}
)

set(GICC_OFI_LIBS
    hip::device
    ${LIBFABRIC_LIBRARY}
    ${HWLOC_LIBRARY}
    pthread
    ${GICC_BOOTSTRAP_LIBS}
)

function(gicc_ofi_example name source)
    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${GICC_OFI_INCLUDES})
    target_compile_definitions(${name} PRIVATE GICC_PLATFORM_OFI ${GICC_BOOTSTRAP_DEFINE})
    target_link_libraries(${name} PRIVATE ${GICC_OFI_LIBS})
endfunction()

gicc_ofi_example(test_gicc           test_gicc.cpp)
gicc_ofi_example(benchmark           benchmark.cpp)
gicc_ofi_example(mm_minimal          mm_minimal.cpp)
gicc_ofi_example(barrier_test        barrier_test.cpp)
gicc_ofi_example(am_pingpong         am_pingpong.cpp)
gicc_ofi_example(jacobi              jacobi.cpp)
gicc_ofi_example(barrier_bench       barrier_bench.cpp)
gicc_ofi_example(test_ofi_get        ${CMAKE_SOURCE_DIR}/legacy/ofi/test_ofi_get.cpp)
gicc_ofi_example(test_lookup         ${CMAKE_SOURCE_DIR}/legacy/ofi/test_lookup.cpp)
gicc_ofi_example(test_launch_skeleton ${CMAKE_SOURCE_DIR}/legacy/ofi/test_launch_skeleton.cpp)

# Pure MPI barrier reference benchmark (CPU-triggered, no HIP/libfabric).
if(GICC_BOOTSTRAP STREQUAL "mpi")
    add_executable(mpi_barrier_bench mpi_barrier_bench.cpp)
    target_link_libraries(mpi_barrier_bench PRIVATE MPI::MPI_CXX)
endif()
