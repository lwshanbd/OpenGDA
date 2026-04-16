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

# Cray PMI2
set(PMI_ROOT "/opt/cray/pe/pmi/6.1.15" CACHE PATH "Cray PMI path")
find_path(PMI_INCLUDE_DIR pmi2.h
    PATHS ${PMI_ROOT} PATH_SUFFIXES include NO_DEFAULT_PATH)
find_library(PMI_LIBRARY pmi2
    PATHS ${PMI_ROOT} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
if(NOT PMI_INCLUDE_DIR OR NOT PMI_LIBRARY)
    message(FATAL_ERROR "Cray PMI not found at ${PMI_ROOT}")
endif()

# hwloc (GPU-NIC affinity)
find_path(HWLOC_INCLUDE_DIR hwloc.h)
find_library(HWLOC_LIBRARY hwloc)
if(NOT HWLOC_INCLUDE_DIR OR NOT HWLOC_LIBRARY)
    message(FATAL_ERROR "hwloc not found")
endif()

find_package(MPI REQUIRED)

set(GICC_OFI_INCLUDES
    ${CMAKE_SOURCE_DIR}/src                                  # for "gicc/..."
    ${CMAKE_SOURCE_DIR}/src/gicc/platform/ofi/internal       # for sibling-relative includes inside internal/
    ${LIBFABRIC_INCLUDE_DIR}
    ${PMI_INCLUDE_DIR}
    ${HWLOC_INCLUDE_DIR}
)

set(GICC_OFI_LIBS
    hip::device
    ${LIBFABRIC_LIBRARY}
    ${PMI_LIBRARY}
    ${HWLOC_LIBRARY}
    pthread
)

function(gicc_ofi_example name source)
    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${GICC_OFI_INCLUDES} ${MPI_CXX_INCLUDE_DIRS})
    target_compile_definitions(${name} PRIVATE GICC_PLATFORM_OFI)
    target_link_libraries(${name} PRIVATE ${GICC_OFI_LIBS} MPI::MPI_CXX)
endfunction()

gicc_ofi_example(test_gicc           test_gicc.cpp)
gicc_ofi_example(gda_benchmark_gicc  gda_benchmark_gicc.cpp)
gicc_ofi_example(mm_gda_minimal_gicc mm_gda_minimal_gicc.cpp)
gicc_ofi_example(barrier_test        barrier_test.cpp)
