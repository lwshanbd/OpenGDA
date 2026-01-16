cmake_minimum_required(VERSION 3.21)

# Must set HIP architecture before enabling HIP language
set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "HIP architectures")

project(gda_minimal LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_HIP_FLAGS_RELEASE "-O3")

# Find ROCm/HIP
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
find_package(hip REQUIRED)

# Find libfabric - prefer Cray libfabric which has CXI extensions
# Set NO_DEFAULT_PATH first to ensure Cray path is used
set(LIBFABRIC_ROOT "/opt/cray/libfabric/2.1" CACHE PATH "Cray libfabric path")
find_path(LIBFABRIC_INCLUDE_DIR rdma/fi_cxi_ext.h
    PATHS ${LIBFABRIC_ROOT}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)
find_library(LIBFABRIC_LIBRARY fabric
    PATHS ${LIBFABRIC_ROOT}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

if(NOT LIBFABRIC_INCLUDE_DIR OR NOT LIBFABRIC_LIBRARY)
    message(FATAL_ERROR "Cray libfabric not found at ${LIBFABRIC_ROOT}")
endif()

# Find PMI2 from Cray PMI
set(PMI_ROOT "/opt/cray/pe/pmi/6.1.15" CACHE PATH "Cray PMI path")
find_path(PMI_INCLUDE_DIR pmi2.h
    PATHS ${PMI_ROOT}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)
find_library(PMI_LIBRARY pmi2
    PATHS ${PMI_ROOT}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

if(NOT PMI_INCLUDE_DIR OR NOT PMI_LIBRARY)
    message(FATAL_ERROR "Cray PMI not found at ${PMI_ROOT}")
endif()

# Common include directories and libraries
set(COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBFABRIC_INCLUDE_DIR}
    ${PMI_INCLUDE_DIR}
)

set(COMMON_LIBS
    hip::device
    ${LIBFABRIC_LIBRARY}
    ${PMI_LIBRARY}
)

# Original benchmark (6 concurrent streams max)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(gda_benchmark main.cpp)
target_include_directories(gda_benchmark PRIVATE ${COMMON_INCLUDES})
target_link_libraries(gda_benchmark PRIVATE ${COMMON_LIBS})

# Batched benchmark (16+ streams via counter pool)
set_source_files_properties(main_batched.cpp PROPERTIES LANGUAGE HIP)
add_executable(gda_benchmark_batched main_batched.cpp)
target_include_directories(gda_benchmark_batched PRIVATE ${COMMON_INCLUDES})
target_link_libraries(gda_benchmark_batched PRIVATE ${COMMON_LIBS})

# Print configuration
message(STATUS "HIP found: ${hip_DIR}")
message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "libfabric include: ${LIBFABRIC_INCLUDE_DIR}")
message(STATUS "libfabric library: ${LIBFABRIC_LIBRARY}")
message(STATUS "PMI include: ${PMI_INCLUDE_DIR}")
message(STATUS "PMI library: ${PMI_LIBRARY}")
