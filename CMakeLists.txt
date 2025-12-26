cmake_minimum_required(VERSION 3.18)
project(OpenGDA VERSION 0.1.0 LANGUAGES CXX)

# ============================================================================
# Build Options
# ============================================================================
option(USE_AMDGPU "Enable AMD GPU support with ROCm/HIP" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(DEBUG_OFI "Enable DEBUG/TRACE log levels (default OFF for production)" OFF)

# Default Memory Region sizes (in bytes, can be overridden by environment variables)
set(DEFAULT_HOST_MR_SIZE "17179869184" CACHE STRING "Default host MR size in bytes (default 16GB)")
set(DEFAULT_GPU_MR_SIZE "17179869184" CACHE STRING "Default GPU MR size in bytes (default 16GB)")

# Bootstrap options
set(BOOTSTRAP_TYPE "PMI2" CACHE STRING "Bootstrap type: PMI2 or PMIX")
set_property(CACHE BOOTSTRAP_TYPE PROPERTY STRINGS "PMI2" "PMIX")

# Network options
set(NETWORK_TYPE "OFI" CACHE STRING "Network backend: OFI or IBV")
set_property(CACHE NETWORK_TYPE PROPERTY STRINGS "OFI" "IBV")

# ============================================================================
# Compiler Settings
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Find Required Dependencies
# ============================================================================

# Common dependencies
# hwloc (Hardware Locality)
find_library(HWLOC_LIBRARY NAMES hwloc REQUIRED)
find_path(HWLOC_INCLUDE_DIR hwloc.h REQUIRED)
message(STATUS "Found hwloc: ${HWLOC_LIBRARY}")

# ============================================================================
# Find Bootstrap Dependencies
# ============================================================================
if(BOOTSTRAP_TYPE STREQUAL "PMI2")
    # PMI2 (Process Management Interface)
    if(DEFINED ENV{CRAY_PMI_PREFIX})
        set(BOOTSTRAP_INCLUDE_DIR "$ENV{CRAY_PMI_PREFIX}/include")
        set(BOOTSTRAP_LIBRARY_DIR "$ENV{CRAY_PMI_PREFIX}/lib")
        find_library(BOOTSTRAP_LIBRARY NAMES pmi2 PATHS ${BOOTSTRAP_LIBRARY_DIR} REQUIRED)
        message(STATUS "Found PMI2 (Cray): ${BOOTSTRAP_LIBRARY}")
    else()
        find_library(BOOTSTRAP_LIBRARY NAMES pmi2 REQUIRED)
        find_path(BOOTSTRAP_INCLUDE_DIR pmi2.h REQUIRED)
        message(STATUS "Found PMI2: ${BOOTSTRAP_LIBRARY}")
    endif()
    set(BOOTSTRAP_COMPILE_DEF "BOOTSTRAP_PMI2")
elseif(BOOTSTRAP_TYPE STREQUAL "PMIX")
    # PMIx (Process Management Interface Exascale)
    find_library(BOOTSTRAP_LIBRARY NAMES pmix REQUIRED)
    find_path(BOOTSTRAP_INCLUDE_DIR pmix.h REQUIRED)
    message(STATUS "Found PMIx: ${BOOTSTRAP_LIBRARY}")
    set(BOOTSTRAP_COMPILE_DEF "BOOTSTRAP_PMIX")
else()
    message(FATAL_ERROR "Invalid BOOTSTRAP_TYPE: ${BOOTSTRAP_TYPE}. Must be PMI2 or PMIX.")
endif()

# ============================================================================
# Find Network Dependencies
# ============================================================================
if(NETWORK_TYPE STREQUAL "OFI")
    # Libfabric (OFI) - Prefer Cray libfabric if available
    if(DEFINED ENV{CRAY_LIBFABRIC_PREFIX})
        set(NETWORK_INCLUDE_DIR "$ENV{CRAY_LIBFABRIC_PREFIX}/include")
        set(NETWORK_LIBRARY "$ENV{CRAY_LIBFABRIC_PREFIX}/lib64/libfabric.so")
        if(NOT EXISTS ${NETWORK_LIBRARY})
            message(FATAL_ERROR "Libfabric not found at ${NETWORK_LIBRARY}")
        endif()
        message(STATUS "Found Libfabric (Cray): ${NETWORK_LIBRARY}")
    elseif(EXISTS "/opt/cray/libfabric/2.1/lib64/libfabric.so")
        # Fallback: use default Cray path
        set(NETWORK_INCLUDE_DIR "/opt/cray/libfabric/2.1/include")
        set(NETWORK_LIBRARY "/opt/cray/libfabric/2.1/lib64/libfabric.so")
        message(STATUS "Found Libfabric (Cray default): ${NETWORK_LIBRARY}")
    else()
        # Use system libfabric
        find_library(NETWORK_LIBRARY NAMES fabric REQUIRED)
        find_path(NETWORK_INCLUDE_DIR rdma/fabric.h REQUIRED)
        message(STATUS "Found Libfabric (system): ${NETWORK_LIBRARY}")
    endif()
    set(NETWORK_COMPILE_DEF "NETWORK_OFI")
elseif(NETWORK_TYPE STREQUAL "IBV")
    # InfiniBand Verbs
    find_library(NETWORK_LIBRARY NAMES ibverbs REQUIRED)
    find_path(NETWORK_INCLUDE_DIR infiniband/verbs.h REQUIRED)
    message(STATUS "Found libibverbs: ${NETWORK_LIBRARY}")
    set(NETWORK_COMPILE_DEF "NETWORK_IBV")
else()
    message(FATAL_ERROR "Invalid NETWORK_TYPE: ${NETWORK_TYPE}. Must be OFI or IBV.")
endif()

# ============================================================================
# Find Optional Dependencies (AMD GPU)
# ============================================================================
if(USE_AMDGPU)
    # Find ROCm/HIP
    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH $ENV{ROCM_PATH})
        else()
            set(ROCM_PATH "/opt/rocm")
        endif()
    endif()

    find_package(hip QUIET)
    if(NOT hip_FOUND)
        # Fallback: manually set HIP paths
        set(HIP_INCLUDE_DIR "${ROCM_PATH}/include")
        set(HIP_LIBRARY_DIR "${ROCM_PATH}/lib")
        find_library(HIP_LIBRARY NAMES amdhip64 PATHS ${HIP_LIBRARY_DIR})
        if(HIP_LIBRARY)
            message(STATUS "Found HIP (manual): ${HIP_LIBRARY}")
        else()
            message(WARNING "HIP not found, disabling GPU support")
            set(USE_AMDGPU OFF)
        endif()
    else()
        message(STATUS "Found HIP (via find_package)")
    endif()
endif()

# ============================================================================
# Subdirectories
# ============================================================================
add_subdirectory(src)

# ============================================================================
# Installation (Simple mode - no CMake package config)
# ============================================================================
include(GNUInstallDirs)

# ============================================================================
# Print Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "OpenGDA Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Bootstrap: ${BOOTSTRAP_TYPE}")
message(STATUS "  Network: ${NETWORK_TYPE}")
if(USE_AMDGPU)
    message(STATUS "  GPU support: ENABLED (AMD/ROCm)")
else()
    message(STATUS "  GPU support: DISABLED")
endif()
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  Bootstrap: ${BOOTSTRAP_LIBRARY}")
message(STATUS "  Network: ${NETWORK_LIBRARY}")
message(STATUS "  hwloc: ${HWLOC_LIBRARY}")
if(USE_AMDGPU)
    if(hip_FOUND)
        message(STATUS "  HIP: Found (via find_package)")
    else()
        message(STATUS "  HIP: ${HIP_LIBRARY}")
    endif()
endif()
message(STATUS "========================================")
message(STATUS "")
