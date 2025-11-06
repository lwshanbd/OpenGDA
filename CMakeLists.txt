cmake_minimum_required(VERSION 3.18)
project(OpenGDA VERSION 0.1.0 LANGUAGES CXX)

# ============================================================================
# Build Options
# ============================================================================
option(USE_AMDGPU "Enable AMD GPU support with ROCm/HIP" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

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

# Libfabric (OFI)
# Prefer Cray libfabric if available
if(DEFINED ENV{CRAY_LIBFABRIC_PREFIX})
    set(LIBFABRIC_INCLUDE_DIR "$ENV{CRAY_LIBFABRIC_PREFIX}/include")
    set(LIBFABRIC_LIBRARY "$ENV{CRAY_LIBFABRIC_PREFIX}/lib64/libfabric.so")
    if(NOT EXISTS ${LIBFABRIC_LIBRARY})
        message(FATAL_ERROR "Libfabric not found at ${LIBFABRIC_LIBRARY}")
    endif()
    message(STATUS "Found Libfabric (Cray): ${LIBFABRIC_LIBRARY}")
elseif(EXISTS "/opt/cray/libfabric/2.1/lib64/libfabric.so")
    # Fallback: use default Cray path
    set(LIBFABRIC_INCLUDE_DIR "/opt/cray/libfabric/2.1/include")
    set(LIBFABRIC_LIBRARY "/opt/cray/libfabric/2.1/lib64/libfabric.so")
    message(STATUS "Found Libfabric (Cray default): ${LIBFABRIC_LIBRARY}")
else()
    # Use system libfabric
    find_library(LIBFABRIC_LIBRARY NAMES fabric REQUIRED)
    find_path(LIBFABRIC_INCLUDE_DIR rdma/fabric.h REQUIRED)
    message(STATUS "Found Libfabric (system): ${LIBFABRIC_LIBRARY}")
endif()

# hwloc (Hardware Locality)
find_library(HWLOC_LIBRARY NAMES hwloc REQUIRED)
find_path(HWLOC_INCLUDE_DIR hwloc.h REQUIRED)
message(STATUS "Found hwloc: ${HWLOC_LIBRARY}")

# PMI2 (Process Management Interface)
if(DEFINED ENV{CRAY_PMI_PREFIX})
    set(PMI2_INCLUDE_DIR "$ENV{CRAY_PMI_PREFIX}/include")
    set(PMI2_LIBRARY_DIR "$ENV{CRAY_PMI_PREFIX}/lib")
    find_library(PMI2_LIBRARY NAMES pmi2 PATHS ${PMI2_LIBRARY_DIR} REQUIRED)
    message(STATUS "Found PMI2: ${PMI2_LIBRARY}")
else()
    find_library(PMI2_LIBRARY NAMES pmi2 REQUIRED)
    find_path(PMI2_INCLUDE_DIR pmi2.h REQUIRED)
    message(STATUS "Found PMI2: ${PMI2_LIBRARY}")
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
message(STATUS "Dependencies:")
message(STATUS "  Libfabric: ${LIBFABRIC_LIBRARY}")
message(STATUS "  hwloc: ${HWLOC_LIBRARY}")
message(STATUS "  PMI2: ${PMI2_LIBRARY}")
if(USE_AMDGPU)
    if(hip_FOUND)
        message(STATUS "  HIP: Found (via find_package)")
    else()
        message(STATUS "  HIP: ${HIP_LIBRARY}")
    endif()
    message(STATUS "  GPU support: ENABLED")
else()
    message(STATUS "  GPU support: DISABLED")
endif()
message(STATUS "========================================")
message(STATUS "")
