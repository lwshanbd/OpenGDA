# ============================================================================
# OpenGDA Library - Source Directory
# ============================================================================

# ============================================================================
# Source Files
# ============================================================================
set(OPENGDA_SOURCES
    gda.cpp
    gda_barrier_proxy.cpp
    common/log.cpp
    bootstrap/common.cpp
)

# Add bootstrap source files based on type
if(BOOTSTRAP_TYPE STREQUAL "PMI2")
    list(APPEND OPENGDA_SOURCES bootstrap/pmi2.cpp)
elseif(BOOTSTRAP_TYPE STREQUAL "PMIX")
    list(APPEND OPENGDA_SOURCES bootstrap/pmix.cpp)
endif()

# Add network source files based on type
if(NETWORK_TYPE STREQUAL "OFI")
    list(APPEND OPENGDA_SOURCES network/ofi.cpp)
elseif(NETWORK_TYPE STREQUAL "OFI_PROXY")
    list(APPEND OPENGDA_SOURCES network/ofi-proxy.cpp)
elseif(NETWORK_TYPE STREQUAL "IBV")
    list(APPEND OPENGDA_SOURCES network/ibv.cpp)
endif()

set(OPENGDA_PUBLIC_HEADERS
    gda.h
    gda_barrier_proxy.h
)

set(OPENGDA_PRIVATE_HEADERS
    common/log.hpp
    bootstrap/common.hpp
)

# Add bootstrap headers based on type
if(BOOTSTRAP_TYPE STREQUAL "PMI2")
    list(APPEND OPENGDA_PRIVATE_HEADERS bootstrap/pmi2.hpp)
elseif(BOOTSTRAP_TYPE STREQUAL "PMIX")
    list(APPEND OPENGDA_PRIVATE_HEADERS bootstrap/pmix.hpp)
endif()

# Add network headers based on type
if(NETWORK_TYPE STREQUAL "OFI")
    list(APPEND OPENGDA_PRIVATE_HEADERS network/ofi.hpp)
elseif(NETWORK_TYPE STREQUAL "IBV")
    list(APPEND OPENGDA_PRIVATE_HEADERS network/ibv.hpp)
endif()

# ============================================================================
# Build Library Target
# ============================================================================
add_library(opengda ${OPENGDA_SOURCES})

# Set library properties
set_target_properties(opengda PROPERTIES
    OUTPUT_NAME "opengda"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${OPENGDA_PUBLIC_HEADERS}"
)

# ============================================================================
# Include Directories
# ============================================================================
target_include_directories(opengda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${NETWORK_INCLUDE_DIR}
        ${BOOTSTRAP_INCLUDE_DIR}
        ${HWLOC_INCLUDE_DIR}
)

# ============================================================================
# Link Libraries
# ============================================================================
target_link_libraries(opengda
    PRIVATE
        ${NETWORK_LIBRARY}
        ${BOOTSTRAP_LIBRARY}
        ${HWLOC_LIBRARY}
        pthread
)

# ============================================================================
# Compiler Options
# ============================================================================
target_compile_options(opengda PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# ============================================================================
# Compile Definitions (Macros)
# ============================================================================

# Bootstrap type macro
target_compile_definitions(opengda PRIVATE ${BOOTSTRAP_COMPILE_DEF})

# Network type macro
target_compile_definitions(opengda PRIVATE ${NETWORK_COMPILE_DEF})

# AMD GPU Support
if(USE_AMDGPU)
    target_compile_definitions(opengda PRIVATE USE_AMDGPU=1)
    target_include_directories(opengda PRIVATE ${HIP_INCLUDE_DIR})

    if(hip_FOUND)
        target_link_libraries(opengda PRIVATE hip::host)
    else()
        target_link_libraries(opengda PRIVATE ${HIP_LIBRARY})
    endif()
endif()

# NVIDIA GPU Support
if(USE_NVGPU)
    target_compile_definitions(opengda PRIVATE USE_NVGPU=1)
    target_include_directories(opengda PRIVATE ${CUDA_INCLUDE_DIR})

    if(CUDAToolkit_FOUND)
        target_link_libraries(opengda PRIVATE CUDA::cudart)
    else()
        target_link_libraries(opengda PRIVATE ${CUDA_RUNTIME_LIBRARY})
    endif()
endif()

# Debug Support
if(DEBUG_OFI)
    target_compile_definitions(opengda PRIVATE DEBUG_OFI=1)
    message(STATUS "Debug logging: ENABLED (DEBUG/TRACE compiled in)")
else()
    message(STATUS "Debug logging: DISABLED (DEBUG/TRACE optimized out, INFO/WARN/ERROR always available)")
endif()

# Default MR sizes (compile-time defaults, can be overridden by environment variables)
target_compile_definitions(opengda PRIVATE
    DEFAULT_HOST_MR_SIZE=${DEFAULT_HOST_MR_SIZE}ULL
    DEFAULT_GPU_MR_SIZE=${DEFAULT_GPU_MR_SIZE}ULL
)

# ============================================================================
# Copy Headers to Build Directory (for development)
# ============================================================================

# Create include directory in build
set(BUILD_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${BUILD_INCLUDE_DIR})

# Copy public headers
add_custom_command(
    TARGET opengda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/gda.h
        ${BUILD_INCLUDE_DIR}/gda.h
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/gda_barrier_proxy.h
        ${BUILD_INCLUDE_DIR}/gda_barrier_proxy.h
    COMMENT "Copying public headers to build directory"
)

# Copy bootstrap headers
add_custom_command(
    TARGET opengda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${BUILD_INCLUDE_DIR}/opengda/bootstrap
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/bootstrap/common.hpp
        ${BUILD_INCLUDE_DIR}/opengda/bootstrap/common.hpp
)

# Copy bootstrap-specific headers
if(BOOTSTRAP_TYPE STREQUAL "PMI2")
    add_custom_command(
        TARGET opengda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/bootstrap/pmi2.hpp
            ${BUILD_INCLUDE_DIR}/opengda/bootstrap/pmi2.hpp
    )
elseif(BOOTSTRAP_TYPE STREQUAL "PMIX")
    add_custom_command(
        TARGET opengda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/bootstrap/pmix.hpp
            ${BUILD_INCLUDE_DIR}/opengda/bootstrap/pmix.hpp
    )
endif()

# Copy network headers
add_custom_command(
    TARGET opengda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${BUILD_INCLUDE_DIR}/opengda/network
)

if(NETWORK_TYPE STREQUAL "OFI")
    add_custom_command(
        TARGET opengda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/network/ofi.hpp
            ${BUILD_INCLUDE_DIR}/opengda/network/ofi.hpp
    )
elseif(NETWORK_TYPE STREQUAL "OFI_PROXY")
    add_custom_command(
        TARGET opengda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/network/ofi-proxy.hpp
            ${BUILD_INCLUDE_DIR}/opengda/network/ofi-proxy.hpp
    )
elseif(NETWORK_TYPE STREQUAL "IBV")
    add_custom_command(
        TARGET opengda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/network/ibv.hpp
            ${BUILD_INCLUDE_DIR}/opengda/network/ibv.hpp
    )
endif()

message(STATUS "Headers will be copied to: ${BUILD_INCLUDE_DIR}")

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

# Install library
install(TARGETS opengda
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install private headers (for C++ API users)
install(DIRECTORY
    bootstrap
    network
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opengda
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)
