# ============================================================================
# OpenGDA Library - Source Directory
# ============================================================================

# ============================================================================
# Source Files
# ============================================================================
set(OPENGDA_SOURCES
    gda.cpp
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
elseif(NETWORK_TYPE STREQUAL "IBV")
    list(APPEND OPENGDA_SOURCES network/ibv.cpp)
endif()

set(OPENGDA_PUBLIC_HEADERS
    gda.h
)

set(OPENGDA_PRIVATE_HEADERS
    bootstrap/common.hpp
    bootstrap/pmi2.hpp
    network/ofi.hpp
)

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
