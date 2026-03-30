cmake_minimum_required(VERSION 3.18)
project(GICC VERSION 1.0.0 LANGUAGES CXX CUDA)

# ============================================================================
# Build Options
# ============================================================================

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build GICC examples" ON)

# ============================================================================
# Subdirectories
# ============================================================================

add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(examples/gicc)
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "GICC Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version:      ${PROJECT_VERSION}")
message(STATUS "Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Examples:     ${BUILD_EXAMPLES}")
message(STATUS "========================================")
message(STATUS "")
