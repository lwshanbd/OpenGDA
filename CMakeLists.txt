cmake_minimum_required(VERSION 3.21)

# ============================================================================
# Backend selection — set BEFORE project() so the language list is correct.
# ============================================================================
set(GICC_BACKEND "mlx5" CACHE STRING "GICC backend: mlx5 (NVIDIA+IB) or ofi (AMD+libfabric)")
set_property(CACHE GICC_BACKEND PROPERTY STRINGS mlx5 ofi)

if(GICC_BACKEND STREQUAL "ofi")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "HIP architectures")
    project(GICC VERSION 1.0.0 LANGUAGES CXX HIP)
elseif(GICC_BACKEND STREQUAL "mlx5")
    project(GICC VERSION 1.0.0 LANGUAGES CXX CUDA)
else()
    message(FATAL_ERROR "GICC_BACKEND must be 'mlx5' or 'ofi' (got '${GICC_BACKEND}')")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES    "Build GICC examples"    ON)

# ============================================================================
# Subdirectories
# ============================================================================
if(GICC_BACKEND STREQUAL "mlx5")
    add_subdirectory(src)
    if(BUILD_EXAMPLES)
        add_subdirectory(examples/gicc)
    endif()
elseif(GICC_BACKEND STREQUAL "ofi")
    option(GICC_BUILD_PLUGIN "Build the gicc-clang-plugin" OFF)
    set(GICC_OFI_PLUGIN_PATH "" CACHE FILEPATH "Path to gicc-clang-plugin.so")
    if(GICC_BUILD_PLUGIN)
        add_subdirectory(tools/gicc-clang-plugin)
    endif()
    # gicc_target_plugin(<tgt>) helper. Always available under the OFI
    # branch; errors at call time if neither GICC_BUILD_PLUGIN nor
    # GICC_OFI_PLUGIN_PATH is set, so projects that don't use the plugin
    # are unaffected.
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GiccPluginTarget.cmake)
    if(BUILD_EXAMPLES)
        add_subdirectory(examples/ofi)
    endif()
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "GICC Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version:      ${PROJECT_VERSION}")
message(STATUS "Backend:      ${GICC_BACKEND}")
message(STATUS "Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Examples:     ${BUILD_EXAMPLES}")
message(STATUS "========================================")
message(STATUS "")
