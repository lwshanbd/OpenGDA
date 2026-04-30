cmake_minimum_required(VERSION 3.21)
project(gicc-passes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Plugin must be built with the same Clang/LLVM that loads it (matching ABI).
set(_GICC_EXPECTED_CXX "/opt/rocm-6.4.0/lib/llvm/bin/clang++")
if(NOT CMAKE_CXX_COMPILER STREQUAL "${_GICC_EXPECTED_CXX}")
    message(FATAL_ERROR
        "gicc-passes must be built with the ROCm Clang to match LLVM's ABI.\n"
        "  -DCMAKE_CXX_COMPILER=${_GICC_EXPECTED_CXX}\n"
        "Current: ${CMAKE_CXX_COMPILER}")
endif()

set(LLVM_DIR "/opt/rocm-6.4.0/lib/llvm/lib/cmake/llvm" CACHE PATH "LLVM cmake dir")
find_package(LLVM REQUIRED CONFIG)
message(STATUS "gicc-passes: using LLVM ${LLVM_VERSION_MAJOR} from ${LLVM_DIR}")

add_library(gicc-passes MODULE
    src/PluginEntry.cpp
    src/GICCPassConfig.cpp
    src/SiteId.cpp
    src/GICCDeviceDiscovery.cpp
    src/HKAnalysis.cpp
    src/GICCHKAnalysis.cpp
    src/MetadataIO.cpp
    src/TraceTemplateBuilder.cpp
    src/GICCDeviceLowering.cpp
    src/GICCHostDiscovery.cpp
)
target_include_directories(gicc-passes PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(gicc-passes PRIVATE ${LLVM_DEFINITIONS})

# LLVM is built with -fno-rtti by default; using RTTI in the plugin causes
# undefined typeinfo references at dlopen time (e.g. _ZTIN4llvm13ErrorInfoBaseE
# from Expected<>). Match LLVM's build configuration.
target_compile_options(gicc-passes PRIVATE -fno-rtti)

# Don't link LLVM libs — the loading process (clang/opt) provides the symbols.
# Default MODULE library naming gives libgicc-passes.so on Linux.

# ---------------------------------------------------------------------------
# Lit tests
# ---------------------------------------------------------------------------
find_program(LIT_EXE lit HINTS ${LLVM_TOOLS_BINARY_DIR})
if(LIT_EXE)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/lit/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/tests/lit/lit.site.cfg.py
        @ONLY)
    add_custom_target(check-gicc-passes
        COMMAND ${LIT_EXE} -v ${CMAKE_CURRENT_BINARY_DIR}/tests/lit
        DEPENDS gicc-passes
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL)
else()
    message(STATUS "gicc-passes: lit not found; skipping check-gicc-passes target")
endif()
