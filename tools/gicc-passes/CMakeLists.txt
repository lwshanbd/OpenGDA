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
)
target_include_directories(gicc-passes PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(gicc-passes PRIVATE ${LLVM_DEFINITIONS})

# Don't link LLVM libs — the loading process (clang/opt) provides the symbols.
# Default MODULE library naming gives libgicc-passes.so on Linux.
