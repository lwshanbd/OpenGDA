cmake_minimum_required(VERSION 3.21)
project(gicc-clang-plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Plugin must be built with the same Clang that will dlopen() it, or
# typeinfo symbols (e.g. clang::PluginASTAction) go undefined at load.
set(_GICC_EXPECTED_CXX "/opt/rocm-6.4.0/lib/llvm/bin/clang++")
if(NOT CMAKE_CXX_COMPILER STREQUAL "${_GICC_EXPECTED_CXX}")
    message(WARNING
        "gicc-clang-plugin: CMAKE_CXX_COMPILER is '${CMAKE_CXX_COMPILER}', "
        "but the plugin should be built with '${_GICC_EXPECTED_CXX}' "
        "to match the Clang that will load it. Re-run cmake with "
        "-DCMAKE_CXX_COMPILER=${_GICC_EXPECTED_CXX} if loading fails.")
endif()

# Use the Clang/LLVM that ships with ROCm so plugin ABI matches hipcc's clang.
set(LLVM_DIR  "/opt/rocm-6.4.0/lib/llvm/lib/cmake/llvm"
    CACHE PATH "LLVM cmake dir")
set(Clang_DIR "/opt/rocm-6.4.0/lib/llvm/lib/cmake/clang"
    CACHE PATH "Clang cmake dir")

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
message(STATUS "gicc-clang-plugin: using LLVM ${LLVM_VERSION_MAJOR} from ${LLVM_DIR}")

add_library(gicc-clang-plugin MODULE
    src/PluginAction.cpp
)

target_include_directories(gicc-clang-plugin PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(gicc-clang-plugin PRIVATE
    ${LLVM_DEFINITIONS}
)

# LLVM/Clang are built with -fno-rtti; plugin TUs that subclass Clang
# classes (PluginASTAction, ASTConsumer) must match or typeinfo symbols
# from the host Clang go undefined at dlopen() time.
target_compile_options(gicc-clang-plugin PRIVATE -fno-rtti)

# Plugins must NOT link against LLVM core libs — Clang loads them at
# runtime; double-link would conflict.
set_target_properties(gicc-clang-plugin PROPERTIES
    PREFIX ""        # produce gicc-clang-plugin.so, not lib...so
    POSITION_INDEPENDENT_CODE ON
)
