# examples/omp/CMakeLists.txt - standalone consumer, NOT part of the main build.
# Proves find_package(gicc-omp) + gicc::omp compiles a public-API-only app.
cmake_minimum_required(VERSION 3.21)
# Compiler must be set BEFORE project() to take effect. ROCm 6.4.0 clang matches
# the pass plugin's LLVM (so the DWQ target can load -fpass-plugin too).
set(CMAKE_CXX_COMPILER /opt/rocm-6.4.0/lib/llvm/bin/clang++)
project(giomp_examples LANGUAGES CXX)
find_package(gicc-omp REQUIRED)
add_executable(hello_giomp hello_giomp.cpp)
target_link_libraries(hello_giomp PRIVATE gicc::omp)
