# examples/omp/CMakeLists.txt - standalone consumer, NOT part of the main build.
# Proves find_package(gicc-omp) + gicc::omp compiles a public-API-only app.
cmake_minimum_required(VERSION 3.21)
project(giomp_examples LANGUAGES CXX)
set(CMAKE_CXX_COMPILER /p/lustre2/shan4/softwares/diomp/bin/clang++)
find_package(gicc-omp REQUIRED)
add_executable(hello_giomp hello_giomp.cpp)
target_link_libraries(hello_giomp PRIVATE gicc::omp)
