# ============================================================================
# gicc_pass — LLVM analysis pass (NewPM plugin) for the GICC-Pilot selector
#
# Builds libgicc_pass.so, a loadable plugin compatible with opt/clang from the
# LLVM version we link against. Also a tiny demo binary (gicc_explain) that
# can run the pass on a single .ll file from the command line for CI/debug.
# ============================================================================
cmake_minimum_required(VERSION 3.21)

# LLVM_DIR must point at a cmake package dir of the SAME LLVM that clang/hipcc
# use to produce bitcode — otherwise IR-version skew will break loading.
if(NOT DEFINED LLVM_DIR)
    set(LLVM_DIR "/opt/rocm-6.4.0/lib/llvm/lib/cmake/llvm" CACHE PATH
        "Path to LLVMConfig.cmake")
endif()

find_package(LLVM CONFIG REQUIRED PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)

message(STATUS "gicc_pass: LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_INSTALL_PREFIX}")

# ---- The pass plugin itself -------------------------------------------------
add_library(gicc_pass SHARED
    GiccPass.cpp)

target_include_directories(gicc_pass PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src)   # for "gicc/policy/policy.hpp"

target_compile_definitions(gicc_pass PRIVATE
    ${LLVM_DEFINITIONS})

# Pass plugin must match LLVM's C++ ABI flags; llvm-config ships the right ones.
# -fno-rtti is required because LLVM is built without RTTI; mixing would fail at link.
target_compile_options(gicc_pass PRIVATE
    -fno-rtti
    -fPIC
    -O2
    -Wall)

# The pass links to libgicc_policy for rule evaluation. Both are in this CMake
# build; propagate PUBLIC include from gicc_policy automatically.
target_link_libraries(gicc_pass PRIVATE gicc_policy)

# Plugin should NOT link all of libLLVM — opt/clang provide those symbols at
# dlopen time. Leaving them undefined is the standard NewPM plugin pattern.
set_target_properties(gicc_pass PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "gicc_pass"
    LINK_FLAGS "-Wl,--allow-shlib-undefined")

# Build-time flag that selects whether infeasible decisions are an error (CAP_PRUNE=1)
# or a warning+fallback (CAP_PRUNE=0). Default ON follows M2-A / GICC_CAP_PRUNE.
target_compile_definitions(gicc_pass PRIVATE
    GICC_PASS_CAP_PRUNE=$<IF:$<BOOL:${GICC_CAP_PRUNE}>,1,0>)

# ---- Small standalone harness: run the pass on a .ll file -------------------
# This exists so CI / toy-kernel tests (M2-I) can exercise the pass without
# rigging the full clang/hipcc pipeline.
add_executable(gicc_explain gicc_explain_main.cpp)
target_include_directories(gicc_explain PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_compile_definitions(gicc_explain PRIVATE ${LLVM_DEFINITIONS})
target_compile_options(gicc_explain PRIVATE -fno-rtti -O2 -Wall)
target_link_libraries(gicc_explain PRIVATE
    gicc_policy
    LLVMCore
    LLVMSupport
    LLVMIRReader
    LLVMBitReader
    LLVMAnalysis
    LLVMPasses
    LLVMTransformUtils)

# ---- Toy-kernel unit tests (Block M2-I) ------------------------------------
if(GICC_BUILD_POLICY_TESTS)
    add_test(NAME gicc_pass_toy_kernels
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/gicc_pass/run_pass_tests.sh
                "${CMAKE_BINARY_DIR}"
                "${LLVM_TOOLS_BINARY_DIR}/clang++"
                "${LLVM_TOOLS_BINARY_DIR}/opt")
endif()
