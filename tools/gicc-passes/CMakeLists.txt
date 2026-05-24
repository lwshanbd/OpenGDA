cmake_minimum_required(VERSION 3.21)
project(gicc-passes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build modes
# ---------------------------------------------------------------------------
# GICC_PASSES_ANALYZE_ONLY=ON drops the two lowering passes that hard-code
# DWQ / MMIO codegen (GICCDeviceLowering, GICCDispatchLowering). The rest
# of the plugin — discovery, HK analysis, feature extraction, host
# discovery, trace synthesis — is target-agnostic IR-level code that
# already recognizes both AMDGPU and NVPTX intrinsics. Use this on
# NVIDIA / non-CXI machines (e.g. MAPLE) to exercise the analysis side
# without needing a DWQ runtime.
option(GICC_PASSES_ANALYZE_ONLY
    "Build only the target-agnostic analysis passes; skip DWQ/MMIO lowering"
    OFF)

# ---------------------------------------------------------------------------
# LLVM discovery
# ---------------------------------------------------------------------------
# The plugin must be ABI-compatible with the LLVM that loads it (clang
# or opt). The historical default points at the ROCm 6.4 LLVM 19 tree
# on Tioga; on other machines pass -DLLVM_DIR=<some>/lib/cmake/llvm
# to a matching install (any LLVM 19.x with full dev headers).
set(_GICC_ROCM_LLVM_DIR "/opt/rocm-6.4.0/lib/llvm/lib/cmake/llvm")
if(NOT DEFINED LLVM_DIR)
    if(EXISTS "${_GICC_ROCM_LLVM_DIR}")
        set(LLVM_DIR "${_GICC_ROCM_LLVM_DIR}" CACHE PATH "LLVM cmake dir")
    else()
        message(WARNING
            "gicc-passes: ROCm LLVM not found at ${_GICC_ROCM_LLVM_DIR}.\n"
            "Pass -DLLVM_DIR=<dir-with-LLVMConfig.cmake> to point at an "
            "LLVM 19.x dev install (any vendor: ROCm, upstream LLVM, "
            "Spack, etc.).")
    endif()
endif()
find_package(LLVM REQUIRED CONFIG)
message(STATUS "gicc-passes: using LLVM ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} "
               "from ${LLVM_DIR}")

if(NOT LLVM_VERSION_MAJOR EQUAL 19)
    message(WARNING
        "gicc-passes was developed against LLVM 19. Found "
        "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} — the plugin will "
        "only load into a host whose LLVM major version matches the "
        "build-time LLVM.")
endif()

# Warn (don't fail) when the C++ compiler isn't the LLVM bin/clang++.
# The plugin uses ${LLVM_INCLUDE_DIRS} for headers and links no LLVM
# libs at build time, so any C++17 compiler with a matching libstdc++
# ABI will produce a loadable .so. We do recommend the matching clang
# to minimise surprises.
get_filename_component(_GICC_LLVM_BIN "${LLVM_TOOLS_BINARY_DIR}" ABSOLUTE)
set(_GICC_LLVM_CXX "${_GICC_LLVM_BIN}/clang++")
if(EXISTS "${_GICC_LLVM_CXX}" AND
   NOT CMAKE_CXX_COMPILER STREQUAL "${_GICC_LLVM_CXX}")
    message(STATUS
        "gicc-passes: CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} does not "
        "match the LLVM bin/clang++ at ${_GICC_LLVM_CXX}. Build will "
        "proceed; pass -DCMAKE_CXX_COMPILER=${_GICC_LLVM_CXX} if you hit "
        "ABI issues when dlopen'ing the plugin.")
endif()

set(_GICC_PASS_SOURCES
    src/PluginEntry.cpp
    src/GICCPassConfig.cpp
    src/SiteId.cpp
    src/HostMirrorAnnotation.cpp
    src/GICCDeviceDiscovery.cpp
    src/HKAnalysis.cpp
    src/GICCHKAnalysis.cpp
    src/MetadataIO.cpp
    src/TraceTemplateBuilder.cpp
    src/GICCHostDiscovery.cpp
    src/GICCFeatureExtraction.cpp
    src/GICCTraceSynthesis.cpp
)
if(NOT GICC_PASSES_ANALYZE_ONLY)
    list(APPEND _GICC_PASS_SOURCES
        src/DispatchDecision.cpp
        src/GICCDeviceLowering.cpp
        src/GICCDispatchLowering.cpp
    )
else()
    message(STATUS "gicc-passes: ANALYZE_ONLY build — skipping "
                   "GICCDeviceLowering / GICCDispatchLowering")
endif()

add_library(gicc-passes MODULE ${_GICC_PASS_SOURCES})

target_include_directories(gicc-passes PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(gicc-passes PRIVATE ${LLVM_DEFINITIONS})
if(GICC_PASSES_ANALYZE_ONLY)
    target_compile_definitions(gicc-passes PRIVATE GICC_PASSES_ANALYZE_ONLY=1)
endif()

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
    if(GICC_PASSES_ANALYZE_ONLY)
        set(GICC_PASSES_HAVE_LOWERING 0)
    else()
        set(GICC_PASSES_HAVE_LOWERING 1)
    endif()
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
