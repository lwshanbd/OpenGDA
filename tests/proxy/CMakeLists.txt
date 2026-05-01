# tests/proxy/CMakeLists.txt - microtests for the CPU Proxy ring buffer.
# Gated on GICC_ENABLE_CPU_PROXY (set by the top-level CMakeLists.txt).

if(NOT GICC_ENABLE_CPU_PROXY)
    return()
endif()

# The ring is header-only — we don't link against the gicc runtime library
# because that would pull in the rest of the OFI backend, which may not
# compile under all GPU API selections during early bring-up.

if(GICC_GPU_API STREQUAL "cuda")
    set_source_files_properties(test_d2h_ring.cu PROPERTIES LANGUAGE CUDA)
    add_executable(test_d2h_ring test_d2h_ring.cu)
elseif(GICC_GPU_API STREQUAL "hip")
    set_source_files_properties(test_d2h_ring.hip.cpp PROPERTIES LANGUAGE HIP)
    add_executable(test_d2h_ring test_d2h_ring.hip.cpp)
else()
    message(FATAL_ERROR
        "tests/proxy: unknown GICC_GPU_API '${GICC_GPU_API}'")
endif()

target_include_directories(test_d2h_ring PRIVATE
    ${CMAKE_SOURCE_DIR}/src)
