# OpenGDA Examples
#
# Examples demonstrating OpenGDA public API usage.
# All examples are built with HIP support when USE_AMDGPU is enabled.

# Find MPI for mm_hip_mpi example
find_package(MPI REQUIRED)

if(USE_AMDGPU)
    # ========================================================================
    # GPU Examples (require HIP)
    # ========================================================================

    # mr_example - Memory registration API (1 rank)
    add_executable(mr_example mr_example.cpp)
    target_link_libraries(mr_example PRIVATE opengda)
    target_compile_definitions(mr_example PRIVATE USE_AMDGPU)
    set_source_files_properties(mr_example.cpp PROPERTIES LANGUAGE HIP)

    # put_example - GPU-triggered put/get operations (2 ranks)
    add_executable(put_example put_example.cpp)
    target_link_libraries(put_example PRIVATE opengda)
    target_compile_definitions(put_example PRIVATE USE_AMDGPU)
    set_source_files_properties(put_example.cpp PROPERTIES LANGUAGE HIP)

    # test_example - Non-blocking completion detection API (2 ranks)
    add_executable(test_example test_example.cpp)
    target_link_libraries(test_example PRIVATE opengda)
    target_compile_definitions(test_example PRIVATE USE_AMDGPU)
    set_source_files_properties(test_example.cpp PROPERTIES LANGUAGE HIP)

    # Link HIP runtime
    if(hip_FOUND)
        target_link_libraries(mr_example PRIVATE hip::device)
        target_link_libraries(put_example PRIVATE hip::device)
        target_link_libraries(test_example PRIVATE hip::device)
    else()
        target_include_directories(mr_example PRIVATE ${HIP_INCLUDE_DIR})
        target_include_directories(put_example PRIVATE ${HIP_INCLUDE_DIR})
        target_include_directories(test_example PRIVATE ${HIP_INCLUDE_DIR})
        target_link_libraries(mr_example PRIVATE ${HIP_LIBRARY})
        target_link_libraries(put_example PRIVATE ${HIP_LIBRARY})
        target_link_libraries(test_example PRIVATE ${HIP_LIBRARY})
    endif()

    # mm_hip_mpi - Matrix multiplication with HIP + MPI (N ranks)
    add_executable(mm_hip_mpi mm_hip_mpi.cpp)
    target_link_libraries(mm_hip_mpi PRIVATE MPI::MPI_CXX)
    target_compile_definitions(mm_hip_mpi PRIVATE USE_AMDGPU)
    set_source_files_properties(mm_hip_mpi.cpp PROPERTIES LANGUAGE HIP)
    # Link GTL library for GPU-aware MPI (HSA for AMD GPUs)
    target_link_libraries(mm_hip_mpi PRIVATE /opt/cray/pe/lib64/libmpi_gtl_hsa.so)

    # mm_hip_gda - Matrix multiplication with HIP + GDA (N ranks)
    add_executable(mm_hip_gda mm_hip_gda.cpp)
    target_link_libraries(mm_hip_gda PRIVATE opengda)
    target_compile_definitions(mm_hip_gda PRIVATE USE_AMDGPU)
    set_source_files_properties(mm_hip_gda.cpp PROPERTIES LANGUAGE HIP)

    # barrier_test - GPU barrier test (N ranks)
    add_executable(barrier_test barrier_test.cpp)
    target_link_libraries(barrier_test PRIVATE opengda)
    target_compile_definitions(barrier_test PRIVATE USE_AMDGPU)
    set_source_files_properties(barrier_test.cpp PROPERTIES LANGUAGE HIP)

    # Link HIP runtime to new examples
    if(hip_FOUND)
        target_link_libraries(mm_hip_mpi PRIVATE hip::device)
        target_link_libraries(mm_hip_gda PRIVATE hip::device)
        target_link_libraries(barrier_test PRIVATE hip::device)
    else()
        target_include_directories(mm_hip_mpi PRIVATE ${HIP_INCLUDE_DIR})
        target_include_directories(mm_hip_gda PRIVATE ${HIP_INCLUDE_DIR})
        target_include_directories(barrier_test PRIVATE ${HIP_INCLUDE_DIR})
        target_link_libraries(mm_hip_mpi PRIVATE ${HIP_LIBRARY})
        target_link_libraries(mm_hip_gda PRIVATE ${HIP_LIBRARY})
        target_link_libraries(barrier_test PRIVATE ${HIP_LIBRARY})
    endif()

    # Installation
    install(TARGETS mr_example put_example test_example mm_hip_mpi mm_hip_gda
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)
else()
    message(STATUS "Skipping examples (GPU support required)")
endif()
