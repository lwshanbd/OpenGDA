# ============================================================================
# GiccPluginTarget.cmake — gicc_target_plugin(<tgt>) helper.
#
# Wires the gicc-clang-plugin into the compile of an OFI-backend target.
# After this call:
#
#   * Every source on <tgt> is compiled with
#       -fplugin=<plugin.so>
#       -fplugin-arg-gicc-sidecar-dir=<build>/<tgt>.gicc
#       -include <build>/<tgt>.gicc/<stem>.gicc.cpp
#     The plugin writes the sidecar as a side effect of the main compile;
#     the -include flag pulls the (previous build's) sidecar into the
#     translation unit so the main TU sees the
#     `gicc::detail::kernel_trace<&K>::run` specialization at the launch
#     call site.
#
# Usage:
#
#   add_executable(my_app my_app.cu)
#   gicc_ofi_example(my_app my_app.cu)        # or your own helper
#   gicc_target_plugin(my_app)
#
# Requirements:
#   * GICC_BUILD_PLUGIN=ON (or GICC_OFI_PLUGIN_PATH set to a built plugin)
#   * Sources listed on the target must exist at configure time.
#
# Two-pass build behaviour:
#   The first `make` of a target compiles the user source against an
#   empty stub sidecar (so the no-op default kernel_trace is what the
#   linker picks up), but the plugin overwrites the stub with the real
#   trace as a side effect. The second `make` recompiles the user source
#   against the new sidecar and now exercises the real trace. This
#   applies on every change to the user source. Incremental builds that
#   touch only headers don't need the second pass.
# ============================================================================

function(gicc_target_plugin tgt)
    # ---- Resolve the plugin .so path ---------------------------------------
    # Two ways to acquire it:
    #   1. -DGICC_OFI_PLUGIN_PATH=<path>      (use a pre-built plugin)
    #   2. -DGICC_BUILD_PLUGIN=ON             (build it in-tree; resolve
    #                                          via the gicc-clang-plugin
    #                                          target that lives in
    #                                          tools/gicc-clang-plugin).
    set(_plugin_path "${GICC_OFI_PLUGIN_PATH}")
    set(_plugin_dep)
    if(NOT _plugin_path AND TARGET gicc-clang-plugin)
        set(_plugin_path "$<TARGET_FILE:gicc-clang-plugin>")
        set(_plugin_dep gicc-clang-plugin)
    endif()
    if(NOT _plugin_path)
        message(FATAL_ERROR
            "gicc_target_plugin(${tgt}): no plugin available.\n"
            "  Pass -DGICC_BUILD_PLUGIN=ON to build it in-tree, or\n"
            "  -DGICC_OFI_PLUGIN_PATH=<path/to/gicc-clang-plugin.so> to use a pre-built one.")
    endif()

    # ---- Per-target sidecar directory (under the build tree) ---------------
    set(_sidecar_dir "${CMAKE_CURRENT_BINARY_DIR}/${tgt}.gicc")
    file(MAKE_DIRECTORY "${_sidecar_dir}")

    set(_stub_wrapper "${CMAKE_SOURCE_DIR}/cmake/gicc-stub-sidecar.sh")

    # ---- Pre-create stub sidecars at configure time ------------------------
    # The main TU is compiled with -include <sidecar>, so the file must
    # exist BEFORE the first compile. We create empty stubs here so the
    # very first `make` doesn't fail. Subsequent builds: the plugin
    # overwrites the stub from inside the main compile, and the next
    # `make` picks up the new content.
    get_target_property(_srcs ${tgt} SOURCES)
    foreach(_src IN LISTS _srcs)
        get_filename_component(_stem  "${_src}" NAME_WE)
        get_filename_component(_absrc "${_src}" ABSOLUTE)
        set(_sidecar_src "${_sidecar_dir}/${_stem}.gicc.cpp")
        if(NOT EXISTS "${_sidecar_src}")
            execute_process(COMMAND bash "${_stub_wrapper}"
                                    "${_sidecar_src}" "${_absrc}")
        endif()
    endforeach()

    # ---- Compile flags: enable plugin + force-include sidecar --------------
    target_compile_options(${tgt} PRIVATE
        -fplugin=${_plugin_path}
        -fplugin-arg-gicc-sidecar-dir=${_sidecar_dir})

    foreach(_src IN LISTS _srcs)
        get_filename_component(_stem "${_src}" NAME_WE)
        set(_sidecar_src "${_sidecar_dir}/${_stem}.gicc.cpp")
        # -include path is per-source: each TU gets its own sidecar.
        # Apply via source-file properties so multi-source targets stay
        # correctly bound.
        set_property(SOURCE "${_src}" APPEND PROPERTY
            COMPILE_OPTIONS "-include" "${_sidecar_src}")
        # OBJECT_DEPENDS makes make re-compile the source whenever the
        # sidecar changes (i.e. whenever a previous compile's plugin run
        # produced a new trace). Without this, the second `make` would
        # be a no-op because make only sees the user source as the input.
        set_property(SOURCE "${_src}" APPEND PROPERTY
            OBJECT_DEPENDS "${_sidecar_src}")
    endforeach()

    # ---- Plugin dependency: rebuild when plugin changes --------------------
    if(_plugin_dep)
        add_dependencies(${tgt} ${_plugin_dep})
    endif()
endfunction()
