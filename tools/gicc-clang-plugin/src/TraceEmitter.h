/**
 * TraceEmitter.h — phase 3 trace function generator for the gicc-clang-plugin.
 *
 * For each validated kernel K, generates the C++ source of a specialization
 *   template<> struct gicc::detail::kernel_trace<&K> {
 *       static void run(gicc::Runtime& rt, int peer,
 *                       dim3 grid, dim3 block,
 *                       <K's args minus DeviceCtx*>) {
 *           // lifted host-side put_no_db / get_no_db calls
 *       }
 *   };
 * and writes it to a sidecar source file. The build wrapper (Phase G)
 * compiles the sidecar and links it alongside the original TU.
 *
 * Sidecar location: <sidecar_dir>/<TU_basename>.gicc.cpp. The plugin
 * accepts `-fplugin-arg-gicc-sidecar-dir=<path>` to override the
 * directory; lit tests rely on the default `/tmp`.
 */
#pragma once

#include "HKAnalysis.h"
#include "KernelDiscovery.h"

#include "clang/Frontend/CompilerInstance.h"

#include <set>
#include <string>

namespace gicc_plugin {

class TraceEmitter {
public:
    TraceEmitter(clang::CompilerInstance& CI, std::string sidecar_dir)
        : CI_(CI), sidecar_dir_(std::move(sidecar_dir)) {}

    // Emit one specialization. Appends to or creates the sidecar file.
    // Returns the sidecar file path it wrote to.
    std::string emit(const KernelInfo& ki, const HKAnalysis& hk);

    // Force-create a sidecar for this TU even if no kernel was lifted.
    // Used by PluginAction so every TU built with -fplugin produces a
    // predictable sidecar file (CMake needs it as a known build artifact).
    // No-op if the sidecar already exists for this TU. Returns the path.
    std::string ensure_sidecar(const std::string& main_file);

private:
    clang::CompilerInstance& CI_;
    std::string sidecar_dir_;
    // Track which sidecar files we've opened this run so we know whether
    // to truncate vs append (each TU gets one sidecar with possibly
    // multiple specializations stacked inside).
    std::set<std::string> opened_sidecars_;
};

} // namespace gicc_plugin
