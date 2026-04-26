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

#include <map>
#include <sstream>
#include <string>

namespace gicc_plugin {

class TraceEmitter {
public:
    TraceEmitter(clang::CompilerInstance& CI, std::string sidecar_dir)
        : CI_(CI), sidecar_dir_(std::move(sidecar_dir)) {}

    // Emit one specialization. Buffered; not written to disk until flush().
    // Returns the (eventual) sidecar file path.
    std::string emit(const KernelInfo& ki, const HKAnalysis& hk);

    // Ensure the buffer for this TU exists even if no kernel was lifted.
    // Called by PluginAction so every TU built with -fplugin produces a
    // predictable sidecar file (CMake needs it as a known build artifact).
    std::string ensure_sidecar(const std::string& main_file);

    // Write all buffered sidecar content to disk. To avoid bumping the
    // sidecar's mtime (which would force a redundant main-TU recompile),
    // a sidecar is only overwritten if its on-disk content differs from
    // the buffered content. Called at end-of-TU by PluginAction.
    void flush();

private:
    clang::CompilerInstance& CI_;
    std::string sidecar_dir_;
    // Per-sidecar buffer: path -> content. Built up across emit() and
    // ensure_sidecar() calls; written to disk in flush().
    std::map<std::string, std::ostringstream> buffers_;

    // Get-or-create buffer for the given sidecar path. Initializes the
    // buffer with the standard header on first access.
    std::ostringstream& buffer_for(const std::string& sidecar,
                                   const std::string& main_file);
};

} // namespace gicc_plugin
