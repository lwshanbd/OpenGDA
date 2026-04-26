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
 * v1 sidecar location: /tmp/<TU_basename>.gicc.cpp. Phase G will switch
 * to a CMake-aware path under the build directory once the build wrapper
 * is in place.
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
    explicit TraceEmitter(clang::CompilerInstance& CI) : CI_(CI) {}

    // Emit one specialization. Appends to or creates the sidecar file.
    // Returns the sidecar file path it wrote to.
    std::string emit(const KernelInfo& ki, const HKAnalysis& hk);

private:
    clang::CompilerInstance& CI_;
    // Track which sidecar files we've opened this run so we know whether
    // to truncate vs append (each TU gets one sidecar with possibly
    // multiple specializations stacked inside).
    std::set<std::string> opened_sidecars_;
};

} // namespace gicc_plugin
