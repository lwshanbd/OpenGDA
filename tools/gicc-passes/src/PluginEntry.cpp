#include "GICCDeviceDiscovery.h"
#include "GICCDeviceLowering.h"
#include "GICCDispatchLowering.h"
#include "GICCFeatureExtraction.h"
#include "GICCHKAnalysis.h"
#include "GICCHostDiscovery.h"
#include "GICCPassConfig.h"
#include "GICCTraceSynthesis.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace gicc::pass;

namespace {

struct GICCSentinelPass : PassInfoMixin<GICCSentinelPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
        const auto &cfg = getConfig();
        errs() << "[gicc-pass] mode=" << modeName(cfg.mode)
               << " target=" << targetName(cfg.target)
               << " triple=" << M.getTargetTriple()
               << " meta-dir=" << cfg.metaDir
               << " module=" << M.getName()
               << "\n";
        return PreservedAnalyses::all();
    }
    static StringRef name() { return "GICCSentinelPass"; }
};

}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "gicc-passes", "0.1",
        [](PassBuilder &PB) {
            // Auto-attach to the default optimizer pipeline so -fpass-plugin
            // builds get the analyses without naming each pass. Each pass
            // internally checks Mode (passthrough = no-op) and the module
            // triple (the wrong-side passes short-circuit harmlessly).
            //
            // PipelineEarlySimplification fires after the early function
            // simplification (mem2reg, SROA, EarlyCSE, simple instcombine)
            // but BEFORE the inliner. We need:
            //   - mem2reg done so kernel formals aren't hidden behind
            //     alloca/store/load pairs (HK analysis would otherwise
            //     reject every loaded operand as "device memory load").
            //   - inliner NOT yet run so gicc::put_no_db / get_no_db /
            //     flush / quiet calls are still callable functions.
            // PipelineStart is too early (mem2reg hasn't run yet);
            // OptimizerLast is too late (inliner has consumed the calls).
            PB.registerPipelineEarlySimplificationEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                    MPM.addPass(GICCDeviceDiscoveryPass());
                    MPM.addPass(GICCHKAnalysisPass());
                    // CRITICAL: order matters — DeviceLowering reads
                    // proxy_aware (set by DispatchLowering) from the
                    // kernel JSON. Within an LTO invocation that
                    // processes both device and host modules together,
                    // the host passes must run first so the JSON is
                    // up-to-date when DeviceLowering consumes it.
                    // (For separate-compilation flows where device and
                    // host are different LLVM modules / clang
                    // invocations, the JSON is only consistent on a
                    // SECOND rebuild — document this limitation in the
                    // build instructions.)
                    MPM.addPass(GICCHostDiscoveryPass());
                    MPM.addPass(GICCFeatureExtractionPass());
                    MPM.addPass(GICCTraceSynthesisPass());
                    MPM.addPass(GICCDispatchLoweringPass());
                    MPM.addPass(GICCDeviceLoweringPass());
                });
            // Sentinel stays at OptimizerLast — it's just a debug probe
            // and we want to see the post-optimization module triple.
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                    MPM.addPass(GICCSentinelPass());
                });
            // Named-pass registration so tests can drive the plugin via
            // opt -passes='gicc-sentinel' / 'gicc-device-discovery'.
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) -> bool {
                    if (Name == "gicc-sentinel") {
                        MPM.addPass(GICCSentinelPass());
                        return true;
                    }
                    if (Name == "gicc-device-discovery") {
                        MPM.addPass(GICCDeviceDiscoveryPass());
                        return true;
                    }
                    if (Name == "gicc-hk-analysis") {
                        MPM.addPass(GICCHKAnalysisPass());
                        return true;
                    }
                    if (Name == "gicc-device-lowering") {
                        MPM.addPass(GICCDeviceLoweringPass());
                        return true;
                    }
                    if (Name == "gicc-host-discovery") {
                        MPM.addPass(GICCHostDiscoveryPass());
                        return true;
                    }
                    if (Name == "gicc-feature-extraction") {
                        MPM.addPass(GICCFeatureExtractionPass());
                        return true;
                    }
                    if (Name == "gicc-trace-synthesis") {
                        MPM.addPass(GICCTraceSynthesisPass());
                        return true;
                    }
                    if (Name == "gicc-dispatch-lowering") {
                        MPM.addPass(GICCDispatchLoweringPass());
                        return true;
                    }
                    return false;
                });
        }};
}
