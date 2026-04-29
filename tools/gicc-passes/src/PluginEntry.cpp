#include "GICCDeviceDiscovery.h"
#include "GICCDeviceLowering.h"
#include "GICCHKAnalysis.h"
#include "GICCPassConfig.h"

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
            // builds get the device-side analyses without naming each pass.
            // Each pass internally checks Mode (passthrough = no-op) and the
            // module triple (host modules are short-circuited).
            //
            // The device-side passes register at PipelineStart so they
            // observe the original gicc:: API calls BEFORE the inliner
            // expands them into the legacy device-side IPC ring writes.
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                    MPM.addPass(GICCDeviceDiscoveryPass());
                    MPM.addPass(GICCHKAnalysisPass());
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
                    return false;
                });
        }};
}
