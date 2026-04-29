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
            // builds get the sentinel for free.
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                    MPM.addPass(GICCSentinelPass());
                });
            // Named-pass registration so tests can drive the plugin via
            // opt -passes='gicc-sentinel'.
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) -> bool {
                    if (Name == "gicc-sentinel") {
                        MPM.addPass(GICCSentinelPass());
                        return true;
                    }
                    return false;
                });
        }};
}
