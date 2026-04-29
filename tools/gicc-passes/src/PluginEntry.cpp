#include "GICCPassConfig.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct GICCSentinelPass : PassInfoMixin<GICCSentinelPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
        errs() << "[gicc-pass] ran on triple=" << M.getTargetTriple()
               << " module=" << M.getName() << "\n";
        return PreservedAnalyses::all();
    }
    static StringRef name() { return "GICCSentinelPass"; }
};

}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "gicc-passes", "0.1",
        [](PassBuilder &PB) {
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                    MPM.addPass(GICCSentinelPass());
                });
        }};
}
