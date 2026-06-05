#include "GICCDeviceDiscovery.h"
#include "GICCFeatureExtraction.h"
#include "GICCHKAnalysis.h"
#include "GICCHostDiscovery.h"
#include "GICCOmpDeviceDiscovery.h"
#include "GICCOmpHostDiscovery.h"
#include "GICCPassConfig.h"
#include "GICCTraceSynthesis.h"
#ifndef GICC_PASSES_ANALYZE_ONLY
#  include "GICCDeviceLowering.h"
#  include "GICCDispatchLowering.h"
#endif

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace gicc::pass;

// PassBuilder EP-callback signature changed in LLVM 20: a third
// `ThinOrFullLTOPhase` parameter was added. Provide a single macro for
// the lambda prologue so both the early-simplification and
// optimizer-last callbacks stay one-line-per-callsite on both versions.
#if LLVM_VERSION_MAJOR >= 20
#  define GICC_EP_LAMBDA_HEAD(MPM_NAME) \
       [](ModulePassManager &MPM_NAME, OptimizationLevel, ThinOrFullLTOPhase)
#else
#  define GICC_EP_LAMBDA_HEAD(MPM_NAME) \
       [](ModulePassManager &MPM_NAME, OptimizationLevel)
#endif

namespace {

struct GICCSentinelPass : PassInfoMixin<GICCSentinelPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
        const auto &cfg = getConfig();
        // Module::getTargetTriple() returns std::string in LLVM ≤19 and
        // Triple in LLVM ≥20. Wrap-then-.str() builds on both: in ≤19
        // it uses Triple's Twine ctor, in ≥20 it uses the copy ctor.
        errs() << "[gicc-pass] mode=" << modeName(cfg.mode)
               << " target=" << targetName(cfg.target)
               << " triple=" << Triple(M.getTargetTriple()).str()
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
                GICC_EP_LAMBDA_HEAD(MPM) {
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
#ifndef GICC_PASSES_ANALYZE_ONLY
                    MPM.addPass(GICCDispatchLoweringPass());
                    MPM.addPass(GICCDeviceLoweringPass());
#endif
                });
            // Sentinel stays at OptimizerLast — it's just a debug probe
            // and we want to see the post-optimization module triple.
            PB.registerOptimizerLastEPCallback(
                GICC_EP_LAMBDA_HEAD(MPM) {
                    MPM.addPass(GICCSentinelPass());
                });
            // Phase 2 (omp-dwq): the OpenMP markers survive only AFTER the
            // inliner has run, so the omp passes attach at OptimizerLast.
            // Each pass self-gates on Mode::OmpDwq, so this block is inert
            // in every other mode (Discover / Lower / Passthrough).
            PB.registerOptimizerLastEPCallback(
                GICC_EP_LAMBDA_HEAD(MPM) {
                    MPM.addPass(GICCOmpDeviceDiscoveryPass());
                    MPM.addPass(GICCOmpHostDiscoveryPass());
                    MPM.addPass(GICCTraceSynthesisPass());
#ifndef GICC_PASSES_ANALYZE_ONLY
                    MPM.addPass(GICCDispatchLoweringPass());
                    MPM.addPass(GICCDeviceLoweringPass());
#endif
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
                    if (Name == "gicc-omp-device-discovery") {
                        MPM.addPass(GICCOmpDeviceDiscoveryPass());
                        return true;
                    }
                    if (Name == "gicc-omp-host-discovery") {
                        MPM.addPass(GICCOmpHostDiscoveryPass());
                        return true;
                    }
#ifndef GICC_PASSES_ANALYZE_ONLY
                    if (Name == "gicc-device-lowering") {
                        MPM.addPass(GICCDeviceLoweringPass());
                        return true;
                    }
                    if (Name == "gicc-dispatch-lowering") {
                        MPM.addPass(GICCDispatchLoweringPass());
                        return true;
                    }
#endif
                    return false;
                });
        }};
}
