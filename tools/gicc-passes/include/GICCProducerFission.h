#pragma once

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

// Host materializer for the compiler-owned two-phase producer schedule. It
// runs before general inlining so every expanded callsite inherits the phase
// guard, while still auditing either the compiler HIP stub or a direct final
// launch. The automatic pipeline includes it only for an explicit transform
// request or the backward-compatible oracle switch.
class GICCProducerFissionHostPass
    : public llvm::PassInfoMixin<GICCProducerFissionHostPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() {
        return "GICCProducerFissionHostPass";
    }
};

// Early device-IR half of the same schedule. Named pipelines remain available
// for focused proof tests, but they still require either the explicit site
// transform or the research-oracle switch before changing IR.
class GICCProducerFissionDevicePass
    : public llvm::PassInfoMixin<GICCProducerFissionDevicePass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() {
        return "GICCProducerFissionDevicePass";
    }
};

// Host/device halves of the guarded early-trigger schedule. The host half
// runs at the pre-inliner LTO extension point so every later-expanded launch
// inherits its guard; named tests may also exercise the final direct-launch
// shape. Both halves keep a separate transform and device attestation.
class GICCGuardedEarlyTriggerHostPass
    : public llvm::PassInfoMixin<GICCGuardedEarlyTriggerHostPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() {
        return "GICCGuardedEarlyTriggerHostPass";
    }
};

class GICCGuardedEarlyTriggerDevicePass
    : public llvm::PassInfoMixin<GICCGuardedEarlyTriggerDevicePass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() {
        return "GICCGuardedEarlyTriggerDevicePass";
    }
};

}  // namespace gicc::pass
