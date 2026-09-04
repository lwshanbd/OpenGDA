#pragma once

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

// Final-host-IR materializer for the compiler-owned two-phase producer
// schedule. The automatic pipeline includes it only for an explicit
// compiler-owned transform request or the backward-compatible oracle switch.
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

}  // namespace gicc::pass
