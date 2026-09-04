#pragma once

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

// Final-host-IR materializer for the compiler-owned two-phase producer
// schedule. This pass is intentionally available only by its explicit
// pipeline name until the matching device-body materializer is complete.
class GICCProducerFissionHostPass
    : public llvm::PassInfoMixin<GICCProducerFissionHostPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() {
        return "GICCProducerFissionHostPass";
    }
};

}  // namespace gicc::pass
