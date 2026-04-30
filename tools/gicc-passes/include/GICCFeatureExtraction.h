#pragma once

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

class GICCFeatureExtractionPass
    : public llvm::PassInfoMixin<GICCFeatureExtractionPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCFeatureExtractionPass"; }
};

}  // namespace gicc::pass
