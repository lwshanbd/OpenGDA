#pragma once

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

class GICCHKAnalysisPass
    : public llvm::PassInfoMixin<GICCHKAnalysisPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCHKAnalysisPass"; }
};

}  // namespace gicc::pass
