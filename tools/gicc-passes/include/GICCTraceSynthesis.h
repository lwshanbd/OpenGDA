#pragma once

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

class GICCTraceSynthesisPass
    : public llvm::PassInfoMixin<GICCTraceSynthesisPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCTraceSynthesisPass"; }
};

}  // namespace gicc::pass
