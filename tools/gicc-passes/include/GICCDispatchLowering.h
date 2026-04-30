#pragma once

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

class GICCDispatchLoweringPass
    : public llvm::PassInfoMixin<GICCDispatchLoweringPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCDispatchLoweringPass"; }
};

}  // namespace gicc::pass
