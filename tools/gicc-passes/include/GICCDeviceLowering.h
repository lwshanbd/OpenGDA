#pragma once

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

class GICCDeviceLoweringPass
    : public llvm::PassInfoMixin<GICCDeviceLoweringPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCDeviceLoweringPass"; }
};

}  // namespace gicc::pass
