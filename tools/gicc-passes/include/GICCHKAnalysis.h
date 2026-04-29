#pragma once

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

class GICCHKAnalysisPass
    : public llvm::PassInfoMixin<GICCHKAnalysisPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCHKAnalysisPass"; }

    // True if any GICC call site had a non-host-knowable argument; the
    // pass also prints structured diagnostics to errs() for each one.
    bool sawError = false;
};

}  // namespace gicc::pass
