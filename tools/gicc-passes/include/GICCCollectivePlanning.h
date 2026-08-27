#pragma once

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

// Discover compiler-declared collective algorithm catalogs and materialize a
// strictly compiler-generated plan.  The external decision contains opaque
// candidate IDs; this pass repeats catalog, ABI, family, contract, threshold,
// and content-ID checks before changing any call.
class GICCCollectivePlanningPass
    : public llvm::PassInfoMixin<GICCCollectivePlanningPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &MAM);
    static llvm::StringRef name() { return "GICCCollectivePlanningPass"; }
};

}  // namespace gicc::pass
