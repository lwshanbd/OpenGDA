#pragma once
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
namespace gicc::pass {
// GICC_MODE=write-summary: for every offload kernel, what its worksharing
// loop writes -- 1-D ranges or collapsed boxes (extents and byte strides in
// terms of kernel values) -- and why a store could not be described.
class GICCWriteSummaryPass : public llvm::PassInfoMixin<GICCWriteSummaryPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCWriteSummaryPass"; }
};
}  // namespace gicc::pass
