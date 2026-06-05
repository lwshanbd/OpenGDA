#pragma once
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
namespace gicc::pass {
class GICCOmpHostDiscoveryPass
    : public llvm::PassInfoMixin<GICCOmpHostDiscoveryPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCOmpHostDiscoveryPass"; }
};
}  // namespace gicc::pass
