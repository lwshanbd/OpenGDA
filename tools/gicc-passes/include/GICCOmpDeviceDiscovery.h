#pragma once
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
namespace gicc::pass {
class GICCOmpDeviceDiscoveryPass
    : public llvm::PassInfoMixin<GICCOmpDeviceDiscoveryPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCOmpDeviceDiscoveryPass"; }
};
}  // namespace gicc::pass
