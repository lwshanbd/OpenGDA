#pragma once

#include "KernelInventory.h"

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

class GICCDeviceDiscoveryPass
    : public llvm::PassInfoMixin<GICCDeviceDiscoveryPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCDeviceDiscoveryPass"; }

    GICCKernelInventory Inventory;
};

}  // namespace gicc::pass
