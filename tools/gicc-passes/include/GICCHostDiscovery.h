#pragma once

#include "LaunchSiteInventory.h"

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

class GICCHostDiscoveryPass
    : public llvm::PassInfoMixin<GICCHostDiscoveryPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCHostDiscoveryPass"; }

    GICCLaunchInventory Inventory;
};

// Extract the kernel mangled name encoded as a non-type template argument
// in a `gicc::launch<&kernel>(...)` instantiation's mangled symbol. Returns
// the empty string if no kernel pointer NTTP is present (caller should
// skip the launch wrapper). Public to keep the parser unit-testable.
std::string extractKernelMangledName(llvm::StringRef launchWrapperMangled);

}  // namespace gicc::pass
