#pragma once

#include "LaunchSiteInventory.h"

#include "llvm/IR/Module.h"
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

// Build a fresh GICCLaunchInventory by walking M's
// llvm.global.annotations and loading per-kernel meta JSONs from
// `metaDir`. Reusable from FeatureExtraction / TraceSynthesis so they
// don't need a separate analysis-result plumbing in v1.
GICCLaunchInventory collectLaunchInventory(llvm::Module &M,
                                           const std::string &metaDir);

}  // namespace gicc::pass
