#include "KernelInventory.h"
#include "SiteId.h"
#include "GICCPassConfig.h"
#include "GICCDeviceDiscovery.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace gicc::pass {

PreservedAnalyses GICCDeviceDiscoveryPass::run(Module &M, ModuleAnalysisManager &) {
    const auto &cfg = getConfig();
    if (cfg.mode == Mode::Passthrough) return PreservedAnalyses::all();

    Inventory.kernels.clear();
    Inventory.valid = false;

    for (Function &F : M) {
        if (F.isDeclaration() || !isGPUKernel(F)) continue;

        GICCKernelInfo info;
        collectGICCSites(F, info);
        if (info.sites.empty()) continue;

        errs() << "[discovery] kernel " << info.simpleName
               << " has " << info.sites.size() << " sites\n";
        for (const auto &s : info.sites) {
            errs() << "[discovery]   site_id=" << s.siteId
                   << " kind=" << opKindName(s.kind) << "\n";
        }
        Inventory.kernels.push_back(std::move(info));
    }
    Inventory.valid = true;
    return PreservedAnalyses::all();
}

}  // namespace gicc::pass
