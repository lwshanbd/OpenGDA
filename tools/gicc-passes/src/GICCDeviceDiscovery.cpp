#include "KernelInventory.h"
#include "SiteId.h"
#include "GICCPassConfig.h"

#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include "GICCDeviceDiscovery.h"

using namespace llvm;

namespace gicc::pass {

namespace {

bool isKernel(const Function &F) {
    auto cc = F.getCallingConv();
    return cc == CallingConv::AMDGPU_KERNEL || cc == CallingConv::PTX_Kernel;
}

// Mangled-name substring match. The Itanium ABI encodes
//   gicc::put_no_db   → "_ZN4gicc10put_no_db..."
//   gicc::get_no_db   → "_ZN4gicc10get_no_db..."
//   gicc::flush       → "_ZN4gicc5flushE..."
//   gicc::quiet       → "_ZN4gicc5quietE..."
// Substrings unique to gicc:: identifiers — won't collide with user code.
bool isGICCCall(const CallInst &CI, GICCOpKind &out) {
    auto *F = CI.getCalledFunction();
    if (!F) return false;
    StringRef n = F->getName();
    if (n.contains("10put_no_db")) { out = GICCOpKind::PutNoDb; return true; }
    if (n.contains("10get_no_db")) { out = GICCOpKind::GetNoDb; return true; }
    if (n.contains("5flushE"))     { out = GICCOpKind::Flush;   return true; }
    if (n.contains("5quietE"))     { out = GICCOpKind::Quiet;   return true; }
    return false;
}

// Strip Itanium "_ZN4gicc..." nested-name prefix and template suffix to
// recover the simple kernel identifier. For un-mangled or unrecognized
// names, return the input unchanged.
std::string simpleNameOf(StringRef mangled) {
    // Cheap heuristic: take the last "<digits><identifier>" group.
    // Good enough for kernels named like _Z11halo_kernel...; demangling
    // upgrades land in Phase 2.
    if (!mangled.starts_with("_Z")) return mangled.str();
    StringRef s = mangled.drop_front(2);  // past "_Z"
    if (s.starts_with("N")) s = s.drop_front();  // nested name
    while (!s.empty() && (s.front() == 'K' || s.front() == 'V')) {
        s = s.drop_front();  // CV-quals
    }
    // Read length prefix.
    size_t i = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    if (i == 0) return mangled.str();
    unsigned len = 0;
    s.substr(0, i).getAsInteger(10, len);
    s = s.drop_front(i);
    if (len == 0 || len > s.size()) return mangled.str();
    return s.substr(0, len).str();
}

}  // namespace

PreservedAnalyses GICCDeviceDiscoveryPass::run(Module &M, ModuleAnalysisManager &) {
    const auto &cfg = getConfig();
    if (cfg.mode == Mode::Passthrough) return PreservedAnalyses::all();

    Inventory.kernels.clear();
    Inventory.valid = false;

    for (Function &F : M) {
        if (F.isDeclaration() || !isKernel(F)) continue;

        GICCKernelInfo info;
        info.kernel      = &F;
        info.mangledName = F.getName().str();
        info.simpleName  = simpleNameOf(F.getName());

        unsigned idx = 0;
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                auto *CI = dyn_cast<CallInst>(&I);
                if (!CI) continue;
                GICCOpKind kind;
                if (!isGICCCall(*CI, kind)) continue;

                GICCCallSite site;
                site.CI                  = CI;
                site.kind                = kind;
                site.callIndexInKernel   = idx;
                site.siteId              = buildSiteId(CI, &F, idx);
                info.sites.push_back(site);
                ++idx;
            }
        }

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
