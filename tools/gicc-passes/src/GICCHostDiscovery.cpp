#include "GICCHostDiscovery.h"
#include "GICCPassConfig.h"
#include "LaunchSiteInventory.h"
#include "MetadataIO.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace gicc::pass {

namespace {

// Walk the global llvm.global.annotations array; return the set of
// Function* whose annotation string equals "gicc.launch_site".
SmallVector<Function *, 4> findLaunchWrappers(Module &M) {
    SmallVector<Function *, 4> out;
    auto *gv = M.getNamedGlobal("llvm.global.annotations");
    if (!gv) return out;
    auto *arr = dyn_cast_or_null<ConstantArray>(gv->getInitializer());
    if (!arr) return out;

    for (Use &u : arr->operands()) {
        auto *entry = dyn_cast<ConstantStruct>(u.get());
        if (!entry || entry->getNumOperands() < 2) continue;

        // Operand 0: ptr to the annotated function (possibly through a bitcast).
        Value *fn = entry->getOperand(0)->stripPointerCasts();
        // Operand 1: ptr to a private constant string with the annotation.
        Value *str = entry->getOperand(1)->stripPointerCasts();
        auto *gvStr = dyn_cast<GlobalVariable>(str);
        if (!gvStr) continue;
        auto *cda = dyn_cast_or_null<ConstantDataArray>(gvStr->getInitializer());
        if (!cda || !cda->isString()) continue;

        StringRef ann = cda->getAsCString();
        if (ann != "gicc.launch_site") continue;

        if (auto *F = dyn_cast<Function>(fn))
            out.push_back(F);
    }
    return out;
}

}  // namespace

std::string extractKernelMangledName(StringRef launchWrapperMangled) {
    // Itanium-mangled gicc::launch<K, Args...> instantiations encode the
    // kernel template argument as `XadL_Z<kernel>E` (address-of-literal-of
    // function symbol). Locate that substring and read the function-name
    // segment, balancing inner N/I/J brackets so we stop at the L wrapper's
    // own E.
    size_t start = launchWrapperMangled.find("XadL_Z");
    if (start == StringRef::npos) return {};
    start += 4;  // skip "XadL", now at "_Z..."

    int    depth = 0;
    size_t i     = start;
    const size_t n = launchWrapperMangled.size();
    while (i < n) {
        char c = launchWrapperMangled[i];
        if (c == 'N' || c == 'I' || c == 'J') {
            ++depth;
        } else if (c == 'E') {
            if (depth == 0) break;
            --depth;
        }
        ++i;
    }
    if (i == n) return {};
    return launchWrapperMangled.substr(start, i - start).str();
}

PreservedAnalyses GICCHostDiscoveryPass::run(Module &M,
                                             ModuleAnalysisManager &) {
    const auto &cfg = getConfig();
    if (cfg.mode == Mode::Passthrough) return PreservedAnalyses::all();

    Inventory.sites.clear();
    Inventory.valid = false;

    auto wrappers = findLaunchWrappers(M);
    if (wrappers.empty()) {
        Inventory.valid = true;
        return PreservedAnalyses::all();
    }

    for (Function *W : wrappers) {
        std::string kernelMangled = extractKernelMangledName(W->getName());
        if (kernelMangled.empty()) {
            errs() << "[host-discovery] skip wrapper " << W->getName()
                   << " (no kernel NTTP)\n";
            continue;
        }

        unsigned siteCount = 0;
        for (User *U : W->users()) {
            auto *CI = dyn_cast<CallInst>(U);
            if (!CI || CI->getCalledFunction() != W) continue;

            GICCLaunchSite site;
            site.callsite      = CI;
            site.launchWrapper = W;
            site.kernelMangled = kernelMangled;

            KernelTemplate t;
            if (readKernelTemplate(cfg.metaDir, kernelMangled, t)) {
                site.kernelTemplate = std::move(t);
                site.haveTemplate   = true;
            } else {
                errs() << "[host-discovery] WARN: no template for "
                       << kernelMangled << " under " << cfg.metaDir << "\n";
            }
            Inventory.sites.push_back(std::move(site));
            ++siteCount;
        }

        // Recover the kernel's simple name from the loaded template if
        // available; otherwise fall back to the mangled name for logs.
        StringRef simple = !Inventory.sites.empty() &&
                           Inventory.sites.back().haveTemplate
            ? StringRef(Inventory.sites.back().kernelTemplate.simpleName)
            : StringRef(kernelMangled);
        errs() << "[host-discovery] found " << siteCount
               << " launch site(s) for kernel " << simple << "\n";
    }

    Inventory.valid = true;
    return PreservedAnalyses::all();
}

}  // namespace gicc::pass
