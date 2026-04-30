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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace gicc::pass {

namespace {

// Walk the global llvm.global.annotations array; return the set of
// Function* whose annotation string equals "gicc.launch_site".
SmallVector<Function *, 4> findLaunchWrappers(Module &M) {
    SmallVector<Function *, 4> out;
    auto *gv = M.getNamedGlobal("llvm.global.annotations");
    if (!gv || !gv->hasInitializer()) return out;
    Constant *init = gv->getInitializer();

    for (unsigned i = 0; i < init->getNumOperands(); ++i) {
        auto *entry = dyn_cast_or_null<ConstantStruct>(
            init->getAggregateElement(i));
        if (!entry || entry->getNumOperands() < 2) continue;

        Value *fn   = entry->getOperand(0)->stripPointerCasts();
        Value *str  = entry->getOperand(1)->stripPointerCasts();
        auto  *gvS  = dyn_cast<GlobalVariable>(str);
        if (!gvS) continue;
        auto *cda = dyn_cast_or_null<ConstantDataArray>(gvS->getInitializer());
        if (!cda || !cda->isString()) continue;
        if (cda->getAsCString() != "gicc.launch_site") continue;
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

// Itanium NTTP encoding compresses already-seen substrings via S_, S0_,
// S1_... back-references. The kernel mangled name in the launch
// wrapper's NTTP therefore differs from the kernel's standalone mangled
// name (which Discovery used as the meta JSON filename). Both share the
// "_Z<len><simple-name>" prefix, so look up the meta file by that
// prefix and use the on-disk filename as the canonical mangled name.
static std::string resolveKernelMangledName(const std::string &metaDir,
                                            const std::string &extracted) {
    // Direct hit first — covers tests with hand-crafted names.
    {
        SmallString<256> p(metaDir);
        sys::path::append(p, extracted + ".json");
        if (sys::fs::exists(p)) return extracted;
    }

    // Compute the "_Z<len><simple>" prefix of the extracted name.
    StringRef s = extracted;
    if (!s.starts_with("_Z")) return extracted;
    StringRef body = s.drop_front(2);
    size_t i = 0;
    while (i < body.size() && body[i] >= '0' && body[i] <= '9') ++i;
    if (i == 0) return extracted;
    unsigned len = 0;
    body.substr(0, i).getAsInteger(10, len);
    if (len == 0 || (i + len) > body.size()) return extracted;
    StringRef prefix = s.take_front(2 + i + len);

    // Scan metaDir for a single file whose stem starts with the prefix.
    std::error_code ec;
    for (sys::fs::directory_iterator it(metaDir, ec), end;
         !ec && it != end; it.increment(ec)) {
        StringRef base = sys::path::filename(StringRef(it->path()));
        if (!base.ends_with(".json")) continue;
        StringRef stem = base.drop_back(5);
        if (stem.starts_with(prefix)) return stem.str();
    }
    return extracted;
}

GICCLaunchInventory collectLaunchInventory(Module &M,
                                           const std::string &metaDir) {
    GICCLaunchInventory inv;
    auto wrappers = findLaunchWrappers(M);

    for (Function *W : wrappers) {
        std::string extracted = extractKernelMangledName(W->getName());
        if (extracted.empty()) continue;
        std::string kernelMangled = resolveKernelMangledName(metaDir, extracted);

        for (User *U : W->users()) {
            // Catch both `call` (CallInst) and `invoke` (InvokeInst);
            // C++ exception support turns may-throw calls into invokes.
            auto *CI = dyn_cast<CallBase>(U);
            if (!CI || CI->getCalledFunction() != W) continue;

            GICCLaunchSite site;
            site.callsite      = CI;
            site.launchWrapper = W;
            site.kernelMangled = kernelMangled;

            KernelTemplate t;
            if (readKernelTemplate(metaDir, kernelMangled, t)) {
                site.kernelTemplate = std::move(t);
                site.haveTemplate   = true;
            }
            inv.sites.push_back(std::move(site));
        }
    }
    inv.valid = true;
    return inv;
}

PreservedAnalyses GICCHostDiscoveryPass::run(Module &M,
                                             ModuleAnalysisManager &) {
    const auto &cfg = getConfig();
    if (cfg.mode == Mode::Passthrough) return PreservedAnalyses::all();

    Inventory = collectLaunchInventory(M, cfg.metaDir);
    if (Inventory.sites.empty()) return PreservedAnalyses::all();

    // Group log lines by wrapper for readability.
    Function *cur = nullptr;
    unsigned   count = 0;
    StringRef  lastSimple;
    auto flush = [&] {
        if (!cur) return;
        errs() << "[host-discovery] found " << count
               << " launch site(s) for kernel " << lastSimple << "\n";
    };
    for (const auto &s : Inventory.sites) {
        if (s.launchWrapper != cur) {
            flush();
            cur   = s.launchWrapper;
            count = 0;
            lastSimple = s.haveTemplate ? StringRef(s.kernelTemplate.simpleName)
                                        : StringRef(s.kernelMangled);
            if (!s.haveTemplate) {
                errs() << "[host-discovery] WARN: no template for "
                       << s.kernelMangled << " under " << cfg.metaDir << "\n";
            }
        }
        ++count;
    }
    flush();
    return PreservedAnalyses::all();
}

}  // namespace gicc::pass
