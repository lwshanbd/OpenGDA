#include "SiteId.h"
#include "KernelInventory.h"

#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Path.h"

#include <sstream>

using namespace llvm;

namespace gicc::pass {

const char *opKindName(GICCOpKind k) {
    switch (k) {
        case GICCOpKind::PutNoDb: return "put_no_db";
        case GICCOpKind::GetNoDb: return "get_no_db";
        case GICCOpKind::Flush:   return "flush";
        case GICCOpKind::Quiet:   return "quiet";
    }
    return "?";
}

bool isGPUKernel(const Function &F) {
    auto cc = F.getCallingConv();
    return cc == CallingConv::AMDGPU_KERNEL || cc == CallingConv::PTX_Kernel;
}

bool classifyGICCCall(const CallInst &CI, GICCOpKind &out) {
    auto *F = CI.getCalledFunction();
    if (!F) return false;
    StringRef n = F->getName();
    // Itanium length-prefixes the unmangled identifier:
    //   put_no_db / get_no_db are 9 chars  → "9put_no_db" / "9get_no_db"
    //   flush / quiet      are 5 chars     → "5flushE"   / "5quietE"
    // The trailing E for flush/quiet rules out accidental matches against
    // longer identifiers that happen to start with "flush" / "quiet".
    if (n.contains("9put_no_db")) { out = GICCOpKind::PutNoDb; return true; }
    if (n.contains("9get_no_db")) { out = GICCOpKind::GetNoDb; return true; }
    if (n.contains("5flushE"))    { out = GICCOpKind::Flush;   return true; }
    if (n.contains("5quietE"))    { out = GICCOpKind::Quiet;   return true; }
    return false;
}

namespace {

// Strip Itanium "_ZN4gicc..." nested-name prefix and template suffix to
// recover the simple kernel identifier.
std::string simpleNameOf(StringRef mangled) {
    if (!mangled.starts_with("_Z")) return mangled.str();
    StringRef s = mangled.drop_front(2);
    if (s.starts_with("N")) s = s.drop_front();
    while (!s.empty() && (s.front() == 'K' || s.front() == 'V')) {
        s = s.drop_front();
    }
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

void collectGICCSites(Function &F, GICCKernelInfo &info) {
    info.kernel      = &F;
    info.mangledName = F.getName().str();
    info.simpleName  = simpleNameOf(F.getName());
    info.sites.clear();

    unsigned idx = 0;
    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            auto *CI = dyn_cast<CallInst>(&I);
            if (!CI) continue;
            GICCOpKind kind;
            if (!classifyGICCCall(*CI, kind)) continue;

            GICCCallSite s;
            s.CI                = CI;
            s.kind              = kind;
            s.callIndexInKernel = idx;
            s.siteId            = buildSiteId(CI, &F, idx);
            info.sites.push_back(s);
            ++idx;
        }
    }
}

std::string buildSiteId(const Instruction *I, const Function *kernel,
                        unsigned callIndex) {
    std::string tuBase = "?";
    std::string line   = "?";
    if (const auto loc = I->getDebugLoc()) {
        if (auto *scope = loc->getScope()) {
            if (auto *file = scope->getFile()) {
                tuBase = sys::path::filename(file->getFilename()).str();
            }
        }
        line = std::to_string(loc.getLine());
    }
    std::ostringstream os;
    os << tuBase << ':' << line << ':' << kernel->getName().str()
       << "::" << callIndex;
    return os.str();
}

}  // namespace gicc::pass
