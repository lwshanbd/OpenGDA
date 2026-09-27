#include "GICCWriteSummary.h"
#include "GICCPassConfig.h"
#include "OmpKernel.h"
#include "WriteSet.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace gicc::pass {

namespace {

void summarizeKernel(Function &F, FunctionAnalysisManager &FAM) {
    raw_ostream &out = errs();
    const char *tag = "[gicc-write]   ";
    out << "[gicc-write] kernel " << F.getName() << "\n";
    std::string why;
    auto KI = OmpKernel::locate(F, why);
    auto W = KI ? collectWrites(*KI, FAM, UnknownStore::Record, why) : std::nullopt;
    if (!W) {
        out << tag << "not analysable: " << why << "\n";
        return;
    }
    auto list = [](auto &v) {
        std::string r = "[";
        for (size_t i = 0; i < v.size(); ++i) r += (i ? ", " : "") + scevStr(v[i]);
        return r + "]";
    };
    if (!W->preconditions.empty()) {
        out << tag << "when";
        for (const EntryFact &f : W->preconditions)
            out << " (" << scevStr(f.x) << (f.nonzeroOnly ? " != 0)" : " > 0)");
        out << "\n";
    }
    for (auto &b : W->boxes)
        out << tag << objName(b.obj) << ": " << b.stride.size() << "-D box"
            << " extent=" << list(b.extent) << " stride=" << list(b.stride)
            << " offset=" << scevStr(b.offset)
            << (b.unconditional ? "" : " (conditional)")
            << (b.trustedTrunc ? " (trusts loop-variable trunc)" : "") << "\n";
    for (auto &[St, reason] : W->unknown)
        out << tag << "store to " << *St->getPointerOperand()->getType()
            << " not analysable: " << reason << "\n";
}

}  // namespace

PreservedAnalyses GICCWriteSummaryPass::run(Module &M, ModuleAnalysisManager &MAM) {
    if (getConfig().mode != Mode::WriteSummary) return PreservedAnalyses::all();
    Triple T(M.getTargetTriple());
    if (!T.isNVPTX() && !T.isAMDGPU()) return PreservedAnalyses::all();
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    for (Function &F : M)
        if (isOffloadKernel(F)) summarizeKernel(F, FAM);
    return PreservedAnalyses::all();
}

}  // namespace gicc::pass
