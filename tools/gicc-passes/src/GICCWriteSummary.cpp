#include "GICCWriteSummary.h"
#include "GICCPassConfig.h"
#include "OmpKernel.h"
#include "WriteSet.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
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
    auto &SK = FAM.getResult<ScalarEvolutionAnalysis>(F);
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    Type *I64 = Type::getInt64Ty(F.getContext());

    // Iteration count N = ub0 - lb0 + 1 of the distribute loop.
    const SCEV *N = nullptr;
    if (auto bounds = KI->distBounds(DT)) {
        const SCEV *lb0 = KI->normalize(SK, SK.getSCEV(bounds->first->getValueOperand()));
        const SCEV *ub0 = KI->normalize(SK, SK.getSCEV(bounds->second->getValueOperand()));
        if (lb0->isZero()) N = SK.getAddExpr(KI->extDist(SK, ub0, I64), SK.getOne(I64));
    }
    auto list = [](auto &v) {
        std::string r = "[";
        for (size_t i = 0; i < v.size(); ++i)
            r += (i ? ", " : "") + (v[i] ? scevStr(v[i]) : std::string("?"));
        return r + "]";
    };

    for (auto &[obj, dw] : W->dense)
        out << tag << objName(obj) << ": 1-D stride=" << scevStr(dw.c)
            << " offset=" << scevStr(dw.d) << " extent=[" << (N ? scevStr(N) : "?") << "]\n";
    for (auto &b : W->boxes) {
        SmallVector<const SCEV *, 4> ext, stride;
        for (size_t d = 0; d < b.box.stride.size(); ++d) {
            stride.push_back(KI->toKernel(b.box.stride[d], SK));
            ext.push_back(d ? KI->toKernel(b.box.extent[d], SK) : nullptr);
        }
        // n_1 = N / R_1 (a 1-D box has n_1 = N).
        if (N) {
            if (b.box.radix.empty()) ext[0] = N;
            else if (const SCEV *R1 = KI->toKernel(b.box.radix[0], SK))
                ext[0] = exactDivide(SK, N, SK.getNoopOrSignExtend(R1, I64));
        }
        const SCEV *off = KI->toKernel(b.box.offset, SK);
        out << tag << objName(b.obj) << ": " << b.box.stride.size() << "-D box"
            << " extent=" << list(ext) << " stride=" << list(stride)
            << " offset=" << (off ? scevStr(off) : std::string("?"))
            << (b.unconditional ? "" : " (conditional)")
            << (b.box.trustedTrunc ? " (trusts loop-variable trunc)" : "") << "\n";
    }
    for (auto &[St, reason] : W->unknown)
        out << tag << "store to " << *St->getPointerOperand()->getType()
            << " not analysable: " << reason << "\n";
}

}  // namespace

PreservedAnalyses GICCWriteSummaryPass::run(Module &M, ModuleAnalysisManager &MAM) {
    if (getConfig().mode != Mode::WriteSummary) return PreservedAnalyses::all();
    if (!Triple(M.getTargetTriple()).isGPU()) return PreservedAnalyses::all();
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    for (Function &F : M)
        if (isOffloadKernel(F)) summarizeKernel(F, FAM);
    return PreservedAnalyses::all();
}

}  // namespace gicc::pass
