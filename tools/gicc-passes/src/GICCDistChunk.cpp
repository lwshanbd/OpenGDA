#include "GICCDistChunk.h"
#include "GICCPassConfig.h"
#include "OmpKernel.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace gicc::pass {

namespace {

// Whether F, or anything defined here it calls or hands to a call (the
// outlined region __kmpc_parallel runs), asks which team or block it runs in.
bool asksTeam(const Function &F, SmallPtrSetImpl<const Function *> &seen) {
    if (!seen.insert(&F).second) return false;
    for (const Instruction &I : instructions(F)) {
        const auto *CB = dyn_cast<CallBase>(&I);
        if (!CB) continue;
        for (const Value *A : CB->args())
            if (const auto *H = dyn_cast<Function>(A->stripPointerCasts()))
                if (!H->isDeclaration() && asksTeam(*H, seen)) return true;
        const Function *G = CB->getCalledFunction();
        if (!G) {
            if (!CB->isInlineAsm()) return true;   // an indirect call: unknown
            continue;
        }
        StringRef n = G->getName();
        if (n == "omp_get_team_num" || n.starts_with("ompx_block_id") ||
            n.starts_with("llvm.nvvm.read.ptx.sreg.ctaid") ||
            n.starts_with("llvm.amdgcn.workgroup.id") ||
            n == "__kmpc_get_hardware_num_blocks")
            return true;
        if (!G->isDeclaration() && asksTeam(*G, seen)) return true;
    }
    return false;
}

// The chunk clang deals when the source names none: the team's thread count,
// a constant or the runtime's query.
bool defaultChunk(Value *chunk) {
    if (isa<ConstantInt>(chunk)) return true;
    auto *CB = dyn_cast<CallBase>(stripIntCasts(chunk));
    return CB && calleeName(*CB) == "__kmpc_get_hardware_num_threads_in_block";
}

}  // namespace

PreservedAnalyses GICCDistChunkPass::run(Module &M, ModuleAnalysisManager &) {
    const Config &cfg = getConfig();
    if (!cfg.distChunkSet || cfg.distChunk == 0) return PreservedAnalyses::all();
    const Triple T(M.getTargetTriple());
    if (!T.isNVPTX() && !T.isAMDGPU()) return PreservedAnalyses::all();
    bool changed = false;
    for (Function &K : M) {
        if (K.isDeclaration() || !isOffloadKernel(K)) continue;
        std::string why;
        auto KI = OmpKernel::locate(K, why);
        if (!KI || !KI->spmd()) continue;
        auto *sched = dyn_cast<ConstantInt>(KI->distInit->getArgOperand(kmpc::InitSched));
        if (!sched || sched->getSExtValue() != kmpc::SchedDistChunked) continue;
        Value *chunk = KI->distInit->getArgOperand(kmpc::InitChunk);
        if (!defaultChunk(chunk)) continue;
        SmallPtrSet<const Function *, 16> seen;
        if (asksTeam(K, seen)) {
            errs() << "[gicc-dist-chunk] " << K.getName() << ": asks its team, left alone\n";
            continue;
        }
        DominatorTree DT(K);
        auto bounds = KI->distBounds(DT);
        if (!bounds) continue;
        Type *I64 = Type::getInt64Ty(M.getContext());
        auto wide = [&](StoreInst *St) -> Value * {
            Value *V = St->getValueOperand();
            if (V->getType() == I64) return V;
            IRBuilder<> B(St->getNextNode());
            return KI->distSigned ? B.CreateSExt(V, I64) : B.CreateZExt(V, I64);
        };
        Value *lb0 = wide(bounds->first), *ub0 = wide(bounds->second);
        KI->widenDistChunk(lb0, ub0, cfg.distChunk);
        errs() << "[gicc-dist-chunk] " << K.getName() << ": chunk " << *chunk
               << " -> up to " << cfg.distChunk << "\n";
        changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
