#include "GICCDistChunk.h"
#include "TuneConfig.h"

#include <cstdlib>
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
    const char *sitesOut = std::getenv("GICC_SITES_OUT");
    const bool reporting = sitesOut && *sitesOut;
    if (!cfg.distChunkSet && !tune::anyDistChunk() && !reporting) return PreservedAnalyses::all();
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
        if (asksTeam(K)) {
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
        // The chunk the tuning file names for this kernel, else
        // GICC_DIST_CHUNK's; 0 leaves the kernel as clang made it.
        const uint64_t want =
            tune::distChunk(K.getName()).value_or(cfg.distChunkSet ? cfg.distChunk : 0);
        json::Object rec{{"kind", "kernel"}, {"id", K.getName()}, {"option", "dist_chunk"}};
        if (want) rec["chosen"] = static_cast<int64_t>(want);
        tune::record(std::move(rec));
        if (want == 0) continue;
        Value *lb0 = wide(bounds->first), *ub0 = wide(bounds->second);
        KI->widenDistChunk(lb0, ub0, want);
        errs() << "[gicc-dist-chunk] " << K.getName() << ": chunk " << *chunk
               << " -> up to " << want << "\n";
        changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
