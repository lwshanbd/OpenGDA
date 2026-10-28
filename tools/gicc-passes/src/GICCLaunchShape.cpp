#include "GICCLaunchShape.h"
#include "GICCAfterPut.h"
#include "TuneConfig.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdlib>

using namespace llvm;

namespace gicc::pass {

namespace {

// __tgt_target_kernel(loc, device, num_teams, thread_limit, region, args),
// and in __tgt_kernel_arguments the first of NumTeams[3] and of
// ThreadLimit[3] (LLVM 19 to 22).
constexpr unsigned kOpTeams = 2, kOpThreads = 3, kOpArgs = 5;
constexpr uint64_t kArgsTeams = 72, kArgsThreads = 84;

bool zero(const Value *V) {
    auto *C = dyn_cast<ConstantInt>(V);
    return C && C->isZero();
}

}  // namespace

PreservedAnalyses GICCLaunchShapePass::run(Module &M, ModuleAnalysisManager &) {
    const Triple T(M.getTargetTriple());
    if (T.isNVPTX() || T.isAMDGPU()) return PreservedAnalyses::all();
    const char *sitesOut = std::getenv("GICC_SITES_OUT");
    if (!tune::anyLaunchShape() && !(sitesOut && *sitesOut)) return PreservedAnalyses::all();
    Function *launch = M.getFunction("__tgt_target_kernel");
    if (!launch) return PreservedAnalyses::all();
    SmallVector<CallInst *, 16> calls;
    for (User *U : launch->users())
        if (auto *CI = dyn_cast<CallInst>(U); CI && CI->getCalledFunction() == launch)
            calls.push_back(CI);
    Type *I32 = Type::getInt32Ty(M.getContext()), *I8 = Type::getInt8Ty(M.getContext());
    StringSet<> recorded;
    bool changed = false;
    for (CallInst *L : calls) {
        const std::string K = launchedKernelOf(L);
        if (K.empty()) continue;
        const bool teamsFree = zero(L->getArgOperand(kOpTeams));
        const bool threadsFree = zero(L->getArgOperand(kOpThreads));
        if (recorded.insert(K).second) {
            tune::record(json::Object{{"kind", "kernel"}, {"id", K}, {"option", "teams"},
                                      {"teams_fixed", !teamsFree}});
            tune::record(json::Object{{"kind", "kernel"}, {"id", K}, {"option", "threads"},
                                      {"threads_fixed", !threadsFree}});
        }
        IRBuilder<> B(L);
        auto set = [&](std::optional<uint32_t> want, bool free, unsigned op, uint64_t at,
                       const char *what) {
            if (!want) return;
            if (!free) {
                errs() << "[gicc-launch-shape] " << K << ": the source fixes its " << what
                       << ", left alone\n";
                return;
            }
            Value *N = ConstantInt::get(I32, *want);
            L->setArgOperand(op, N);
            B.CreateStore(N, B.CreateConstInBoundsGEP1_64(I8, L->getArgOperand(kOpArgs), at));
            changed = true;
        };
        set(tune::teams(K), teamsFree, kOpTeams, kArgsTeams, "team count");
        set(tune::threads(K), threadsFree, kOpThreads, kArgsThreads, "thread limit");
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
