#include "GICCDynamicPut.h"

#include "GICCAfterPut.h"
#include "GICCPassConfig.h"
#include "OmpKernel.h"
#include "TuneConfig.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include <cstdlib>
#include <string>

using namespace llvm;

namespace gicc::pass {
namespace {

constexpr StringLiteral kBegin = "ompx__dyn_begin";
constexpr StringLiteral kFlush = "ompx__dyn_flush";

raw_ostream &out() { return errs(); }

// Functions of the GICC and OpenMP device runtimes: never the kernel's own
// code, so neither instrumented nor followed.
bool isRuntime(StringRef n) {
    return n.starts_with("ompx_") || n.starts_with("__kmpc") || n.starts_with("__omp_rtl") ||
           n.starts_with("_ZN4ompx") || n.starts_with("_ZN4gicc") || n.starts_with("llvm.") ||
           n.starts_with("omp_") || n.starts_with("__nv_") || n.starts_with("__ocml_") ||
           n.starts_with("__llvm_");
}

// Runtime calls that write no memory the kernel's stores could be counted
// against, beyond what the trusted contract of isBenignCall covers.
bool benignRuntimeCall(const CallBase &CB) {
    if (isBenignCall(CB)) return true;
    StringRef n = calleeName(CB);
    return n.starts_with("__kmpc_barrier") || n == "__kmpc_alloc_shared" ||
           n == "__kmpc_free_shared" || n.starts_with("__kmpc_begin_sharing_variables") ||
           n == "__kmpc_end_sharing_variables" || n == "__kmpc_get_shared_variables" ||
           n == "__kmpc_syncwarp" || n == "__kmpc_flush" || n == kBegin || n == kFlush ||
           // Queries: the team, the thread, their counts.
           n.starts_with("omp_get_") || n.starts_with("omp_in_");
}

// Memory a store into which is never part of a put: a stack slot, or team
// memory.
bool privateStore(const Value *Ptr) {
    if (Ptr->getType()->getPointerAddressSpace() == kmpc::SharedAddrSpace) return true;
    const Value *obj = getUnderlyingObject(Ptr, 12);
    if (isa<AllocaInst>(obj)) return true;
    if (obj->getType()->isPointerTy() &&
        obj->getType()->getPointerAddressSpace() == kmpc::SharedAddrSpace)
        return true;
    if (auto *G = dyn_cast<GlobalVariable>(obj))
        return G->getAddressSpace() == kmpc::SharedAddrSpace;
    return false;
}

// K and every function defined here that it reaches by a call or by
// handing it to one (the outlined region a parallel call runs).
SetVector<Function *> reach(Function &K) {
    SetVector<Function *> fs;
    fs.insert(&K);
    for (unsigned n = 0; n < fs.size(); ++n)
        for (Instruction &I : instructions(*fs[n])) {
            auto *CB = dyn_cast<CallBase>(&I);
            if (!CB) continue;
            for (Value *op : CB->operands())
                if (auto *F = dyn_cast<Function>(op->stripPointerCasts());
                    F && !F->isDeclaration() && !isRuntime(F->getName()))
                    fs.insert(F);
        }
    return fs;
}

// Why K cannot be counted, or empty.
std::string refuse(const SetVector<Function *> &fs) {
    const SmallPtrSet<Function *, 8> own(fs.begin(), fs.end());
    for (Function *F : fs) {
        if (F != fs[0])
            for (User *U : F->users()) {
                auto *I = dyn_cast<Instruction>(U);
                if (!I || !own.count(I->getFunction()))
                    return "function " + F->getName().str() + " is shared with other code";
            }
        for (Instruction &I : instructions(*F)) {
            if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I)) {
                if (!privateStore(getLoadStorePointerOperand(&I) ? getLoadStorePointerOperand(&I)
                                                                 : I.getOperand(0)))
                    return "an atomic in " + F->getName().str();
                continue;
            }
            if (auto *S = dyn_cast<StoreInst>(&I)) {
                if (S->isAtomic() && !privateStore(S->getPointerOperand()))
                    return "an atomic store in " + F->getName().str();
                continue;
            }
            auto *CB = dyn_cast<CallBase>(&I);
            if (!CB || isa<DbgInfoIntrinsic>(CB)) continue;
            if (auto *MI = dyn_cast<AnyMemIntrinsic>(CB)) {
                if (!privateStore(MI->getRawDest()))
                    return "a memset or memcpy in " + F->getName().str();
                continue;
            }
            Function *callee = CB->getCalledFunction();
            if (callee && own.count(callee)) continue;
            if (benignRuntimeCall(*CB)) continue;
            return "a call to " + calleeName(*CB).str() + " in " + F->getName().str() +
                   " may write memory";
        }
    }
    return {};
}

// Where the kernel's user code starts: after __kmpc_target_init, on the
// branch every thread that runs user code takes.
Instruction *userStart(Function &K) {
    for (Instruction &I : instructions(K)) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || calleeName(*CI) != "__kmpc_target_init") continue;
        for (User *U : CI->users()) {
            auto *Cmp = dyn_cast<ICmpInst>(U);
            if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_EQ) continue;
            auto *M1 = dyn_cast<ConstantInt>(Cmp->getOperand(1));
            if (!M1 || !M1->isMinusOne()) continue;
            for (User *V : Cmp->users())
                if (auto *Br = dyn_cast<BranchInst>(V); Br && Br->isConditional())
                    return &*Br->getSuccessor(0)->getFirstInsertionPt();
        }
        return CI->getNextNode();
    }
    return nullptr;
}

// Counts F's stores: the thread keeps the piece it is storing into and the
// bytes stored there so far (cur, pend: promoted to registers at the end);
// a store into another piece, and a return, flush the count. Only the range
// check and an add stay on the path of a store into the current piece.
void countStores(Function &F, ArrayRef<StoreInst *> stores, Constant *team, Function *flush,
                 const DataLayout &DL) {
    LLVMContext &Ctx = F.getContext();
    Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx), *I8 = Type::getInt8Ty(Ctx);
    auto *Ptr = PointerType::get(Ctx, 0);
    IRBuilder<> E(&*F.getEntryBlock().getFirstInsertionPt());
    AllocaInst *cur = E.CreateAlloca(I32, nullptr, "gicc.dyn.cur");
    AllocaInst *pend = E.CreateAlloca(I64, nullptr, "gicc.dyn.pend");
    E.CreateStore(ConstantInt::get(I32, -1), cur);
    E.CreateStore(ConstantInt::get(I64, 0), pend);
    SmallVector<CallInst *, 8> flushes;
    auto flushIf = [&](Instruction *at) {
        IRBuilder<> B(at);
        Value *c = B.CreateLoad(I32, cur);
        Instruction *then = SplitBlockAndInsertIfThen(B.CreateICmpSGE(c, B.getInt32(0)), at, false);
        IRBuilder<> T(then);
        flushes.push_back(T.CreateCall(flush, {team, c, T.CreateLoad(I64, pend)}));
    };
    for (StoreInst *S : stores) {
        const uint64_t size = DL.getTypeStoreSize(S->getValueOperand()->getType());
        Instruction *next = S->getNextNode();
        IRBuilder<> B(next);
        Value *src = B.CreateLoad(Ptr, team, "gicc.dyn.src");
        Value *bytes = B.CreateLoad(I64, B.CreateConstGEP1_64(I8, team, 16), "gicc.dyn.bytes");
        Value *shift = B.CreateLoad(I32, B.CreateConstGEP1_64(I8, team, 44), "gicc.dyn.shift");
        Value *addr = B.CreatePtrToInt(
            B.CreatePointerBitCastOrAddrSpaceCast(S->getPointerOperand(), Ptr), I64);
        Value *off = B.CreateSub(addr, B.CreatePtrToInt(src, I64), "gicc.dyn.off");
        Instruction *in = SplitBlockAndInsertIfThen(B.CreateICmpULT(off, bytes), next, false);
        IRBuilder<> T(in);
        Value *p = T.CreateTrunc(T.CreateLShr(off, T.CreateZExt(shift, I64)), I32, "gicc.dyn.piece");
        Value *c = T.CreateLoad(I32, cur);
        Instruction *same = nullptr, *other = nullptr;
        SplitBlockAndInsertIfThenElse(T.CreateICmpEQ(p, c), in, &same, &other);
        IRBuilder<> A(same);
        A.CreateStore(A.CreateAdd(A.CreateLoad(I64, pend), ConstantInt::get(I64, size)), pend);
        flushIf(other);
        IRBuilder<> O(other);
        O.CreateStore(p, cur);
        O.CreateStore(ConstantInt::get(I64, size), pend);
    }
    SmallVector<ReturnInst *, 4> rets;
    for (BasicBlock &BB : F)
        if (auto *R = dyn_cast<ReturnInst>(BB.getTerminator())) rets.push_back(R);
    for (ReturnInst *R : rets) flushIf(R);
    DominatorTree DT(F);
    PromoteMemToReg({cur, pend}, DT);
    // Inline, on its cold branch: a call would add the posts' registers to
    // the kernel's (see ompx__dyn_flush).
    for (CallInst *C : flushes) {
        InlineFunctionInfo IFI;
        (void)InlineFunction(*C, IFI);
    }
}

}  // namespace

PreservedAnalyses GICCDynamicPutPass::run(Module &M, ModuleAnalysisManager &) {
    const Config &cfg = getConfig();
    if (cfg.mode != Mode::ChunkLower || !cfg.metaDirSet) return PreservedAnalyses::all();
    Triple T(M.getTargetTriple());
    if (!T.isNVPTX() && !T.isAMDGPU()) return PreservedAnalyses::all();
    const StringMap<AfterPutSite> sites = readAfterPutSites();
    if (sites.empty()) return PreservedAnalyses::all();
    const char *all = std::getenv("GICC_DYNAMIC");
    const bool everyKernel = all && *all && *all != '0';
    Function *begin = M.getFunction(kBegin), *flush = M.getFunction(kFlush);

    bool changed = false;
    for (Function &K : M) {
        if (!isOffloadKernel(K)) continue;
        auto it = sites.find(K.getName());
        if (it == sites.end()) continue;
        const bool wanted = tune::dynamic(K.getName()).value_or(everyKernel);
        // The report says whether the kernel could count (for the tuner);
        // only a kernel that is asked to does.
        json::Object rec{{"kind", "kernel"}, {"id", K.getName()}, {"option", "dynamic"}};
        auto keep = [&](const Twine &why) {
            rec["dynamicable"] = false;
            rec["why"] = why.str();
            tune::record(std::move(rec));
            if (wanted)
                out() << "[gicc-dyn] " << K.getName() << ": the put after it stays on the host: "
                      << why << "\n";
        };
        if (!begin || begin->isDeclaration() || !flush || flush->isDeclaration()) {
            keep("this unit does not include gicc/omp_pipeline.h");
            continue;
        }
        // A post the chunk lowering already reads goes from the kernel.
        SmallSet<int64_t, 4> lowered;
        for (Instruction &I : instructions(K))
            if (auto *CI = dyn_cast<CallInst>(&I); CI && calleeName(*CI) == "ompx__after_put")
                if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(1)))
                    lowered.insert(C->getSExtValue());
        // The first post nothing else sends; the team memory holds one.
        const AfterPutSite &site = it->second;
        int post = -1;
        for (unsigned i = 0; i < site.srcArgs.size(); ++i)
            if (!lowered.count(i)) {
                post = static_cast<int>(i);
                break;
            }
        if (post < 0) {
            keep("its puts after the launch already go from the kernel (pipelined sites)");
            continue;
        }
        const unsigned j = site.srcArgs[post];
        if (j + 1 >= K.arg_size() || !K.getArg(j + 1)->getType()->isPointerTy()) {
            keep("the host hands the kernel no pointer as argument " + std::to_string(j));
            continue;
        }
        const SetVector<Function *> fs = reach(K);
        if (std::string why = refuse(fs); !why.empty()) {
            keep(why);
            continue;
        }
        Instruction *start = userStart(K);
        if (!start) {
            keep("cannot find where its user code starts");
            continue;
        }
        rec["dynamicable"] = true;
        rec["signal"] = post < static_cast<int>(site.signals.size()) && site.signals[post];
        tune::record(std::move(rec));
        if (!wanted) continue;
        LLVMContext &Ctx = M.getContext();
        Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
        auto *Ptr = PointerType::get(Ctx, 0);
        // The post as the team reads it (ompx_dyn_post), in team memory.
        Type *slotTy = ArrayType::get(I64, 8);
        auto *slot = new GlobalVariable(M, slotTy, false, GlobalValue::InternalLinkage,
                                        PoisonValue::get(slotTy), "gicc.dyn.post", nullptr,
                                        GlobalValue::NotThreadLocal, kmpc::SharedAddrSpace);
        slot->setAlignment(Align(16));
        Constant *team = ConstantExpr::getAddrSpaceCast(slot, Ptr);
        IRBuilder<> B(start);
        B.CreateCall(begin, {ConstantInt::get(I64, afterPutHash(K.getName())),
                             ConstantInt::get(I32, post), ConstantInt::get(I32, j),
                             B.CreatePointerBitCastOrAddrSpaceCast(K.getArg(j + 1), Ptr), team});
        const DataLayout &DL = M.getDataLayout();
        // Debug: GICC_DYN_NOCOUNT=1 reads the post but counts nothing.
        const char *nc = std::getenv("GICC_DYN_NOCOUNT");
        const bool noCount = nc && *nc && *nc != '0';
        unsigned counted = 0;
        for (Function *F : fs) {
            SmallVector<StoreInst *, 16> stores;
            for (Instruction &I : instructions(*F))
                if (auto *S = dyn_cast<StoreInst>(&I); S && !privateStore(S->getPointerOperand()))
                    stores.push_back(S);
            if (stores.empty() || noCount) continue;
            counted += stores.size();
            countStores(*F, stores, team, flush, DL);
        }
        out() << "[gicc-dyn] " << K.getName() << ": put " << post
              << (post < static_cast<int>(site.signals.size()) && site.signals[post]
                      ? " (ompx_put_signal)"
                      : "")
              << " goes from the kernel by run-time counting, " << counted
              << " store(s) counted in " << fs.size() << " function(s)\n";
        changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
