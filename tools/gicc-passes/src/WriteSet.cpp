#include "WriteSet.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"

#include <functional>

using namespace llvm;

namespace gicc::pass {

namespace {

// The slot a (possibly cast) SCEVUnknown load reads, or null.
const Value *loadedSlot(const SCEV *S) {
    if (auto *C = dyn_cast<SCEVCastExpr>(S)) S = C->getOperand();
    auto *U = dyn_cast<SCEVUnknown>(S);
    if (!U) return nullptr;
    auto *LI = dyn_cast<LoadInst>(U->getValue());
    return LI ? LI->getPointerOperand()->stripPointerCasts() : nullptr;
}

// Split S = c * ext(load slot) into (c, ext(load slot)); c = 1 when no mul.
bool splitScaledLoad(const SCEV *S, const Value *slot, const SCEV *&c,
                     const SCEV *&term, ScalarEvolution &SE) {
    if (auto *M = dyn_cast<SCEVMulExpr>(S)) {
        if (M->getNumOperands() == 2 && isa<SCEVConstant>(M->getOperand(0)) &&
            loadedSlot(M->getOperand(1)) == slot) {
            c = M->getOperand(0);
            term = M->getOperand(1);
            return true;
        }
        return false;
    }
    if (loadedSlot(S) == slot) {
        c = SE.getOne(S->getType());
        term = S;
        return true;
    }
    return false;
}

class Collector {
    const OmpKernel &KI;
    FunctionAnalysisManager &FAM;
    UnknownStore policy;
    std::string &why;
    Function &O;
    ScalarEvolution &SE;
    LoopInfo &LI;
    const Value *lbSlot = nullptr, *ubSlot = nullptr, *strideSlot = nullptr;
    std::optional<AccessDecomposer> decomposer;
    WriteSet W;

public:
    Collector(const OmpKernel &KI, FunctionAnalysisManager &FAM, UnknownStore policy,
              std::string &why)
        : KI(KI), FAM(FAM), policy(policy), why(why), O(*KI.outlined),
          SE(FAM.getResult<ScalarEvolutionAnalysis>(O)),
          LI(FAM.getResult<LoopAnalysis>(O)) {}

    std::optional<WriteSet> run() {
        if (!findThreadLoop()) return std::nullopt;
        for (Instruction &I : instructions(O)) {
            if (auto *CB = dyn_cast<CallBase>(&I)) {
                if (!isBenignCall(*CB)) {
                    why = calleeName(*CB).empty()
                              ? std::string("indirect call in the parallel region")
                              : ("call to '" + calleeName(*CB) +
                                 "' in the parallel region may write memory").str();
                    return std::nullopt;
                }
                continue;
            }
            if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I)) {
                why = "atomic write in the parallel region";
                return std::nullopt;
            }
            auto *St = dyn_cast<StoreInst>(&I);
            if (!St || isa<AllocaInst>(getUnderlyingObject(St->getPointerOperand())))
                continue;   // thread-private
            if (!store(St)) return std::nullopt;
        }
        return std::move(W);
    }

private:
    // __kmpc_for_static_init, and the thread range starting from the block
    // bounds the kernel hands in.
    bool findThreadLoop() {
        CallInst *forInit = nullptr;
        for (Instruction &I : instructions(O))
            if (isCallTo(I, "__kmpc_for_static_init")) {
                if (forInit) {
                    why = "more than one worksharing loop in the parallel region";
                    return false;
                }
                forInit = cast<CallInst>(&I);
            }
        if (!forInit) { why = "parallel region has no worksharing loop"; return false; }
        lbSlot     = forInit->getArgOperand(kmpc::InitLower)->stripPointerCasts();
        ubSlot     = forInit->getArgOperand(kmpc::InitUpper)->stripPointerCasts();
        strideSlot = forInit->getArgOperand(kmpc::InitStride)->stripPointerCasts();
        auto storedFromFormal = [&](const Value *slot, unsigned argNo) {
            StoreInst *SI = uniqueStoreTo(O, slot);
            if (!SI) return false;
            Value *v = SI->getValueOperand();
            if (auto *CI = dyn_cast<CastInst>(v)) v = CI->getOperand(0);
            return v == O.getArg(argNo);
        };
        if (O.arg_size() <= kmpc::OutlinedUB ||
            !storedFromFormal(lbSlot, kmpc::OutlinedLB) ||
            !storedFromFormal(ubSlot, kmpc::OutlinedUB)) {
            why = "worksharing loop bounds are not the distribute block bounds";
            return false;
        }
        return true;
    }

    // A store the analysis cannot describe: listed under Record (go on),
    // fatal under Fail.
    bool unknown(StoreInst *St, const std::string &reason) {
        if (policy == UnknownStore::Record) { W.unknown.push_back({St, reason}); return true; }
        why = reason;
        return false;
    }

    bool store(StoreInst *St) {
        Value *ptr = St->getPointerOperand();
        Loop *inner = LI.getLoopFor(St->getParent());
        if (!inner) return unknown(St, "store outside the worksharing loop body");
        // The worksharing loop is the outermost one.
        Loop *L = inner;
        while (L->getParentLoop()) L = L->getParentLoop();
        const SCEVAddRecExpr *iv = loopCoversBlock(L);
        if (!iv) return false;
        if (inner != L) {
            // Inside a sequential loop nested in the worksharing loop (a
            // per-element compute loop): fine if the address does not move
            // with it, as it then depends on the worksharing IV alone.
            // Conditional: the inner loop may not run.
            for (Loop *l = inner; l != L; l = l->getParentLoop())
                if (!SE.isLoopInvariant(SE.getSCEV(ptr), l))
                    return unknown(St, "store address moves with a sequential loop "
                                       "nested in the worksharing loop");
            return box(St, L, iv, /*inInnerLoop=*/true);
        }
        auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(ptr));
        if (!AR || AR->getLoop() != L || !AR->isAffine())
            // Not base + c*i: try the multi-dimensional (collapse) form.
            return box(St, L, iv, /*inInnerLoop=*/false);

        // addr = base + c*iv + d, with the same IV the exit tests.
        const SCEV *c = nullptr, *stepTerm = nullptr;
        if (!splitScaledLoad(AR->getStepRecurrence(SE), strideSlot, c, stepTerm, SE) ||
            stepTerm != iv->getStepRecurrence(SE))
            return unknown(St, "store address does not advance with the loop IV");
        auto *base = dyn_cast<SCEVUnknown>(SE.getPointerBase(AR->getStart()));
        if (!base) return unknown(St, "store has no single base pointer");
        const SCEV *d = SE.getMinusSCEV(SE.getMinusSCEV(AR, SE.getMulExpr(c, iv)), base);
        if (!isa<SCEVConstant>(d))
            return unknown(St, "store offset is not constant relative to base + c*i");
        Value *obj = KI.resolveOutlined(base->getValue());
        if (!obj || !isa<Argument>(obj)) return unknown(St, "store base is not a kernel argument");
        DenseWrite dw{c, d, KI.DL.getTypeStoreSize(St->getValueOperand()->getType())};
        auto [it, fresh] = W.dense.try_emplace(obj, dw);
        if (!fresh && (it->second.c != c || it->second.d != d ||
                       it->second.storeSize != dw.storeSize))
            return unknown(St, "object '" + objName(obj) +
                               "' is written with more than one access pattern");
        W.storesTo[obj].push_back(St);
        return true;
    }

    // A store whose address is a box over the collapsed loop's digits.
    bool box(StoreInst *St, Loop *L, const SCEVAddRecExpr *iv, bool inInnerLoop) {
        auto fail = [&](const std::string &reason) {
            return unknown(St, "store address: " + reason);
        };
        // The phi may be narrower than the IV the exit test sees (an i32
        // loop compared through a sext); match it widened.
        PHINode *ivPhi = nullptr;
        for (PHINode &PN : L->getHeader()->phis())
            if (SE.isSCEVable(PN.getType()) &&
                PN.getType()->getScalarSizeInBits() <= iv->getType()->getScalarSizeInBits() &&
                SE.getNoopOrSignExtend(SE.getSCEV(&PN), iv->getType()) == iv)
                ivPhi = &PN;
        if (!ivPhi) return fail("the loop IV is not a header phi");
        if (!decomposer) decomposer.emplace(SE, KI.DL, L, ivPhi);
        std::string reason;
        auto b = decomposer->decompose(St->getPointerOperand(), reason);
        if (!b) return fail(reason);
        for (auto &[R, R2] : b->assumedEqual)
            if (!provenEqual(L, R, R2))
                return fail("collapse radix " + scevStr(R) + " and " + scevStr(R2) +
                            " differ only by a cast, but they cannot be proven equal");
        Value *obj = KI.resolveOutlined(b->base);
        if (!obj || !isa<Argument>(obj)) return fail("store base is not a kernel argument");
        auto &DT = FAM.getResult<DominatorTreeAnalysis>(O);
        BasicBlock *latch = L->getLoopLatch();
        const bool always = !inInnerLoop && latch && DT.dominates(St->getParent(), latch);
        W.boxes.push_back({St, obj, *b, always});
        W.storesTo[obj].push_back(St);
        return true;
    }

    // The loop must run exactly the thread's share of the block: its IV
    // starts at the thread lower bound, steps by the thread stride, the
    // entry is guarded by lb <= UB and the latch continues while
    // i.next <= UB. Returns the IV of the current iteration, or null.
    const SCEVAddRecExpr *loopCoversBlock(Loop *L) {
        auto isUB = [&](Value *v) {
            if (auto *CI = dyn_cast<CastInst>(v)) v = CI->getOperand(0);
            if (v == O.getArg(kmpc::OutlinedUB)) return true;
            auto *LdI = dyn_cast<LoadInst>(v);
            return LdI && LdI->getPointerOperand()->stripPointerCasts() == ubSlot;
        };
        BasicBlock *latch = L->getLoopLatch();
        auto *BI = latch ? dyn_cast<BranchInst>(latch->getTerminator()) : nullptr;
        auto *Cmp = BI && BI->isConditional() ? dyn_cast<ICmpInst>(BI->getCondition()) : nullptr;
        if (!Cmp) { why = "loop latch is not a compare-and-branch"; return nullptr; }

        // The bound is UB itself, or UB + 1 (64-bit loops are emitted as
        // "continue while i.next < UB + 1").
        auto isBound = [&](Value *v, bool &plusOne) {
            plusOne = false;
            if (auto *BO = dyn_cast<BinaryOperator>(v); BO && BO->getOpcode() == Instruction::Add)
                if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1)); C && C->isOne()) {
                    plusOne = true;
                    v = BO->getOperand(0);
                }
            return isUB(v);
        };
        ICmpInst::Predicate P = Cmp->getPredicate();
        if (BI->getSuccessor(0) != L->getHeader()) P = ICmpInst::getInversePredicate(P);
        Value *ivSide = Cmp->getOperand(0), *bSide = Cmp->getOperand(1);
        bool plusOne = false;
        if (isBound(ivSide, plusOne)) { std::swap(ivSide, bSide); P = ICmpInst::getSwappedPredicate(P); }
        if (!isBound(bSide, plusOne)) {
            why = "loop exit does not compare against the block upper bound";
            return nullptr;
        }
        const bool inclusive = P == ICmpInst::ICMP_SLE || P == ICmpInst::ICMP_ULE;
        const bool strict    = P == ICmpInst::ICMP_SLT || P == ICmpInst::ICMP_ULT;
        if (!(plusOne ? strict : inclusive)) {
            why = "loop exit is not 'continue while i <= UB'";
            return nullptr;
        }
        W.ubPlusOne |= plusOne;
        // The compared value must be the IV of the next iteration: the value
        // a header phi takes on the back edge (true by construction, even
        // where SCEV cannot fold the casts on it), or an add recurrence SCEV
        // proves to be one step ahead.
        const SCEVAddRecExpr *cur = nullptr;
        for (PHINode &PN : L->getHeader()->phis()) {
            if (PN.getIncomingValueForBlock(latch) != ivSide) continue;
            auto *ar = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(&PN));
            if (ar && ar->getLoop() == L && ar->isAffine()) cur = ar;
        }
        if (!cur) {
            auto *next = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(ivSide));
            if (next && next->getLoop() == L && next->isAffine()) {
                const SCEV *s = next->getStepRecurrence(SE);
                cur = dyn_cast<SCEVAddRecExpr>(SE.getAddRecExpr(
                    SE.getMinusSCEV(next->getStart(), s), s, L, SCEV::FlagAnyWrap));
            }
        }
        if (!cur) { why = "loop exit compare is not on the IV"; return nullptr; }
        if (loadedSlot(cur->getStart()) != lbSlot ||
            loadedSlot(cur->getStepRecurrence(SE)) != strideSlot) {
            why = "loop IV is not the thread's lb stepped by its stride";
            return nullptr;
        }
        const SCEV *ub = SE.getSCEV(bSide);
        if (!SE.isLoopEntryGuardedByCond(L, P, cur->getStart(),
                                         SE.getNoopOrSignExtend(ub, cur->getType()))) {
            why = "first iteration is not guarded by lb <= UB";
            return nullptr;
        }
        return cur;
    }

    // Values x with x > 0 on entry to L: the signed icmp x > 0 (or x >= 1)
    // conditions of branches whose taken edge dominates the header.
    SmallVector<const SCEV *, 4> positiveOnEntry(Loop *L) {
        SmallVector<const SCEV *, 4> facts;
        auto &DT = FAM.getResult<DominatorTreeAnalysis>(O);
        std::function<void(Value *, bool)> collect = [&](Value *c, bool holds) {
            if (auto *BO = dyn_cast<BinaryOperator>(c)) {
                // a & b true, or a | b false: both halves hold (negated).
                if ((BO->getOpcode() == Instruction::And && holds) ||
                    (BO->getOpcode() == Instruction::Or && !holds)) {
                    collect(BO->getOperand(0), holds);
                    collect(BO->getOperand(1), holds);
                }
                return;
            }
            auto *Cmp = dyn_cast<ICmpInst>(c);
            if (!Cmp) return;
            ICmpInst::Predicate P = holds ? Cmp->getPredicate() : Cmp->getInversePredicate();
            Value *x = Cmp->getOperand(0), *y = Cmp->getOperand(1);
            if (isa<ConstantInt>(x)) { std::swap(x, y); P = ICmpInst::getSwappedPredicate(P); }
            auto *K = dyn_cast<ConstantInt>(y);
            if (!K) return;
            if ((P == ICmpInst::ICMP_SGT && K->getSExtValue() >= 0) ||
                (P == ICmpInst::ICMP_SGE && K->getSExtValue() >= 1))
                facts.push_back(SE.getSCEV(x));
        };
        for (BasicBlock &BB : O) {
            auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
            if (!BI || !BI->isConditional()) continue;
            for (unsigned s = 0; s < 2; ++s)
                if (DT.dominates(BasicBlockEdge(&BB, BI->getSuccessor(s)), L->getHeader()))
                    collect(BI->getCondition(), s == 0);
        }
        return facts;
    }

    // R == R2, two radices that differ only by casts. Mapped to kernel
    // values they must be equal, or be sext(t) and zext(t) of one narrow t
    // that a guard of the loop shows positive.
    bool provenEqual(Loop *L, const SCEV *R, const SCEV *R2) {
        auto &SK = FAM.getResult<ScalarEvolutionAnalysis>(KI.K);
        const SCEV *a = KI.toKernel(R, SK), *b = KI.toKernel(R2, SK);
        if (!a || !b) return false;
        if (sameSCEV(SK, a, b)) return true;
        auto narrow = [](const SCEV *s, bool &isSext) -> const SCEV * {
            if (auto *X = dyn_cast<SCEVSignExtendExpr>(s)) { isSext = true; return X->getOperand(); }
            if (auto *X = dyn_cast<SCEVZeroExtendExpr>(s)) { isSext = false; return X->getOperand(); }
            return nullptr;
        };
        bool sa = false, sb = false;
        const SCEV *ta = narrow(a, sa), *tb = narrow(b, sb);
        if (!ta || !tb || ta != tb || sa == sb) return false;
        for (const SCEV *x : positiveOnEntry(L))
            if (const SCEV *xk = KI.toKernel(x, SK); xk && xk == ta) return true;
        return false;
    }
};

}  // namespace

std::optional<WriteSet> collectWrites(const OmpKernel &KI, FunctionAnalysisManager &FAM,
                                      UnknownStore policy, std::string &why) {
    return Collector(KI, FAM, policy, why).run();
}

}  // namespace gicc::pass
