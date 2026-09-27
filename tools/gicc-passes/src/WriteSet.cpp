#include "WriteSet.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"

#include <functional>

using namespace llvm;

namespace gicc::pass {

SmallVector<EntryFact, 4> factsOnEntry(BasicBlock *target, DominatorTree &DT,
                                      ScalarEvolution &SE) {
    SmallVector<EntryFact, 4> facts;
    auto add = [&](const SCEV *x, bool nonzeroOnly = false) {
        if (llvm::none_of(facts, [&](const EntryFact &f) { return f.x == x; }))
            facts.push_back({x, nonzeroOnly});
    };
    std::function<void(Value *, bool)> collect = [&](Value *c, bool holds) {
        // a & b true, or a | b false: both halves hold (the latter negated).
        // Also in the select form instcombine uses for logical and / or.
        Value *a = nullptr, *b = nullptr;
        bool isAnd = false, isOr = false;
        if (auto *BO = dyn_cast<BinaryOperator>(c)) {
            a = BO->getOperand(0);
            b = BO->getOperand(1);
            isAnd = BO->getOpcode() == Instruction::And;
            isOr = BO->getOpcode() == Instruction::Or;
        } else if (auto *Sel = dyn_cast<SelectInst>(c)) {
            a = Sel->getCondition();
            if (auto *K = dyn_cast<ConstantInt>(Sel->getFalseValue()); K && K->isZero()) {
                b = Sel->getTrueValue();
                isAnd = true;
            } else if (auto *K = dyn_cast<ConstantInt>(Sel->getTrueValue()); K && K->isOne()) {
                b = Sel->getFalseValue();
                isOr = true;
            }
        }
        if (a && b) {
            if ((isAnd && holds) || (isOr && !holds)) {
                collect(a, holds);
                collect(b, holds);
            }
            return;
        }
        auto *Cmp = dyn_cast<ICmpInst>(c);
        if (!Cmp || !SE.isSCEVable(Cmp->getOperand(0)->getType())) return;
        ICmpInst::Predicate P = holds ? Cmp->getPredicate() : Cmp->getInversePredicate();
        const SCEV *x = SE.getSCEV(Cmp->getOperand(0)), *y = SE.getSCEV(Cmp->getOperand(1));
        const SCEV *one = SE.getOne(x->getType());
        switch (P) {
        case ICmpInst::ICMP_SGT: add(SE.getMinusSCEV(x, y)); break;
        case ICmpInst::ICMP_SGE: add(SE.getAddExpr(SE.getMinusSCEV(x, y), one)); break;
        case ICmpInst::ICMP_SLT: add(SE.getMinusSCEV(y, x)); break;
        case ICmpInst::ICMP_SLE: add(SE.getAddExpr(SE.getMinusSCEV(y, x), one)); break;
        case ICmpInst::ICMP_NE:
            if (y->isZero()) add(x, true);
            else if (x->isZero()) add(y, true);
            break;
        case ICmpInst::ICMP_UGT: if (y->isZero()) add(x, true); break;
        case ICmpInst::ICMP_ULT: if (x->isZero()) add(y, true); break;
        default: break;
        }
    };
    for (BasicBlock &BB : *target->getParent()) {
        auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
        if (!BI || !BI->isConditional()) continue;
        for (unsigned s = 0; s < 2; ++s)
            if (DT.dominates(BasicBlockEdge(&BB, BI->getSuccessor(s)), target))
                collect(BI->getCondition(), s == 0);
    }
    return facts;
}

namespace {

// The slot a (possibly cast) SCEVUnknown load reads, or null.
const Value *loadedSlot(const SCEV *S) {
    if (auto *C = dyn_cast<SCEVCastExpr>(S)) S = C->getOperand();
    auto *U = dyn_cast<SCEVUnknown>(S);
    if (!U) return nullptr;
    auto *LI = dyn_cast<LoadInst>(U->getValue());
    return LI ? LI->getPointerOperand()->stripPointerCasts() : nullptr;
}

class Collector {
    const OmpKernel &KI;
    FunctionAnalysisManager &FAM;
    UnknownStore policy;
    std::string &why;
    Function &O;
    ScalarEvolution &SE, &SK;
    LoopInfo &LI;
    DominatorTree &DT;
    Type *I64;
    const Value *lbSlot = nullptr, *ubSlot = nullptr, *strideSlot = nullptr;
    // The distribute loop's iteration count, when it starts at 0.
    const SCEV *N = nullptr;
    SmallVector<EntryFact, 4> kernelFacts;
    DenseMap<Loop *, SmallVector<EntryFact, 4>> loopFacts;
    WriteSet W;

public:
    Collector(const OmpKernel &KI, FunctionAnalysisManager &FAM, UnknownStore policy,
              std::string &why)
        : KI(KI), FAM(FAM), policy(policy), why(why), O(*KI.outlined),
          SE(FAM.getResult<ScalarEvolutionAnalysis>(O)),
          SK(FAM.getResult<ScalarEvolutionAnalysis>(KI.K)),
          LI(FAM.getResult<LoopAnalysis>(O)), DT(FAM.getResult<DominatorTreeAnalysis>(O)),
          I64(Type::getInt64Ty(O.getContext())) {}

    std::optional<WriteSet> run() {
        if (!findThreadLoop()) return std::nullopt;
        auto &DTK = FAM.getResult<DominatorTreeAnalysis>(KI.K);
        kernelFacts = factsOnEntry(KI.distInit->getParent(), DTK, SK);
        W.preconditions = kernelFacts;
        if (auto bounds = KI.distBounds(DTK)) {
            const SCEV *lb0 = KI.normalize(SK, SK.getSCEV(bounds->first->getValueOperand()));
            const SCEV *ub0 = KI.normalize(SK, SK.getSCEV(bounds->second->getValueOperand()));
            if (lb0->isZero()) N = SK.getAddExpr(KI.extDist(SK, ub0, I64), SK.getOne(I64));
        }
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
                if (!SE.isLoopInvariant(SE.getSCEV(St->getPointerOperand()), l))
                    return unknown(St, "store address moves with a sequential loop "
                                       "nested in the worksharing loop");
        }
        return box(St, L, iv, /*inInnerLoop=*/inner != L);
    }

    // x > 0 for an outlined-function value, whenever the loop body runs:
    // SCEV knows it, a guard of the loop or of the kernel's distribute
    // loop says so (x != 0 suffices, radices being trip counts below 2^63),
    // or it is a product / extension of such values.
    bool isPositive(const SCEV *x, Loop *L) {
        if (SE.isKnownPositive(x)) return true;
        auto [it, fresh] = loopFacts.try_emplace(L);
        if (fresh) it->second = factsOnEntry(L->getHeader(), DT, SE);
        if (llvm::any_of(it->second, [&](const EntryFact &f) { return sameSCEV(SE, f.x, x); }))
            return true;
        if (const SCEV *k = KI.toKernel(x, SK); k && kernelPositive(k)) return true;
        if (auto *M = dyn_cast<SCEVMulExpr>(x))
            return llvm::all_of(M->operands(), [&](const SCEV *op) { return isPositive(op, L); });
        // sext(t) > 0 iff t > 0; zext(t) > 0 if t > 0 read signed.
        if (isa<SCEVSignExtendExpr>(x) || isa<SCEVZeroExtendExpr>(x))
            return isPositive(cast<SCEVCastExpr>(x)->getOperand(), L);
        return false;
    }

    // The same for a kernel value, against the distribute loop's guards.
    bool kernelPositive(const SCEV *k) {
        if (SK.isKnownPositive(k) ||
            llvm::any_of(kernelFacts, [&](const EntryFact &f) { return sameSCEV(SK, f.x, k); }))
            return true;
        if (auto *M = dyn_cast<SCEVMulExpr>(k))
            return llvm::all_of(M->operands(), [&](const SCEV *op) { return kernelPositive(op); });
        if (isa<SCEVSignExtendExpr>(k) || isa<SCEVZeroExtendExpr>(k))
            return kernelPositive(cast<SCEVCastExpr>(k)->getOperand());
        return false;
    }

    // Each store as a box over the loop's digits, in kernel values.
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
        std::string reason;
        auto b = AccessDecomposer(SE, KI.DL, L, ivPhi).decompose(St->getPointerOperand(), reason);
        if (!b) return fail(reason);
        for (auto &[R, R2] : b->assumedEqual)
            if (!provenEqual(L, R, R2))
                return fail("collapse radix " + scevStr(R) + " and " + scevStr(R2) +
                            " differ only by a cast, but they cannot be proven equal");
        Value *obj = KI.resolveOutlined(b->base);
        if (!obj || !isa<Argument>(obj)) return fail("store base is not a kernel argument");

        // The digits are a box only for positive radices and an iteration
        // count N = n_1 * R_1.
        for (const SCEV *R : b->radix)
            if (!isPositive(R, L))
                return fail("collapse radix " + scevStr(R) +
                            " cannot be proven positive (a division the source wrote?)");
        if (!N) return fail("the distribute loop's iteration count is unknown");
        BoxWrite bw{St, obj, {}, {}, nullptr,
                    KI.DL.getTypeStoreSize(St->getValueOperand()->getType()), false,
                    b->trustedTrunc};
        const SCEV *R1 = b->radix.empty() ? nullptr : KI.toKernel(b->radix[0], SK);
        const SCEV *n1 = b->radix.empty() ? N
                         : R1 ? exactDivide(SK, N, SK.getNoopOrSignExtend(R1, I64)) : nullptr;
        if (!n1)
            return fail("the iteration count " + scevStr(N) +
                        " is not a proven multiple of collapse radix " + scevStr(b->radix[0]));
        bw.extent.push_back(n1);
        for (size_t d = 0; d < b->stride.size(); ++d) {
            const SCEV *s = KI.toKernel(b->stride[d], SK);
            const SCEV *e = d ? KI.toKernel(b->extent[d], SK) : n1;
            if (!s || !e) return fail("the box has no counterpart in kernel values");
            bw.stride.push_back(SK.getNoopOrSignExtend(s, I64));
            if (d) bw.extent.push_back(SK.getNoopOrSignExtend(e, I64));
        }
        bw.offset = KI.toKernel(b->offset, SK);
        if (!bw.offset) return fail("the box has no counterpart in kernel values");
        BasicBlock *latch = L->getLoopLatch();
        bw.unconditional = !inInnerLoop && latch && DT.dominates(St->getParent(), latch);
        W.boxes.push_back(bw);
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
            // Also an extension of it: a narrow IV compared through a sext.
            const SCEV *n = SE.getSCEV(ivSide);
            if (isa<SCEVSignExtendExpr>(n) || isa<SCEVZeroExtendExpr>(n))
                n = cast<SCEVCastExpr>(n)->getOperand();
            auto *next = dyn_cast<SCEVAddRecExpr>(n);
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
        // Compare in the wider of the two types.
        const SCEV *ub = SE.getSCEV(bSide), *lb = cur->getStart();
        Type *T = SE.getWiderType(ub->getType(), lb->getType());
        if (!SE.isLoopEntryGuardedByCond(L, P, SE.getNoopOrSignExtend(lb, T),
                                         SE.getNoopOrSignExtend(ub, T))) {
            why = "first iteration is not guarded by lb <= UB";
            return nullptr;
        }
        return cur;
    }

    // R == R2, two radices that differ only by casts. Mapped to kernel
    // values they must be equal, or be sext(t) and zext(t) of one narrow t
    // that a guard shows positive.
    bool provenEqual(Loop *L, const SCEV *R, const SCEV *R2) {
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
        // t > 0: the outlined side's narrow value, or the kernel's.
        for (const SCEV *r : {R, R2})
            if (auto *C = dyn_cast<SCEVCastExpr>(r); C && isPositive(C->getOperand(), L))
                return true;
        return llvm::any_of(kernelFacts, [&](const EntryFact &f) { return f.x == ta; });
    }
};

}  // namespace

std::optional<WriteSet> collectWrites(const OmpKernel &KI, FunctionAnalysisManager &FAM,
                                      UnknownStore policy, std::string &why) {
    return Collector(KI, FAM, policy, why).run();
}

}  // namespace gicc::pass
