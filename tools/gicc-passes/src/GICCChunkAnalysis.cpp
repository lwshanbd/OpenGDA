// Chunk legality analysis for OpenMP offload kernels.
//
// Input shape (generic-mode teams region, as clang emits it):
//
//   #pragma omp target teams
//   {
//       #pragma omp distribute parallel for [dist_schedule(static, C)]
//       for (i = lb0; i <= ub0; ++i)  src[i + d'] = ...;
//       ompx_pipelined_put(peer, dst, src + off, bytes);
//   }
//
// ompx_pipelined_put means "send these bytes once the kernel's writes to
// them are complete" -- the same as a host put after the kernel. It is
// legal to split it into one put per distribute block, issued when that
// block's parallel region joins, iff
//
//   (P1) every write to the source object inside the worksharing loop is
//        dense and affine in the loop IV: addr = base + c*i + d with
//        c == store size, so block [LB, UB] owns exactly
//        [base + c*LB + d, base + c*(UB+1) + d);
//   (P2) the put covers exactly the union of the blocks:
//        off == c*lb0 + d and bytes == c*(ub0 - lb0 + 1);
//   (P3) nothing else writes the source object: no store outside the
//        worksharing loop, no opaque call that could;
//   (P4) peer and dst are the same in every team (kernel formals and
//        constants only), so each team can issue its own blocks;
//   (P5) no synchronization lies between the first block's join and the
//        put, so every block still reaches the peer inside the same
//        synchronization epoch as the original put -- the epoch in which
//        the program already guarantees the peer leaves dst alone.
//
// Trusted, not proven here: the OpenMP runtime contract that
// __kmpc_distribute_static_init partitions [lb0, ub0] among teams without
// gaps or overlap and __kmpc_for_static_init partitions a block [LB, UB]
// among a team's threads the same way; and clang's convention of passing
// the current block's LB/UB in slots 0 and 1 of __kmpc_parallel_51's
// argument array. Distinct kernel pointer arguments are assumed not to
// overlap (is_device_ptr gives no noalias guarantee).
//
// A plain ompx_put in the same position is diagnosed as a race: every
// team runs it once, after only its own blocks are written.

#include "GICCChunkAnalysis.h"
#include "GICCPassConfig.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include <cstdlib>
#include <optional>
#include <string>

using namespace llvm;

namespace gicc::pass {

namespace {

constexpr StringLiteral kPipelinedPut = "ompx_pipelined_put";
constexpr StringLiteral kPlainPut     = "ompx_put";

raw_ostream &out() { return errs(); }

std::string objName(const Value *V) {
    if (V->hasName()) return V->getName().str();
    if (auto *A = dyn_cast<Argument>(V)) return "arg#" + std::to_string(A->getArgNo());
    return "<unnamed>";
}

std::string str(const SCEV *S) {
    std::string s;
    raw_string_ostream os(s);
    S->print(os);
    return s;
}

StringRef calleeName(const CallBase &CB) {
    if (const Function *F = CB.getCalledFunction()) return F->getName();
    return "";
}

bool isCallTo(const Instruction &I, StringRef prefix) {
    const auto *CB = dyn_cast<CallBase>(&I);
    return CB && calleeName(*CB).starts_with(prefix);
}

// Calls that cannot write user memory, or whose writes are covered by the
// trusted OpenMP runtime contract (they only touch the bound slots).
bool isBenignCall(const CallBase &CB) {
    if (const auto *II = dyn_cast<IntrinsicInst>(&CB)) {
        if (II->isLifetimeStartOrEnd() || II->isAssumeLikeIntrinsic())
            return true;
        return !II->mayWriteToMemory();
    }
    StringRef n = calleeName(CB);
    return n.starts_with("__kmpc_for_static_init") ||
           n.starts_with("__kmpc_for_static_fini") ||
           n.starts_with("__kmpc_distribute_static_init") ||
           n.starts_with("__kmpc_distribute_static_fini") ||
           n == "__kmpc_global_thread_num" ||
           n == "__kmpc_target_init" || n == "__kmpc_target_deinit" ||
           n == "__kmpc_parallel_51" || n == kPipelinedPut;
}

// The unique store to Obj (after pointer-cast stripping) in F, or null.
StoreInst *uniqueStoreTo(Function &F, const Value *Obj) {
    StoreInst *found = nullptr;
    for (Instruction &I : instructions(F)) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI || SI->getPointerOperand()->stripPointerCasts() != Obj)
            continue;
        if (found) return nullptr;
        found = SI;
    }
    return found;
}

// One offload kernel, the pieces of its worksharing structure, and the
// value provenance needed to see through generic-mode globalization.
struct Kernel {
    Function     &K;
    CallInst     *distInit  = nullptr;
    CallInst     *distFini  = nullptr;
    CallInst     *parallel  = nullptr;
    Function     *outlined  = nullptr;
    AllocaInst   *argsArray = nullptr;
    const DataLayout &DL;
    bool          distSigned = true;
    bool          ubPlusOne  = false;   // a loop exits on i.next < UB + 1
    std::string   error;
    // Stores in the outlined region, by the kernel object they write.
    DenseMap<Value *, SmallVector<StoreInst *, 2>> storesTo;

    Kernel(Function &K) : K(K), DL(K.getParent()->getDataLayout()) {}

    // Follow a kernel-side value back through phis whose inputs agree and
    // loads of team-shared globals with a single store. Returns V itself
    // when nothing simpler is known.
    Value *resolve(Value *V, unsigned depth = 0) const {
        if (depth > 16) return V;
        V = V->stripPointerCasts();
        if (auto *PN = dyn_cast<PHINode>(V)) {
            Value *common = nullptr;
            for (Value *in : PN->incoming_values()) {
                Value *r = resolve(in, depth + 1);
                if (common && r != common) return V;
                common = r;
            }
            return common ? common : V;
        }
        if (auto *LI = dyn_cast<LoadInst>(V)) {
            auto *GV = dyn_cast<GlobalVariable>(
                LI->getPointerOperand()->stripPointerCasts());
            if (!GV) return V;
            if (StoreInst *SI = uniqueStoreTo(K, GV))
                return resolve(SI->getValueOperand(), depth + 1);
        }
        return V;
    }

    // Kernel-side value stored in slot `idx` of the parallel argument array.
    Value *slotValue(unsigned idx) const {
        Value *found = nullptr;
        for (Instruction &I : instructions(K)) {
            auto *SI = dyn_cast<StoreInst>(&I);
            if (!SI) continue;
            APInt off(DL.getIndexTypeSizeInBits(SI->getPointerOperandType()), 0);
            const Value *base = SI->getPointerOperand()
                ->stripAndAccumulateConstantOffsets(DL, off, true);
            if (base->stripPointerCasts() != argsArray) continue;
            if (off.getZExtValue() != idx * DL.getPointerSize()) continue;
            if (found) return nullptr;
            found = SI->getValueOperand();
        }
        return found;
    }

    // Map an outlined-function value that is a captured variable (a
    // pointer to it, loaded) back to the kernel value it holds.
    Value *resolveOutlined(Value *V) const {
        V = V->stripPointerCasts();
        auto viaSlot = [&](Argument *A) -> Value * {
            if (A->getParent() != outlined || A->getArgNo() < 2) return nullptr;
            return slotValue(A->getArgNo() - 2);
        };
        if (auto *LI = dyn_cast<LoadInst>(V)) {
            // Captured by reference: the slot holds the variable's address.
            if (auto *A = dyn_cast<Argument>(
                    LI->getPointerOperand()->stripPointerCasts())) {
                Value *addr = viaSlot(A);
                if (!addr) return nullptr;
                Value *obj = addr->stripPointerCasts();
                if (!isa<GlobalVariable>(obj) && !isa<AllocaInst>(obj))
                    return nullptr;
                StoreInst *SI = uniqueStoreTo(K, obj);
                return SI ? resolve(SI->getValueOperand()) : nullptr;
            }
            return nullptr;
        }
        if (auto *A = dyn_cast<Argument>(V)) {
            // Captured by value.
            Value *v = viaSlot(A);
            return v ? resolve(v) : nullptr;
        }
        return nullptr;
    }
};

// Rewrite every SCEVUnknown whose value resolves to something simpler, so
// that equal quantities reached through globalized copies compare equal.
class ResolveRewriter : public SCEVRewriteVisitor<ResolveRewriter> {
    const Kernel &KI;
public:
    ResolveRewriter(ScalarEvolution &SE, const Kernel &KI)
        : SCEVRewriteVisitor(SE), KI(KI) {}
    const SCEV *visitUnknown(const SCEVUnknown *U) {
        Value *r = KI.resolve(U->getValue());
        if (r == U->getValue() || !SE.isSCEVable(r->getType())) return U;
        return visit(SE.getSCEV(r));
    }
};

const SCEV *normalize(ScalarEvolution &SE, const Kernel &KI, const SCEV *S) {
    ResolveRewriter RW(SE, KI);
    return RW.visit(S);
}

bool isZero(ScalarEvolution &SE, const SCEV *a, const SCEV *b) {
    if (a->getType() != b->getType()) {
        Type *T = SE.getWiderType(a->getType(), b->getType());
        a = SE.getNoopOrSignExtend(a, T);
        b = SE.getNoopOrSignExtend(b, T);
    }
    return SE.getMinusSCEV(a, b)->isZero();
}

// True when every SCEVUnknown in S is a formal of F (the value is then the
// same in every team).
bool onlyFormals(const SCEV *S, const Function &F) {
    bool ok = true;
    struct V {
        bool &ok; const Function &F;
        bool follow(const SCEV *S) {
            if (auto *U = dyn_cast<SCEVUnknown>(S)) {
                auto *A = dyn_cast<Argument>(U->getValue());
                if (!A || A->getParent() != &F) ok = false;
            }
            return ok;
        }
        bool isDone() const { return !ok; }
    } vis{ok, F};
    visitAll(S, vis);
    return ok;
}

// The write pattern of one kernel object inside the worksharing loop.
struct WritePattern {
    const SCEV *c = nullptr;   // bytes per iteration
    const SCEV *d = nullptr;   // constant byte offset from base + c*i
    uint64_t    storeSize = 0;
};

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

// A put proven splittable, with everything needed to emit one send per
// block. SCEVs are in terms of kernel formals only (lb0 too), so they can be
// expanded anywhere in the kernel.
// Attribute marking outlined parallel regions GICCChunkPrepPass kept out of
// their wrappers; the lowering removes it together with the noinline.
constexpr StringLiteral kPrepAttr = "gicc-chunk-noinline";

struct Plan {
    CallInst   *put;
    Value      *srcObj;   // kernel formal the put reads
    const SCEV *c, *d;    // block [LB,UB] owns srcObj + c*LB + d, c*(UB-LB+1) bytes
    const SCEV *lb0;      // first distribute iteration (i64)
    const SCEV *peer, *dst;
};

class Analyzer {
    Module &M;
    FunctionAnalysisManager &FAM;
    bool lower;
    bool changed = false;
public:
    Analyzer(Module &M, FunctionAnalysisManager &FAM, bool lower)
        : M(M), FAM(FAM), lower(lower) {}
    bool modified() const { return changed; }

    void run() {
        for (Function &F : M) {
            if (F.isDeclaration() || !F.getName().starts_with("__omp_offloading_"))
                continue;
            if (F.hasLocalLinkage()) continue;   // outlined helpers
            analyzeKernel(F);
        }
        // Lowering done: let the device link inline these again.
        for (Function &F : M)
            if (F.hasFnAttribute(kPrepAttr)) {
                const bool always =
                    F.getFnAttribute(kPrepAttr).getValueAsString() == "alwaysinline";
                F.removeFnAttr(kPrepAttr);
                F.removeFnAttr(Attribute::NoInline);
                if (always) F.addFnAttr(Attribute::AlwaysInline);
                changed = true;
            }
    }

private:
    bool locateStructure(Kernel &KI) {
        for (Instruction &I : instructions(KI.K)) {
            auto *CI = dyn_cast<CallInst>(&I);
            if (!CI) continue;
            StringRef n = calleeName(*CI);
            if (n.starts_with("__kmpc_distribute_static_init")) {
                if (KI.distInit) { KI.error = "more than one distribute loop"; return false; }
                KI.distInit = CI;
                KI.distSigned = !n.ends_with("u");
            } else if (n.starts_with("__kmpc_distribute_static_fini")) {
                KI.distFini = CI;
            } else if (n == "__kmpc_parallel_51") {
                if (KI.parallel) { KI.error = "more than one parallel region"; return false; }
                KI.parallel = CI;
            }
        }
        if (!KI.distInit || !KI.distFini || !KI.parallel) {
            KI.error = "no distribute parallel for worksharing loop";
            return false;
        }
        KI.outlined = dyn_cast<Function>(
            KI.parallel->getArgOperand(5)->stripPointerCasts());
        KI.argsArray = dyn_cast<AllocaInst>(
            KI.parallel->getArgOperand(7)->stripPointerCasts());
        if (!KI.outlined || KI.outlined->isDeclaration() || !KI.argsArray) {
            KI.error = "unrecognized __kmpc_parallel_51 operands";
            return false;
        }
        return true;
    }

    // Distribute bound slot (plower = operand 4, pupper = operand 5) and
    // the value stored into it before the init call.
    static const Value *boundSlot(CallInst *init, unsigned opnd) {
        return init->getArgOperand(opnd)->stripPointerCasts();
    }

    // Is V (a value passed in slot 0/1) the current block bound derived
    // from `slot`? Accepts the load itself, the clamp to ub0, the advance
    // by the stride, and phis over those.
    bool isBlockBound(Value *V, const Value *slot, const Value *strideSlot,
                      SmallPtrSetImpl<Value *> &seen) {
        V = V->stripPointerCasts();
        if (auto *I2P = dyn_cast<IntToPtrInst>(V)) V = I2P->getOperand(0);
        if (auto *CI = dyn_cast<CastInst>(V)) V = CI->getOperand(0);
        if (!seen.insert(V).second) return true;
        if (auto *LI = dyn_cast<LoadInst>(V))
            return LI->getPointerOperand()->stripPointerCasts() == slot;
        if (auto *MM = dyn_cast<MinMaxIntrinsic>(V))
            return isBlockBound(MM->getLHS(), slot, strideSlot, seen);
        if (auto *BO = dyn_cast<BinaryOperator>(V)) {
            if (BO->getOpcode() != Instruction::Add) return false;
            for (unsigned k = 0; k < 2; ++k) {
                auto *LI = dyn_cast<LoadInst>(BO->getOperand(1 - k));
                if (LI && LI->getPointerOperand()->stripPointerCasts() == strideSlot)
                    return isBlockBound(BO->getOperand(k), slot, strideSlot, seen);
            }
            return false;
        }
        if (auto *PN = dyn_cast<PHINode>(V)) {
            for (Value *in : PN->incoming_values())
                if (!isBlockBound(in, slot, strideSlot, seen)) return false;
            return true;
        }
        return false;
    }

    // Collect the write pattern of every kernel object written inside the
    // outlined parallel region. Returns false (with KI.error) on anything
    // the analysis cannot account for.
    bool collectWrites(Kernel &KI, DenseMap<Value *, WritePattern> &writes) {
        Function &O = *KI.outlined;
        auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(O);
        auto &LI = FAM.getResult<LoopAnalysis>(O);

        CallInst *forInit = nullptr;
        for (Instruction &I : instructions(O))
            if (isCallTo(I, "__kmpc_for_static_init")) {
                if (forInit) { KI.error = "more than one worksharing loop in the parallel region"; return false; }
                forInit = cast<CallInst>(&I);
            }
        if (!forInit) { KI.error = "parallel region has no worksharing loop"; return false; }
        const Value *lbSlot     = forInit->getArgOperand(4)->stripPointerCasts();
        const Value *ubSlot     = forInit->getArgOperand(5)->stripPointerCasts();
        const Value *strideSlot = forInit->getArgOperand(6)->stripPointerCasts();

        // The thread range must start from the block bounds handed in by
        // the kernel (formals 2 and 3 of the outlined function).
        auto storedFromFormal = [&](const Value *slot, unsigned argNo) {
            StoreInst *SI = uniqueStoreTo(O, slot);
            if (!SI) return false;
            Value *v = SI->getValueOperand();
            if (auto *CI = dyn_cast<CastInst>(v)) v = CI->getOperand(0);
            return v == O.getArg(argNo);
        };
        if (O.arg_size() < 4 || !storedFromFormal(lbSlot, 2) ||
            !storedFromFormal(ubSlot, 3)) {
            KI.error = "worksharing loop bounds are not the distribute block bounds";
            return false;
        }

        for (Instruction &I : instructions(O)) {
            if (auto *CB = dyn_cast<CallBase>(&I)) {
                if (!isBenignCall(*CB)) {
                    KI.error = ("call to '" + calleeName(*CB) +
                                "' in the parallel region may write memory").str();
                    if (calleeName(*CB).empty()) KI.error = "indirect call in the parallel region";
                    return false;
                }
                continue;
            }
            if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I)) {
                KI.error = "atomic write in the parallel region";
                return false;
            }
            auto *St = dyn_cast<StoreInst>(&I);
            if (!St) continue;
            Value *ptr = St->getPointerOperand();
            if (isa<AllocaInst>(getUnderlyingObject(ptr))) continue;   // thread-private

            Loop *L = LI.getLoopFor(St->getParent());
            if (!L) { KI.error = "store outside the worksharing loop body"; return false; }
            // Inner loops (e.g. a per-element compute loop) are fine as long
            // as the store itself sits in the worksharing loop: its address
            // then depends on that loop's IV alone.
            if (L->getParentLoop()) {
                KI.error = "store inside a loop nested in the worksharing loop";
                return false;
            }
            auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(ptr));
            if (!AR || AR->getLoop() != L || !AR->isAffine()) {
                KI.error = "store address is not affine in the loop IV";
                return false;
            }
            const SCEVAddRecExpr *iv = checkLoopCoversBlock(
                L, O, lbSlot, ubSlot, strideSlot, SE, KI);
            if (!iv) return false;

            // addr = base + c*iv + d, with the same IV the exit tests.
            const SCEV *c = nullptr, *stepTerm = nullptr;
            if (!splitScaledLoad(AR->getStepRecurrence(SE), strideSlot, c, stepTerm, SE) ||
                stepTerm != iv->getStepRecurrence(SE)) {
                KI.error = "store address does not advance with the loop IV";
                return false;
            }
            auto *base = dyn_cast<SCEVUnknown>(SE.getPointerBase(AR->getStart()));
            if (!base) { KI.error = "store has no single base pointer"; return false; }
            const SCEV *d = SE.getMinusSCEV(
                SE.getMinusSCEV(AR, SE.getMulExpr(c, iv)), base);
            if (!isa<SCEVConstant>(d)) {
                KI.error = "store offset is not constant relative to base + c*i";
                return false;
            }
            Value *obj = KI.resolveOutlined(base->getValue());
            if (!obj || !isa<Argument>(obj)) {
                KI.error = "store base is not a kernel argument";
                return false;
            }
            WritePattern wp{c, d, KI.DL.getTypeStoreSize(
                                      St->getValueOperand()->getType())};
            auto [it, fresh] = writes.try_emplace(obj, wp);
            if (!fresh && (it->second.c != c || it->second.d != d ||
                           it->second.storeSize != wp.storeSize)) {
                KI.error = ("object '" + objName(obj) +
                            "' is written with more than one access pattern");
                return false;
            }
            KI.storesTo[obj].push_back(St);
        }
        return true;
    }

    // The loop must run exactly the thread's share of the block: its IV
    // starts at the thread lower bound, steps by the thread stride, the
    // entry is guarded by lb <= UB and the latch continues while
    // i.next <= UB. Returns the IV of the current iteration, or null.
    const SCEVAddRecExpr *checkLoopCoversBlock(Loop *L, Function &O,
                                               const Value *lbSlot,
                                               const Value *ubSlot,
                                               const Value *strideSlot,
                                               ScalarEvolution &SE, Kernel &KI) {
        auto isUB = [&](Value *v) {
            if (auto *CI = dyn_cast<CastInst>(v)) v = CI->getOperand(0);
            if (v == O.getArg(3)) return true;
            auto *LdI = dyn_cast<LoadInst>(v);
            return LdI && LdI->getPointerOperand()->stripPointerCasts() == ubSlot;
        };
        BasicBlock *latch = L->getLoopLatch();
        auto *BI = latch ? dyn_cast<BranchInst>(latch->getTerminator()) : nullptr;
        auto *Cmp = BI && BI->isConditional() ? dyn_cast<ICmpInst>(BI->getCondition()) : nullptr;
        if (!Cmp) { KI.error = "loop latch is not a compare-and-branch"; return nullptr; }

        // The bound is UB itself, or UB + 1 (64-bit loops are emitted as
        // "continue while i.next < UB + 1"; equivalent while UB + 1 does
        // not wrap, which the kernel side proves when ubPlusOne is set).
        auto isBound = [&](Value *v, bool &plusOne) {
            plusOne = false;
            if (auto *BO = dyn_cast<BinaryOperator>(v);
                BO && BO->getOpcode() == Instruction::Add)
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
        if (!isBound(bSide, plusOne)) { KI.error = "loop exit does not compare against the block upper bound"; return nullptr; }
        const bool inclusive = P == ICmpInst::ICMP_SLE || P == ICmpInst::ICMP_ULE;
        const bool strict    = P == ICmpInst::ICMP_SLT || P == ICmpInst::ICMP_ULT;
        if (!(plusOne ? strict : inclusive)) {
            KI.error = "loop exit is not 'continue while i <= UB'";
            return nullptr;
        }
        KI.ubPlusOne |= plusOne;
        // The compared value must be the IV of the next iteration: either
        // the value a header phi takes on the back edge (true by
        // construction, even where SCEV cannot fold the casts on it), or
        // an add recurrence SCEV proves to be one step ahead.
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
        if (!cur) {
            KI.error = "loop exit compare is not on the IV";
            return nullptr;
        }
        const SCEV *step = cur->getStepRecurrence(SE);
        if (!cur || loadedSlot(cur->getStart()) != lbSlot ||
            loadedSlot(step) != strideSlot) {
            KI.error = "loop IV is not the thread's lb stepped by its stride";
            return nullptr;
        }
        const SCEV *ub = SE.getSCEV(bSide);
        if (!SE.isLoopEntryGuardedByCond(L, P, cur->getStart(),
                                         SE.getNoopOrSignExtend(ub, cur->getType()))) {
            KI.error = "first iteration is not guarded by lb <= UB";
            return nullptr;
        }
        return cur;
    }

    void analyzeKernel(Function &F) {
        Kernel KI(F);
        SmallVector<CallInst *, 4> puts, plain;
        for (Instruction &I : instructions(F)) {
            if (isCallTo(I, kPipelinedPut)) puts.push_back(cast<CallInst>(&I));
            else if (auto *CB = dyn_cast<CallBase>(&I); CB && calleeName(*CB) == kPlainPut)
                plain.push_back(cast<CallInst>(&I));
        }
        if (puts.empty() && plain.empty()) return;

        out() << "[gicc-chunk] kernel " << F.getName() << "\n";
        DenseMap<Value *, WritePattern> writes;
        if (!locateStructure(KI)) {
            for (CallInst *CI : puts) report(CI, "ILLEGAL", KI.error);
            if (std::getenv("GICC_CHUNK_DUMP")) F.print(errs());
            return;
        }

        auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
        auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
        auto &LI = FAM.getResult<LoopAnalysis>(F);

        // Checked first: it is the one condition about the peer rather
        // than this rank, and its message should not be masked by the
        // generic "call may write memory" of the checks below.
        SmallVector<CallInst *, 4> unsynced;
        for (CallInst *CI : puts) {
            if (std::string sync = syncBeforePut(KI, CI, DT, LI); !sync.empty())
                report(CI, "ILLEGAL",
                       "synchronization (" + sync + ") between the first block and "
                       "the put: early blocks would reach the peer before it");
            else
                unsynced.push_back(CI);
        }
        puts = std::move(unsynced);
        if (puts.empty() && plain.empty()) return;

        if (!collectWrites(KI, writes)) {
            for (CallInst *CI : puts) report(CI, "ILLEGAL", KI.error);
            return;
        }

        // Distribute iteration space [lb0, ub0].
        const Value *lbSlot = boundSlot(KI.distInit, 4);
        const Value *ubSlot = boundSlot(KI.distInit, 5);
        StoreInst *lbSt = nullptr, *ubSt = nullptr;
        for (Instruction &I : instructions(F)) {
            auto *SI = dyn_cast<StoreInst>(&I);
            if (!SI || !DT.dominates(SI, KI.distInit)) continue;
            if (SI->getPointerOperand()->stripPointerCasts() == lbSlot) lbSt = SI;
            if (SI->getPointerOperand()->stripPointerCasts() == ubSlot) ubSt = SI;
        }
        const Value *strideSlot = boundSlot(KI.distInit, 6);
        Value *slot0 = KI.slotValue(0), *slot1 = KI.slotValue(1);
        SmallPtrSet<Value *, 8> seen0, seen1;
        if (!lbSt || !ubSt || !slot0 || !slot1 ||
            !isBlockBound(slot0, lbSlot, strideSlot, seen0) ||
            !isBlockBound(slot1, ubSlot, strideSlot, seen1)) {
            for (CallInst *CI : puts)
                report(CI, "ILLEGAL", "cannot identify the distribute block bounds");
            return;
        }
        auto ext = [&](const SCEV *S, Type *T) {
            return KI.distSigned ? SE.getNoopOrSignExtend(S, T) : SE.getNoopOrZeroExtend(S, T);
        };

        for (CallInst *CI : plain) {
            Value *obj = stripToObject(KI, CI->getArgOperand(2), nullptr);
            if (obj && writes.count(obj) && !isInOutlined(CI, KI))
                report(CI, "RACE",
                       "ompx_put in the teams region runs once per team, after "
                       "only that team's blocks of '" + objName(obj) +
                       "' are written; use ompx_pipelined_put or put after the kernel");
        }

        // The put's arguments as they are when the distribute loop ran. A
        // phi merging in a zero-trip path (clang folds n * size to 0 there)
        // is read on the loop side only: the lowered form sends nothing on
        // the zero-trip path, which the range check below then compares
        // with a loop that wrote nothing.
        auto onLoopPath = [&](Value *V) -> Value * {
            auto *PN = dyn_cast<PHINode>(V);
            if (!PN) return V;
            Value *pick = nullptr;
            for (unsigned k = 0; k < PN->getNumIncomingValues(); ++k) {
                if (!isPotentiallyReachable(KI.distInit->getParent(),
                                            PN->getIncomingBlock(k), nullptr, &DT, &LI))
                    continue;
                Value *r = KI.resolve(PN->getIncomingValue(k));
                if (pick && r != pick) return V;
                pick = r;
            }
            return pick ? pick : V;
        };

        SmallVector<Plan, 2> plans;
        for (CallInst *CI : puts) {
            if (!isAfterDistribute(KI, CI, DT, LI)) {
                report(CI, "ILLEGAL", "put is not after the distribute loop");
                continue;
            }
            int64_t off = 0;
            Value *obj = stripToObject(KI, CI->getArgOperand(2), &off);
            if (!obj || !isa<Argument>(obj)) {
                report(CI, "ILLEGAL", "put source is not a kernel argument plus a constant");
                continue;
            }
            auto it = writes.find(obj);
            if (it == writes.end()) {
                report(CI, "ILLEGAL", "put source '" + objName(obj) +
                                      "' is not written by the worksharing loop");
                continue;
            }
            if (std::string why = writesOutsideLoop(KI, obj); !why.empty()) {
                report(CI, "ILLEGAL", why);
                continue;
            }
            const WritePattern &wp = it->second;
            auto *cC = cast<SCEVConstant>(wp.c);
            if (cC->getAPInt().getZExtValue() != wp.storeSize) {
                report(CI, "ILLEGAL",
                       "writes to '" + objName(obj) + "' are strided (" +
                       std::to_string(cC->getAPInt().getSExtValue()) + " bytes per iteration, " +
                       std::to_string(wp.storeSize) + "-byte stores): a block does not own a contiguous range");
                continue;
            }
            // Trip count in the bounds' own width. It equals the true count
            // only if ub0 - lb0 + 1 does not wrap there; prove that where
            // the loop actually starts (a zero-trip path writes nothing).
            Type *I64 = Type::getInt64Ty(F.getContext());
            const SCEV *lb0n = normalize(SE, KI, SE.getSCEV(lbSt->getValueOperand()));
            const SCEV *ub0n = normalize(SE, KI, SE.getSCEV(ubSt->getValueOperand()));
            const SCEV *span = SE.getMinusSCEV(ub0n, lb0n);
            const SCEV *one  = SE.getOne(span->getType());
            if (KI.ubPlusOne &&
                !SE.willNotOverflow(Instruction::Add, KI.distSigned, ub0n, one, KI.distInit)) {
                report(CI, "ILLEGAL", "cannot prove the block bound " + str(ub0n) +
                                      " + 1 used by the loop exit does not wrap");
                continue;
            }
            if (!SE.willNotOverflow(Instruction::Sub, KI.distSigned, ub0n, lb0n, KI.distInit) ||
                !SE.willNotOverflow(Instruction::Add, KI.distSigned, span, one, KI.distInit)) {
                report(CI, "ILLEGAL", "cannot prove the distribute trip count " +
                                      str(ub0n) + " - " + str(lb0n) + " + 1 does not wrap");
                continue;
            }
            const SCEV *lb0 = ext(lb0n, I64);
            const SCEV *ub0 = ext(ub0n, I64);
            const SCEV *c   = SE.getNoopOrSignExtend(wp.c, I64);
            const SCEV *d   = SE.getNoopOrSignExtend(wp.d, I64);
            const SCEV *wantOff   = SE.getAddExpr(SE.getMulExpr(c, lb0), d);
            const SCEV *wantBytes = SE.getMulExpr(c, ext(SE.getAddExpr(span, one), I64));
            const SCEV *bytes = normalize(SE, KI, SE.getSCEV(onLoopPath(CI->getArgOperand(3))));
            const SCEV *offS  = SE.getConstant(I64, off, true);
            if (!isZero(SE, offS, wantOff) || !isZero(SE, bytes, wantBytes)) {
                report(CI, "ILLEGAL",
                       "put range [src+" + str(offS) + ", +" + str(bytes) +
                       ") is not the loop's write range [src+" + str(wantOff) +
                       ", +" + str(wantBytes) + ")");
                continue;
            }
            const SCEV *peer = normalize(SE, KI, SE.getSCEV(CI->getArgOperand(0)));
            const SCEV *dst  = normalize(SE, KI, SE.getSCEV(CI->getArgOperand(1)));
            if (!onlyFormals(peer, F) || !onlyFormals(dst, F)) {
                report(CI, "ILLEGAL", "peer or dst differs between teams");
                continue;
            }
            report(CI, "LEGAL", "");
            int64_t sched = cast<ConstantInt>(KI.distInit->getArgOperand(2))->getSExtValue();
            out() << "[gicc-chunk]   blocks: distribute sched=" << sched;
            if (sched == 91)
                out() << " chunk=" << str(SE.getSCEV(KI.distInit->getArgOperand(8)));
            out() << " over [" << str(lb0) << ", " << str(ub0) << "]\n"
                  << "[gicc-chunk]   block [LB,UB]: put(peer, dst + " << str(c)
                  << "*(LB - " << str(lb0) << "), " << objName(obj) << " + "
                  << str(c) << "*LB + " << str(d) << ", " << str(c)
                  << "*(UB - LB + 1)) after the block's parallel region joins\n";
            plans.push_back({CI, obj, c, d, lb0, peer, dst});
        }
        if (!lower) return;
        for (const Plan &P : plans) lowerPut(KI, P, SE);
        if (!plans.empty()) {
            changed = true;
            if (verifyFunction(F, &errs()))
                report_fatal_error("gicc-chunk: lowering produced invalid IR");
        }
    }

    // The integer block bound behind parallel slot 0 or 1:
    // inttoptr(ext(bound)) -> bound.
    static Value *slotBound(Value *V) {
        V = V->stripPointerCasts();
        if (auto *I2P = dyn_cast<IntToPtrInst>(V)) V = I2P->getOperand(0);
        if (auto *CI = dyn_cast<CastInst>(V)) V = CI->getOperand(0);
        return V;
    }

    // Replace the put with sends issued as the data becomes final, at the
    // grain GICC_CHUNK_GRAIN selects:
    //
    //   block    one ompx__block_put per block, right after the block's
    //            __kmpc_parallel_51 returns: the region has joined, so all
    //            of the block's words are written.
    //   element  every store to the source is followed by the same store
    //            into a same-node peer's copy. P1 makes that store the
    //            word's only write, so its value is final when stored. The
    //            peer address comes from ompx__peer_addr, once per team;
    //            when it is null (peer not IPC-mapped) the kernel falls
    //            back at run time to one single-thread ompx_put per block
    //            (ompx__block_put_one), which keeps the kernel SPMD-able.
    //
    // On a zero-trip path nothing is sent, where the original put sent
    // c*(ub0 - lb0 + 1) <= 0 bytes.
    void lowerPut(Kernel &KI, const Plan &P, ScalarEvolution &SE) {
        LLVMContext &Ctx = M.getContext();
        Type *I64 = Type::getInt64Ty(Ctx);
        Type *I32 = Type::getInt32Ty(Ctx);
        Type *I8  = Type::getInt8Ty(Ctx);
        auto *Ptr = cast<PointerType>(P.put->getArgOperand(1)->getType());
        SCEVExpander X(SE, KI.DL, "gicc.chunk");

        bool element = getConfig().chunkGrain != "block";
        if (element && !wrapperCallsOutlined(KI)) {
            out() << "[gicc-chunk]   the parallel region is inlined into its wrapper "
                     "(GICCChunkPrepPass did not run): using block grain\n";
            element = false;
        }

        // Per-block send after the region joins; under the element grain
        // only when the team could not map the peer.
        BasicBlock::iterator at = std::next(KI.parallel->getIterator());
        Value *mirror = nullptr;
        if (element) {
            // Once per team, before the loop: where the source's first byte
            // lands in the peer, as a byte delta from the local address.
            BasicBlock::iterator pre = KI.distInit->getIterator();
            const SCEV *srcStart = SE.getAddExpr(
                SE.getSCEV(P.srcObj), SE.getAddExpr(SE.getMulExpr(P.c, P.lb0), P.d));
            Value *peerV = X.expandCodeFor(P.peer, I32, pre);
            Value *dstV  = X.expandCodeFor(P.dst, Ptr, pre);
            Value *srcV  = X.expandCodeFor(srcStart, Ptr, pre);
            IRBuilder<> B(KI.distInit);
            FunctionCallee peerAddr = M.getOrInsertFunction(
                "ompx__peer_addr", Ptr, I32, Ptr);
            Value *pd = B.CreateCall(peerAddr, {peerV, dstV});
            mirror = B.CreateICmpNE(pd, ConstantPointerNull::get(Ptr));
            Value *delta = B.CreateSub(B.CreatePtrToInt(pd, I64), B.CreatePtrToInt(srcV, I64));
            // Team-shared, like clang's own globalized captures: the main
            // thread writes them before any parallel region starts.
            auto shared = [&](Type *T, const char *name) {
                return new GlobalVariable(M, T, false, GlobalValue::InternalLinkage,
                                          PoisonValue::get(T), name, nullptr,
                                          GlobalValue::NotThreadLocal, 3);
            };
            GlobalVariable *gDelta  = shared(I64, "gicc.chunk.delta");
            GlobalVariable *gMirror = shared(I8, "gicc.chunk.mirror");
            B.CreateStore(delta, gDelta);
            B.CreateStore(B.CreateZExt(mirror, I8), gMirror);

            Function &O = *KI.outlined;
            IRBuilder<> BO(&*O.getEntryBlock().getFirstInsertionPt());
            Value *oDelta  = BO.CreateLoad(I64, gDelta, "gicc.chunk.delta");
            Value *oMirror = BO.CreateICmpNE(BO.CreateLoad(I8, gMirror), ConstantInt::get(I8, 0));
            for (StoreInst *St : KI.storesTo.lookup(P.srcObj)) {
                IRBuilder<> BS(St->getNextNode());
                Value *remote = BS.CreateGEP(I8, St->getPointerOperand(), oDelta);
                Value *target = BS.CreateSelect(oMirror, remote, St->getPointerOperand());
                BS.CreateAlignedStore(St->getValueOperand(), target, St->getAlign());
            }
        }

        const SCEV *LB = extTo64(SE, KI, slotBound(KI.slotValue(0)));
        const SCEV *UB = extTo64(SE, KI, slotBound(KI.slotValue(1)));
        const SCEV *srcBlk = SE.getAddExpr(
            SE.getSCEV(P.srcObj), SE.getAddExpr(SE.getMulExpr(P.c, LB), P.d));
        const SCEV *dstBlk = SE.getAddExpr(
            P.dst, SE.getMulExpr(P.c, SE.getMinusSCEV(LB, P.lb0)));
        const SCEV *len = SE.getMulExpr(
            P.c, SE.getAddExpr(SE.getMinusSCEV(UB, LB), SE.getOne(I64)));
        // Expand everything before touching the CFG: SCEV, the expander and
        // their dominance information describe the function as it is now.
        SCEVExpander XB(SE, KI.DL, "gicc.chunk");
        Value *vPeer = XB.expandCodeFor(P.peer, I32, at);
        Value *vDst  = XB.expandCodeFor(dstBlk, Ptr, at);
        Value *vSrc  = XB.expandCodeFor(srcBlk, Ptr, at);
        Value *vLen  = XB.expandCodeFor(len, I64, at);
        // The element grain's fallback is a single-thread put: a parallel
        // region there would keep the device link from making the kernel
        // SPMD, and the per-block state-machine round trips cost ~8% at
        // compute-heavy sizes.
        FunctionCallee send = M.getOrInsertFunction(
            element ? "ompx__block_put_one" : "ompx__block_put",
            Type::getVoidTy(Ctx), I32, Ptr, Ptr, I64);
        CallInst *call =
            IRBuilder<>(at->getParent(), at).CreateCall(send, {vPeer, vDst, vSrc, vLen});
        if (mirror) {
            Instruction *then = SplitBlockAndInsertIfThen(
                IRBuilder<>(call).CreateNot(mirror), call->getIterator(), false);
            call->moveBefore(then->getIterator());
        }

        out() << "[gicc-chunk]   lowered: "
              << (element ? "element grain (store mirrored to the peer; one put "
                            "per block if the peer is not mapped)"
                          : "block grain (ompx__block_put after each block's "
                            "parallel region)")
              << "\n";
        P.put->eraseFromParent();
    }

    static const SCEV *extTo64(ScalarEvolution &SE, const Kernel &KI, Value *V) {
        Type *I64 = Type::getInt64Ty(V->getContext());
        const SCEV *S = SE.getSCEV(V);
        return KI.distSigned ? SE.getNoopOrSignExtend(S, I64)
                             : SE.getNoopOrZeroExtend(S, I64);
    }

    // Does the worker entry of the parallel region still call the outlined
    // function? If the body was inlined into it, rewriting the outlined
    // function alone would miss the code that runs.
    static bool wrapperCallsOutlined(const Kernel &KI) {
        auto *W = dyn_cast<Function>(KI.parallel->getArgOperand(6)->stripPointerCasts());
        if (!W) return false;
        for (const Instruction &I : instructions(*W))
            if (auto *CB = dyn_cast<CallBase>(&I); CB && CB->getCalledFunction() == KI.outlined)
                return true;
        return false;
    }

    // Kernel object behind a put's source pointer, with its constant
    // byte offset. Null when it is not argument + constant.
    Value *stripToObject(const Kernel &KI, Value *V, int64_t *off) {
        int64_t total = 0;
        for (unsigned depth = 0; depth < 16; ++depth) {
            APInt o(KI.DL.getIndexTypeSizeInBits(V->getType()), 0);
            Value *b = V->stripAndAccumulateConstantOffsets(KI.DL, o, true);
            total += o.getSExtValue();
            Value *r = KI.resolve(b);
            if (r == b) {
                if (off) *off = total;
                return b;
            }
            V = r;
        }
        return nullptr;
    }

    // Every path from the distribute loop to the put leaves it through
    // __kmpc_distribute_static_fini, and the loop cannot run again after
    // the put. A path that skips the loop entirely (zero trip count)
    // writes nothing and is fine.
    bool isAfterDistribute(const Kernel &KI, CallInst *CI, DominatorTree &DT,
                           LoopInfo &LI) {
        if (LI.getLoopFor(CI->getParent())) return false;
        if (isPotentiallyReachable(CI, KI.distInit, nullptr, &DT, &LI))
            return false;
        SmallPtrSet<BasicBlock *, 1> finiBlock{KI.distFini->getParent()};
        return !isPotentiallyReachable(KI.distInit->getParent(), CI->getParent(),
                                       &finiBlock, &DT, &LI);
    }

    // The original program guarantees the peer leaves dst alone for the
    // whole synchronization epoch the put lands in: with no ordering
    // between the ranks inside an epoch, the put may land at any moment of
    // it. Sending a block earlier but inside the same epoch keeps that
    // guarantee. Sending it before a synchronization the put came after
    // does not -- e.g. waiting for the peer's "done reading dst" signal.
    // So nothing between the first block's join and the put may order
    // this rank with another: no ompx_ call other than the put itself,
    // no fence, no atomic or volatile access.
    static std::string syncKind(const Instruction &I) {
        if (isa<FenceInst>(I)) return "fence";
        if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I))
            return "atomic read-modify-write";
        if (auto *LdI = dyn_cast<LoadInst>(&I); LdI && (LdI->isAtomic() || LdI->isVolatile()))
            return "atomic or volatile load";
        if (auto *SI = dyn_cast<StoreInst>(&I); SI && (SI->isAtomic() || SI->isVolatile()))
            return "atomic or volatile store";
        if (auto *CB = dyn_cast<CallBase>(&I)) {
            StringRef n = calleeName(*CB);
            if (n.starts_with("ompx_") && n != kPipelinedPut && n != "ompx_trigger")
                return n.str();
        }
        return "";
    }

    std::string syncBeforePut(const Kernel &KI, CallInst *put, DominatorTree &DT,
                              LoopInfo &LI) {
        // Anything in the parallel region runs after some block has joined.
        for (Instruction &I : instructions(*KI.outlined))
            if (std::string k = syncKind(I); !k.empty()) return k;
        for (Instruction &I : instructions(KI.K)) {
            if (&I == put) continue;
            std::string k = syncKind(I);
            if (k.empty()) continue;
            if (isPotentiallyReachable(KI.distInit, &I, nullptr, &DT, &LI) &&
                isPotentiallyReachable(&I, put, nullptr, &DT, &LI))
                return k;
        }
        return "";
    }

    bool isInOutlined(const Instruction *I, const Kernel &KI) {
        return I->getFunction() == KI.outlined;
    }

    // Any store or unaccounted call in the kernel body that could write obj.
    std::string writesOutsideLoop(const Kernel &KI, Value *obj) {
        for (Instruction &I : instructions(KI.K)) {
            if (auto *SI = dyn_cast<StoreInst>(&I)) {
                if (stripToObject(KI, SI->getPointerOperand(), nullptr) == obj)
                    return "'" + objName(obj) +
                           "' is written in the teams region outside the worksharing loop";
                continue;
            }
            auto *CB = dyn_cast<CallBase>(&I);
            if (CB && !isBenignCall(*CB) && calleeName(*CB) != kPlainPut)
                return ("call to '" + calleeName(*CB) +
                        "' in the teams region may write '").str() + objName(obj) + "'";
        }
        return "";
    }

    void report(CallInst *CI, StringRef verdict, const std::string &why) {
        if (lower && verdict == "ILLEGAL" && calleeName(*CI) == kPipelinedPut)
            CI->getContext().emitError(
                CI, "cannot pipeline ompx_pipelined_put: " + why);
        out() << "[gicc-chunk]   " << calleeName(*CI);
        if (const DebugLoc &DLc = CI->getDebugLoc())
            out() << " (line " << DLc.getLine() << ")";
        out() << ": " << verdict;
        if (!why.empty()) out() << ": " << why;
        out() << "\n";
    }
};

}  // namespace

PreservedAnalyses GICCChunkPrepPass::run(Module &M, ModuleAnalysisManager &) {
    if (getConfig().mode != Mode::ChunkLower) return PreservedAnalyses::all();
    if (!Triple(M.getTargetTriple()).isGPU()) return PreservedAnalyses::all();
    bool changed = false;
    for (Function &F : M) {
        bool hasPut = false;
        for (Instruction &I : instructions(F))
            hasPut |= isCallTo(I, kPipelinedPut);
        if (!hasPut) continue;
        for (Instruction &I : instructions(F)) {
            auto *CB = dyn_cast<CallBase>(&I);
            if (!CB || calleeName(*CB) != "__kmpc_parallel_51") continue;
            auto *O = dyn_cast<Function>(CB->getArgOperand(5)->stripPointerCasts());
            if (!O || O->isDeclaration() || O->hasFnAttribute(kPrepAttr)) continue;
            // Clang marks outlined regions alwaysinline; remember that so
            // the lowering can put it back.
            const bool always = O->hasFnAttribute(Attribute::AlwaysInline);
            O->removeFnAttr(Attribute::AlwaysInline);
            O->addFnAttr(Attribute::NoInline);
            O->addFnAttr(kPrepAttr, always ? "alwaysinline" : "");
            changed = true;
        }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

PreservedAnalyses GICCChunkAnalysisPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
    const Mode mode = getConfig().mode;
    if (mode != Mode::ChunkAnalyze && mode != Mode::ChunkLower)
        return PreservedAnalyses::all();
    if (!Triple(M.getTargetTriple()).isGPU()) return PreservedAnalyses::all();
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    Analyzer A(M, FAM, mode == Mode::ChunkLower);
    A.run();
    return A.modified() ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
