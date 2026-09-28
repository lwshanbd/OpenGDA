// Pipelined puts in OpenMP offload kernels.
//
// Input shape (a teams region, as clang emits it):
//
//   #pragma omp target teams
//   {
//       #pragma omp distribute parallel for [dist_schedule(static, C)]
//       for (i = lb0; i <= ub0; ++i)  src[i + d'] = ...;
//       ompx_pipelined_put(peer, dst, src + off, bytes);
//   }
//
// ompx_pipelined_put means "send these bytes once the kernel's writes to
// them are complete" -- the same as a host put after the kernel. It may be
// split into sends issued as the data becomes final iff
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
//        constants only), so each team can issue its own sends;
//   (P5) no synchronization lies between the first block's join and the
//        put, so every send still reaches the peer inside the same
//        synchronization epoch as the original put -- the epoch in which
//        the program already guarantees the peer leaves dst alone.
//
// A loop that writes the source as a multi-dimensional box (collapse(D),
// in any loop order) takes the box form instead of P1/P2: the put may be
// any contiguous range of the written object, dst - src must be the same
// in every team, and
//
//   (B1) every store to the object writes the same box, on every
//        iteration, so each word of the range is either stored exactly once
//        by the loop or not written by the kernel at all;
//   (B2) the range, the box and the conditions under which the loop runs
//        are kernel formals only, so the kernel can evaluate them before
//        the loop.
//
// P3-P5 hold as before, and the kernel checks at run time that the range
// and the box are aligned to the store size. The lowering is always element
// grain: each store
// that lands in the range is repeated into the peer's copy as it happens,
// and the bytes of the range the box does not cover -- final before the
// kernel starts, and sent by no mirrored store -- are copied once at its
// start (ompx__box_residual). The peer must be same-node.
//
// The put may also be stated inside the worksharing loop's body, so that
// the kernel can be a combined `target teams distribute parallel for`,
// which clang emits in SPMD mode; a marker after the loop keeps the teams
// region generic, and the kernel is then slower even with nothing else in
// it. Such a put runs on every iteration with the same arguments, and means
// the same as one after the loop in a loop that runs at least once: it
// takes the box form whatever the dimension of the write.
//
// GICC_MODE=chunk-analyze reports the verdict; chunk-lower also rewrites
// each proven put at the grain GICC_CHUNK_GRAIN selects (see lowerPut) and
// makes an unprovable one a compile error. GICCChunkPrepPass keeps the
// outlined region out of its wrapper until then, so an element-grain
// rewrite reaches the code the workers run.
//
// Trusted, not proven: the runtime and codegen contract in OmpKernel.h.
// Distinct kernel pointer arguments are assumed not to overlap
// (is_device_ptr gives no noalias guarantee).
//
// A plain ompx_put in the same position is diagnosed as a race: every team
// runs it once, after only its own blocks are written.

#include "GICCChunkAnalysis.h"
#include "AccessDecomposition.h"
#include "GICCPassConfig.h"
#include "OmpKernel.h"
#include "WriteSet.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include <optional>
#include <string>

using namespace llvm;

namespace gicc::pass {

namespace {

raw_ostream &out() { return errs(); }

// Attribute marking the outlined regions GICCChunkPrepPass kept out of
// their wrappers; its value records whether they were alwaysinline.
constexpr StringLiteral kPrepAttr = "gicc-chunk-noinline";

enum class Verdict { Legal, Illegal, Race };

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

// Is V (a value passed in slot 0/1) the current block bound derived from
// `slot`? Accepts the load itself, its clamp against ub0 (`clamp`, the value
// stored as the upper distribute bound), the advance by the stride, and
// phis over those.
bool isBlockBound(Value *V, const Value *slot, const Value *strideSlot, Value *clamp,
                  SmallPtrSetImpl<Value *> &seen) {
    V = stripIntCasts(V);
    if (!seen.insert(V).second) return true;
    if (auto *LI = dyn_cast<LoadInst>(V))
        return LI->getPointerOperand()->stripPointerCasts() == slot;
    if (auto *MM = dyn_cast<MinMaxIntrinsic>(V)) {
        Value *c = stripIntCasts(clamp);
        for (auto [bound, other] : {std::pair(MM->getLHS(), MM->getRHS()),
                                    std::pair(MM->getRHS(), MM->getLHS())})
            if (stripIntCasts(other) == c)
                return isBlockBound(bound, slot, strideSlot, clamp, seen);
        return false;
    }
    if (auto *BO = dyn_cast<BinaryOperator>(V)) {
        if (BO->getOpcode() != Instruction::Add) return false;
        for (unsigned k = 0; k < 2; ++k) {
            auto *LI = dyn_cast<LoadInst>(BO->getOperand(1 - k));
            if (LI && LI->getPointerOperand()->stripPointerCasts() == strideSlot)
                return isBlockBound(BO->getOperand(k), slot, strideSlot, clamp, seen);
        }
        return false;
    }
    if (auto *PN = dyn_cast<PHINode>(V)) {
        for (Value *in : PN->incoming_values())
            if (!isBlockBound(in, slot, strideSlot, clamp, seen)) return false;
        return true;
    }
    return false;
}

// Kernel object behind a put's source pointer, with its constant byte
// offset. Null when it is not argument + constant.
Value *stripToObject(const OmpKernel &KI, Value *V, int64_t *off) {
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
// __kmpc_distribute_static_fini, and the loop cannot run again after the
// put. A path that skips the loop entirely (zero trip count) writes nothing
// and is fine.
bool isAfterDistribute(const OmpKernel &KI, CallInst *CI, DominatorTree &DT, LoopInfo &LI) {
    if (LI.getLoopFor(CI->getParent())) return false;
    if (isPotentiallyReachable(CI, KI.distInit, nullptr, &DT, &LI)) return false;
    // The put in the fini call's own block (a loop whose trip count folded
    // away): every path to it runs the fini first iff the fini comes first.
    if (CI->getParent() == KI.distFini->getParent()) return KI.distFini->comesBefore(CI);
    SmallPtrSet<BasicBlock *, 1> finiBlock{KI.distFini->getParent()};
    return !isPotentiallyReachable(KI.distInit->getParent(), CI->getParent(), &finiBlock,
                                   &DT, &LI);
}

// The original program guarantees the peer leaves dst alone for the whole
// synchronization epoch the put lands in: with no ordering between the
// ranks inside an epoch, the put may land at any moment of it. Sending a
// block earlier but inside the same epoch keeps that guarantee. Sending it
// before a synchronization the put came after does not -- e.g. waiting for
// the peer's "done reading dst" signal. So nothing between the first
// block's join and the put may order this rank with another: no ompx_ call
// other than the put itself and ompx_trigger (which only releases staged
// work), no fence, no atomic or volatile access.
std::string syncKind(const Instruction &I) {
    if (isa<FenceInst>(I)) return "fence";
    if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I)) return "atomic read-modify-write";
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

std::string syncBeforePut(const OmpKernel &KI, CallInst *put, DominatorTree &DT,
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

// Any store or unaccounted call in the kernel body that could write obj. A
// store may write obj unless every object it may point to is provably
// another one: a different kernel formal (formals are assumed not to
// overlap), a stack slot, or a global.
std::string writesOutsideLoop(const OmpKernel &KI, Value *obj) {
    for (Instruction &I : instructions(KI.K)) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
            SmallVector<const Value *, 4> objs;
            getUnderlyingObjects(SI->getPointerOperand(), objs);
            for (const Value *o : objs) {
                Value *r = KI.resolve(const_cast<Value *>(o));
                if (r == obj)
                    return "'" + objName(obj) +
                           "' is written in the teams region outside the worksharing loop";
                if (!isa<Argument>(r) && !isa<AllocaInst>(r) && !isa<GlobalVariable>(r))
                    return "a store in the teams region may write '" + objName(obj) + "'";
            }
            continue;
        }
        auto *CB = dyn_cast<CallBase>(&I);
        if (CB && !isBenignCall(*CB) && calleeName(*CB) != kPlainPut)
            return ("call to '" + calleeName(*CB) + "' in the teams region may write '")
                       .str() + objName(obj) + "'";
    }
    return "";
}

// Does the worker entry of the parallel region still call the outlined
// function? If the body was inlined into it, rewriting the outlined
// function alone would miss the code that runs.
bool wrapperCallsOutlined(const OmpKernel &KI) {
    // No wrapper (an SPMD kernel): the runtime calls the region directly.
    if (isa<ConstantPointerNull>(KI.parallel->getArgOperand(kmpc::ParallelWrapper)))
        return true;
    auto *W = dyn_cast<Function>(
        KI.parallel->getArgOperand(kmpc::ParallelWrapper)->stripPointerCasts());
    if (!W) return false;
    for (const Instruction &I : instructions(*W))
        if (auto *CB = dyn_cast<CallBase>(&I); CB && CB->getCalledFunction() == KI.outlined)
            return true;
    return false;
}

std::string stridedMessage(Value *obj, const std::string &stride, uint64_t storeSize) {
    return "writes to '" + objName(obj) + "' are strided (" + stride +
           " bytes per iteration, " + std::to_string(storeSize) +
           "-byte stores): a block does not own a contiguous range";
}

// A put proven splittable, with everything needed to emit its sends. SCEVs
// are in terms of kernel formals only (lb0 too), so they can be expanded
// anywhere in the kernel.
struct Plan {
    CallInst   *put;
    Value      *srcObj;   // kernel formal the put reads
    const SCEV *c, *d;    // block [LB,UB] owns srcObj + c*LB + d, c*(UB-LB+1) bytes
    const SCEV *lb0;      // first distribute iteration (i64)
    const SCEV *peer, *dst;
    // Some write covers the range on every iteration. The element grain
    // sends only what is stored, so it needs this; the block grain sends
    // whole blocks and does not.
    bool covered;
    // Box form (a multi-dimensional write): the range [srcObj + rangeOff,
    // + bytes), the box every store writes, and the loop's run conditions.
    bool box = false;
    const SCEV *rangeOff = nullptr, *bytes = nullptr, *boxOffset = nullptr;
    SmallVector<const SCEV *, 4> boxStride, boxExtent;
    SmallVector<EntryFact, 4> runs;
    uint64_t elem = 0;
};

// The most box puts one kernel sends by count (ompx__box_plan's kMaxPuts).
constexpr unsigned kMaxCountedPuts = 8;

class ChunkAnalyzer {
    Module &M;
    FunctionAnalysisManager &FAM;
    bool lower;
    bool changed = false;

public:
    ChunkAnalyzer(Module &M, FunctionAnalysisManager &FAM, bool lower)
        : M(M), FAM(FAM), lower(lower) {}

    bool run() {
        for (Function &F : M)
            if (isOffloadKernel(F)) analyzeKernel(F);
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
        return changed;
    }

private:
    // What the kernel-level counting (lowerBoxCounting) needs of one box
    // put, expanded before the distribute loop.
    struct BoxSend {
        Value *peer, *dst, *src, *len, *box, *stride, *extent;
        Value *counted;   // i1: the kernel sends it by count
        unsigned dims;
        uint64_t elem;
    };

    // State shared by the per-put checks of one kernel.
    struct KernelFacts {
        const OmpKernel &KI;
        const WriteSet &W;
        ScalarEvolution &SE;
        DominatorTree &DT;
        PostDominatorTree &PDT;
        LoopInfo &LI;
        StoreInst *lbSt, *ubSt;
    };

    void analyzeKernel(Function &F) {
        SmallVector<CallInst *, 4> puts, plain;
        for (Instruction &I : instructions(F)) {
            if (isCallTo(I, kPipelinedPut)) puts.push_back(cast<CallInst>(&I));
            else if (auto *CB = dyn_cast<CallBase>(&I); CB && calleeName(*CB) == kPlainPut)
                plain.push_back(cast<CallInst>(&I));
        }
        // Markers in the worksharing loop's body, found through the kernel's
        // parallel region.
        std::string why;
        auto KI = OmpKernel::locate(F, why);
        SmallVector<CallInst *, 4> loopPuts;
        if (KI)
            for (Instruction &I : instructions(*KI->outlined))
                if (isCallTo(I, kPipelinedPut)) loopPuts.push_back(cast<CallInst>(&I));
        if (puts.empty() && plain.empty() && loopPuts.empty()) return;

        out() << "[gicc-chunk] kernel " << F.getName() << "\n";
        if (!KI) {
            for (CallInst *CI : puts) report(CI, Verdict::Illegal, why);
            return;
        }
        auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
        auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
        auto &LI = FAM.getResult<LoopAnalysis>(F);

        // Checked first: it is the one condition about the peer rather than
        // this rank, and its message should not be masked by the generic
        // "call may write memory" of the checks below.
        SmallVector<CallInst *, 4> unsynced;
        for (CallInst *CI : puts) {
            if (std::string sync = syncBeforePut(*KI, CI, DT, LI); !sync.empty())
                report(CI, Verdict::Illegal,
                       "synchronization (" + sync + ") between the first block and "
                       "the put: early blocks would reach the peer before it");
            else
                unsynced.push_back(CI);
        }
        puts = std::move(unsynced);
        // A put in the loop's body stands for one when the loop ends.
        if (!loopPuts.empty())
            if (std::string sync = syncBeforePut(*KI, KI->distFini, DT, LI); !sync.empty()) {
                for (CallInst *CI : loopPuts)
                    report(CI, Verdict::Illegal,
                           "synchronization (" + sync + ") in the loop: early words would "
                           "reach the peer before it");
                loopPuts.clear();
            }
        if (puts.empty() && plain.empty() && loopPuts.empty()) return;

        auto W = collectWrites(*KI, FAM, UnknownStore::Fail, why);
        if (!W) {
            for (CallInst *CI : puts) report(CI, Verdict::Illegal, why);
            for (CallInst *CI : loopPuts) report(CI, Verdict::Illegal, why);
            return;
        }

        auto bounds = KI->distBounds(DT);
        Value *slot0 = KI->slotValue(0), *slot1 = KI->slotValue(1);
        const Value *strideSlot = KI->distSlot(kmpc::InitStride);
        SmallPtrSet<Value *, 8> seen0, seen1;
        if (!bounds || !slot0 || !slot1 ||
            !isBlockBound(slot0, KI->distSlot(kmpc::InitLower), strideSlot,
                          bounds->second->getValueOperand(), seen0) ||
            !isBlockBound(slot1, KI->distSlot(kmpc::InitUpper), strideSlot,
                          bounds->second->getValueOperand(), seen1)) {
            for (CallInst *CI : puts)
                report(CI, Verdict::Illegal, "cannot identify the distribute block bounds");
            if (loopPuts.empty()) return;
            puts.clear();   // a put in the loop needs no block bounds
        }

        for (CallInst *CI : plain) {
            Value *obj = stripToObject(*KI, CI->getArgOperand(2), nullptr);
            if (obj && W->storesTo.count(obj) && CI->getFunction() != KI->outlined)
                report(CI, Verdict::Race,
                       "ompx_put in the teams region runs once per team, after "
                       "only that team's blocks of '" + objName(obj) +
                       "' are written; use ompx_pipelined_put or put after the kernel");
        }

        auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
        KernelFacts KF{*KI, *W, SE, DT, PDT, LI,
                       bounds ? bounds->first : nullptr, bounds ? bounds->second : nullptr};
        SmallVector<Plan, 2> plans;
        for (CallInst *CI : puts)
            if (auto P = checkPut(KF, CI)) {
                report(CI, Verdict::Legal, "");
                printPlan(KF, *P);
                plans.push_back(*P);
            }
        for (CallInst *CI : loopPuts)
            if (auto P = checkLoopPut(KF, CI)) {
                report(CI, Verdict::Legal, "");
                printPlan(KF, *P);
                plans.push_back(*P);
            }
        if (!lower) return;
        // A box put to a peer that is not IPC-mapped is sent by the kernel
        // itself once its range is written, when the kernel can count the
        // teams past it: SPMD, a chunked distribute loop over 64-bit bounds.
        // Otherwise the next quiet sends it.
        const unsigned nBox = llvm::count_if(plans, [](const Plan &P) { return P.box; });
        const bool canCount =
            nBox > 0 && nBox <= kMaxCountedPuts && KI->spmd() && bounds &&
            cast<ConstantInt>(KI->distInit->getArgOperand(kmpc::InitSched))->getSExtValue() ==
                kmpc::SchedDistChunked &&
            KI->distInit->getArgOperand(kmpc::InitChunk)->getType()->isIntegerTy(64) &&
            bounds->first->getValueOperand()->getType()->isIntegerTy(64) &&
            slotStores(*KI).has_value();
        // Expand every plan before any CFG edit: SCEV, the expanders and the
        // dominator tree all describe the function as it is now.
        SmallVector<std::pair<CallInst *, Value *>, 2> fallbacks;
        SmallVector<BoxSend, 2> sends;
        for (const Plan &P : plans)
            if (auto fb = lowerPut(*KI, *W, P, SE, canCount, sends)) fallbacks.push_back(*fb);
        for (auto [call, mirror] : fallbacks) {
            Instruction *then = SplitBlockAndInsertIfThen(
                IRBuilder<>(call).CreateNot(mirror), call->getIterator(), false);
            call->moveBefore(*then->getParent(), then->getIterator());
        }
        if (canCount) {
            lowerBoxCounting(*KI, bounds->first, bounds->second, sends);
            out() << "[gicc-chunk]   a peer that is not IPC-mapped is sent to by the kernel, "
                     "once every team is past the range (chunks rotated to store it first)\n";
        } else if (nBox > 0) {
            out() << "[gicc-chunk]   a peer that is not IPC-mapped is sent to by the next "
                     "ompx_quiet\n";
        }
        if (!plans.empty()) {
            changed = true;
            if (verifyFunction(F, &errs()) || verifyFunction(*KI->outlined, &errs()))
                report_fatal_error("gicc-chunk: lowering produced invalid IR");
        }
    }

    // The put's length as it is when the distribute loop ran. A phi merging
    // in a zero-trip path is read on the loop side only, provided the
    // zero-trip side is 0 (clang folds n * size to 0 there): the lowered
    // form sends nothing on that path. Null when the zero-trip side sends
    // something.
    Value *onLoopPath(const KernelFacts &KF, Value *V) {
        auto *PN = dyn_cast<PHINode>(V);
        if (!PN) return V;
        Value *pick = nullptr;
        for (unsigned k = 0; k < PN->getNumIncomingValues(); ++k) {
            if (!isPotentiallyReachable(KF.KI.distInit->getParent(), PN->getIncomingBlock(k),
                                        nullptr, &KF.DT, &KF.LI)) {
                auto *C = dyn_cast<ConstantInt>(KF.KI.resolve(PN->getIncomingValue(k)));
                if (!C || !C->isZero()) return nullptr;
                continue;
            }
            Value *r = KF.KI.resolve(PN->getIncomingValue(k));
            if (pick && r != pick) return V;
            pick = r;
        }
        return pick ? pick : V;
    }

    // P1-P4 for one put (P5 was checked before the writes were collected).
    std::optional<Plan> checkPut(const KernelFacts &KF, CallInst *CI) {
        const OmpKernel &KI = KF.KI;
        ScalarEvolution &SE = KF.SE;
        auto illegal = [&](const std::string &why) {
            report(CI, Verdict::Illegal, why);
            return std::nullopt;
        };
        if (!isAfterDistribute(KI, CI, KF.DT, KF.LI))
            return illegal("put is not after the distribute loop");
        if (KF.LI.getLoopFor(KI.distInit->getParent()))
            return illegal("the distribute loop is inside a sequential loop of the teams "
                           "region: its blocks would be sent once per trip");
        // Sends issued as blocks complete assume the put happens: it must
        // run on every path through the loop.
        if (!KF.PDT.dominates(CI->getParent(), KI.distInit->getParent()))
            return illegal("the put does not run on every path after the distribute loop "
                           "(it is conditional); guard the whole kernel instead");
        if (auto src = symbolicSource(KF, CI->getArgOperand(2)))
            if (llvm::any_of(KF.W.boxes, [&](const BoxWrite &b) {
                    return b.obj == src->first && b.stride.size() > 1;
                }))
                return checkBoxPut(KF, CI, src->first, src->second);
        int64_t off = 0;
        Value *obj = stripToObject(KI, CI->getArgOperand(2), &off);
        if (!obj || !isa<Argument>(obj))
            return illegal("put source is not a kernel argument plus a constant");
        // Every write to the source is the same dense 1-D range.
        const BoxWrite *first = nullptr;
        bool covered = false;   // some write runs on every iteration
        for (const BoxWrite &b : KF.W.boxes) {
            if (b.obj != obj) continue;
            if (b.stride.size() != 1)
                return illegal("writes to '" + objName(obj) +
                               "' are multi-dimensional (collapse); "
                               "ompx_pipelined_put splits 1-D blocks only");
            auto *cC = dyn_cast<SCEVConstant>(b.stride[0]);
            if (!cC || cC->getAPInt().getZExtValue() != b.storeSize)
                return illegal(stridedMessage(obj, scevStr(b.stride[0]), b.storeSize));
            if (first && (!sameSCEV(SE, b.stride[0], first->stride[0]) ||
                          !sameSCEV(SE, b.offset, first->offset)))
                return illegal("object '" + objName(obj) +
                               "' is written with more than one access pattern");
            first = first ? first : &b;
            covered |= b.unconditional;
        }
        if (!first)
            return illegal("put source '" + objName(obj) +
                           "' is not written by the worksharing loop");
        if (std::string why = writesOutsideLoop(KI, obj); !why.empty()) return illegal(why);

        // Trip count in the bounds' own width. It equals the true count only
        // if ub0 - lb0 + 1 does not wrap there; prove that where the loop
        // actually starts (a zero-trip path writes nothing).
        Type *I64 = Type::getInt64Ty(CI->getContext());
        const SCEV *lb0n = KI.normalize(SE, SE.getSCEV(KF.lbSt->getValueOperand()));
        const SCEV *ub0n = KI.normalize(SE, SE.getSCEV(KF.ubSt->getValueOperand()));
        const SCEV *span = SE.getMinusSCEV(ub0n, lb0n);
        const SCEV *one  = SE.getOne(span->getType());
        if (KF.W.ubPlusOne &&
            !SE.willNotOverflow(Instruction::Add, KI.distSigned, ub0n, one, KI.distInit))
            return illegal("cannot prove the block bound " + scevStr(ub0n) +
                           " + 1 used by the loop exit does not wrap");
        if (!SE.willNotOverflow(Instruction::Sub, KI.distSigned, ub0n, lb0n, KI.distInit) ||
            !SE.willNotOverflow(Instruction::Add, KI.distSigned, span, one, KI.distInit))
            return illegal("cannot prove the distribute trip count " + scevStr(ub0n) +
                           " - " + scevStr(lb0n) + " + 1 does not wrap");
        const SCEV *lb0 = KI.extDist(SE, lb0n, I64);
        const SCEV *c   = first->stride[0];
        const SCEV *d   = first->offset;
        const SCEV *wantOff   = SE.getAddExpr(SE.getMulExpr(c, lb0), d);
        const SCEV *wantBytes = SE.getMulExpr(c, KI.extDist(SE, SE.getAddExpr(span, one), I64));
        Value *len = onLoopPath(KF, CI->getArgOperand(3));
        if (!len) return illegal("the put sends bytes on the path that skips the loop");
        const SCEV *bytes = KI.normalize(SE, SE.getSCEV(len));
        const SCEV *offS  = SE.getConstant(I64, off, true);
        if (!sameSCEV(SE, offS, wantOff) || !sameSCEV(SE, bytes, wantBytes))
            return illegal("put range [src+" + scevStr(offS) + ", +" + scevStr(bytes) +
                           ") is not the loop's write range [src+" + scevStr(wantOff) +
                           ", +" + scevStr(wantBytes) + ")");
        const SCEV *peer = KI.normalize(SE, SE.getSCEV(CI->getArgOperand(0)));
        const SCEV *dst  = KI.normalize(SE, SE.getSCEV(CI->getArgOperand(1)));
        if (!onlyFormals(peer, KI.K) || !onlyFormals(dst, KI.K))
            return illegal("peer or dst differs between teams");
        return Plan{CI, obj, c, d, lb0, peer, dst, covered};
    }

    // The kernel formal a pointer points into and its byte offset from it,
    // symbolic; nullopt when it is not a formal plus an offset.
    std::optional<std::pair<Value *, const SCEV *>> symbolicSource(const KernelFacts &KF,
                                                                  Value *ptr) {
        ScalarEvolution &SE = KF.SE;
        const SCEV *S = KF.KI.normalize(SE, SE.getSCEV(ptr));
        auto *base = dyn_cast<SCEVUnknown>(SE.getPointerBase(S));
        if (!base) return std::nullopt;
        Value *obj = KF.KI.resolve(base->getValue());
        if (!isa<Argument>(obj)) return std::nullopt;
        const SCEV *off = SE.removePointerBase(S);
        if (isa<SCEVCouldNotCompute>(off)) return std::nullopt;
        return std::make_pair(obj, off);
    }

    static bool sameBox(ScalarEvolution &SE, const BoxWrite &a, const BoxWrite &b) {
        if (a.storeSize != b.storeSize || a.stride.size() != b.stride.size() ||
            !sameSCEV(SE, a.offset, b.offset))
            return false;
        for (size_t d = 0; d < a.stride.size(); ++d)
            if (!sameSCEV(SE, a.stride[d], b.stride[d]) || !sameSCEV(SE, a.extent[d], b.extent[d]))
                return false;
        return true;
    }

    // B1, B2 and P4 for a put of an object the loop writes as a box (P3 and
    // P5 as in checkPut).
    std::optional<Plan> checkBoxPut(const KernelFacts &KF, CallInst *CI, Value *obj,
                                    const SCEV *off, const SCEV *bytes = nullptr,
                                    const SCEV *peer = nullptr, const SCEV *dst = nullptr) {
        const OmpKernel &KI = KF.KI;
        ScalarEvolution &SE = KF.SE;
        auto illegal = [&](const std::string &why) {
            report(CI, Verdict::Illegal, why);
            return std::nullopt;
        };
        const std::string name = "'" + objName(obj) + "'";
        const BoxWrite *first = nullptr;
        bool covered = false;
        for (const BoxWrite &b : KF.W.boxes) {
            if (b.obj != obj) continue;
            if (first && !sameBox(SE, *first, b))
                return illegal("object " + name + " is written with more than one access pattern");
            first = first ? first : &b;
            covered |= b.unconditional;
        }
        if (!covered)
            return illegal("no store to " + name + " runs on every iteration: a put of a "
                           "multi-dimensional write sends each word as it is stored, so the "
                           "loop must store every point of its box");
        if (std::string why = writesOutsideLoop(KI, obj); !why.empty()) return illegal(why);
        if (lower && !wrapperCallsOutlined(KI))
            return illegal("the parallel region is inlined into its wrapper (GICCChunkPrepPass "
                           "did not run): its stores cannot be mirrored");
        const uint64_t elem = first->storeSize;
        if (!isPowerOf2_64(elem))
            return illegal("the stores to " + name + " are " + std::to_string(elem) +
                           " bytes, not a power of two");
        // That the range and the box are aligned to the stores (so none
        // straddles the range's edge) is checked at run time, by
        // ompx__box_residual: a length is often a formal SCEV knows nothing of.
        if (!bytes) bytes = KI.normalize(SE, SE.getSCEV(CI->getArgOperand(3)));
        // Everything the kernel evaluates before the loop must be the same in
        // every team and on every path: formals and constants only.
        if (!onlyFormals(off, KI.K) || !onlyFormals(bytes, KI.K))
            return illegal("the put range [src+" + scevStr(off) + ", +" + scevStr(bytes) +
                           ") is not made of kernel formals");
        bool boxFormal = onlyFormals(first->offset, KI.K);
        for (size_t d = 0; d < first->stride.size(); ++d)
            boxFormal &= onlyFormals(first->stride[d], KI.K) &&
                         onlyFormals(first->extent[d], KI.K);
        if (!boxFormal)
            return illegal("the box " + name + " is written in is not made of kernel formals");
        for (const EntryFact &f : KF.W.preconditions)
            if (!onlyFormals(f.x, KI.K))
                return illegal("the condition " + scevStr(f.x) + " under which the loop runs "
                               "is not made of kernel formals");
        if (!peer) peer = KI.normalize(SE, SE.getSCEV(CI->getArgOperand(0)));
        if (!dst) dst = KI.normalize(SE, SE.getSCEV(CI->getArgOperand(1)));
        if (!onlyFormals(peer, KI.K) || !onlyFormals(dst, KI.K))
            return illegal("peer or dst differs between teams");
        Plan P{CI, obj, nullptr, nullptr, nullptr, peer, dst, covered};
        P.box = true;
        P.rangeOff = off;
        P.bytes = bytes;
        P.boxOffset = first->offset;
        P.boxStride.assign(first->stride.begin(), first->stride.end());
        P.boxExtent.assign(first->extent.begin(), first->extent.end());
        P.runs.assign(KF.W.preconditions.begin(), KF.W.preconditions.end());
        P.elem = elem;
        return P;
    }

    // A put in the worksharing loop's body: it must run on every iteration
    // with arguments that are kernel values, the same in every team; it then
    // takes the box form.
    std::optional<Plan> checkLoopPut(const KernelFacts &KF, CallInst *CI) {
        const OmpKernel &KI = KF.KI;
        ScalarEvolution &SE = KF.SE;
        auto illegal = [&](const std::string &why) {
            report(CI, Verdict::Illegal, why);
            return std::nullopt;
        };
        Function &O = *KI.outlined;
        auto &SO = FAM.getResult<ScalarEvolutionAnalysis>(O);
        auto &LO = FAM.getResult<LoopAnalysis>(O);
        auto &DO = FAM.getResult<DominatorTreeAnalysis>(O);
        Loop *L = LO.getLoopFor(CI->getParent());
        if (!L) return illegal("the put is in the parallel region but not in its loop");
        if (L->getParentLoop())
            return illegal("the put is in a loop nested in the worksharing loop");
        BasicBlock *latch = L->getLoopLatch();
        if (!latch || !DO.dominates(CI->getParent(), latch))
            return illegal("the put does not run on every iteration (it is conditional)");
        const SCEV *v[4];
        for (unsigned i = 0; i < 4; ++i) {
            v[i] = KI.toKernel(SO.getSCEV(CI->getArgOperand(i)), SE);
            if (!v[i]) return illegal("argument " + std::to_string(i) + " of the put is not a "
                                      "value of the kernel");
            v[i] = KI.normalize(SE, v[i]);
        }
        auto *base = dyn_cast<SCEVUnknown>(SE.getPointerBase(v[2]));
        Value *obj = base ? KI.resolve(base->getValue()) : nullptr;
        if (!obj || !isa<Argument>(obj))
            return illegal("put source is not a kernel argument plus an offset");
        const SCEV *off = SE.removePointerBase(v[2]);
        if (isa<SCEVCouldNotCompute>(off))
            return illegal("put source is not a kernel argument plus an offset");
        if (llvm::none_of(KF.W.boxes, [&](const BoxWrite &b) { return b.obj == obj; }))
            return illegal("put source '" + objName(obj) +
                           "' is not written by the worksharing loop");
        return checkBoxPut(KF, CI, obj, off, v[3], v[0], v[1]);
    }

    void printPlan(const KernelFacts &KF, const Plan &P) {
        if (P.box) {
            auto list = [](const SmallVectorImpl<const SCEV *> &v) {
                std::string r = "[";
                for (size_t i = 0; i < v.size(); ++i) r += (i ? ", " : "") + scevStr(v[i]);
                return r + "]";
            };
            out() << "[gicc-chunk]   box: " << P.boxStride.size() << "-D extent="
                  << list(P.boxExtent) << " stride=" << list(P.boxStride)
                  << " offset=" << scevStr(P.boxOffset) << "\n"
                  << "[gicc-chunk]   range: [" << objName(P.srcObj) << " + "
                  << scevStr(P.rangeOff) << ", + " << scevStr(P.bytes)
                  << "): stores inside it are mirrored, the rest is sent at kernel start\n";
            return;
        }
        const OmpKernel &KI = KF.KI;
        ScalarEvolution &SE = KF.SE;
        Type *I64 = Type::getInt64Ty(KI.K.getContext());
        const SCEV *ub0 =
            KI.extDist(SE, KI.normalize(SE, SE.getSCEV(KF.ubSt->getValueOperand())), I64);
        int64_t sched =
            cast<ConstantInt>(KI.distInit->getArgOperand(kmpc::InitSched))->getSExtValue();
        out() << "[gicc-chunk]   blocks: distribute sched=" << sched;
        if (sched == kmpc::SchedDistChunked)
            out() << " chunk=" << scevStr(SE.getSCEV(KI.distInit->getArgOperand(kmpc::InitChunk)));
        out() << " over [" << scevStr(P.lb0) << ", " << scevStr(ub0) << "]\n"
              << "[gicc-chunk]   block [LB,UB]: put(peer, dst + " << scevStr(P.c)
              << "*(LB - " << scevStr(P.lb0) << "), " << objName(P.srcObj) << " + "
              << scevStr(P.c) << "*LB + " << scevStr(P.d) << ", " << scevStr(P.c)
              << "*(UB - LB + 1)) after the block's parallel region joins\n";
    }

    // Replace the put with sends issued as the data becomes final, at the
    // grain GICC_CHUNK_GRAIN selects:
    //
    //   block    one ompx__block_put per block, right after the block's
    //            __kmpc_parallel_51 returns: the region has joined, so all
    //            of the block's words are written.
    //   element  every store to the source is followed by the same store
    //            into a same-node peer's copy, so each word leaves with the
    //            value it is stored with. The peer address comes from
    //            ompx__peer_addr, once per team; when it is null (peer not
    //            IPC-mapped) the kernel falls back at run time to one
    //            single-thread ompx_put per block (ompx__block_put_one),
    //            which keeps the kernel SPMD-able.
    //
    // On a zero-trip path nothing is sent; checkPut made sure the original
    // put sent 0 bytes there.
    //
    // Returns the element grain's per-block fallback call and the flag it
    // must be guarded by (run it only when the peer is not mapped); the
    // caller splits the block once every plan is expanded.
    std::optional<std::pair<CallInst *, Value *>>
    lowerPut(const OmpKernel &KI, const WriteSet &W, const Plan &P, ScalarEvolution &SE,
             bool canCount, SmallVectorImpl<BoxSend> &sends) {
        if (P.box) {
            sends.push_back(lowerBoxPut(KI, W, P, SE, canCount));
            out() << "[gicc-chunk]   lowered: element grain over the box (stores in the "
                     "range mirrored to the peer, the rest sent at kernel start)\n";
            P.put->eraseFromParent();
            return std::nullopt;
        }
        bool element = getConfig().chunkGrain == ChunkGrain::Element;
        if (element && !P.covered) {
            out() << "[gicc-chunk]   no write covers the range on every iteration, and "
                     "the element grain sends only what is stored: using block grain\n";
            element = false;
        }
        if (element && !wrapperCallsOutlined(KI)) {
            out() << "[gicc-chunk]   the parallel region is inlined into its wrapper "
                     "(GICCChunkPrepPass did not run): using block grain\n";
            element = false;
        }
        Value *mirror = element ? emitElementMirror(KI, W, P, SE) : nullptr;
        CallInst *send = emitBlockSend(KI, P, SE, mirror);
        out() << "[gicc-chunk]   lowered: "
              << (element ? "element grain (store mirrored to the peer; one put "
                            "per block if the peer is not mapped)"
                          : "block grain (ompx__block_put after each block's "
                            "parallel region)")
              << "\n";
        P.put->eraseFromParent();
        if (!mirror) return std::nullopt;
        return std::make_pair(send, mirror);
    }

    // Element grain: the peer address, once per team before the loop, and a
    // mirrored store after each store to the source. Returns the kernel's
    // "peer is mapped" flag.
    Value *emitElementMirror(const OmpKernel &KI, const WriteSet &W, const Plan &P,
                             ScalarEvolution &SE) {
        LLVMContext &Ctx = M.getContext();
        Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx), *I8 = Type::getInt8Ty(Ctx);
        auto *Ptr = cast<PointerType>(P.put->getArgOperand(1)->getType());
        // Where the source's first byte lands in the peer, as a byte delta
        // from the local address.
        BasicBlock::iterator pre = KI.distInit->getIterator();
        const SCEV *srcStart = SE.getAddExpr(
            SE.getSCEV(P.srcObj), SE.getAddExpr(SE.getMulExpr(P.c, P.lb0), P.d));
        SCEVExpander X(SE, KI.DL, "gicc.chunk");
        Value *peerV = X.expandCodeFor(P.peer, I32, pre);
        Value *dstV  = X.expandCodeFor(P.dst, Ptr, pre);
        Value *srcV  = X.expandCodeFor(srcStart, Ptr, pre);
        IRBuilder<> B(KI.distInit);
        FunctionCallee peerAddr = M.getOrInsertFunction("ompx__peer_addr", Ptr, I32, Ptr);
        Value *pd = B.CreateCall(peerAddr, {peerV, dstV});
        Value *mirror = B.CreateICmpNE(pd, ConstantPointerNull::get(Ptr));
        Value *delta = B.CreateSub(B.CreatePtrToInt(pd, I64), B.CreatePtrToInt(srcV, I64));
        // Team-shared, like clang's own globalized captures: the main thread
        // writes them before any parallel region starts.
        auto shared = [&](Type *T, const char *name) {
            return new GlobalVariable(M, T, false, GlobalValue::InternalLinkage,
                                      PoisonValue::get(T), name, nullptr,
                                      GlobalValue::NotThreadLocal, kmpc::SharedAddrSpace);
        };
        GlobalVariable *gDelta  = shared(I64, "gicc.chunk.delta");
        GlobalVariable *gMirror = shared(I8, "gicc.chunk.mirror");
        B.CreateStore(delta, gDelta);
        B.CreateStore(B.CreateZExt(mirror, I8), gMirror);

        Function &O = *KI.outlined;
        IRBuilder<> BO(&*O.getEntryBlock().getFirstInsertionPt());
        Value *oDelta  = BO.CreateLoad(I64, gDelta, "gicc.chunk.delta");
        Value *oMirror = BO.CreateICmpNE(BO.CreateLoad(I8, gMirror), ConstantInt::get(I8, 0));
        for (StoreInst *St : W.storesTo.lookup(P.srcObj)) {
            IRBuilder<> BS(St->getNextNode());
            Value *remote = BS.CreateGEP(I8, St->getPointerOperand(), oDelta);
            Value *target = BS.CreateSelect(oMirror, remote, St->getPointerOperand());
            StoreInst *Copy = BS.CreateAlignedStore(St->getValueOperand(), target,
                                                    St->getAlign());
            // It repeats St, which the put_no_db mirror already checks.
            Copy->setMetadata("gicc.repeat", MDNode::get(Copy->getContext(), {}));
        }
        return mirror;
    }

    // Box element grain. Before the loop, in the kernel: the peer address,
    // the "loop runs" flag, and one call that sends the part of the range
    // the box does not write. In the parallel region: each store to the
    // source that lands in [start, start + len) is repeated at the same
    // offset in the peer.
    BoxSend lowerBoxPut(const OmpKernel &KI, const WriteSet &W, const Plan &P,
                        ScalarEvolution &SE, bool canCount) {
        LLVMContext &Ctx = M.getContext();
        Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx), *I8 = Type::getInt8Ty(Ctx);
        auto *Ptr = cast<PointerType>(P.put->getArgOperand(1)->getType());
        BasicBlock::iterator pre = KI.distInit->getIterator();
        SCEVExpander X(SE, KI.DL, "gicc.chunk");
        const SCEV *obj = SE.getSCEV(P.srcObj);
        Value *peerV = X.expandCodeFor(P.peer, I32, pre);
        Value *dstV  = X.expandCodeFor(P.dst, Ptr, pre);
        Value *srcV  = X.expandCodeFor(SE.getAddExpr(obj, P.rangeOff), Ptr, pre);
        Value *lenV  = X.expandCodeFor(P.bytes, I64, pre);
        Value *boxV  = X.expandCodeFor(SE.getAddExpr(obj, P.boxOffset), Ptr, pre);
        SmallVector<Value *, 4> strides, extents, facts;
        for (size_t d = 0; d < P.boxStride.size(); ++d) {
            strides.push_back(X.expandCodeFor(P.boxStride[d], I64, pre));
            extents.push_back(X.expandCodeFor(P.boxExtent[d], I64, pre));
        }
        for (const EntryFact &f : P.runs)
            facts.push_back(X.expandCodeFor(f.x, f.x->getType(), pre));

        IRBuilder<> B(KI.distInit);
        IRBuilder<> EB(&*KI.K.getEntryBlock().getFirstInsertionPt());
        AllocaInst *countedSlot =
            EB.CreateAlloca(I32, KI.DL.getAllocaAddrSpace(), nullptr, "gicc.box.counted");
        // The peer's copy of the range when it is IPC-mapped; otherwise the
        // kernel sends by count (canCount) or the next quiet does.
        FunctionCallee peerFn = M.getOrInsertFunction("ompx__box_peer", Ptr, I32, Ptr, Ptr,
                                                      I64, I32, Ptr);
        Value *pd = B.CreateCall(peerFn, {peerV, dstV, srcV, lenV,
                                          ConstantInt::get(I32, canCount),
                                          B.CreateAddrSpaceCast(countedSlot, Ptr)});
        Value *counted = B.CreateICmpNE(B.CreateLoad(I32, countedSlot), ConstantInt::get(I32, 0));
        Value *mirror = B.CreateICmpNE(pd, ConstantPointerNull::get(Ptr));
        Value *delta = B.CreateSub(B.CreatePtrToInt(pd, I64), B.CreatePtrToInt(srcV, I64));
        Value *runs = B.getTrue();
        for (size_t i = 0; i < facts.size(); ++i) {
            Value *z = Constant::getNullValue(facts[i]->getType());
            runs = B.CreateAnd(runs, P.runs[i].nonzeroOnly ? B.CreateICmpNE(facts[i], z)
                                                           : B.CreateICmpSGT(facts[i], z));
        }
        // The dims go by reference, in stack arrays of the kernel.
        const unsigned D = P.boxStride.size();
        auto array = [&](ArrayRef<Value *> vals, const char *name) {
            IRBuilder<> EB(&*KI.K.getEntryBlock().getFirstInsertionPt());
            auto *AT = ArrayType::get(I64, D);
            AllocaInst *A = EB.CreateAlloca(AT, KI.DL.getAllocaAddrSpace(), nullptr, name);
            for (unsigned d = 0; d < D; ++d)
                B.CreateStore(vals[d], B.CreateConstInBoundsGEP2_32(AT, A, 0, d));
            return B.CreateAddrSpaceCast(A, Ptr);
        };
        Value *strideArr = array(strides, "gicc.box.stride");
        Value *extentArr = array(extents, "gicc.box.extent");
        // Every thread of an SPMD kernel's team runs the code before its
        // loop -- clang emitted it so, or the SPMD conversion that guards
        // sequential code has already run -- so the team shares the copy.
        // A generic kernel leaves it to one thread, even one the device
        // link makes SPMD later.
        FunctionCallee residual = M.getOrInsertFunction(
            "ompx__box_residual", Type::getVoidTy(Ctx), Ptr, Ptr, I64, Ptr, I32, Ptr, Ptr, I64,
            I32, I32);
        B.CreateCall(residual, {pd, srcV, lenV, boxV, ConstantInt::get(I32, D), strideArr,
                                extentArr, ConstantInt::get(I64, P.elem),
                                B.CreateZExt(runs, I32), ConstantInt::get(I32, KI.spmd())});

        auto shared = [&](Type *T, const char *name) {
            return new GlobalVariable(M, T, false, GlobalValue::InternalLinkage,
                                      PoisonValue::get(T), name, nullptr,
                                      GlobalValue::NotThreadLocal, kmpc::SharedAddrSpace);
        };
        GlobalVariable *gDelta  = shared(I64, "gicc.box.delta");
        GlobalVariable *gStart  = shared(I64, "gicc.box.start");
        GlobalVariable *gLen    = shared(I64, "gicc.box.len");
        GlobalVariable *gMirror = shared(I8, "gicc.box.mirror");
        B.CreateStore(delta, gDelta);
        B.CreateStore(B.CreatePtrToInt(srcV, I64), gStart);
        B.CreateStore(lenV, gLen);
        B.CreateStore(B.CreateZExt(mirror, I8), gMirror);

        Function &O = *KI.outlined;
        IRBuilder<> BO(&*O.getEntryBlock().getFirstInsertionPt());
        Value *oDelta  = BO.CreateLoad(I64, gDelta, "gicc.box.delta");
        Value *oStart  = BO.CreateLoad(I64, gStart, "gicc.box.start");
        Value *oLen    = BO.CreateLoad(I64, gLen, "gicc.box.len");
        Value *oMirror = BO.CreateICmpNE(BO.CreateLoad(I8, gMirror), ConstantInt::get(I8, 0));
        for (StoreInst *St : W.storesTo.lookup(P.srcObj)) {
            Instruction *next = St->getNextNode();
            IRBuilder<> BS(next);
            Value *rel = BS.CreateSub(BS.CreatePtrToInt(St->getPointerOperand(), I64), oStart);
            Value *in = BS.CreateAnd(oMirror, BS.CreateICmpULT(rel, oLen), "gicc.box.in");
            Instruction *then = SplitBlockAndInsertIfThen(in, next->getIterator(), false);
            IRBuilder<> BT(then);
            Value *remote = BT.CreateGEP(I8, St->getPointerOperand(), oDelta);
            StoreInst *Copy = BT.CreateAlignedStore(St->getValueOperand(), remote, St->getAlign());
            // It repeats St, which the put_no_db mirror already checks.
            Copy->setMetadata("gicc.repeat", MDNode::get(Ctx, {}));
        }
        return BoxSend{peerV, dstV, srcV, lenV, boxV, strideArr, extentArr, counted, D, P.elem};
    }

    // The stores that hand the parallel region its block bounds (slots 0
    // and 1 of the argument array), in the parallel call's block; nullopt
    // when they are not both there.
    static std::optional<std::pair<StoreInst *, StoreInst *>> slotStores(const OmpKernel &KI) {
        StoreInst *st[2] = {nullptr, nullptr};
        const uint64_t slot =
            KI.DL.getTypeAllocSize(cast<ArrayType>(KI.argsArray->getAllocatedType())->getElementType());
        for (Instruction &I : *KI.parallel->getParent()) {
            if (&I == KI.parallel) break;
            auto *S = dyn_cast<StoreInst>(&I);
            if (!S) continue;
            APInt off(KI.DL.getIndexTypeSizeInBits(S->getPointerOperand()->getType()), 0);
            const Value *base = S->getPointerOperand()->stripAndAccumulateConstantOffsets(
                KI.DL, off, /*AllowNonInbounds=*/true);
            if (base != KI.argsArray) continue;
            if (off == 0) st[0] = S;
            else if (off == slot) st[1] = S;
        }
        if (!st[0] || !st[1]) return std::nullopt;
        return std::make_pair(st[0], st[1]);
    }

    // A box put the kernel sends by count (BoxSend::counted, decided at run
    // time): its peer is not IPC-mapped and the CPU proxy is on. The
    // distribute loop runs its chunks rotated so that those storing into the
    // counted ranges come first (ompx__box_plan): the block bounds handed to
    // the parallel region become lb + shift, wrapped, and the outlined loop
    // is left alone. Each chunk then compares lb with the smallest point at
    // which a counted put is due, and past it calls ompx__box_due, which
    // counts the team past those ranges; the last team past one sends it.
    // All of it sits behind one test of whether the launch counts anything,
    // weighted as rare: once the device link inlines the region into the
    // kernel this is the hot loop, one chunk per element of each thread,
    // and a launch that counts nothing should pay no more than that test.
    // SPMD only: every thread of the team runs this code, and the count
    // synchronizes them.
    void lowerBoxCounting(const OmpKernel &KI, StoreInst *lbSt, StoreInst *ubSt,
                          ArrayRef<BoxSend> sends) {
        LLVMContext &Ctx = M.getContext();
        Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
        auto *Ptr = PointerType::get(Ctx, 0);
        const unsigned n = sends.size();
        const unsigned AS = KI.DL.getAllocaAddrSpace();
        auto [lbSlot, ubSlot] = *slotStores(KI);
        IRBuilder<> EB(&*KI.K.getEntryBlock().getFirstInsertionPt());
        auto array = [&](Type *T, const char *name) {
            return EB.CreateAlloca(ArrayType::get(T, n), AS, nullptr, name);
        };
        AllocaInst *countedArr = array(I32, "gicc.box.counted.all");
        AllocaInst *loArr = array(I64, "gicc.box.lo"), *hiArr = array(I64, "gicc.box.hi");
        AllocaInst *dueArr = array(I64, "gicc.box.due");
        AllocaInst *peerArr = array(I32, "gicc.box.peer");
        AllocaInst *dstArr = array(Ptr, "gicc.box.dst"), *srcArr = array(Ptr, "gicc.box.src");
        AllocaInst *bytesArr = array(I64, "gicc.box.bytes");
        AllocaInst *counterArr = array(Ptr, "gicc.box.counter");
        AllocaInst *shiftSlot = EB.CreateAlloca(I64, AS, nullptr, "gicc.box.shift");
        AllocaInst *nextSlot = EB.CreateAlloca(I64, AS, nullptr, "gicc.box.next");
        auto at = [&](IRBuilder<> &B, AllocaInst *A, unsigned i) {
            return B.CreateConstInBoundsGEP2_32(A->getAllocatedType(), A, 0, i);
        };
        // A value the same in every thread, read back from private memory:
        // said so on AMDGPU, where it would otherwise take a vector register
        // in the loop -- enough of them cost the kernel a wave per SIMD.
        const bool amdgpu = Triple(M.getTargetTriple()).isAMDGPU();
        auto uniform = [&](IRBuilder<> &B, Value *V) -> Value * {
            return amdgpu ? B.CreateIntrinsic(Intrinsic::amdgcn_readfirstlane, {V->getType()}, {V})
                          : V;
        };
        auto flat = [&](IRBuilder<> &B, AllocaInst *A) {
            return B.CreateAddrSpaceCast(at(B, A, 0), Ptr);
        };

        // Before the loop: what each put needs, its hull, the rotation, and
        // when each counted put is due.
        IRBuilder<> B(KI.distInit);
        FunctionCallee hullFn = M.getOrInsertFunction(
            "ompx__box_hull", Type::getVoidTy(Ctx), Ptr, I64, Ptr, I32, Ptr, Ptr, I64, Ptr, Ptr);
        for (unsigned p = 0; p < n; ++p) {
            const BoxSend &S = sends[p];
            auto *counter = new GlobalVariable(
                M, I32, false, GlobalValue::InternalLinkage, ConstantInt::get(I32, 0),
                "gicc.box.count", nullptr, GlobalValue::NotThreadLocal, kmpc::GlobalAddrSpace);
            B.CreateStore(B.CreateZExt(S.counted, I32), at(B, countedArr, p));
            B.CreateStore(S.peer, at(B, peerArr, p));
            B.CreateStore(S.dst, at(B, dstArr, p));
            B.CreateStore(S.src, at(B, srcArr, p));
            B.CreateStore(S.len, at(B, bytesArr, p));
            B.CreateStore(ConstantExpr::getAddrSpaceCast(counter, Ptr), at(B, counterArr, p));
            B.CreateCall(hullFn, {S.src, S.len, S.box, ConstantInt::get(I32, S.dims), S.stride,
                                  S.extent, ConstantInt::get(I64, S.elem),
                                  B.CreateAddrSpaceCast(at(B, loArr, p), Ptr),
                                  B.CreateAddrSpaceCast(at(B, hiArr, p), Ptr)});
        }
        Value *lb0 = lbSt->getValueOperand(), *ub0 = ubSt->getValueOperand();
        Value *chunk = KI.distInit->getArgOperand(kmpc::InitChunk);
        Value *one = ConstantInt::get(I64, 1);
        Value *iters = B.CreateAdd(B.CreateSub(ub0, lb0), one);
        Value *span = B.CreateMul(B.CreateUDiv(B.CreateAdd(iters, B.CreateSub(chunk, one)), chunk),
                                  chunk);
        FunctionCallee planFn = M.getOrInsertFunction(
            "ompx__box_plan", Type::getVoidTy(Ctx), I32, Ptr, Ptr, Ptr, I64, I64, I64, Ptr, Ptr);
        B.CreateCall(planFn, {ConstantInt::get(I32, n), flat(B, countedArr), flat(B, loArr),
                              flat(B, hiArr), lb0, iters, chunk,
                              B.CreateAddrSpaceCast(shiftSlot, Ptr), flat(B, dueArr)});
        Value *shift = uniform(B, B.CreateLoad(I64, shiftSlot, "gicc.box.shift"));
        Value *next = B.CreateLoad(I64, at(B, dueArr, 0));
        for (unsigned p = 1; p < n; ++p)
            next = B.CreateBinaryIntrinsic(Intrinsic::smin, next, B.CreateLoad(I64, at(B, dueArr, p)));
        next = uniform(B, next);
        B.CreateStore(next, nextSlot);

        // Whether this launch counts anything at all: on most ranks (every
        // peer IPC-mapped) nothing is, and a chunk then costs one test more.
        Value *active = B.CreateICmpNE(next, ConstantInt::get(I64, INT64_MAX));
        MDNode *rarely = MDBuilder(Ctx).createBranchWeights(1, 1000);

        // Each chunk of a counting launch: the bounds of the chunk it now
        // stands for, stored over the loop's own, and the counts now due.
        Value *lb = IRBuilder<>(KI.parallel)
                        .CreatePtrToInt(lbSlot->getValueOperand(), I64, "gicc.box.lb");
        Instruction *counting =
            SplitBlockAndInsertIfThen(active, KI.parallel->getIterator(), false, rarely);
        IRBuilder<> RB(counting);
        Value *lbR = RB.CreateAdd(lb, shift);
        lbR = RB.CreateSelect(RB.CreateICmpUGE(RB.CreateSub(lbR, lb0), span),
                              RB.CreateSub(lbR, span), lbR);
        Value *ubR = RB.CreateBinaryIntrinsic(
            Intrinsic::umin, RB.CreateAdd(lbR, RB.CreateSub(chunk, one)), ub0);
        Type *slotT = lbSlot->getValueOperand()->getType();
        RB.CreateStore(RB.CreateIntToPtr(lbR, slotT), lbSlot->getPointerOperand());
        RB.CreateStore(RB.CreateIntToPtr(ubR, slotT), ubSlot->getPointerOperand());

        FunctionCallee dueFn = M.getOrInsertFunction(
            "ompx__box_due", I64, I32, I64, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr);
        auto dueCall = [&](Instruction *where, Value *pos, Value *go) {
            Instruction *then = SplitBlockAndInsertIfThen(go, where->getIterator(), false, rarely);
            IRBuilder<> T(then);
            Value *left = T.CreateCall(dueFn, {ConstantInt::get(I32, n), pos, flat(T, dueArr),
                                               flat(T, peerArr), flat(T, dstArr), flat(T, srcArr),
                                               flat(T, bytesArr), flat(T, counterArr)});
            T.CreateStore(uniform(T, left), nextSlot);
        };
        dueCall(counting, lb, RB.CreateICmpSGE(lb, RB.CreateLoad(I64, nextSlot)));
        // Past the loop: every counted put not yet counted.
        {
            IRBuilder<> FB(KI.distFini);
            Value *left = FB.CreateICmpNE(FB.CreateLoad(I64, nextSlot),
                                          ConstantInt::get(I64, INT64_MAX));
            dueCall(KI.distFini, ConstantInt::get(I64, INT64_MAX - 1), left);
        }

        DominatorTree DT(KI.K);
        PromoteMemToReg({nextSlot}, DT);
    }

    // One send per block right after its parallel region joins. Under the
    // element grain (`mirror` set) it is a single-thread put -- a parallel
    // region there would keep the device link from making the kernel SPMD --
    // which the caller guards to run only when the peer is not mapped.
    CallInst *emitBlockSend(const OmpKernel &KI, const Plan &P, ScalarEvolution &SE,
                            Value *mirror) {
        LLVMContext &Ctx = M.getContext();
        Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
        auto *Ptr = cast<PointerType>(P.put->getArgOperand(1)->getType());
        auto blockBound = [&](unsigned slot) {
            return KI.extDist(SE, SE.getSCEV(stripIntCasts(KI.slotValue(slot))), I64);
        };
        const SCEV *LB = blockBound(0), *UB = blockBound(1);
        const SCEV *srcBlk = SE.getAddExpr(
            SE.getSCEV(P.srcObj), SE.getAddExpr(SE.getMulExpr(P.c, LB), P.d));
        const SCEV *dstBlk = SE.getAddExpr(P.dst, SE.getMulExpr(P.c, SE.getMinusSCEV(LB, P.lb0)));
        const SCEV *len = SE.getMulExpr(P.c, SE.getAddExpr(SE.getMinusSCEV(UB, LB), SE.getOne(I64)));
        BasicBlock::iterator at = std::next(KI.parallel->getIterator());
        SCEVExpander X(SE, KI.DL, "gicc.chunk");
        Value *vPeer = X.expandCodeFor(P.peer, I32, at);
        Value *vDst  = X.expandCodeFor(dstBlk, Ptr, at);
        Value *vSrc  = X.expandCodeFor(srcBlk, Ptr, at);
        Value *vLen  = X.expandCodeFor(len, I64, at);
        FunctionCallee send = M.getOrInsertFunction(
            mirror ? "ompx__block_put_one" : "ompx__block_put", Type::getVoidTy(Ctx), I32,
            Ptr, Ptr, I64);
        return IRBuilder<>(at->getParent(), at).CreateCall(send, {vPeer, vDst, vSrc, vLen});
    }

    void report(CallInst *CI, Verdict v, const std::string &why) {
        const char *name = v == Verdict::Legal ? "LEGAL" : v == Verdict::Race ? "RACE" : "ILLEGAL";
        if (lower && v == Verdict::Illegal && calleeName(*CI) == kPipelinedPut)
            CI->getContext().emitError(CI, "cannot pipeline ompx_pipelined_put: " + why);
        out() << "[gicc-chunk]   " << calleeName(*CI);
        if (const DebugLoc &DLc = CI->getDebugLoc()) out() << " (line " << DLc.getLine() << ")";
        out() << ": " << name;
        if (!why.empty()) out() << ": " << why;
        out() << "\n";
    }
};

}  // namespace

PreservedAnalyses GICCChunkPrepPass::run(Module &M, ModuleAnalysisManager &) {
    // Also when only analysing: a put in a loop's body is read in the shape
    // this prepares.
    if (getConfig().mode != Mode::ChunkLower && getConfig().mode != Mode::ChunkAnalyze)
        return PreservedAnalyses::all();
    Triple T(M.getTargetTriple());
    if (!T.isNVPTX() && !T.isAMDGPU()) return PreservedAnalyses::all();
    bool changed = false;
    // The marker is only read by this pass, which replaces it. Said to touch
    // no memory the program can see, it stays in the loop but no longer
    // keeps the optimizer from hoisting the loop's own loads past it (the
    // thread stride, above all), so a marker in the loop's body leaves the
    // loop in the shape the write analysis reads.
    if (Function *Marker = M.getFunction(kPipelinedPut);
        Marker && !Marker->onlyAccessesInaccessibleMemory()) {
        Marker->setMemoryEffects(MemoryEffects::inaccessibleMemOnly());
        Marker->addFnAttr(Attribute::NoUnwind);
        changed = true;
    }
    auto hasPut = [](Function &F) {
        for (Instruction &I : instructions(F))
            if (isCallTo(I, kPipelinedPut)) return true;
        return false;
    };
    for (Function &F : M) {
        if (F.isDeclaration()) continue;
        const bool kernelHasPut = hasPut(F);
        for (Instruction &I : instructions(F)) {
            auto *CB = dyn_cast<CallBase>(&I);
            if (!CB || calleeName(*CB) != "__kmpc_parallel_51") continue;
            auto *O = dyn_cast<Function>(CB->getArgOperand(kmpc::ParallelFn)->stripPointerCasts());
            if (!O || O->isDeclaration() || O->hasFnAttribute(kPrepAttr)) continue;
            // The marker after the loop, or in the region's loop itself.
            if (!kernelHasPut && !hasPut(*O)) continue;
            // Clang marks outlined regions alwaysinline; remember that so the
            // lowering can put it back.
            const bool always = O->hasFnAttribute(Attribute::AlwaysInline);
            O->removeFnAttr(Attribute::AlwaysInline);
            O->addFnAttr(Attribute::NoInline);
            O->addFnAttr(kPrepAttr, always ? "alwaysinline" : "");
            changed = true;
        }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

PreservedAnalyses GICCChunkAnalysisPass::run(Module &M, ModuleAnalysisManager &MAM) {
    const Mode mode = getConfig().mode;
    if (mode != Mode::ChunkAnalyze && mode != Mode::ChunkLower) return PreservedAnalyses::all();
    Triple T(M.getTargetTriple());
    if (!T.isNVPTX() && !T.isAMDGPU()) return PreservedAnalyses::all();
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    return ChunkAnalyzer(M, FAM, mode == Mode::ChunkLower).run() ? PreservedAnalyses::none()
                                                                 : PreservedAnalyses::all();
}

}  // namespace gicc::pass
