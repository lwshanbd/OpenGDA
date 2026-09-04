#include "TraceTemplateBuilder.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/CFG.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "HKAnalysis.h"
#include "HostMirrorAnnotation.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"

#include <algorithm>
#include <array>
#include <optional>

using namespace llvm;

namespace gicc::pass {

namespace {

// Translate an LLVM Type to a short string used in the kernel-template
// JSON params list. Rich enough to round-trip the cases we lower.
std::string typeStr(Type *T) {
    if (!T) return "";
    if (T->isPointerTy())  return "ptr";
    if (T->isIntegerTy()) {
        std::string s = "i";
        s += std::to_string(T->getIntegerBitWidth());
        return s;
    }
    if (T->isFloatTy())    return "f32";
    if (T->isDoubleTy())   return "f64";
    if (T->isVoidTy())     return "void";
    return "?";
}

// Prove that every use of a pointer formal is either address formation, a
// pointer-only merge/compare, or a load. Unknown calls and captures fail
// closed. This recovers the same local fact that later optimization may add
// as `readonly`, while remaining valid at the early discovery extension
// point where frontend `restrict` has already supplied `noalias` but
// function-attribute inference has not yet run.
bool pointerFormalOnlyReadsMemory(const Argument &argument) {
    if (!argument.getType()->isPointerTy()) return false;
    SmallPtrSet<const Value *, 16> visited;
    SmallVector<const Value *, 16> work{&argument};
    while (!work.empty()) {
        const Value *value = work.pop_back_val();
        if (!visited.insert(value).second) continue;
        for (const User *user : value->users()) {
            if (const auto *load = dyn_cast<LoadInst>(user)) {
                if (load->getPointerOperand() == value) continue;
                return false;
            }
            if (const auto *gep = dyn_cast<GetElementPtrInst>(user)) {
                if (gep->getPointerOperand() != value) return false;
                work.push_back(gep);
                continue;
            }
            if (const auto *cast = dyn_cast<CastInst>(user)) {
                if (cast->getOperand(0) != value ||
                    !cast->getType()->isPointerTy())
                    return false;
                work.push_back(cast);
                continue;
            }
            if (const auto *phi = dyn_cast<PHINode>(user)) {
                if (!phi->getType()->isPointerTy()) return false;
                work.push_back(phi);
                continue;
            }
            if (const auto *select = dyn_cast<SelectInst>(user)) {
                if (!select->getType()->isPointerTy() ||
                    select->getCondition() == value)
                    return false;
                work.push_back(select);
                continue;
            }
            if (const auto *freeze = dyn_cast<FreezeInst>(user)) {
                if (!freeze->getType()->isPointerTy()) return false;
                work.push_back(freeze);
                continue;
            }
            if (isa<ICmpInst>(user)) continue;
            return false;
        }
    }
    return true;
}

// Canonical argument names for each GICC operation. Index matches the
// formal index in the gicc:: signature; entry [0] is always "ctx" and
// is skipped when emitting (callers start from index 1).
const std::array<const char *, 7> &putGetArgNames() {
    static const std::array<const char *, 7> names = {
        "ctx", "target_rank", "dst_buf", "dst_off", "src_buf", "src_off", "size"};
    return names;
}
const std::array<const char *, 1> &flushQuietArgNames() {
    static const std::array<const char *, 1> names = {"ctx"};
    return names;
}

// Map a BinaryOperator opcode to its IR mnemonic.
const char *binopName(unsigned op) {
    switch (op) {
        case Instruction::Add:  return "add";
        case Instruction::Sub:  return "sub";
        case Instruction::Mul:  return "mul";
        case Instruction::SDiv: return "sdiv";
        case Instruction::UDiv: return "udiv";
        case Instruction::SRem: return "srem";
        case Instruction::URem: return "urem";
        case Instruction::Shl:  return "shl";
        case Instruction::AShr: return "ashr";
        case Instruction::LShr: return "lshr";
        case Instruction::And:  return "and";
        case Instruction::Or:   return "or";
        case Instruction::Xor:  return "xor";
        default: return "binop";
    }
}

const char *castName(unsigned op) {
    switch (op) {
        case Instruction::Trunc:    return "trunc";
        case Instruction::ZExt:     return "zext";
        case Instruction::SExt:     return "sext";
        case Instruction::FPToUI:   return "fptoui";
        case Instruction::FPToSI:   return "fptosi";
        case Instruction::UIToFP:   return "uitofp";
        case Instruction::SIToFP:   return "sitofp";
        case Instruction::FPTrunc:  return "fptrunc";
        case Instruction::FPExt:    return "fpext";
        case Instruction::PtrToInt: return "ptrtoint";
        case Instruction::IntToPtr: return "inttoptr";
        case Instruction::BitCast:  return "bitcast";
        default: return "cast";
    }
}

std::optional<unsigned> directParamRef(const ArgRef &ref) {
    if (ref.kind == ArgRef::Kind::Param) return ref.paramIdx;
    if (ref.kind == ArgRef::Kind::Cast && ref.children.size() == 1)
        return directParamRef(ref.children.front());
    return std::nullopt;
}

// `ivPhi` (optional): if non-null, walking encounters this PHI we emit
// an ArgRef::Kind::LoopIv leaf so the host trace synthesizer can
// substitute the host-side loop counter instead of treating the PHI
// as a kernel formal (which it is not).
//
// `hostMirrored` (optional): per-formal "is host-mirrored" bits. When
// present and `V` is a LoadInst that reduces to host_mirror[iv].field
// on a host-mirrored formal, we emit an ArgRef::Kind::FieldLoad so the
// trace synthesizer can read the field at trace time.
ArgRef toArgRef(Value *V, const Function *K, const Value *ivPhi = nullptr,
                const std::vector<bool> *hostMirrored = nullptr) {
    ArgRef out;
    if (!V) {
        out.kind = ArgRef::Kind::Derived;
        return out;
    }
    if (ivPhi && V == ivPhi) {
        out.kind = ArgRef::Kind::LoopIv;
        return out;
    }
    if (auto *C = dyn_cast<ConstantInt>(V)) {
        out.kind = ArgRef::Kind::ConstI64;
        out.constVal = C->getSExtValue();
        return out;
    }
    if (auto *A = dyn_cast<Argument>(V)) {
        if (A->getParent() == K) {
            out.kind = ArgRef::Kind::Param;
            out.paramIdx = A->getArgNo();
            return out;
        }
    }
    if (auto *BO = dyn_cast<BinaryOperator>(V)) {
        out.kind = ArgRef::Kind::BinOp;
        out.opStr = binopName(BO->getOpcode());
        out.children.push_back(toArgRef(BO->getOperand(0), K, ivPhi, hostMirrored));
        out.children.push_back(toArgRef(BO->getOperand(1), K, ivPhi, hostMirrored));
        return out;
    }
    if (auto *CI = dyn_cast<CastInst>(V)) {
        out.kind = ArgRef::Kind::Cast;
        out.opStr = castName(CI->getOpcode());
        out.children.push_back(toArgRef(CI->getOperand(0), K, ivPhi, hostMirrored));
        return out;
    }
    if (auto *LI = dyn_cast<LoadInst>(V)) {
        if (hostMirrored) {
            FieldLoadMatch m = matchHostMirroredFieldLoad(LI, K, *hostMirrored);
            if (m.matched) {
                out.kind            = ArgRef::Kind::FieldLoad;
                out.paramIdx        = m.formalIdx;
                out.structElemSize  = m.structElemSize;
                out.fieldByteOffset = m.fieldByteOffset;
                out.fieldTypeStr    = m.fieldTypeStr;
                out.children.push_back(
                    toArgRef(m.iv, K, ivPhi, hostMirrored));
                return out;
            }
        }
        out.kind = ArgRef::Kind::Derived;
        return out;
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
        out.kind = ArgRef::Kind::Cast;
        out.opStr = "gep";
        for (Value *op : GEP->operands()) {
            out.children.push_back(toArgRef(op, K, ivPhi, hostMirrored));
        }
        return out;
    }
    out.kind = ArgRef::Kind::Derived;
    return out;
}

GuardSpec deriveGuard(const CallInst *CI, const Function *K,
                      const Value *ivPhi = nullptr,
                      const std::vector<bool> *hostMirrored = nullptr) {
    GuardSpec g;
    g.kind = GuardSpec::Kind::Always;

    const BasicBlock *parent = CI->getParent();
    if (!parent) return g;

    const BasicBlock *pred = parent->getSinglePredecessor();
    if (!pred) {
        // Multi-pred: dominating branch is harder to recover; mark as
        // unknown so the host pass can decide whether to fail or fall
        // back to a runtime test.
        return g;
    }

    const auto *br = dyn_cast<BranchInst>(pred->getTerminator());
    if (!br || !br->isConditional()) return g;

    Value *cond = br->getCondition();
    bool   takeWhenCondTrue = (br->getSuccessor(0) == parent);

    // Direct truth test on a kernel formal: if (param) { call }
    if (auto *A = dyn_cast<Argument>(cond)) {
        if (A->getParent() == K && takeWhenCondTrue) {
            g.kind = GuardSpec::Kind::ParamTruthy;
            g.paramIdx = A->getArgNo();
            return g;
        }
    }

    if (auto *icmp = dyn_cast<ICmpInst>(cond)) {
        Value *lhs = icmp->getOperand(0);
        Value *rhs = icmp->getOperand(1);

        // icmp ne / eq vs zero of a kernel formal → ParamTruthy.
        auto *RC = dyn_cast<ConstantInt>(rhs);
        auto *LA = dyn_cast<Argument>(lhs);
        if (RC && LA && LA->getParent() == K) {
            if (RC->isZero()) {
                bool truthy = (icmp->getPredicate() == ICmpInst::ICMP_NE)
                                  ? takeWhenCondTrue
                                  : !takeWhenCondTrue;
                if (truthy) {
                    g.kind = GuardSpec::Kind::ParamTruthy;
                    g.paramIdx = LA->getArgNo();
                    return g;
                }
            }
            if (icmp->getPredicate() == ICmpInst::ICMP_EQ && takeWhenCondTrue) {
                g.kind = GuardSpec::Kind::ParamEqConst;
                g.paramIdx = LA->getArgNo();
                g.constVal = RC->getSExtValue();
                return g;
            }
        }

        // icmp <eq/ne> ptr <field_load>, null — the ASF IPC-skip
        // pattern. The trace should emit the call only for entries
        // where the field IS null (cross-node peers in ASF's
        // transfers[]).
        if (hostMirrored) {
            LoadInst *fieldLoad = nullptr;
            if (auto *L = dyn_cast<LoadInst>(lhs);
                L && isa<ConstantPointerNull>(rhs)) {
                fieldLoad = L;
            } else if (auto *L = dyn_cast<LoadInst>(rhs);
                       L && isa<ConstantPointerNull>(lhs)) {
                fieldLoad = L;
            }
            if (fieldLoad) {
                FieldLoadMatch m =
                    matchHostMirroredFieldLoad(fieldLoad, K, *hostMirrored);
                if (m.matched) {
                    // Does branching into `parent` mean "field IS null"?
                    //   pred EQ ⇒ cond-true means field == null
                    //   pred NE ⇒ cond-true means field != null
                    bool trueMeansNull =
                        (icmp->getPredicate() == ICmpInst::ICMP_EQ);
                    bool enterMeansNull = (trueMeansNull == takeWhenCondTrue);
                    if (enterMeansNull) {
                        g.kind = GuardSpec::Kind::FieldNotNull;
                        ArgRef fl;
                        fl.kind            = ArgRef::Kind::FieldLoad;
                        fl.paramIdx        = m.formalIdx;
                        fl.structElemSize  = m.structElemSize;
                        fl.fieldByteOffset = m.fieldByteOffset;
                        fl.fieldTypeStr    = m.fieldTypeStr;
                        fl.children.push_back(
                            toArgRef(m.iv, K, ivPhi, hostMirrored));
                        g.fieldArg.clear();
                        g.fieldArg.push_back(std::move(fl));
                        return g;
                    }
                }
            }
        }
    }

    g.kind = GuardSpec::Kind::Unknown;
    return g;
}

void fillArgs(const CallInst *CI, GICCOpKind kind, OpTemplate &op,
              const Function *K, const Value *ivPhi = nullptr,
              const std::vector<bool> *hostMirrored = nullptr) {
    const auto *names =
        (kind == GICCOpKind::PutNoDb || kind == GICCOpKind::GetNoDb)
            ? reinterpret_cast<const char *const *>(putGetArgNames().data())
            : reinterpret_cast<const char *const *>(flushQuietArgNames().data());
    size_t maxArgs =
        (kind == GICCOpKind::PutNoDb || kind == GICCOpKind::GetNoDb)
            ? putGetArgNames().size()
            : flushQuietArgNames().size();

    // Skip arg 0 (ctx) — not part of the trace template.
    for (unsigned i = 1; i < CI->arg_size() && i < maxArgs; ++i) {
        op.args[names[i]] =
            toArgRef(CI->getArgOperand(i), K, ivPhi, hostMirrored);
    }
}

// Check if a Value is a kernel-formal Argument or a sign/zero-extend of
// one. Used while sniffing the loop bound out of `icmp slt %iv, %bound`.
bool isKernelFormal(const Value *V, const Function *K, unsigned &paramOut) {
    if (!V) return false;
    if (auto *A = dyn_cast<Argument>(V)) {
        if (A->getParent() == K) {
            paramOut = A->getArgNo();
            return true;
        }
        return false;
    }
    if (auto *CI = dyn_cast<CastInst>(V)) {
        return isKernelFormal(CI->getOperand(0), K, paramOut);
    }
    return false;
}

// Identify the canonical induction PHI of a loop. Returns the PHI and
// fills `start`/`step` if both are integer constants. Pattern matched:
//
//   loop.header:
//     %iv = phi <ty> [ <const start>, %preheader ],
//                   [ %iv.next,       %latch ]
//     %cmp = icmp <pred> %iv (or sext/zext %iv), <bound>
//     br i1 %cmp, label %body, label %exit   ; OR exit/body
//   ...
//   loop.latch:
//     %iv.next = add nsw <ty> %iv, <const step>
//     br label %loop.header
//
// On success returns (phi, start, step, boundParamIdx).
struct LoopShape {
    PHINode *iv         = nullptr;
    int64_t  start      = 0;
    int64_t  step       = 1;
    bool     ivBoundKnown = false;
    bool     boundIsConst = false;
    int64_t  constBound   = 0;
    unsigned ivParamIdx = 0;
    bool     valid      = false;   // true → safe to emit a loop in the trace
};

LoopShape analyzeLoop(const Loop *L, const Function *K) {
    LoopShape S;
    if (!L) return S;
    BasicBlock *header   = L->getHeader();
    BasicBlock *latch    = L->getLoopLatch();
    BasicBlock *preheader = L->getLoopPreheader();
    if (!header || !latch || !preheader) return S;

    // Find a PHI in the header whose incoming pair is
    //   [ const, preheader ], [ <user>, latch ]
    for (PHINode &phi : header->phis()) {
        if (phi.getNumIncomingValues() != 2) continue;
        Value *fromPreheader = phi.getIncomingValueForBlock(preheader);
        Value *fromLatch     = phi.getIncomingValueForBlock(latch);
        if (!fromPreheader || !fromLatch) continue;
        auto *startC = dyn_cast<ConstantInt>(fromPreheader);
        if (!startC) continue;

        auto *bo = dyn_cast<BinaryOperator>(fromLatch);
        if (!bo || bo->getOpcode() != Instruction::Add) continue;
        Value *bo0 = bo->getOperand(0);
        Value *bo1 = bo->getOperand(1);
        ConstantInt *stepC = nullptr;
        if (bo0 == &phi)      stepC = dyn_cast<ConstantInt>(bo1);
        else if (bo1 == &phi) stepC = dyn_cast<ConstantInt>(bo0);
        if (!stepC) continue;

        S.iv    = &phi;
        S.start = startC->getSExtValue();
        S.step  = stepC->getSExtValue();
        break;
    }
    if (!S.iv) return S;

    // Find the loop's exit-branching icmp. Prefer the latch's icmp (do/while
    // shape) or the header's icmp (canonical for-loop). Walk the conditional
    // branches in the loop blocks looking for `icmp <pred> %iv-or-cast,
    // <kernel formal-or-cast>` whose outcome controls a loop exit.
    SmallVector<BasicBlock *, 4> exiting;
    L->getExitingBlocks(exiting);
    for (BasicBlock *EB : exiting) {
        auto *br = dyn_cast<BranchInst>(EB->getTerminator());
        if (!br || !br->isConditional()) continue;
        auto *icmp = dyn_cast<ICmpInst>(br->getCondition());
        if (!icmp) continue;
        Value *l = icmp->getOperand(0);
        Value *r = icmp->getOperand(1);

        // Strip casts when matching against the IV (so `sext i32 %iv to
        // i64` still resolves as the iv).
        auto stripsToIV = [&](Value *V) -> bool {
            if (V == S.iv) return true;
            if (auto *CI = dyn_cast<CastInst>(V))
                if (CI->getOperand(0) == S.iv) return true;
            return false;
        };

        unsigned boundParam = 0;
        bool ivOnLeft = stripsToIV(l);
        bool ivOnRight = stripsToIV(r);
        if (ivOnLeft && isKernelFormal(r, K, boundParam)) {
            S.ivParamIdx   = boundParam;
            S.ivBoundKnown = true;
            break;
        }
        if (ivOnRight && isKernelFormal(l, K, boundParam)) {
            S.ivParamIdx   = boundParam;
            S.ivBoundKnown = true;
            break;
        }
        // A literal bound is just as host-knowable as a formal, and it
        // is the common shape for a fixed-size halo or a small fan-out.
        // Treating it as unrecoverable made the trace synthesizer skip
        // the op, which silently dropped the transfer.
        if (auto *CB = dyn_cast<ConstantInt>(ivOnLeft ? r : l)) {
            if (ivOnLeft || ivOnRight) {
                S.boundIsConst = true;
                S.constBound   = CB->getSExtValue();
                S.ivBoundKnown = true;
                break;
            }
        }
    }

    // We accept "ivBoundKnown=false" paths (e.g. constant trip count)
    // as degraded so the host trace skips the op rather than emitting
    // wrong code. valid=true means we have all the pieces needed to
    // emit a host-side loop.
    S.valid = (S.iv != nullptr && S.ivBoundKnown);
    return S;
}

// Count instructions in `BB` that look like arithmetic / FP compute.
// Int+FP binops, the typical math intrinsics, and fmuladd/fma. We
// intentionally ignore memory ops, casts, GEPs, and per-thread
// intrinsics — those are address arithmetic / dispatch glue, not the
// "FLOPs" the ML decider cares about. `stopAt`, if non-null, makes the
// scan stop just before that instruction (used for the call's own BB).
bool isArith(const Instruction &I) {
    if (isa<BinaryOperator>(&I)) return true;
    if (const auto *II = dyn_cast<IntrinsicInst>(&I)) {
        switch (II->getIntrinsicID()) {
            case Intrinsic::fmuladd:
            case Intrinsic::fma:
            case Intrinsic::sqrt:
            case Intrinsic::sin:
            case Intrinsic::cos:
            case Intrinsic::pow:
            case Intrinsic::exp:
            case Intrinsic::exp2:
            case Intrinsic::log:
            case Intrinsic::log2:
            case Intrinsic::log10:
            case Intrinsic::fabs:
            case Intrinsic::minnum:
            case Intrinsic::maxnum:
                return true;
            default:
                return false;
        }
    }
    return false;
}

int countArithInBB(const BasicBlock &BB, const Instruction *stopAt) {
    int n = 0;
    for (const Instruction &I : BB) {
        if (stopAt && &I == stopAt) break;
        if (isArith(I)) ++n;
    }
    return n;
}

// Sum of arithmetic ops in every BB that dominates the call's parent
// BB (proper dominators), plus the prefix of the call's own BB up to
// (but excluding) the call instruction. Coarse static "compute before
// comm" estimate.
int computeBeforeFor(const CallInst *CI, const Function *K,
                     const DominatorTree *DT) {
    if (!CI || !K || !DT) return -1;
    const BasicBlock *parent = CI->getParent();
    if (!parent) return -1;

    int total = 0;
    for (const BasicBlock &BB : *K) {
        if (&BB == parent) continue;
        if (DT->dominates(&BB, parent))
            total += countArithInBB(BB, /*stopAt=*/nullptr);
    }
    total += countArithInBB(*parent, /*stopAt=*/CI);
    return total;
}

// Sum of arithmetic ops that execute AFTER this call and before the
// kernel's completion point.
//
// Dominance is the wrong relation here: a call inside a loop body does
// not dominate the code after the loop, so a dominance walk reports
// almost nothing. What we want is "reachable from the call and not
// already behind it", bounded by the completion point.
//
// Arithmetic inside a loop is weighted by that loop's trip count when
// ScalarEvolution can prove one, so a short loop body that runs many
// times counts as the work it actually is. Loops with runtime bounds
// stay unweighted, which makes this an under-estimate for exactly the
// kernels whose extent is a formal -- the decider is told as much by
// `distance_exact`.
int computeAfterFor(const CallInst *CI, const CallInst *stop,
                    const Function *K, const DominatorTree *DT,
                    LoopInfo *LI, ScalarEvolution *SE, bool *exact) {
    if (!CI || !K || !DT) return -1;
    const BasicBlock *parent = CI->getParent();
    if (!parent) return -1;
    const BasicBlock *stopBB = stop ? stop->getParent() : nullptr;
    if (exact) *exact = true;

    // Per-BB multiplier: the trip count of the innermost enclosing loop
    // with a provable constant one.
    auto weightOf = [&](const BasicBlock &BB) -> long long {
        if (!LI || !SE) return 1;
        const Loop *L = LI->getLoopFor(&BB);
        if (!L) return 1;
        unsigned tc = SE->getSmallConstantTripCount(const_cast<Loop *>(L));
        if (!tc) { if (exact) *exact = false; return 1; }
        return static_cast<long long>(tc);
    };

    // Forward reachability from the call's block, stopping at the block
    // that holds the completion point.
    SmallPtrSet<const BasicBlock *, 32> seen;
    SmallVector<const BasicBlock *, 32> work;
    for (const BasicBlock *succ : successors(parent))
        if (seen.insert(succ).second) work.push_back(succ);

    long long total = 0;
    // Suffix of the call's own block.
    {
        bool counting = false;
        for (const Instruction &I : *parent) {
            if (&I == CI) { counting = true; continue; }
            if (&I == stop) break;
            if (counting && isArith(I)) total += weightOf(*parent);
        }
    }
    while (!work.empty()) {
        const BasicBlock *BB = work.pop_back_val();
        if (BB == stopBB) {
            total += (long long)countArithInBB(*BB, stop) * weightOf(*BB);
            continue;   // nothing past the completion point can hide anything
        }
        total += (long long)countArithInBB(*BB, nullptr) * weightOf(*BB);
        for (const BasicBlock *succ : successors(BB))
            if (seen.insert(succ).second) work.push_back(succ);
    }
    // Saturate rather than overflow the int field.
    if (total > (long long)INT32_MAX) total = INT32_MAX;
    return static_cast<int>(total);
}

// Compile-time trip count of the loop containing `CI`, when
// ScalarEvolution can prove a constant one. This is the per-phase op
// count K, which the proxy-vs-trigger decision is strongly sensitive to
// (issue cost is per-op on both paths, with different constants).
// Returns -1 when the bound is a runtime value, which is the common case
// for kernels whose extent is a formal.
long long tripCountFor(const CallInst *CI, LoopInfo *LI, ScalarEvolution *SE) {
    if (!CI || !LI || !SE) return -1;
    const Loop *L = LI->getLoopFor(CI->getParent());
    if (!L) return -1;
    // What we want is how many times THIS CALL runs, which is not
    // LLVM's "trip count": that counts header executions, i.e. one more
    // than the body for a loop the pass sees before rotation (measured:
    // a source bound of N reported N+1). The backedge-taken count is the
    // number of body executions; only a call in the header itself runs
    // the extra time.
    const SCEV *BTC = SE->getBackedgeTakenCount(const_cast<Loop *>(L));
    if (!BTC) return -1;
    const auto *C = dyn_cast<SCEVConstant>(BTC);
    if (!C) return -1;
    const uint64_t btc = C->getAPInt().getZExtValue();
    const bool inHeader = (CI->getParent() == L->getHeader());
    const uint64_t n = inHeader ? btc + 1 : btc;
    return n ? static_cast<long long>(n) : -1;
}

// What fence scope does this completion point actually need?
//
// The fence after a quiet exists so that reads issued AFTER it observe
// what the NIC or the proxy wrote. If nothing on the device reads
// anything after this point, the kernel ends and the host's stream sync
// supplies the ordering, so the fence is doing no work -- and it is not
// free, because a system fence writes back what the kernel just wrote.
//
// Deliberately coarse: any load, atomic, or call that might read counts,
// because without alias analysis there is no way to tell a read of the
// landed payload from any other read. Coarse in the SAFE direction --
// the answer is "keep the strongest fence" whenever anything at all could
// read, and a wrong answer in the weak direction gives stale data with
// nothing at runtime to catch it.
int fenceScopeFor(const CallInst *CI, const Function *K) {
    if (!CI || !K) return 3;                       // FENCE_SYSTEM

    auto mightRead = [](const Instruction &I) {
        if (isa<LoadInst>(&I) || isa<AtomicRMWInst>(&I) ||
            isa<AtomicCmpXchgInst>(&I))
            return true;
        if (const auto *C = dyn_cast<CallInst>(&I)) {
            const Function *F = C->getCalledFunction();
            // An indirect call, or one whose body is not here, could read
            // anything. Intrinsics that only write are the exception worth
            // making, since fences and stores are common right after.
            if (!F) return true;
            return !F->onlyWritesMemory() && !F->doesNotAccessMemory();
        }
        return false;
    };

    // Rest of the quiet's own block, then everything reachable forward.
    bool seen = false;
    for (const Instruction &I : *CI->getParent()) {
        if (&I == CI) { seen = true; continue; }
        if (seen && mightRead(I)) return 3;
    }
    SmallPtrSet<const BasicBlock *, 32> visited;
    SmallVector<const BasicBlock *, 32> work(succ_begin(CI->getParent()),
                                             succ_end(CI->getParent()));
    while (!work.empty()) {
        const BasicBlock *BB = work.pop_back_val();
        if (!visited.insert(BB).second) continue;
        for (const Instruction &I : *BB)
            if (mightRead(I)) return 3;
        work.append(succ_begin(BB), succ_end(BB));
    }
    return 0;                                      // FENCE_NONE
}

// Some HIP builtin accessors and shuffle helpers have not yet acquired
// readnone attributes at the early-simplification extension point even
// though their available IR bodies contain no externally visible write.
// Inspect those bodies recursively instead of treating every such call as an
// unknown producer. Unknown/indirect/recursive calls remain writes. Stores to
// a callee's own allocas are compiler-local scratch and do not contribute to
// the kernel's producer footprint.
bool functionMayWriteNonLocalMemory(
        const Function *F, SmallPtrSetImpl<const Function *> &visiting) {
    if (!F) return true;
    if (F->doesNotAccessMemory() || F->onlyReadsMemory()) return false;
    if (F->isDeclaration() || !visiting.insert(F).second) return true;

    bool mayWrite = false;
    for (const BasicBlock &BB : *F) {
        for (const Instruction &I : BB) {
            if (!I.mayWriteToMemory()) continue;
            const Value *pointer = nullptr;
            if (const auto *store = dyn_cast<StoreInst>(&I))
                pointer = store->getPointerOperand();
            else if (const auto *rmw = dyn_cast<AtomicRMWInst>(&I))
                pointer = rmw->getPointerOperand();
            else if (const auto *cmp = dyn_cast<AtomicCmpXchgInst>(&I))
                pointer = cmp->getPointerOperand();
            if (pointer) {
                const Value *root = getUnderlyingObject(
                    pointer->stripPointerCasts())->stripPointerCasts();
                if (isa<AllocaInst>(root)) continue;
                mayWrite = true;
                break;
            }
            const auto *call = dyn_cast<CallBase>(&I);
            if (!call || functionMayWriteNonLocalMemory(
                             call->getCalledFunction(), visiting)) {
                mayWrite = true;
                break;
            }
        }
        if (mayWrite) break;
    }
    visiting.erase(F);
    return mayWrite;
}

bool callMayWriteNonLocalMemory(const CallBase &call) {
    SmallPtrSet<const Function *, 16> visiting;
    return functionMayWriteNonLocalMemory(call.getCalledFunction(), visiting);
}

DeviceExpr deviceUnknown(StringRef type = "") {
    DeviceExpr out;
    out.kind = DeviceExpr::Kind::Unknown;
    out.typeStr = type.str();
    return out;
}

DeviceExpr deviceConst(int64_t value, StringRef type = "i64") {
    DeviceExpr out;
    out.kind = DeviceExpr::Kind::ConstI64;
    out.constVal = value;
    out.typeStr = type.str();
    return out;
}

DeviceExpr deviceBinOp(StringRef op, DeviceExpr lhs, DeviceExpr rhs) {
    if (op == "add" && lhs.kind == DeviceExpr::Kind::ConstI64 &&
        lhs.constVal == 0)
        return rhs;
    if (op == "add" && rhs.kind == DeviceExpr::Kind::ConstI64 &&
        rhs.constVal == 0)
        return lhs;
    if (op == "mul" && lhs.kind == DeviceExpr::Kind::ConstI64 &&
        lhs.constVal == 1)
        return rhs;
    if (op == "mul" && rhs.kind == DeviceExpr::Kind::ConstI64 &&
        rhs.constVal == 1)
        return lhs;
    DeviceExpr out;
    out.kind = DeviceExpr::Kind::BinOp;
    out.opStr = op.str();
    out.typeStr = "i64";
    out.children.push_back(std::move(lhs));
    out.children.push_back(std::move(rhs));
    return out;
}

bool deviceExprExact(const DeviceExpr &expr) {
    if (expr.typeStr.empty()) return false;
    switch (expr.kind) {
        case DeviceExpr::Kind::Param:
        case DeviceExpr::Kind::ConstI64:
            if (!expr.children.empty()) return false;
            break;
        case DeviceExpr::Kind::Builtin:
            if (expr.opStr.empty() || !expr.children.empty()) return false;
            break;
        case DeviceExpr::Kind::BinOp:
            if (expr.opStr.empty() || expr.opStr == "binop" ||
                expr.children.size() != 2)
                return false;
            break;
        case DeviceExpr::Kind::Cast:
            if (expr.opStr.empty() || expr.opStr == "cast" ||
                expr.children.size() != 1)
                return false;
            break;
        case DeviceExpr::Kind::Compare:
            if (expr.opStr.empty() || expr.children.size() != 2)
                return false;
            break;
        case DeviceExpr::Kind::Select:
            if (expr.children.size() != 3) return false;
            break;
        case DeviceExpr::Kind::Unknown:
            return false;
    }
    for (const auto &child : expr.children)
        if (!deviceExprExact(child)) return false;
    return true;
}

bool blockCanReach(const BasicBlock *from, const BasicBlock *target) {
    SmallPtrSet<const BasicBlock *, 32> visited;
    SmallVector<const BasicBlock *, 32> work;
    work.push_back(from);
    while (!work.empty()) {
        const BasicBlock *current = work.pop_back_val();
        if (current == target) return true;
        if (!visited.insert(current).second) continue;
        work.append(succ_begin(current), succ_end(current));
    }
    return false;
}

const char *dimensionName(uint64_t dimension) {
    switch (dimension) {
        case 0: return "x";
        case 1: return "y";
        case 2: return "z";
        default: return nullptr;
    }
}

std::optional<std::string> gpuBuiltinName(const CallBase &call) {
    const Function *callee = call.getCalledFunction();
    if (!callee) return std::nullopt;
    const StringRef name = callee->getName();
    for (const auto &entry : {
             std::pair<StringRef, StringRef>{"llvm.amdgcn.workgroup.id.x",
                                              "block_id_x"},
             {"llvm.amdgcn.workgroup.id.y", "block_id_y"},
             {"llvm.amdgcn.workgroup.id.z", "block_id_z"},
             {"llvm.amdgcn.workitem.id.x", "thread_id_x"},
             {"llvm.amdgcn.workitem.id.y", "thread_id_y"},
             {"llvm.amdgcn.workitem.id.z", "thread_id_z"},
         }) {
        if (name.starts_with(entry.first)) return entry.second.str();
    }

    // At the early optimization extension point used by clang, HIP's C++
    // builtin accessors may still be direct calls.  Match their exact ABI
    // spellings before they inline to the AMDGPU intrinsics handled above.
    for (const auto &entry : {
             std::pair<StringRef, StringRef>{
                 "_ZN24__hip_builtin_blockIdx_t7__get_xEv", "block_id_x"},
             {"_ZN24__hip_builtin_blockIdx_t7__get_yEv", "block_id_y"},
             {"_ZN24__hip_builtin_blockIdx_t7__get_zEv", "block_id_z"},
             {"_ZN24__hip_builtin_blockDim_t7__get_xEv", "block_size_x"},
             {"_ZN24__hip_builtin_blockDim_t7__get_yEv", "block_size_y"},
             {"_ZN24__hip_builtin_blockDim_t7__get_zEv", "block_size_z"},
             {"_ZN25__hip_builtin_threadIdx_t7__get_xEv", "thread_id_x"},
             {"_ZN25__hip_builtin_threadIdx_t7__get_yEv", "thread_id_y"},
             {"_ZN25__hip_builtin_threadIdx_t7__get_zEv", "thread_id_z"},
         }) {
        if (name == entry.first) return entry.second.str();
    }

    StringRef prefix;
    if (name == "__ockl_get_group_id") prefix = "block_id_";
    else if (name == "__ockl_get_local_id") prefix = "thread_id_";
    else if (name == "__ockl_get_local_size") prefix = "block_size_";
    else return std::nullopt;
    if (call.arg_size() != 1) return std::nullopt;
    const auto *dimension = dyn_cast<ConstantInt>(call.getArgOperand(0));
    if (!dimension) return std::nullopt;
    const char *suffix = dimensionName(dimension->getZExtValue());
    if (!suffix) return std::nullopt;
    return (prefix + suffix).str();
}

std::optional<std::string> implicitBlockSizeName(const LoadInst &load,
                                                 const DataLayout &DL) {
    const Value *pointer = load.getPointerOperand()->stripPointerCasts();
    const auto *gep = dyn_cast<GEPOperator>(pointer);
    if (!gep) return std::nullopt;
    const Value *base = gep->getPointerOperand()->stripPointerCasts();
    const auto *call = dyn_cast<CallBase>(base);
    if (!call || !call->getCalledFunction() ||
        !call->getCalledFunction()->getName().starts_with(
            "llvm.amdgcn.implicitarg.ptr"))
        return std::nullopt;
    const unsigned bitWidth = DL.getIndexSizeInBits(
        gep->getPointerAddressSpace());
    APInt offset(bitWidth, 0);
    if (!gep->accumulateConstantOffset(DL, offset) ||
        !offset.isSignedIntN(64))
        return std::nullopt;
    switch (offset.getSExtValue()) {
        case 12: return std::string("block_size_x");
        case 14: return std::string("block_size_y");
        case 16: return std::string("block_size_z");
        default: return std::nullopt;
    }
}

DeviceExpr buildDeviceExpr(const Value *value, const Function *kernel,
                           const DataLayout &DL,
                           SmallPtrSetImpl<const Value *> &visiting) {
    if (!value) return deviceUnknown();
    if (!visiting.insert(value).second)
        return deviceUnknown(typeStr(value->getType()));
    auto finish = [&](DeviceExpr out) {
        visiting.erase(value);
        return out;
    };
    if (const auto *constant = dyn_cast<ConstantInt>(value))
        return finish(deviceConst(constant->getSExtValue(),
                                  typeStr(constant->getType())));
    if (const auto *argument = dyn_cast<Argument>(value)) {
        if (argument->getParent() != kernel)
            return finish(deviceUnknown(typeStr(argument->getType())));
        DeviceExpr out;
        out.kind = DeviceExpr::Kind::Param;
        out.paramIdx = argument->getArgNo();
        out.typeStr = typeStr(argument->getType());
        return finish(std::move(out));
    }
    if (const auto *binary = dyn_cast<BinaryOperator>(value)) {
        DeviceExpr out;
        out.kind = DeviceExpr::Kind::BinOp;
        out.opStr = binopName(binary->getOpcode());
        out.typeStr = typeStr(binary->getType());
        out.children.push_back(buildDeviceExpr(binary->getOperand(0), kernel,
                                               DL, visiting));
        out.children.push_back(buildDeviceExpr(binary->getOperand(1), kernel,
                                               DL, visiting));
        return finish(std::move(out));
    }
    if (const auto *cast = dyn_cast<CastInst>(value)) {
        DeviceExpr out;
        out.kind = DeviceExpr::Kind::Cast;
        out.opStr = castName(cast->getOpcode());
        out.typeStr = typeStr(cast->getType());
        out.children.push_back(buildDeviceExpr(cast->getOperand(0), kernel,
                                               DL, visiting));
        return finish(std::move(out));
    }
    if (const auto *freeze = dyn_cast<FreezeInst>(value))
        return finish(buildDeviceExpr(freeze->getOperand(0), kernel, DL,
                                      visiting));
    if (const auto *compare = dyn_cast<ICmpInst>(value)) {
        DeviceExpr out;
        out.kind = DeviceExpr::Kind::Compare;
        out.opStr = ICmpInst::getPredicateName(compare->getPredicate()).str();
        out.typeStr = typeStr(compare->getType());
        out.children.push_back(buildDeviceExpr(compare->getOperand(0), kernel,
                                               DL, visiting));
        out.children.push_back(buildDeviceExpr(compare->getOperand(1), kernel,
                                               DL, visiting));
        return finish(std::move(out));
    }
    if (const auto *select = dyn_cast<SelectInst>(value)) {
        DeviceExpr out;
        out.kind = DeviceExpr::Kind::Select;
        out.typeStr = typeStr(select->getType());
        for (const Value *operand : select->operand_values())
            out.children.push_back(buildDeviceExpr(operand, kernel, DL,
                                                   visiting));
        return finish(std::move(out));
    }
    if (const auto *call = dyn_cast<CallBase>(value)) {
        auto builtin = gpuBuiltinName(*call);
        if (!builtin) return finish(deviceUnknown(typeStr(call->getType())));
        DeviceExpr out;
        out.kind = DeviceExpr::Kind::Builtin;
        out.opStr = std::move(*builtin);
        out.typeStr = typeStr(call->getType());
        return finish(std::move(out));
    }
    if (const auto *load = dyn_cast<LoadInst>(value)) {
        auto builtin = implicitBlockSizeName(*load, DL);
        if (!builtin) return finish(deviceUnknown(typeStr(load->getType())));
        DeviceExpr out;
        out.kind = DeviceExpr::Kind::Builtin;
        out.opStr = std::move(*builtin);
        out.typeStr = typeStr(load->getType());
        return finish(std::move(out));
    }
    return finish(deviceUnknown(typeStr(value->getType())));
}

DeviceExpr buildDeviceExpr(const Value *value, const Function *kernel,
                           const DataLayout &DL) {
    SmallPtrSet<const Value *, 32> visiting;
    return buildDeviceExpr(value, kernel, DL, visiting);
}

struct PointerBytes {
    const Argument *base = nullptr;
    DeviceExpr offset = deviceUnknown("i64");
    bool exact = false;
};

PointerBytes decomposePointerBytes(const Value *pointer,
                                   const Function *kernel,
                                   const DataLayout &DL,
                                   SmallPtrSetImpl<const Value *> &visiting) {
    PointerBytes result;
    if (!pointer || !visiting.insert(pointer).second) return result;
    const Value *visitedPointer = pointer;
    auto finish = [&](PointerBytes out) {
        visiting.erase(visitedPointer);
        return out;
    };
    pointer = pointer->stripPointerCasts();
    if (const auto *argument = dyn_cast<Argument>(pointer)) {
        if (argument->getParent() == kernel &&
            argument->getType()->isPointerTy()) {
            result.base = argument;
            result.offset = deviceConst(0);
            result.exact = true;
        }
        return finish(std::move(result));
    }

    const auto *gep = dyn_cast<GEPOperator>(pointer);
    if (!gep) return finish(std::move(result));
    result = decomposePointerBytes(gep->getPointerOperand(), kernel, DL,
                                   visiting);
    if (!result.exact) return finish(std::move(result));

    const unsigned bitWidth = DL.getIndexSizeInBits(
        gep->getPointerAddressSpace());
    MapVector<Value *, APInt> variables;
    APInt constant(bitWidth, 0);
    if (!gep->collectOffset(DL, bitWidth, variables, constant) ||
        !constant.isSignedIntN(64)) {
        result.exact = false;
        result.offset = deviceUnknown("i64");
        return finish(std::move(result));
    }
    DeviceExpr local = deviceConst(constant.getSExtValue());
    for (const auto &entry : variables) {
        if (!entry.second.isSignedIntN(64)) {
            result.exact = false;
            result.offset = deviceUnknown("i64");
            return finish(std::move(result));
        }
        DeviceExpr variable = buildDeviceExpr(entry.first, kernel, DL);
        if (!deviceExprExact(variable)) {
            result.exact = false;
            result.offset = deviceUnknown("i64");
            return finish(std::move(result));
        }
        DeviceExpr term = deviceBinOp(
            "mul", std::move(variable),
            deviceConst(entry.second.getSExtValue()));
        local = deviceBinOp("add", std::move(local), std::move(term));
    }
    result.offset = deviceBinOp("add", std::move(result.offset),
                                std::move(local));
    result.exact = deviceExprExact(result.offset);
    return finish(std::move(result));
}

PointerBytes decomposePointerBytes(const Value *pointer,
                                   const Function *kernel,
                                   const DataLayout &DL) {
    SmallPtrSet<const Value *, 16> visiting;
    return decomposePointerBytes(pointer, kernel, DL, visiting);
}

struct ControlPredicateDomain {
    std::vector<ProducerPredicateFact> predicates;
    std::vector<const BranchInst *> branches;
    bool exact = false;
};

ControlPredicateDomain controllingPredicates(const Instruction &instruction,
                                             const Function &kernel,
                                             DominatorTree *DT,
                                             LoopInfo *LI) {
    ControlPredicateDomain result;
    result.exact = DT && LI && !LI->getLoopFor(instruction.getParent());
    if (!result.exact) return result;

    const DataLayout &DL = kernel.getParent()->getDataLayout();
    for (const BasicBlock &BB : kernel) {
        if (&BB == instruction.getParent() ||
            !DT->dominates(&BB, instruction.getParent()))
            continue;
        const Instruction *term = BB.getTerminator();
        if (!term || term->getNumSuccessors() < 2) continue;
        SmallVector<bool, 4> reachesInstruction;
        unsigned reachable = 0;
        for (const BasicBlock *successor : successors(&BB)) {
            const bool reaches = blockCanReach(successor,
                                               instruction.getParent());
            reachesInstruction.push_back(reaches);
            if (reaches) ++reachable;
        }
        if (reachable == 0 || reachable == term->getNumSuccessors())
            continue;
        const auto *branch = dyn_cast<BranchInst>(term);
        if (!branch || !branch->isConditional() || reachable != 1) {
            result.exact = false;
            continue;
        }
        ProducerPredicateFact predicate;
        predicate.condition = buildDeviceExpr(
            branch->getCondition(), &kernel, DL);
        predicate.requiredValue = reachesInstruction[0];
        if (!deviceExprExact(predicate.condition)) result.exact = false;
        result.predicates.push_back(std::move(predicate));
        result.branches.push_back(branch);
    }
    return result;
}

std::optional<unsigned> partitionRegionPredicate(
        const StoreInst &store, const ControlPredicateDomain &control,
        DominatorTree *DT, PostDominatorTree *PDT,
        std::string &reason) {
    if (!control.exact || !DT || !PDT ||
        control.predicates.size() != control.branches.size()) {
        reason = "store control predicates are not exact";
        return std::nullopt;
    }

    std::optional<unsigned> best;
    unsigned bestLevel = 0;
    for (unsigned i = 0; i < control.branches.size(); ++i) {
        const BranchInst *branch = control.branches[i];
        const unsigned targetIndex =
            control.predicates[i].requiredValue ? 0 : 1;
        const BasicBlock *target = branch->getSuccessor(targetIndex);
        const auto *domNode = DT->getNode(branch->getParent());
        const auto *postNode = PDT->getNode(branch->getParent());
        if (!target->hasNPredecessors(1) ||
            !DT->dominates(target, store.getParent()) || !domNode ||
            !postNode || !postNode->getIDom())
            continue;
        const BasicBlock *merge = postNode->getIDom()->getBlock();
        if (!merge || merge == target ||
            !PDT->dominates(merge, target))
            continue;

        SmallPtrSet<const BasicBlock *, 16> region;
        SmallVector<const BasicBlock *, 16> work{target};
        bool valid = true;
        while (!work.empty() && valid) {
            const BasicBlock *BB = work.pop_back_val();
            if (BB == merge || !region.insert(BB).second) continue;
            if (!DT->dominates(target, BB) || BB == branch->getParent()) {
                valid = false;
                break;
            }
            for (const BasicBlock *successor : successors(BB))
                work.push_back(successor);
        }
        if (!valid || !region.contains(store.getParent())) continue;
        for (const BasicBlock *BB : region) {
            for (const BasicBlock *predecessor : predecessors(BB)) {
                if (BB == target && predecessor == branch->getParent())
                    continue;
                if (!region.contains(predecessor)) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
        }
        if (!valid) continue;

        const unsigned level = domNode->getLevel();
        if (!best || level > bestLevel) {
            best = i;
            bestLevel = level;
        }
    }
    if (!best) {
        reason =
            "no single-entry producer region with one post-dominating merge";
        return std::nullopt;
    }
    reason =
        "one exact controlling edge gates a single-entry producer region";
    return best;
}

ProducerStoreDomainFact producerStoreDomain(const StoreInst &store,
                                             const Function &kernel,
                                             DominatorTree *DT,
                                             PostDominatorTree *PDT,
                                             LoopInfo *LI) {
    ProducerStoreDomainFact domain;
    const DataLayout &DL = kernel.getParent()->getDataLayout();
    PointerBytes address = decomposePointerBytes(
        store.getPointerOperand(), &kernel, DL);
    if (address.base) domain.pointerParam = address.base->getArgNo();
    domain.byteOffset = std::move(address.offset);
    domain.addressExact = address.exact && address.base;

    TypeSize size = DL.getTypeStoreSize(store.getValueOperand()->getType());
    if (!size.isScalable()) domain.byteSize = size.getFixedValue();
    else domain.addressExact = false;

    ControlPredicateDomain control =
        controllingPredicates(store, kernel, DT, LI);
    if (auto predicate =
            partitionRegionPredicate(store, control, DT, PDT,
                                     domain.partitionRegionReason)) {
        domain.partitionRegionExact = true;
        domain.partitionPredicateIndex = *predicate;
    }
    domain.predicates = std::move(control.predicates);
    domain.predicatesExact = control.exact;

    domain.domainExact = domain.addressExact && domain.predicatesExact;
    if (!domain.addressExact)
        domain.reason = "store byte address is not an exact formal-rooted expression";
    else if (LI && LI->getLoopFor(store.getParent()))
        domain.reason = "producer store is inside an unmodeled device loop";
    else if (!domain.predicatesExact)
        domain.reason = "a controlling predicate is not exactly modeled";
    else
        domain.reason = "exact formal-rooted byte interval and controlling predicates recovered";
    return domain;
}

ProducerAtomicDomainFact producerAtomicDomain(
        const Instruction &atomic, const Value *pointer, StringRef operation,
        const Function &kernel, DominatorTree *DT, LoopInfo *LI) {
    ProducerAtomicDomainFact domain;
    const DataLayout &DL = kernel.getParent()->getDataLayout();
    PointerBytes address = decomposePointerBytes(pointer, &kernel, DL);
    if (address.base) domain.pointerParam = address.base->getArgNo();
    domain.operation = operation.str();
    domain.resultUnused = atomic.use_empty();
    ControlPredicateDomain control =
        controllingPredicates(atomic, kernel, DT, LI);
    domain.predicates = std::move(control.predicates);
    domain.predicatesExact = control.exact;
    domain.domainExact = address.exact && address.base &&
        domain.resultUnused && domain.predicatesExact;
    if (!address.exact || !address.base)
        domain.reason =
            "atomic pointer is not an exact formal-rooted expression";
    else if (!domain.resultUnused)
        domain.reason = "atomic return value is observed";
    else if (LI && LI->getLoopFor(atomic.getParent()))
        domain.reason = "atomic is inside an unmodeled device loop";
    else if (!domain.predicatesExact)
        domain.reason = "an atomic controlling predicate is not exactly modeled";
    else
        domain.reason =
            "exact formal-rooted atomic and controlling predicates recovered";
    return domain;
}

std::vector<ProducerPredicateFact> dominatingDirectI1FormalGuards(
        const Instruction &instruction, const Function &kernel,
        DominatorTree *DT) {
    std::vector<ProducerPredicateFact> guards;
    if (!DT) return guards;
    const DataLayout &DL = kernel.getParent()->getDataLayout();
    for (const BasicBlock &BB : kernel) {
        if (&BB == instruction.getParent() ||
            !DT->dominates(&BB, instruction.getParent()))
            continue;
        const auto *branch = dyn_cast<BranchInst>(BB.getTerminator());
        if (!branch || !branch->isConditional()) continue;
        const bool trueDominates =
            DT->dominates(branch->getSuccessor(0), instruction.getParent());
        const bool falseDominates =
            DT->dominates(branch->getSuccessor(1), instruction.getParent());
        if (trueDominates == falseDominates) continue;

        DeviceExpr condition = buildDeviceExpr(
            branch->getCondition(), &kernel, DL);
        if (condition.kind != DeviceExpr::Kind::Param ||
            condition.typeStr != "i1" ||
            condition.paramIdx >= kernel.arg_size() ||
            !kernel.getArg(condition.paramIdx)->getType()->isIntegerTy(1))
            continue;
        const bool requiredValue = trueDominates;
        const bool duplicate = std::any_of(
            guards.begin(), guards.end(), [&](const auto &guard) {
                return guard.condition.paramIdx == condition.paramIdx &&
                       guard.requiredValue == requiredValue;
            });
        if (!duplicate)
            guards.push_back({std::move(condition), requiredValue});
    }
    return guards;
}

ProducerPhaseSensitiveDomainFact producerPhaseSensitiveDomain(
        const CallBase &call, const Function &kernel, DominatorTree *DT) {
    ProducerPhaseSensitiveDomainFact domain;
    const Function *callee = call.getCalledFunction();
    domain.operation = callee
        ? callee->getName().str()
        : std::string("indirect_convergent_call");
    domain.guardPredicates =
        dominatingDirectI1FormalGuards(call, kernel, DT);
    domain.guardPredicatesExact = DT != nullptr;
    domain.domainExact = domain.guardPredicatesExact;
    if (!DT)
        domain.reason = "dominator tree is unavailable";
    else if (domain.guardPredicates.empty())
        domain.reason =
            "exact scan found no direct i1 kernel-formal disabling guard";
    else
        domain.reason =
            "exact dominating direct i1 kernel-formal guards recovered";
    return domain;
}

// How many transfers share a completion point.
//
// A flush or quiet is what actually releases staged work, so a transfer
// belongs to the group of the first completion point REACHABLE from it.
// Reachability, not a linear block order: any depth-first numbering of a
// CFG containing a loop can rank the loop's exit block ahead of its body
// (the body finishes last in post-order), which would put the loop's own
// transfers in the group after the flush that actually releases them.
//
// Transfers with no completion point downstream still form a group — the
// host's reset() drains them.
void assignBatchSizes(const GICCKernelInfo &info, KernelTemplate &t,
                      DominatorTree *DT, PostDominatorTree *PDT,
                      LoopInfo *LI) {
    if (!info.kernel) return;

    // Position within a block, so two sites in the same block can be
    // ordered against each other.
    DenseMap<const Instruction *, unsigned> slot;
    for (const BasicBlock &BB : *info.kernel) {
        unsigned n = 0;
        for (const Instruction &I : BB) slot[&I] = n++;
    }

    SmallVector<unsigned, 4> completions;
    for (unsigned i = 0; i < t.ops.size(); ++i)
        if (t.ops[i].kind == "flush" || t.ops[i].kind == "quiet")
            completions.push_back(i);

    // Earliest completion point inside `BB` that is strictly after
    // `after` (pass null to accept any position in the block).
    auto inBlock = [&](const BasicBlock *BB, const Instruction *after) -> int {
        int      best     = -1;
        unsigned bestSlot = ~0u;
        for (unsigned c : completions) {
            const Instruction *ci = info.sites[c].CI;
            if (ci->getParent() != BB) continue;
            unsigned s = slot.lookup(ci);
            if (after && s <= slot.lookup(after)) continue;
            if (s < bestSlot) { best = static_cast<int>(c); bestSlot = s; }
        }
        return best;
    };

    auto releasedBy = [&](const Instruction *from) -> int {
        if (int c = inBlock(from->getParent(), from); c >= 0) return c;
        SmallPtrSet<const BasicBlock *, 16>  seen;
        SmallVector<const BasicBlock *, 16>  work(succ_begin(from->getParent()),
                                                 succ_end(from->getParent()));
        while (!work.empty()) {
            const BasicBlock *BB = work.pop_back_val();
            if (!seen.insert(BB).second) continue;
            if (int c = inBlock(BB, nullptr); c >= 0) return c;
            work.append(succ_begin(BB), succ_end(BB));
        }
        return -1;   // nothing downstream; the host drain closes this group
    };

    constexpr int kNotATransfer = -2;
    std::vector<int>          owner(t.ops.size(), kNotATransfer);
    std::map<int, long long>  total;
    for (unsigned i = 0; i < t.ops.size(); ++i) {
        if (t.ops[i].kind != "put_no_db" && t.ops[i].kind != "get_no_db")
            continue;
        owner[i] = releasedBy(info.sites[i].CI);
        // A transfer inside a loop reaches the wire once per iteration.
        // Without a provable trip count the best honest answer is one.
        total[owner[i]] += t.ops[i].trip_count > 0 ? t.ops[i].trip_count : 1;
    }
    for (unsigned i = 0; i < t.ops.size(); ++i) {
        if (owner[i] == kNotATransfer) continue;
        t.ops[i].batch_size = total[owner[i]];
        if (owner[i] >= 0)
            t.ops[i].completion_site_id = t.ops[owner[i]].siteId;
    }

    // Prove the narrow completion-group early-trigger shape and recover the
    // matching producer frontier. The group bridge currently advertises only
    // multi-site choices, but producer facts are equally meaningful for one
    // transfer and must not disappear merely because its group has one member.
    // A registered buffer is represented in the
    // device API by an integer handle, so LLVM AA cannot relate it to ordinary
    // pointer stores.  Refuse to move the trigger across *any* instruction
    // which may write memory.  This loses opportunities but cannot turn a
    // producer store into a stale or racing RDMA read.
    for (const auto &group : total) {
        const int completion = group.first;
        SmallVector<unsigned, 4> members;
        for (unsigned i = 0; i < owner.size(); ++i)
            if (owner[i] == completion) members.push_back(i);
        if (members.empty()) continue;

        auto finish = [&](bool legal, StringRef reason) {
            for (unsigned i : members) {
                t.ops[i].group_early_trigger_analyzed = true;
                t.ops[i].group_early_trigger_legal = legal;
                t.ops[i].group_early_trigger_reason = reason.str();
            }
        };
        if (completion < 0 ||
            t.ops[static_cast<unsigned>(completion)].kind != "flush") {
            finish(false, "group has no compiler-identified flush completion");
            continue;
        }
        bool supportedMembers = true;
        for (unsigned i : members) {
            const OpTemplate &op = t.ops[i];
            if (op.kind != "put_no_db" || !op.hk_capable || op.loop.inLoop ||
                op.guard.kind != GuardSpec::Kind::Always) {
                supportedMembers = false;
                break;
            }
        }
        if (!supportedMembers) {
            finish(false, "group members are not unconditional non-loop host-knowable PUTs");
            continue;
        }
        if (!DT || !PDT) {
            finish(false, "dominance analyses unavailable");
            continue;
        }

        // Select a last member only when every other group member dominates
        // it.  That gives one insertion frontier after all descriptors have
        // logically been issued, without inventing a join block.
        CallInst *last = nullptr;
        for (unsigned candidate : members) {
            bool afterAll = true;
            for (unsigned other : members) {
                if (other == candidate) continue;
                if (!DT->dominates(info.sites[other].CI,
                                   info.sites[candidate].CI)) {
                    afterAll = false;
                    break;
                }
            }
            if (afterAll) {
                last = info.sites[candidate].CI;
                break;
            }
        }
        if (!last || !last->getNextNode()) {
            finish(false, "group has no unique post-issue insertion frontier");
            continue;
        }
        CallInst *flush = info.sites[static_cast<unsigned>(completion)].CI;
        if (!DT->dominates(last, flush) ||
            !PDT->dominates(flush->getParent(), last->getParent()) ||
            (last->getParent() == flush->getParent() &&
             !last->comesBefore(flush))) {
            finish(false, "flush is not a mandatory completion after every group member");
            continue;
        }

        Instruction *insertBefore = last->getNextNode();
        bool operandsDominate = true;
        for (Value *operand : flush->args()) {
            auto *definition = dyn_cast<Instruction>(operand);
            if (definition && !DT->dominates(definition, insertBefore)) {
                operandsDominate = false;
                break;
            }
        }
        if (!operandsDominate) {
            finish(false, "a flush operand does not dominate the group frontier");
            continue;
        }
        BasicBlock *startBB = last->getParent();
        BasicBlock *flushBB = flush->getParent();

        // Recover the memory-write footprint crossed by the tempting early
        // trigger.  Unlike the old binary mayWrite result, this preserves
        // which kernel pointer formals receive ordinary stores and which
        // receive atomics.  It remains only a source-free analysis fact:
        // without a host proof connecting src_buf to one of these pointer
        // formals, and without exact byte/domain proofs, fission is not legal.
        ProducerFrontierFacts frontier;
        frontier.analyzed = true;
        frontier.completion_site_id =
            t.ops[static_cast<unsigned>(completion)].siteId;
        SmallVector<unsigned, 4> ordinaryParams;
        SmallVector<unsigned, 4> atomicParams;
        SmallPtrSet<const Instruction *, 32> classified;
        SmallPtrSet<const Instruction *, 32> phaseSensitiveClassified;

        auto pointerFormal = [&](const Value *pointer)
                -> const Argument * {
            if (!pointer || !info.kernel) return nullptr;
            const Value *root = getUnderlyingObject(
                pointer->stripPointerCasts())->stripPointerCasts();
            const auto *arg = dyn_cast<Argument>(root);
            if (!arg || arg->getParent() != info.kernel ||
                !arg->getType()->isPointerTy())
                return nullptr;
            return arg;
        };
        auto classifyWrite = [&](const Instruction &I) {
            if (!I.mayWriteToMemory() || !classified.insert(&I).second)
                return;
            const Value *pointer = nullptr;
            bool atomic = false;
            std::string atomicOperation;
            const StoreInst *ordinaryStore = nullptr;
            if (const auto *store = dyn_cast<StoreInst>(&I)) {
                pointer = store->getPointerOperand();
                ordinaryStore = store;
            } else if (const auto *rmw = dyn_cast<AtomicRMWInst>(&I)) {
                pointer = rmw->getPointerOperand();
                atomic = true;
                atomicOperation =
                    ("atomicrmw_" +
                     AtomicRMWInst::getOperationName(rmw->getOperation())).str();
            } else if (const auto *cmp = dyn_cast<AtomicCmpXchgInst>(&I)) {
                pointer = cmp->getPointerOperand();
                atomic = true;
                atomicOperation = "cmpxchg";
            } else if (const auto *call = dyn_cast<CallBase>(&I)) {
                // HIP keeps source-level atomicAdd as a small device helper
                // until after this pre-inlining analysis point. Recognize
                // only its exact Itanium base name and pointer operand;
                // every other memory-writing call remains unknown.
                const Function *callee = call->getCalledFunction();
                if (!callee || !callee->getName().starts_with("_Z9atomicAdd") ||
                    call->arg_empty() ||
                    !call->getArgOperand(0)->getType()->isPointerTy()) {
                    if (!callMayWriteNonLocalMemory(*call)) return;
                    ++frontier.unknown_write_sites;
                    return;
                }
                pointer = call->getArgOperand(0);
                atomic = true;
                atomicOperation = "atomic_add";
            } else {
                ++frontier.unknown_write_sites;
                return;
            }

            const Value *root = getUnderlyingObject(
                pointer->stripPointerCasts())->stripPointerCasts();
            if (isa<AllocaInst>(root)) return;  // compiler-local scratch
            const Argument *arg = pointerFormal(pointer);
            if (!arg) {
                ++frontier.unknown_write_sites;
                return;
            }
            if (atomic) {
                ++frontier.atomic_write_sites;
                atomicParams.push_back(arg->getArgNo());
                frontier.producer_atomic_domains.push_back(
                    producerAtomicDomain(I, pointer, atomicOperation,
                                         *info.kernel, DT, LI));
            } else {
                ++frontier.ordinary_store_sites;
                ordinaryParams.push_back(arg->getArgNo());
                if (ordinaryStore)
                    frontier.producer_store_domains.push_back(
                        producerStoreDomain(*ordinaryStore, *info.kernel,
                                            DT, PDT, LI));
            }
        };
        auto classifyPhaseSensitive = [&](const Instruction &I) {
            const auto *call = dyn_cast<CallBase>(&I);
            if (!call || (!call->isConvergent() &&
                          !call->cannotDuplicate()) ||
                !phaseSensitiveClassified.insert(&I).second)
                return;
            // Grid/block/thread identity queries are deterministic for a
            // launch and may be recomputed in both phase kernels. Every
            // other convergent or noduplicate call is conservatively
            // non-duplicable.
            if (gpuBuiltinName(*call)) return;
            ++frontier.phase_sensitive_sites;
            frontier.producer_phase_sensitive_domains.push_back(
                producerPhaseSensitiveDomain(*call, *info.kernel, DT));
        };
        auto classifyFrontierInstruction = [&](const Instruction &I) {
            classifyWrite(I);
            classifyPhaseSensitive(I);
        };

        bool afterLastForFacts = false;
        for (const Instruction &I : *startBB) {
            if (&I == last) {
                afterLastForFacts = true;
                continue;
            }
            if (&I == flush) break;
            if (afterLastForFacts) classifyFrontierInstruction(I);
        }
        if (startBB != flushBB) {
            SmallPtrSet<const BasicBlock *, 32> visited;
            SmallVector<const BasicBlock *, 32> work(succ_begin(startBB),
                                                       succ_end(startBB));
            while (!work.empty()) {
                const BasicBlock *BB = work.pop_back_val();
                if (!visited.insert(BB).second) continue;
                for (const Instruction &I : *BB) {
                    if (&I == flush) break;
                    classifyFrontierInstruction(I);
                }
                if (BB != flushBB)
                    work.append(succ_begin(BB), succ_end(BB));
            }
        }
        llvm::sort(ordinaryParams);
        ordinaryParams.erase(
            std::unique(ordinaryParams.begin(), ordinaryParams.end()),
            ordinaryParams.end());
        llvm::sort(atomicParams);
        atomicParams.erase(
            std::unique(atomicParams.begin(), atomicParams.end()),
            atomicParams.end());
        frontier.ordinary_store_params.assign(ordinaryParams.begin(),
                                               ordinaryParams.end());
        frontier.atomic_write_params.assign(atomicParams.begin(),
                                             atomicParams.end());
        frontier.write_footprint_known =
            frontier.ordinary_store_sites > 0 &&
            frontier.unknown_write_sites == 0;
        frontier.producer_domains_known =
            frontier.write_footprint_known &&
            frontier.producer_store_domains.size() ==
                frontier.ordinary_store_sites &&
            llvm::all_of(frontier.producer_store_domains,
                         [](const auto &domain) {
                             return domain.domainExact;
                         });
        frontier.atomic_domains_known =
            frontier.unknown_write_sites == 0 &&
            frontier.producer_atomic_domains.size() ==
                frontier.atomic_write_sites &&
            llvm::all_of(frontier.producer_atomic_domains,
                         [](const auto &domain) {
                             return domain.domainExact;
                         });
        frontier.phase_sensitive_domains_known =
            frontier.producer_phase_sensitive_domains.size() ==
                frontier.phase_sensitive_sites &&
            llvm::all_of(frontier.producer_phase_sensitive_domains,
                         [](const auto &domain) {
                             return domain.domainExact;
                         });
        SmallVector<unsigned, 4> sourceBufferParams;
        bool sourceBufferShapeKnown = true;
        for (unsigned i : members) {
            auto source = t.ops[i].args.find("src_buf");
            if (source == t.ops[i].args.end()) {
                sourceBufferShapeKnown = false;
                break;
            }
            auto param = directParamRef(source->second);
            if (!param || *param >= t.params.size() ||
                t.params[*param].typeStr != "i32") {
                sourceBufferShapeKnown = false;
                break;
            }
            sourceBufferParams.push_back(*param);
        }
        llvm::sort(sourceBufferParams);
        sourceBufferParams.erase(
            std::unique(sourceBufferParams.begin(), sourceBufferParams.end()),
            sourceBufferParams.end());
        const bool onePointer = ordinaryParams.size() == 1 &&
            ordinaryParams.front() < t.params.size() &&
            t.params[ordinaryParams.front()].typeStr == "ptr";
        const bool oneSourceBuffer = sourceBufferShapeKnown &&
            sourceBufferParams.size() == 1;
        frontier.buffer_identity_guardable =
            frontier.write_footprint_known && onePointer && oneSourceBuffer;
        if (frontier.buffer_identity_guardable) {
            frontier.producer_pointer_param = ordinaryParams.front();
            frontier.source_buffer_index_param = sourceBufferParams.front();
            frontier.buffer_identity_guard_reason =
                "one producer pointer and one shared i32 source-buffer formal can be guarded at launch";
        } else {
            frontier.buffer_identity_guard_reason =
                "requires one producer pointer and one shared i32 source-buffer formal";
        }

        if (oneSourceBuffer && frontier.unknown_write_sites == 0) {
            for (unsigned param = 0; param < t.params.size(); ++param) {
                const ParamInfo &candidate = t.params[param];
                if (candidate.typeStr != "ptr" || !candidate.noalias ||
                    !candidate.readonly ||
                    llvm::is_contained(ordinaryParams, param) ||
                    llvm::is_contained(atomicParams, param))
                    continue;
                frontier.source_pointer_candidates.push_back(param);
            }
        }
        frontier.source_identity_guardable =
            oneSourceBuffer && frontier.unknown_write_sites == 0 &&
            !frontier.source_pointer_candidates.empty();
        if (frontier.source_identity_guardable) {
            frontier.source_identity_buffer_index_param =
                sourceBufferParams.front();
            frontier.source_identity_guard_reason =
                "readonly noalias pointer candidates can be matched to the shared source-buffer formal at launch; write-allocation disjointness remains unproved";
        } else if (!oneSourceBuffer) {
            frontier.source_identity_guard_reason =
                "requires one shared i32 source-buffer formal";
        } else if (frontier.unknown_write_sites != 0) {
            frontier.source_identity_guard_reason =
                "an intervening write has no kernel pointer formal";
        } else {
            frontier.source_identity_guard_reason =
                "no readonly noalias pointer formal can serve as a registered-source identity candidate";
        }
        if (frontier.unknown_write_sites != 0) {
            frontier.reason =
                "an intervening write is not rooted in a kernel pointer formal";
        } else if (frontier.ordinary_store_sites == 0) {
            frontier.reason =
                "no ordinary producer store exists between the transfer group and flush";
        } else if (!frontier.producer_domains_known) {
            frontier.reason =
                "formal-rooted writes recovered; exact domains and host buffer identity remain unproved";
        } else {
            frontier.reason =
                "formal-rooted writes and exact local store domains recovered; transfer matching and side-effect partition remain unproved";
        }
        for (unsigned i : members)
            t.ops[i].producer_frontier = frontier;

        auto mayWrite = [](const Instruction &I) {
            return I.mayWriteToMemory();
        };
        bool interveningWrite = false;
        bool afterLast = false;
        for (const Instruction &I : *startBB) {
            if (&I == last) {
                afterLast = true;
                continue;
            }
            if (&I == flush) break;
            if (afterLast && mayWrite(I)) interveningWrite = true;
        }
        if (!interveningWrite && startBB != flushBB) {
            SmallPtrSet<const BasicBlock *, 32> visited;
            SmallVector<const BasicBlock *, 32> work(succ_begin(startBB),
                                                       succ_end(startBB));
            while (!work.empty() && !interveningWrite) {
                const BasicBlock *BB = work.pop_back_val();
                if (!visited.insert(BB).second) continue;
                for (const Instruction &I : *BB) {
                    if (&I == flush) break;
                    if (mayWrite(I)) {
                        interveningWrite = true;
                        break;
                    }
                }
                if (BB != flushBB)
                    work.append(succ_begin(BB), succ_end(BB));
            }
        }
        if (interveningWrite) {
            finish(false, "intervening instruction may write a registered source buffer");
            continue;
        }
        finish(true, "proved mandatory flush and no intervening memory writes");
    }

    // Spread the transfers of a group across blocks so their pushes
    // overlap. Only worth doing when a group has more than one transfer;
    // a single site has nothing to overlap with and keeps block 0, which
    // is what was emitted before this existed.
    //
    // The completion point of a group records how many slots the group
    // used, because it has to drain every ring that was pushed to and not
    // just the first.
    std::map<int, int> next;
    for (unsigned i = 0; i < t.ops.size(); ++i) {
        if (owner[i] == kNotATransfer) continue;
        int members = 0;
        for (unsigned j = 0; j < t.ops.size(); ++j)
            if (owner[j] == owner[i]) ++members;
        if (members < 2) continue;
        t.ops[i].block_slot = next[owner[i]]++;
    }
    for (const auto &kv : next) {
        if (kv.first < 0 || kv.second < 2) continue;   // no completion point
        t.ops[kv.first].block_slot = kv.second;        // slots used by the group
    }
}

}  // namespace

KernelTemplate buildKernelTemplate(const GICCKernelInfo &info,
                                   LoopInfo *LI,
                                   DominatorTree *DT,
                                   ScalarEvolution *SE,
                                   PostDominatorTree *PDT) {
    KernelTemplate t;
    t.mangledName = info.mangledName;
    t.simpleName  = info.simpleName;

    // Discover @llvm.global.annotations entries that flag this kernel's
    // formals as host-mirrored. Name lookup is tried first; positional
    // form is the fallback for builds that strip value names.
    std::vector<bool> hostMirrored;
    if (info.kernel) {
        hostMirrored = computeHostMirroredFormals(*info.kernel);
        for (const Argument &A : info.kernel->args()) {
            ParamInfo p;
            p.name    = A.getName().str();
            p.typeStr = typeStr(A.getType());
            if (A.getArgNo() < hostMirrored.size())
                p.host_mirrored = hostMirrored[A.getArgNo()];
            p.noalias = A.hasAttribute(Attribute::NoAlias);
            p.readonly = A.hasAttribute(Attribute::ReadOnly) ||
                pointerFormalOnlyReadsMemory(A);
            t.params.push_back(std::move(p));
        }
    }

    // The kernel's completion point bounds "work that can hide a
    // transfer": the first quiet/flush in the kernel, if it has one.
    const CallInst *completionSite = nullptr;
    for (const auto &site : info.sites) {
        if (site.kind == GICCOpKind::Quiet || site.kind == GICCOpKind::Flush) {
            completionSite = site.CI;
            break;
        }
    }

    for (const auto &site : info.sites) {
        OpTemplate op;
        op.siteId = site.siteId;
        op.kind   = opKindName(site.kind);
        op.compute_before = computeBeforeFor(site.CI, info.kernel, DT);
        bool distExact = true;
        op.compute_after = computeAfterFor(site.CI, completionSite, info.kernel,
                                           DT, LI, SE, &distExact);
        op.distance_exact = distExact;
        op.trip_count     = tripCountFor(site.CI, LI, SE);
        if (site.kind == GICCOpKind::Quiet || site.kind == GICCOpKind::Flush)
            op.fence_scope = fenceScopeFor(site.CI, info.kernel);

        // Loop analysis runs first so we have the canonical iv PHI
        // before deriving guard / arg ArgRefs (so iv references become
        // LoopIv leaves rather than walked-back kernel formals).
        const Value *ivPhi = nullptr;
        if (LI) {
            BasicBlock *parent = site.CI->getParent();
            Loop      *L       = LI->getLoopFor(parent);
            if (L) {
                LoopShape s = analyzeLoop(L, info.kernel);
                op.loop.inLoop = true;
                if (s.valid) {
                    op.loop.ivBoundKnown   = true;
                    op.loop.ivBoundIsConst = s.boundIsConst;
                    op.loop.ivBoundConst   = s.constBound;
                    op.loop.ivParamIdx     = s.ivParamIdx;
                    op.loop.ivStart      = s.start;
                    op.loop.ivStep       = s.step;
                    ivPhi                = s.iv;
                } else {
                    op.loop.degraded = true;
                }
            }
        }

        op.guard = deriveGuard(site.CI, info.kernel, ivPhi, &hostMirrored);
        fillArgs(site.CI, site.kind, op, info.kernel, ivPhi, &hostMirrored);
        t.ops.push_back(std::move(op));
    }
    assignBatchSizes(info, t, DT, PDT, LI);
    return t;
}

}  // namespace gicc::pass
