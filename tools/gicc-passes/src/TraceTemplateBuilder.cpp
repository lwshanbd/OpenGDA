#include "TraceTemplateBuilder.h"
#include "HKAnalysis.h"
#include "HostMirrorAnnotation.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"

#include <array>

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

    // Look through casts and freeze to the underlying kernel formal, so
    // guards survive the freeze/zext wrappers LLVM inserts on branch
    // conditions at O3.
    auto asFormal = [K](Value *V) -> Argument * {
        while (V) {
            if (auto *A = dyn_cast<Argument>(V))
                return A->getParent() == K ? A : nullptr;
            if (auto *C = dyn_cast<CastInst>(V)) { V = C->getOperand(0); continue; }
            if (auto *F = dyn_cast<FreezeInst>(V)) { V = F->getOperand(0); continue; }
            return nullptr;
        }
        return nullptr;
    };

    if (auto *icmp = dyn_cast<ICmpInst>(cond)) {
        Value *lhs = icmp->getOperand(0);
        Value *rhs = icmp->getOperand(1);

        // Normalize `const CMP formal` to `formal CMP' const`.
        ICmpInst::Predicate normPred = icmp->getPredicate();
        if (isa<ConstantInt>(lhs) && !isa<ConstantInt>(rhs)) {
            std::swap(lhs, rhs);
            normPred = ICmpInst::getSwappedPredicate(normPred);
        }

        // icmp ne / eq vs zero of a kernel formal → ParamTruthy.
        auto *RC = dyn_cast<ConstantInt>(rhs);
        auto *LA = asFormal(lhs);
        if (RC && LA) {
            if (RC->isZero()) {
                bool truthy = (normPred == ICmpInst::ICMP_NE)
                                  ? takeWhenCondTrue
                                  : !takeWhenCondTrue;
                if ((normPred == ICmpInst::ICMP_NE ||
                     normPred == ICmpInst::ICMP_EQ) && truthy) {
                    g.kind = GuardSpec::Kind::ParamTruthy;
                    g.paramIdx = LA->getArgNo();
                    return g;
                }
            }
            if (normPred == ICmpInst::ICMP_EQ && takeWhenCondTrue) {
                g.kind = GuardSpec::Kind::ParamEqConst;
                g.paramIdx = LA->getArgNo();
                g.constVal = RC->getSExtValue();
                return g;
            }
            // General `param CMP const` guard, e.g. `if (I < n)` after LLVM
            // canonicalizes the constant to the RHS. Invert the predicate
            // when the op executes on the false edge.
            g.kind     = GuardSpec::Kind::ParamCmpConst;
            g.paramIdx = LA->getArgNo();
            g.constVal = RC->getSExtValue();
            g.pred     = static_cast<int>(
                takeWhenCondTrue ? normPred
                                 : ICmpInst::getInversePredicate(normPred));
            return g;
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
int countArithInBB(const BasicBlock &BB, const Instruction *stopAt) {
    int n = 0;
    for (const Instruction &I : BB) {
        if (stopAt && &I == stopAt) break;
        if (isa<BinaryOperator>(&I)) {
            ++n;
            continue;
        }
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
                    ++n;
                    break;
                default:
                    break;
            }
        }
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

}  // namespace

KernelTemplate buildKernelTemplate(const GICCKernelInfo &info,
                                   LoopInfo *LI,
                                   DominatorTree *DT) {
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
            t.params.push_back(std::move(p));
        }
    }

    for (const auto &site : info.sites) {
        OpTemplate op;
        op.siteId = site.siteId;
        op.kind   = opKindName(site.kind);
        op.compute_before = computeBeforeFor(site.CI, info.kernel, DT);

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
                    op.loop.ivBoundKnown = true;
                    op.loop.ivParamIdx   = s.ivParamIdx;
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
    return t;
}

}  // namespace gicc::pass
