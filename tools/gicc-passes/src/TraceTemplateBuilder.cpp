#include "TraceTemplateBuilder.h"

#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
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

ArgRef toArgRef(Value *V, const Function *K) {
    ArgRef out;
    if (!V) {
        out.kind = ArgRef::Kind::Derived;
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
        out.children.push_back(toArgRef(BO->getOperand(0), K));
        out.children.push_back(toArgRef(BO->getOperand(1), K));
        return out;
    }
    if (auto *CI = dyn_cast<CastInst>(V)) {
        out.kind = ArgRef::Kind::Cast;
        out.opStr = castName(CI->getOpcode());
        out.children.push_back(toArgRef(CI->getOperand(0), K));
        return out;
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
        out.kind = ArgRef::Kind::Cast;
        out.opStr = "gep";
        for (Value *op : GEP->operands()) {
            out.children.push_back(toArgRef(op, K));
        }
        return out;
    }
    out.kind = ArgRef::Kind::Derived;
    return out;
}

GuardSpec deriveGuard(const CallInst *CI, const Function *K) {
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
    }

    g.kind = GuardSpec::Kind::Unknown;
    return g;
}

void fillArgs(const CallInst *CI, GICCOpKind kind, OpTemplate &op,
              const Function *K) {
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
        op.args[names[i]] = toArgRef(CI->getArgOperand(i), K);
    }
}

}  // namespace

KernelTemplate buildKernelTemplate(const GICCKernelInfo &info) {
    KernelTemplate t;
    t.mangledName = info.mangledName;
    t.simpleName  = info.simpleName;

    if (info.kernel) {
        for (const Argument &A : info.kernel->args()) {
            ParamInfo p;
            p.name    = A.getName().str();
            p.typeStr = typeStr(A.getType());
            t.params.push_back(std::move(p));
        }
    }

    for (const auto &site : info.sites) {
        OpTemplate op;
        op.siteId = site.siteId;
        op.kind   = opKindName(site.kind);
        op.guard  = deriveGuard(site.CI, info.kernel);
        fillArgs(site.CI, site.kind, op, info.kernel);
        t.ops.push_back(std::move(op));
    }
    return t;
}

}  // namespace gicc::pass
