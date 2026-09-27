#include "OmpKernel.h"
#include "KernelInventory.h"

#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace gicc::pass {

StringRef calleeName(const CallBase &CB) {
    if (const Function *F = CB.getCalledFunction()) return F->getName();
    return "";
}

bool isBenignCall(const CallBase &CB) {
    if (const auto *II = dyn_cast<IntrinsicInst>(&CB)) {
        if (II->isLifetimeStartOrEnd() || II->isAssumeLikeIntrinsic()) return true;
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

bool isCallTo(const Instruction &I, StringRef prefix) {
    const auto *CB = dyn_cast<CallBase>(&I);
    return CB && calleeName(*CB).starts_with(prefix);
}

bool isOffloadKernel(const Function &F) {
    return !F.isDeclaration() && isGPUKernel(F) &&
           F.getName().starts_with("__omp_offloading_");
}

std::string objName(const Value *V) {
    if (V->hasName()) return V->getName().str();
    if (auto *A = dyn_cast<Argument>(V)) return "arg#" + std::to_string(A->getArgNo());
    return "<unnamed>";
}

std::string scevStr(const SCEV *S) {
    std::string s;
    raw_string_ostream os(s);
    S->print(os);
    return s;
}

Value *stripIntCasts(Value *V) {
    V = V->stripPointerCasts();
    if (auto *I2P = dyn_cast<IntToPtrInst>(V)) V = I2P->getOperand(0);
    if (auto *CI = dyn_cast<CastInst>(V)) V = CI->getOperand(0);
    return V;
}

StoreInst *uniqueStoreTo(Function &F, const Value *Obj) {
    StoreInst *found = nullptr;
    for (Instruction &I : instructions(F)) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI || SI->getPointerOperand()->stripPointerCasts() != Obj) continue;
        if (found) return nullptr;
        found = SI;
    }
    return found;
}

std::optional<OmpKernel> OmpKernel::locate(Function &K, std::string &why) {
    OmpKernel KI(K);
    for (Instruction &I : instructions(K)) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) continue;
        StringRef n = calleeName(*CI);
        if (n.starts_with("__kmpc_distribute_static_init")) {
            if (KI.distInit) { why = "more than one distribute loop"; return std::nullopt; }
            KI.distInit = CI;
            KI.distSigned = !n.ends_with("u");
        } else if (n.starts_with("__kmpc_distribute_static_fini")) {
            KI.distFini = CI;
        } else if (n == "__kmpc_parallel_51") {
            if (KI.parallel) { why = "more than one parallel region"; return std::nullopt; }
            KI.parallel = CI;
        }
    }
    if (!KI.distInit || !KI.distFini || !KI.parallel) {
        why = "no distribute parallel for worksharing loop";
        return std::nullopt;
    }
    KI.outlined = dyn_cast<Function>(
        KI.parallel->getArgOperand(kmpc::ParallelFn)->stripPointerCasts());
    KI.argsArray = dyn_cast<AllocaInst>(
        KI.parallel->getArgOperand(kmpc::ParallelArgs)->stripPointerCasts());
    if (!KI.outlined || KI.outlined->isDeclaration() || !KI.argsArray) {
        why = "unrecognized __kmpc_parallel_51 operands";
        return std::nullopt;
    }
    return KI;
}

Value *OmpKernel::resolve(Value *V, unsigned depth) const {
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
        auto *GV = dyn_cast<GlobalVariable>(LI->getPointerOperand()->stripPointerCasts());
        if (!GV) return V;
        if (StoreInst *SI = uniqueStoreTo(K, GV))
            return resolve(SI->getValueOperand(), depth + 1);
    }
    return V;
}

Value *OmpKernel::slotValue(unsigned idx) const {
    Value *found = nullptr;
    for (Instruction &I : instructions(K)) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI) continue;
        APInt off(DL.getIndexTypeSizeInBits(SI->getPointerOperandType()), 0);
        const Value *base =
            SI->getPointerOperand()->stripAndAccumulateConstantOffsets(DL, off, true);
        if (base->stripPointerCasts() != argsArray) continue;
        if (off.getZExtValue() != idx * DL.getPointerSize()) continue;
        if (found) return nullptr;
        found = SI->getValueOperand();
    }
    return found;
}

Value *OmpKernel::resolveOutlined(Value *V) const {
    V = V->stripPointerCasts();
    auto viaSlot = [&](Argument *A) -> Value * {
        if (A->getParent() != outlined || A->getArgNo() < kmpc::OutlinedFirstSlot)
            return nullptr;
        return slotValue(A->getArgNo() - kmpc::OutlinedFirstSlot);
    };
    if (auto *LI = dyn_cast<LoadInst>(V)) {
        // Captured by reference: the slot holds the variable's address.
        auto *A = dyn_cast<Argument>(LI->getPointerOperand()->stripPointerCasts());
        if (!A) return nullptr;
        Value *addr = viaSlot(A);
        if (!addr) return nullptr;
        Value *obj = addr->stripPointerCasts();
        if (!isa<GlobalVariable>(obj) && !isa<AllocaInst>(obj)) return nullptr;
        StoreInst *SI = uniqueStoreTo(K, obj);
        return SI ? resolve(SI->getValueOperand()) : nullptr;
    }
    if (auto *A = dyn_cast<Argument>(V)) {
        // Captured by value.
        Value *v = viaSlot(A);
        return v ? resolve(v) : nullptr;
    }
    return nullptr;
}

namespace {
// Rewrite every unknown whose value resolves to something simpler, so that
// equal quantities reached through globalized copies compare equal.
class ResolveRewriter : public SCEVRewriteVisitor<ResolveRewriter> {
    const OmpKernel &KI;
public:
    ResolveRewriter(ScalarEvolution &SE, const OmpKernel &KI)
        : SCEVRewriteVisitor(SE), KI(KI) {}
    const SCEV *visitUnknown(const SCEVUnknown *U) {
        Value *r = KI.resolve(U->getValue());
        if (r == U->getValue() || !SE.isSCEVable(r->getType())) return U;
        return visit(SE.getSCEV(r));
    }
};
}  // namespace

const SCEV *OmpKernel::normalize(ScalarEvolution &SE, const SCEV *S) const {
    ResolveRewriter RW(SE, *this);
    return RW.visit(S);
}

const SCEV *OmpKernel::toKernel(const SCEV *S, ScalarEvolution &SK) const {
    if (auto *C = dyn_cast<SCEVConstant>(S)) return SK.getConstant(C->getAPInt());
    if (auto *U = dyn_cast<SCEVUnknown>(S)) {
        Value *v = U->getValue(), *kv = nullptr;
        if (auto *A = dyn_cast<Argument>(v);
            A && A->getParent() == outlined && A->getArgNo() >= kmpc::OutlinedFirstSlot) {
            if (Value *s = slotValue(A->getArgNo() - kmpc::OutlinedFirstSlot)) {
                s = s->stripPointerCasts();
                if (auto *I2P = dyn_cast<IntToPtrInst>(s)) s = I2P->getOperand(0);
                kv = resolve(s);
            }
        } else if (isa<LoadInst>(v)) {
            kv = resolveOutlined(v);
        }
        if (!kv || !SK.isSCEVable(kv->getType())) return nullptr;
        return SK.getTruncateOrSignExtend(SK.getSCEV(kv), S->getType());
    }
    if (auto *C = dyn_cast<SCEVCastExpr>(S)) {
        const SCEV *op = toKernel(C->getOperand(), SK);
        if (!op) return nullptr;
        switch (S->getSCEVType()) {
        case scTruncate:   return SK.getTruncateExpr(op, S->getType());
        case scZeroExtend: return SK.getZeroExtendExpr(op, S->getType());
        case scSignExtend: return SK.getSignExtendExpr(op, S->getType());
        default:           return nullptr;
        }
    }
    if (auto *N = dyn_cast<SCEVNAryExpr>(S)) {
        SmallVector<const SCEV *, 4> ops;
        for (const SCEV *op : N->operands()) {
            const SCEV *k = toKernel(op, SK);
            if (!k) return nullptr;
            ops.push_back(k);
        }
        if (isa<SCEVAddExpr>(S))  return SK.getAddExpr(ops);
        if (isa<SCEVMulExpr>(S))  return SK.getMulExpr(ops);
        if (isa<SCEVSMaxExpr>(S)) return SK.getSMaxExpr(ops);
        if (isa<SCEVUMaxExpr>(S)) return SK.getUMaxExpr(ops);
        if (isa<SCEVSMinExpr>(S)) return SK.getSMinExpr(ops);
        if (isa<SCEVUMinExpr>(S)) return SK.getUMinExpr(ops);
        return nullptr;
    }
    if (auto *D = dyn_cast<SCEVUDivExpr>(S)) {
        const SCEV *a = toKernel(D->getLHS(), SK), *b = toKernel(D->getRHS(), SK);
        return a && b ? SK.getUDivExpr(a, b) : nullptr;
    }
    return nullptr;
}

const SCEV *OmpKernel::extDist(ScalarEvolution &SE, const SCEV *S, Type *T) const {
    return distSigned ? SE.getNoopOrSignExtend(S, T) : SE.getNoopOrZeroExtend(S, T);
}

const Value *OmpKernel::distSlot(unsigned opnd) const {
    return distInit->getArgOperand(opnd)->stripPointerCasts();
}

std::optional<std::pair<StoreInst *, StoreInst *>>
OmpKernel::distBounds(DominatorTree &DT) const {
    const Value *lbSlot = distSlot(kmpc::InitLower), *ubSlot = distSlot(kmpc::InitUpper);
    StoreInst *lbSt = nullptr, *ubSt = nullptr;
    for (Instruction &I : instructions(K)) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI || !DT.dominates(SI, distInit)) continue;
        if (SI->getPointerOperand()->stripPointerCasts() == lbSlot) lbSt = SI;
        if (SI->getPointerOperand()->stripPointerCasts() == ubSlot) ubSt = SI;
    }
    if (!lbSt || !ubSt) return std::nullopt;
    return std::make_pair(lbSt, ubSt);
}

}  // namespace gicc::pass
