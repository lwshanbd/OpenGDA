#include "HKAnalysis.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"

#include <string>

using namespace llvm;

namespace gicc::pass {

namespace {

// Set of intrinsic IDs whose result is intrinsically per-thread / per-CTA
// — i.e. NOT host-knowable.
bool isPerThreadIntrinsic(Intrinsic::ID id) {
    switch (id) {
        case Intrinsic::amdgcn_workitem_id_x:
        case Intrinsic::amdgcn_workitem_id_y:
        case Intrinsic::amdgcn_workitem_id_z:
        case Intrinsic::amdgcn_workgroup_id_x:
        case Intrinsic::amdgcn_workgroup_id_y:
        case Intrinsic::amdgcn_workgroup_id_z:
        case Intrinsic::nvvm_read_ptx_sreg_tid_x:
        case Intrinsic::nvvm_read_ptx_sreg_tid_y:
        case Intrinsic::nvvm_read_ptx_sreg_tid_z:
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
        case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
        case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
        case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_x:
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_y:
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_z:
            return true;
        default:
            return false;
    }
}

// HK-pure intrinsics whose output is host-knowable iff the inputs are.
bool isHKPureIntrinsic(Intrinsic::ID id) {
    switch (id) {
        // Arithmetic / math that lift over HK arguments.
        case Intrinsic::abs:
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
        case Intrinsic::minimum:
        case Intrinsic::maximum:
        case Intrinsic::umin:
        case Intrinsic::umax:
        case Intrinsic::smin:
        case Intrinsic::smax:
        case Intrinsic::fma:
            return true;
        default:
            return false;
    }
}

const char *intrinsicReason(Intrinsic::ID id) {
    switch (id) {
        case Intrinsic::amdgcn_workitem_id_x: return "depends on llvm.amdgcn.workitem.id.x";
        case Intrinsic::amdgcn_workitem_id_y: return "depends on llvm.amdgcn.workitem.id.y";
        case Intrinsic::amdgcn_workitem_id_z: return "depends on llvm.amdgcn.workitem.id.z";
        case Intrinsic::amdgcn_workgroup_id_x: return "depends on llvm.amdgcn.workgroup.id.x";
        case Intrinsic::amdgcn_workgroup_id_y: return "depends on llvm.amdgcn.workgroup.id.y";
        case Intrinsic::amdgcn_workgroup_id_z: return "depends on llvm.amdgcn.workgroup.id.z";
        case Intrinsic::nvvm_read_ptx_sreg_tid_x: return "depends on threadIdx.x";
        case Intrinsic::nvvm_read_ptx_sreg_tid_y: return "depends on threadIdx.y";
        case Intrinsic::nvvm_read_ptx_sreg_tid_z: return "depends on threadIdx.z";
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_x: return "depends on blockIdx.x";
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_y: return "depends on blockIdx.y";
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_z: return "depends on blockIdx.z";
        default: return "depends on per-thread intrinsic";
    }
}

HKResult fail(const std::string &reason, Instruction *origin) {
    return HKResult{false, reason, origin};
}

std::string fieldTypeStrOf(Type *T) {
    if (!T) return "i64";
    if (T->isPointerTy())  return "ptr";
    if (T->isIntegerTy())  return ("i" + std::to_string(T->getIntegerBitWidth()));
    if (T->isFloatTy())    return "f32";
    if (T->isDoubleTy())   return "f64";
    return "i64";
}

HKResult check(Value *V, Function *K,
               SmallPtrSetImpl<Value *> &visited,
               const std::vector<bool> *hostMirrored);

HKResult checkAllOperands(User *U, Function *K,
                          SmallPtrSetImpl<Value *> &visited,
                          const std::vector<bool> *hostMirrored) {
    for (Use &op : U->operands()) {
        HKResult r = check(op.get(), K, visited, hostMirrored);
        if (!r.ok) return r;
    }
    return {};
}

HKResult check(Value *V, Function *K,
               SmallPtrSetImpl<Value *> &visited,
               const std::vector<bool> *hostMirrored) {
    if (!V) return fail("null value", nullptr);
    if (!visited.insert(V).second) return {};  // cycle (e.g. PHI back-edge)

    if (isa<Constant>(V)) return {};
    if (auto *A = dyn_cast<Argument>(V)) {
        if (A->getParent() == K) return {};
        return fail("references an argument from another function", nullptr);
    }

    auto *I = dyn_cast<Instruction>(V);
    if (!I) return fail("non-instruction value of unrecognized kind", nullptr);

    if (isa<GetElementPtrInst>(I) || isa<CastInst>(I) ||
        isa<BinaryOperator>(I)    || isa<SelectInst>(I) ||
        isa<ICmpInst>(I)          || isa<FCmpInst>(I))
        return checkAllOperands(I, K, visited, hostMirrored);

    if (auto *PHI = dyn_cast<PHINode>(I)) {
        // Only canonical induction-variable PHIs are HK in v1: exactly
        // two incoming edges, one a host-knowable initial value (const
        // or kernel formal, possibly through casts/binops over those),
        // the other a `phi + const_step` self-recurrence. This matches
        // what TraceTemplateBuilder recognizes and what TraceSynthesis
        // can materialize as a host loop with `add i64 %iv, step`.
        //
        // Other PHI shapes (multi-incoming reductions, ternary
        // selections, non-constant step) might be host-knowable in
        // principle but the trace synthesizer would silently emit
        // wrong code for them. Reject conservatively.
        if (PHI->getNumIncomingValues() != 2)
            return fail("PHI has != 2 incoming values "
                        "(only canonical loop iv supported)", I);

        // Find the self-recurrence edge (`phi + const_step`).
        Value *initVal = nullptr;
        Value *recVal  = nullptr;
        for (Value *inc : PHI->incoming_values()) {
            auto *bo = dyn_cast<BinaryOperator>(inc);
            if (bo && bo->getOpcode() == Instruction::Add) {
                Value *lhs = bo->getOperand(0);
                Value *rhs = bo->getOperand(1);
                bool   lhsIsPhi = (lhs == PHI);
                bool   rhsIsPhi = (rhs == PHI);
                Value *other    = lhsIsPhi ? rhs : (rhsIsPhi ? lhs : nullptr);
                if (other && isa<ConstantInt>(other)) {
                    recVal = inc;
                    continue;
                }
            }
            initVal = inc;
        }
        if (!recVal || !initVal)
            return fail("PHI is not a canonical loop iv "
                        "(no `phi + const_step` self-recurrence)", I);

        // Init value must be host-knowable.
        HKResult r = check(initVal, K, visited, hostMirrored);
        if (!r.ok) return r;
        return {};
    }

    if (auto *CI = dyn_cast<CallInst>(I)) {
        if (auto *callee = CI->getCalledFunction()) {
            Intrinsic::ID id = callee->getIntrinsicID();
            if (id != Intrinsic::not_intrinsic) {
                if (isPerThreadIntrinsic(id))
                    return fail(intrinsicReason(id), I);
                if (isHKPureIntrinsic(id))
                    return checkAllOperands(CI, K, visited, hostMirrored);
            }
            return fail(("calls non-HK function '" +
                         callee->getName().str() + "'"), I);
        }
        return fail("indirect call", I);
    }

    if (auto *LI = dyn_cast<LoadInst>(I)) {
        // Loads are non-HK in general (they read device memory). The
        // sole exception is the host-mirrored pattern: when the load's
        // pointer reduces to `host_mirror[iv].field` on a formal flagged
        // host_mirrored, the trace function reads that field from the
        // host mirror at trace time. The iv itself must still be HK.
        if (hostMirrored) {
            FieldLoadMatch m = matchHostMirroredFieldLoad(LI, K, *hostMirrored);
            if (m.matched) {
                if (!m.iv) return {};  // const-iv load: trivially HK
                return check(m.iv, K, visited, hostMirrored);
            }
        }
        return fail("loads from device memory", I);
    }

    return fail("unsupported instruction kind", I);
}

}  // namespace

HKResult isHK(Value *V, Function *kernelF,
              const std::vector<bool> *hostMirrored) {
    SmallPtrSet<Value *, 16> visited;
    return check(V, kernelF, visited, hostMirrored);
}

FieldLoadMatch
matchHostMirroredFieldLoad(const LoadInst *LI, const Function *K,
                           const std::vector<bool> &hostMirrored) {
    FieldLoadMatch out;
    if (!LI || !K) return out;
    const auto *gep = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
    if (!gep) return out;

    const Value *base       = gep->getPointerOperand();
    Type        *containerTy = gep->getSourceElementType();
    Value       *ivVal      = nullptr;
    // GEP indices we should "consume" as struct-field offsets after the
    // iv. `fieldStart` is the index into `gep`'s operands where the
    // field walk begins (operand 0 is the pointer, operand 1 the first
    // explicit index).
    unsigned     fieldStart = 2;

    // Pattern P2: outer GEP's first index is constant 0 and the base is
    // another GEP. Collapse by taking the iv from the inner GEP and
    // beginning the field walk at outer operand 2 (which we already had).
    if (const auto *innerGep = dyn_cast<GetElementPtrInst>(base)) {
        if (auto *outerFirst = dyn_cast<ConstantInt>(gep->getOperand(1));
            outerFirst && outerFirst->isZero()) {
            // Inner must have exactly one explicit index (the iv).
            if (innerGep->getNumOperands() == 2 &&
                innerGep->getSourceElementType() == containerTy) {
                base  = innerGep->getPointerOperand();
                ivVal = const_cast<Value *>(innerGep->getOperand(1));
            }
        }
    }

    // Pattern P1: iv lives in outer GEP's first explicit index.
    if (!ivVal) {
        ivVal      = const_cast<Value *>(gep->getOperand(1));
        // Field walk starts at operand 2 (already the default).
    }

    const auto *arg = dyn_cast<Argument>(base);
    if (!arg || arg->getParent() != K) return out;
    unsigned formalIdx = arg->getArgNo();
    if (formalIdx >= hostMirrored.size() || !hostMirrored[formalIdx])
        return out;

    const DataLayout &DL = K->getParent()->getDataLayout();
    int64_t structSize   = DL.getTypeAllocSize(containerTy).getFixedValue();
    int64_t fieldOff     = 0;

    // Walk remaining indices through the container type (typically a
    // struct), accumulating byte offsets. All field indices must be
    // constants for the trace synthesizer to materialize the offset.
    Type *curTy = containerTy;
    for (unsigned i = fieldStart; i < gep->getNumOperands(); ++i) {
        Value *idx = gep->getOperand(i);
        auto  *ci  = dyn_cast<ConstantInt>(idx);
        if (!ci) return out;
        if (auto *st = dyn_cast<StructType>(curTy)) {
            unsigned f = static_cast<unsigned>(ci->getZExtValue());
            if (f >= st->getNumElements()) return out;
            const StructLayout *SL = DL.getStructLayout(st);
            fieldOff += static_cast<int64_t>(SL->getElementOffset(f));
            curTy = st->getElementType(f);
        } else if (auto *at = dyn_cast<ArrayType>(curTy)) {
            uint64_t k = ci->getZExtValue();
            fieldOff += static_cast<int64_t>(
                k * DL.getTypeAllocSize(at->getElementType()).getFixedValue());
            curTy = at->getElementType();
        } else {
            return out;
        }
    }

    out.matched          = true;
    out.formalIdx        = formalIdx;
    out.structElemSize   = structSize;
    out.fieldByteOffset  = fieldOff;
    out.fieldTypeStr     = fieldTypeStrOf(LI->getType());
    out.iv               = ivVal;
    return out;
}

}  // namespace gicc::pass
