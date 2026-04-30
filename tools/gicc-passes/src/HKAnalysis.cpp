#include "HKAnalysis.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

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

HKResult check(Value *V, Function *K, SmallPtrSetImpl<Value *> &visited);

HKResult checkAllOperands(User *U, Function *K, SmallPtrSetImpl<Value *> &visited) {
    for (Use &op : U->operands()) {
        HKResult r = check(op.get(), K, visited);
        if (!r.ok) return r;
    }
    return {};
}

HKResult check(Value *V, Function *K, SmallPtrSetImpl<Value *> &visited) {
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
        return checkAllOperands(I, K, visited);

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
        HKResult r = check(initVal, K, visited);
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
                    return checkAllOperands(CI, K, visited);
            }
            return fail(("calls non-HK function '" +
                         callee->getName().str() + "'"), I);
        }
        return fail("indirect call", I);
    }

    if (isa<LoadInst>(I))
        return fail("loads from device memory", I);

    return fail("unsupported instruction kind", I);
}

}  // namespace

HKResult isHK(Value *V, Function *kernelF) {
    SmallPtrSet<Value *, 16> visited;
    return check(V, kernelF, visited);
}

}  // namespace gicc::pass
