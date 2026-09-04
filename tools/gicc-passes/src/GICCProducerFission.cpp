#include "GICCProducerFission.h"

#include "DispatchDecision.h"
#include "GICCHostDiscovery.h"
#include "GICCPassConfig.h"
#include "KernelInventory.h"
#include "MetadataIO.h"
#include "TraceTemplateBuilder.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <optional>
#include <vector>

using namespace llvm;

namespace gicc::pass {

namespace {

constexpr uint32_t kScheduleOriginal = 0;
constexpr uint32_t kScheduleProducerFrontier = 1;
constexpr uint32_t kScheduleRemainder = 2;
constexpr uint64_t kSchedulePhaseOffset = 16;
constexpr StringLiteral kSyntheticFlushMD =
    "gicc.producer_fission.synthetic_flush";

std::optional<unsigned> metadataIntegerBits(StringRef metadataType) {
    if (!metadataType.consume_front("i")) return std::nullopt;
    unsigned bits = 0;
    if (metadataType.empty() || metadataType.getAsInteger(10, bits) ||
        bits == 0)
        return std::nullopt;
    return bits;
}

bool parameterCellTypeMatches(const Type *cellType, StringRef metadataType) {
    if (metadataType == "ptr") return cellType->isPointerTy();
    if (metadataType == "f32") return cellType->isFloatTy();
    if (metadataType == "f64") return cellType->isDoubleTy();
    if (auto bits = metadataIntegerBits(metadataType)) {
        if (cellType->isIntegerTy(*bits)) return true;
        return *bits == 1 && cellType->isIntegerTy(8);
    }
    return false;
}

struct AuditedLaunch {
    CallInst *launch = nullptr;
    AllocaInst *params = nullptr;
    std::vector<AllocaInst *> cells;
    std::string reason;
};

AuditedLaunch auditFinalLaunch(Function &wrapper,
                               const KernelTemplate &kernel) {
    AuditedLaunch result;
    SmallVector<CallInst *, 2> launches;
    for (BasicBlock &BB : wrapper) {
        for (Instruction &I : BB) {
            auto *call = dyn_cast<CallInst>(&I);
            const Function *callee = call ? call->getCalledFunction() : nullptr;
            if (callee && callee->getName() == "hipLaunchKernel")
                launches.push_back(call);
        }
    }
    if (launches.size() != 1) {
        result.reason =
            "final annotated wrapper does not own exactly one HIP launch";
        return result;
    }
    CallInst *launch = launches.front();
    if (launch->arg_size() != 8 || !launch->use_empty() ||
        launch->getNumOperandBundles() != 0 ||
        !launch->getArgOperand(0)->getType()->isPointerTy() ||
        !launch->getArgOperand(5)->getType()->isPointerTy() ||
        !launch->getArgOperand(7)->getType()->isPointerTy()) {
        result.reason = "HIP launch does not match the cloneable final ABI";
        return result;
    }
    const Value *target = launch->getArgOperand(0)->stripPointerCasts();
    if (!target->hasName() || target->getName() != kernel.mangledName) {
        result.reason = "HIP launch target disagrees with device metadata";
        return result;
    }

    Value *paramsBase = getUnderlyingObject(
        launch->getArgOperand(5)->stripPointerCasts());
    auto *params = dyn_cast<AllocaInst>(paramsBase);
    if (!params || params->getFunction() != &wrapper ||
        kernel.params.empty()) {
        result.reason = "kernel parameter array is not wrapper-local";
        return result;
    }
    const DataLayout &layout = wrapper.getParent()->getDataLayout();
    const uint64_t pointerBytes = layout.getPointerSize(
        params->getType()->getPointerAddressSpace());
    const auto allocationSize = params->getAllocationSize(layout);
    const uint64_t requiredBytes =
        pointerBytes * static_cast<uint64_t>(kernel.params.size());
    if (pointerBytes == 0 || !allocationSize ||
        allocationSize->isScalable() ||
        allocationSize->getFixedValue() < requiredBytes) {
        result.reason = "kernel parameter array is smaller than metadata";
        return result;
    }

    result.cells.resize(kernel.params.size(), nullptr);
    for (BasicBlock &BB : wrapper) {
        for (Instruction &I : BB) {
            auto *store = dyn_cast<StoreInst>(&I);
            if (!store ||
                getUnderlyingObject(store->getPointerOperand()) != params)
                continue;
            // The generated final wrapper initializes the whole parameter
            // array immediately before the launch. Any cross-block, late, or
            // dynamic write is an unaudited alias and must fail closed.
            if (store->getParent() != launch->getParent() ||
                !store->comesBefore(launch)) {
                result.reason = "kernel parameter array has a non-local write";
                result.cells.clear();
                return result;
            }
            int64_t offset = 0;
            const Value *base = GetPointerBaseWithConstantOffset(
                store->getPointerOperand(), offset, layout);
            if (base != params || offset < 0 ||
                static_cast<uint64_t>(offset) % pointerBytes != 0) {
                result.reason = "kernel parameter array has a dynamic write";
                result.cells.clear();
                return result;
            }
            const uint64_t slot = static_cast<uint64_t>(offset) / pointerBytes;
            if (slot >= result.cells.size()) {
                result.reason =
                    "kernel parameter array has an out-of-range store";
                result.cells.clear();
                return result;
            }
            auto *cell = dyn_cast<AllocaInst>(
                store->getValueOperand()->stripPointerCasts());
            if (!cell || cell->getFunction() != &wrapper ||
                (result.cells[slot] && result.cells[slot] != cell)) {
                result.reason = "kernel parameter slot is not uniquely local";
                result.cells.clear();
                return result;
            }
            result.cells[slot] = cell;
        }
    }

    SmallPtrSet<AllocaInst *, 16> distinct;
    for (unsigned i = 0; i < result.cells.size(); ++i) {
        AllocaInst *cell = result.cells[i];
        if (!cell || !distinct.insert(cell).second ||
            !parameterCellTypeMatches(cell->getAllocatedType(),
                                      kernel.params[i].typeStr)) {
            result.reason =
                "kernel parameter cells are missing, aliased, or mistyped";
            result.cells.clear();
            return result;
        }
    }
    result.launch = launch;
    result.params = params;
    result.reason = "final HIP launch and every parameter slot re-proved";
    return result;
}

bool sameDeviceExpr(const DeviceExpr &left, const DeviceExpr &right) {
    if (left.kind != right.kind || left.paramIdx != right.paramIdx ||
        left.constVal != right.constVal || left.opStr != right.opStr ||
        left.typeStr != right.typeStr ||
        left.children.size() != right.children.size())
        return false;
    for (size_t i = 0; i < left.children.size(); ++i)
        if (!sameDeviceExpr(left.children[i], right.children[i]))
            return false;
    return true;
}

bool sameArgRef(const ArgRef &left, const ArgRef &right) {
    if (left.kind != right.kind || left.paramIdx != right.paramIdx ||
        left.constVal != right.constVal || left.opStr != right.opStr ||
        left.structElemSize != right.structElemSize ||
        left.fieldByteOffset != right.fieldByteOffset ||
        left.fieldTypeStr != right.fieldTypeStr ||
        left.children.size() != right.children.size())
        return false;
    for (size_t i = 0; i < left.children.size(); ++i)
        if (!sameArgRef(left.children[i], right.children[i])) return false;
    return true;
}

bool samePredicates(const std::vector<ProducerPredicateFact> &left,
                    const std::vector<ProducerPredicateFact> &right) {
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i)
        if (left[i].requiredValue != right[i].requiredValue ||
            !sameDeviceExpr(left[i].condition, right[i].condition))
            return false;
    return true;
}

bool sameStoreDomain(const ProducerStoreDomainFact &left,
                     const ProducerStoreDomainFact &right) {
    return left.pointerParam == right.pointerParam &&
        left.byteSize == right.byteSize &&
        left.addressExact == right.addressExact &&
        left.predicatesExact == right.predicatesExact &&
        left.domainExact == right.domainExact &&
        left.partitionRegionExact == right.partitionRegionExact &&
        left.partitionPredicateIndex == right.partitionPredicateIndex &&
        sameDeviceExpr(left.byteOffset, right.byteOffset) &&
        samePredicates(left.predicates, right.predicates);
}

struct FormalGuard {
    bool present = false;
    unsigned param = 0;
    bool value = false;
};

std::optional<FormalGuard> disablingGuard(
        const KernelTemplate &kernel,
        const ProducerFrontierFacts &frontier) {
    if (frontier.atomic_write_sites == 0 &&
        frontier.phase_sensitive_sites == 0)
        return FormalGuard{};
    if ((frontier.atomic_write_sites != 0 &&
         (!frontier.atomic_domains_known ||
          frontier.producer_atomic_domains.empty())) ||
        (frontier.phase_sensitive_sites != 0 &&
         (!frontier.phase_sensitive_domains_known ||
          frontier.producer_phase_sensitive_domains.empty())))
        return std::nullopt;

    const std::vector<ProducerPredicateFact> &candidates =
        frontier.phase_sensitive_sites != 0
            ? frontier.producer_phase_sensitive_domains.front()
                  .guardPredicates
            : frontier.producer_atomic_domains.front().predicates;
    for (const ProducerPredicateFact &candidate : candidates) {
        const DeviceExpr &condition = candidate.condition;
        if (condition.kind != DeviceExpr::Kind::Param ||
            condition.typeStr != "i1" ||
            condition.paramIdx >= kernel.params.size() ||
            kernel.params[condition.paramIdx].typeStr != "i1")
            continue;
        const bool commonAtomic = std::all_of(
            frontier.producer_atomic_domains.begin(),
            frontier.producer_atomic_domains.end(),
            [&](const auto &domain) {
                return domain.domainExact && domain.resultUnused &&
                    std::any_of(domain.predicates.begin(),
                                domain.predicates.end(),
                                [&](const auto &predicate) {
                                    return predicate.requiredValue ==
                                               candidate.requiredValue &&
                                        sameDeviceExpr(predicate.condition,
                                                       condition);
                                });
            });
        const bool commonPhase = std::all_of(
            frontier.producer_phase_sensitive_domains.begin(),
            frontier.producer_phase_sensitive_domains.end(),
            [&](const auto &domain) {
                return domain.domainExact &&
                    std::any_of(domain.guardPredicates.begin(),
                                domain.guardPredicates.end(),
                                [&](const auto &predicate) {
                                    return predicate.requiredValue ==
                                               candidate.requiredValue &&
                                        sameDeviceExpr(predicate.condition,
                                                       condition);
                                });
            });
        if (commonAtomic && commonPhase)
            return FormalGuard{true, condition.paramIdx,
                               !candidate.requiredValue};
    }
    return std::nullopt;
}

bool materializableIntervalExpr(const ArgRef &expr,
                                const KernelTemplate &kernel) {
    switch (expr.kind) {
        case ArgRef::Kind::ConstI64:
            return true;
        case ArgRef::Kind::Param:
            return expr.paramIdx < kernel.params.size() &&
                   kernel.params[expr.paramIdx].typeStr == "i64";
        case ArgRef::Kind::Cast:
            // ArgRef v1 does not carry the source/result bit widths of a
            // cast. Replaying it as an i64 no-op could change zext/sext/trunc
            // semantics, so the first materializer rejects every cast.
            return false;
        case ArgRef::Kind::BinOp:
            if (expr.children.size() != 2 ||
                !materializableIntervalExpr(expr.children[0], kernel) ||
                !materializableIntervalExpr(expr.children[1], kernel))
                return false;
            if (expr.opStr == "shl")
                return expr.children[1].kind == ArgRef::Kind::ConstI64 &&
                       expr.children[1].constVal >= 0 &&
                       expr.children[1].constVal < 64;
            return expr.opStr == "add" || expr.opStr == "sub" ||
                   expr.opStr == "mul" || expr.opStr == "and" ||
                   expr.opStr == "or" || expr.opStr == "xor";
        case ArgRef::Kind::Derived:
        case ArgRef::Kind::LoopIv:
        case ArgRef::Kind::FieldLoad:
            return false;
    }
    return false;
}

struct TransferInterval {
    ArgRef offset;
    ArgRef size;
};

struct FissionPlan {
    unsigned pointerParam = 0;
    unsigned bufferParam = 0;
    FormalGuard sideGuard;
    ProducerStoreDomainFact producerStore;
    std::string completionSiteId;
    std::vector<std::string> transferSiteIds;
    std::vector<TransferInterval> intervals;
};

std::optional<FissionPlan> buildFissionPlan(const KernelTemplate &kernel,
                                            std::string &reason) {
    std::vector<const OpTemplate *> transfers;
    std::string completion;
    for (const OpTemplate &op : kernel.ops) {
        if (op.kind != "put_no_db" && op.kind != "get_no_db") continue;
        if (!op.hk_capable || op.loop.inLoop ||
            op.guard.kind != GuardSpec::Kind::Always) {
            reason =
                "first materializer requires unconditional non-loop HK transfers";
            return std::nullopt;
        }
        if (op.completion_site_id.empty()) {
            reason = "a transfer has no compiler-proved completion";
            return std::nullopt;
        }
        if (completion.empty()) completion = op.completion_site_id;
        if (op.completion_site_id != completion) {
            reason = "first materializer supports one completion group";
            return std::nullopt;
        }
        if (op.kind != "put_no_db") {
            reason = "mixed PUT/GET groups are not materialized";
            return std::nullopt;
        }
        transfers.push_back(&op);
    }
    if (transfers.empty()) {
        reason = "kernel has no completed PUT group";
        return std::nullopt;
    }

    const ProducerFrontierFacts &frontier =
        transfers.front()->producer_frontier;
    if (!frontier.analyzed || !frontier.write_footprint_known ||
        frontier.unknown_write_sites != 0 ||
        !frontier.buffer_identity_guardable ||
        !frontier.producer_domains_known ||
        frontier.ordinary_store_sites != 1 ||
        frontier.producer_store_domains.size() != 1 ||
        !frontier.producer_store_domains.front().domainExact ||
        !frontier.producer_store_domains.front().partitionRegionExact) {
        reason = "producer frontier lacks an exact whole-region proof";
        return std::nullopt;
    }
    if (frontier.producer_pointer_param >= kernel.params.size() ||
        frontier.source_buffer_index_param >= kernel.params.size() ||
        kernel.params[frontier.producer_pointer_param].typeStr != "ptr" ||
        kernel.params[frontier.source_buffer_index_param].typeStr != "i32") {
        reason = "producer identity guard formals are mistyped";
        return std::nullopt;
    }
    auto guard = disablingGuard(kernel, frontier);
    if (!guard) {
        reason = "non-duplicable operations have no shared disabling guard";
        return std::nullopt;
    }

    FissionPlan result;
    result.pointerParam = frontier.producer_pointer_param;
    result.bufferParam = frontier.source_buffer_index_param;
    result.sideGuard = *guard;
    result.producerStore = frontier.producer_store_domains.front();
    result.completionSiteId = completion;
    for (const OpTemplate *op : transfers) {
        const ProducerFrontierFacts &other = op->producer_frontier;
        auto otherGuard = disablingGuard(kernel, other);
        if (!other.analyzed || !other.buffer_identity_guardable ||
            !other.producer_domains_known ||
            other.producer_pointer_param != result.pointerParam ||
            other.source_buffer_index_param != result.bufferParam ||
            other.atomic_write_sites != frontier.atomic_write_sites ||
            other.phase_sensitive_sites != frontier.phase_sensitive_sites ||
            other.producer_store_domains.size() != 1 ||
            !sameStoreDomain(other.producer_store_domains.front(),
                             result.producerStore) ||
            !otherGuard || otherGuard->present != result.sideGuard.present ||
            (otherGuard->present &&
             (otherGuard->param != result.sideGuard.param ||
              otherGuard->value != result.sideGuard.value))) {
            reason = "completion-group producer facts disagree";
            return std::nullopt;
        }
        const auto source = op->args.find("src_buf");
        const auto offset = op->args.find("src_off");
        const auto size = op->args.find("size");
        if (source == op->args.end() || offset == op->args.end() ||
            size == op->args.end() ||
            source->second.kind != ArgRef::Kind::Param ||
            source->second.paramIdx != result.bufferParam ||
            !materializableIntervalExpr(offset->second, kernel) ||
            !materializableIntervalExpr(size->second, kernel)) {
            reason = "a transfer interval is not exactly host-materializable";
            return std::nullopt;
        }
        result.transferSiteIds.push_back(op->siteId);
        result.intervals.push_back({offset->second, size->second});
    }
    reason = "all host-side producer-fission guards re-proved";
    return result;
}

bool sameFissionPlan(const FissionPlan &left, const FissionPlan &right) {
    if (left.pointerParam != right.pointerParam ||
        left.bufferParam != right.bufferParam ||
        left.sideGuard.present != right.sideGuard.present ||
        left.sideGuard.param != right.sideGuard.param ||
        left.sideGuard.value != right.sideGuard.value ||
        left.completionSiteId != right.completionSiteId ||
        left.transferSiteIds != right.transferSiteIds ||
        left.intervals.size() != right.intervals.size() ||
        !sameStoreDomain(left.producerStore, right.producerStore))
        return false;
    for (size_t i = 0; i < left.intervals.size(); ++i)
        if (!sameArgRef(left.intervals[i].offset,
                        right.intervals[i].offset) ||
            !sameArgRef(left.intervals[i].size, right.intervals[i].size))
            return false;
    return true;
}

bool auditDelayedDwqRoute(const FissionPlan &plan, std::string &reason) {
    const auto &config = getConfig();
    if (config.hintIn.empty()) {
        reason = "producer fission requires an explicit DWQ hint file";
        return false;
    }
    HintFile hint;
    if (!readHintFile(config.hintIn, hint)) {
        reason = "producer fission could not validate its hint file";
        return false;
    }
    for (const std::string &siteId : plan.transferSiteIds) {
        SiteHint selected = hintFor(hint, siteId);
        if (selected.dispatch != DispatchKind::DwqTrigger) {
            reason =
                "every fission transfer must use DWQ_TRIGGER";
            return false;
        }
        const bool explicitRequest = selected.transform ==
            CommunicationTransform::ProducerFrontierTwoPhase;
        if ((!config.producerFissionOracle && !explicitRequest) ||
            (config.producerFissionOracle &&
             selected.transform != CommunicationTransform::None &&
             !explicitRequest)) {
            reason =
                "every fission transfer must request the compiler-owned "
                "PRODUCER_FRONTIER_TWO_PHASE transform";
            return false;
        }
    }
    reason = "every transfer is staged for delayed DWQ trigger";
    return true;
}

Value *evaluateIntervalExpr(IRBuilder<> &builder, const ArgRef &expr,
                            const std::vector<AllocaInst *> &cells) {
    Type *i64 = builder.getInt64Ty();
    switch (expr.kind) {
        case ArgRef::Kind::ConstI64:
            return ConstantInt::get(i64, expr.constVal, true);
        case ArgRef::Kind::Param: {
            if (expr.paramIdx >= cells.size()) return nullptr;
            Value *value = builder.CreateLoad(
                cells[expr.paramIdx]->getAllocatedType(), cells[expr.paramIdx],
                "gicc.interval.param");
            if (!value->getType()->isIntegerTy()) return nullptr;
            return builder.CreateIntCast(value, i64, false,
                                         "gicc.interval.i64");
        }
        case ArgRef::Kind::Cast:
            return nullptr;
        case ArgRef::Kind::BinOp:
            break;
        case ArgRef::Kind::Derived:
        case ArgRef::Kind::LoopIv:
        case ArgRef::Kind::FieldLoad:
            return nullptr;
    }
    if (expr.children.size() != 2) return nullptr;
    Value *left = evaluateIntervalExpr(builder, expr.children[0], cells);
    Value *right = evaluateIntervalExpr(builder, expr.children[1], cells);
    if (!left || !right) return nullptr;
    if (expr.opStr == "add") return builder.CreateAdd(left, right);
    if (expr.opStr == "sub") return builder.CreateSub(left, right);
    if (expr.opStr == "mul") return builder.CreateMul(left, right);
    if (expr.opStr == "shl") return builder.CreateShl(left, right);
    if (expr.opStr == "and") return builder.CreateAnd(left, right);
    if (expr.opStr == "or") return builder.CreateOr(left, right);
    if (expr.opStr == "xor") return builder.CreateXor(left, right);
    return nullptr;
}

CallInst *cloneLaunch(IRBuilder<> &builder, const CallInst &launch) {
    auto *clone = cast<CallInst>(launch.clone());
    clone->setTailCallKind(CallInst::TCK_None);
    builder.Insert(clone);
    return clone;
}

bool materializeHostFission(Function &wrapper, const FissionPlan &plan,
                            const AuditedLaunch &audited) {
    if (!audited.launch || wrapper.arg_empty() ||
        !wrapper.getArg(0)->getType()->isPointerTy())
        return false;
    CallInst *launch = audited.launch;
    IRBuilder<> guardBuilder(launch);
    Module &module = *wrapper.getParent();
    Type *ptr = guardBuilder.getPtrTy();
    Type *i32 = guardBuilder.getInt32Ty();
    Type *i64 = guardBuilder.getInt64Ty();
    FunctionCallee identity = module.getOrInsertFunction(
        "gicc_runtime_kernel_arg_matches_local_buffer",
        FunctionType::get(i32, {ptr, ptr, i32, i32}, false));
    FunctionCallee interval = module.getOrInsertFunction(
        "gicc_runtime_local_buffer_contains_interval",
        FunctionType::get(i32, {ptr, ptr, i32, i64, i64}, false));
    FunctionCallee setPhase = module.getOrInsertFunction(
        "gicc_runtime_set_schedule_phase_from_kernel_args",
        FunctionType::get(guardBuilder.getVoidTy(), {ptr, i32, ptr}, false));

    Value *runtime = wrapper.getArg(0);
    Value *params = launch->getArgOperand(5);
    Value *stream = launch->getArgOperand(7);
    Value *guard = guardBuilder.CreateICmpNE(
        guardBuilder.CreateCall(
            identity,
            {runtime, params, guardBuilder.getInt32(plan.pointerParam),
             guardBuilder.getInt32(plan.bufferParam)},
            "gicc.buffer.identity"),
        guardBuilder.getInt32(0), "gicc.identity.ok");

    for (const TransferInterval &transfer : plan.intervals) {
        Value *offset = evaluateIntervalExpr(
            guardBuilder, transfer.offset, audited.cells);
        Value *size = evaluateIntervalExpr(
            guardBuilder, transfer.size, audited.cells);
        if (!offset || !size) return false;
        Value *valid = guardBuilder.CreateICmpNE(
            guardBuilder.CreateCall(
                interval,
                {runtime, params, guardBuilder.getInt32(plan.bufferParam),
                 offset, size},
                "gicc.interval.valid"),
            guardBuilder.getInt32(0), "gicc.interval.ok");
        guard = guardBuilder.CreateAnd(guard, valid, "gicc.guards");
    }
    if (plan.sideGuard.present) {
        if (plan.sideGuard.param >= audited.cells.size()) return false;
        AllocaInst *cell = audited.cells[plan.sideGuard.param];
        Value *value = guardBuilder.CreateLoad(cell->getAllocatedType(), cell,
                                               "gicc.side.guard");
        if (!value->getType()->isIntegerTy()) return false;
        Value *enabled = plan.sideGuard.value
            ? guardBuilder.CreateICmpNE(
                  value, ConstantInt::get(value->getType(), 0))
            : guardBuilder.CreateICmpEQ(
                  value, ConstantInt::get(value->getType(), 0));
        guard = guardBuilder.CreateAnd(guard, enabled, "gicc.guards");
    }

    BasicBlock *guardBlock = launch->getParent();
    Instruction *afterLaunch = launch->getNextNode();
    if (!afterLaunch) return false;
    BasicBlock *fusedBlock = guardBlock->splitBasicBlock(
        launch, "gicc.fission.fused");
    BasicBlock *continuation = fusedBlock->splitBasicBlock(
        afterLaunch, "gicc.fission.cont");
    BasicBlock *phasedBlock = BasicBlock::Create(
        module.getContext(), "gicc.fission.phased", &wrapper, fusedBlock);

    guardBlock->getTerminator()->eraseFromParent();
    IRBuilder<> branchBuilder(guardBlock);
    branchBuilder.CreateCondBr(guard, phasedBlock, fusedBlock);

    IRBuilder<> phasedBuilder(phasedBlock);
    phasedBuilder.CreateCall(
        setPhase,
        {params, phasedBuilder.getInt32(kScheduleProducerFrontier), stream});
    cloneLaunch(phasedBuilder, *launch);
    phasedBuilder.CreateCall(
        setPhase,
        {params, phasedBuilder.getInt32(kScheduleRemainder), stream});
    cloneLaunch(phasedBuilder, *launch);
    phasedBuilder.CreateCall(
        setPhase,
        {params, phasedBuilder.getInt32(kScheduleOriginal), stream});
    phasedBuilder.CreateBr(continuation);
    return true;
}

Type *integerTypeForDeviceExpr(IRBuilder<> &builder, StringRef type) {
    auto bits = metadataIntegerBits(type);
    return bits ? builder.getIntNTy(*bits) : nullptr;
}

Value *amdgcnDimension(IRBuilder<> &builder, Module &module,
                       StringRef builtin) {
    StringRef axis;
    if (builtin.ends_with("_x")) axis = "x";
    else if (builtin.ends_with("_y")) axis = "y";
    else if (builtin.ends_with("_z")) axis = "z";
    else return nullptr;

    if (builtin.starts_with("thread_id_")) {
        std::string name = ("llvm.amdgcn.workitem.id." + axis).str();
        return builder.CreateCall(module.getOrInsertFunction(
            name, FunctionType::get(builder.getInt32Ty(), {}, false)));
    }
    if (builtin.starts_with("block_id_")) {
        std::string name = ("llvm.amdgcn.workgroup.id." + axis).str();
        return builder.CreateCall(module.getOrInsertFunction(
            name, FunctionType::get(builder.getInt32Ty(), {}, false)));
    }
    if (!builtin.starts_with("block_size_")) return nullptr;

    unsigned axisIndex = axis == "x" ? 0 : axis == "y" ? 1 : 2;
    Function *dispatch = Intrinsic::getDeclaration(
        &module, Intrinsic::amdgcn_dispatch_ptr);
    Value *packet = builder.CreateCall(dispatch, {}, "gicc.dispatch.ptr");
    Value *field = builder.CreateGEP(
        builder.getInt8Ty(), packet,
        builder.getInt64(4 + 2 * axisIndex), "gicc.block.size.ptr");
    Value *size = builder.CreateAlignedLoad(
        builder.getInt16Ty(), field, Align(2), false, "gicc.block.size.i16");
    return builder.CreateZExt(size, builder.getInt32Ty(),
                             "gicc.block.size");
}

Value *evaluateDeviceExpr(IRBuilder<> &builder, Function &kernel,
                          const DeviceExpr &expr) {
    Type *resultType = integerTypeForDeviceExpr(builder, expr.typeStr);
    if (!resultType) return nullptr;
    switch (expr.kind) {
        case DeviceExpr::Kind::Param: {
            if (expr.paramIdx >= kernel.arg_size()) return nullptr;
            Value *value = kernel.getArg(expr.paramIdx);
            return value->getType() == resultType ? value : nullptr;
        }
        case DeviceExpr::Kind::ConstI64:
            return ConstantInt::get(resultType, expr.constVal, true);
        case DeviceExpr::Kind::Builtin: {
            Value *value = amdgcnDimension(
                builder, *kernel.getParent(), expr.opStr);
            return value && value->getType() == resultType ? value : nullptr;
        }
        case DeviceExpr::Kind::Cast: {
            if (expr.children.size() != 1) return nullptr;
            Value *value = evaluateDeviceExpr(
                builder, kernel, expr.children.front());
            if (!value || !value->getType()->isIntegerTy()) return nullptr;
            if (expr.opStr == "sext")
                return builder.CreateSExtOrTrunc(value, resultType);
            if (expr.opStr == "zext")
                return builder.CreateZExtOrTrunc(value, resultType);
            if (expr.opStr == "trunc")
                return builder.CreateTrunc(value, resultType);
            return nullptr;
        }
        case DeviceExpr::Kind::BinOp:
        case DeviceExpr::Kind::Compare:
            break;
        case DeviceExpr::Kind::Select:
        case DeviceExpr::Kind::Unknown:
            return nullptr;
    }

    if (expr.children.size() != 2) return nullptr;
    Value *left = evaluateDeviceExpr(builder, kernel, expr.children[0]);
    Value *right = evaluateDeviceExpr(builder, kernel, expr.children[1]);
    if (!left || !right || left->getType() != right->getType()) return nullptr;
    if (expr.kind == DeviceExpr::Kind::BinOp) {
        if (left->getType() != resultType) return nullptr;
        if (expr.opStr == "add") return builder.CreateAdd(left, right);
        if (expr.opStr == "sub") return builder.CreateSub(left, right);
        if (expr.opStr == "mul") return builder.CreateMul(left, right);
        if (expr.opStr == "and") return builder.CreateAnd(left, right);
        if (expr.opStr == "or") return builder.CreateOr(left, right);
        if (expr.opStr == "xor") return builder.CreateXor(left, right);
        return nullptr;
    }
    if (!resultType->isIntegerTy(1)) return nullptr;
    if (expr.opStr == "eq") return builder.CreateICmpEQ(left, right);
    if (expr.opStr == "ne") return builder.CreateICmpNE(left, right);
    if (expr.opStr == "slt") return builder.CreateICmpSLT(left, right);
    if (expr.opStr == "sle") return builder.CreateICmpSLE(left, right);
    if (expr.opStr == "sgt") return builder.CreateICmpSGT(left, right);
    if (expr.opStr == "sge") return builder.CreateICmpSGE(left, right);
    if (expr.opStr == "ult") return builder.CreateICmpULT(left, right);
    if (expr.opStr == "ule") return builder.CreateICmpULE(left, right);
    if (expr.opStr == "ugt") return builder.CreateICmpUGT(left, right);
    if (expr.opStr == "uge") return builder.CreateICmpUGE(left, right);
    return nullptr;
}

Value *evaluateDeviceInterval(IRBuilder<> &builder, Function &kernel,
                              const ArgRef &expr) {
    Type *i64 = builder.getInt64Ty();
    switch (expr.kind) {
        case ArgRef::Kind::ConstI64:
            return ConstantInt::get(i64, expr.constVal, true);
        case ArgRef::Kind::Param:
            if (expr.paramIdx >= kernel.arg_size() ||
                !kernel.getArg(expr.paramIdx)->getType()->isIntegerTy(64))
                return nullptr;
            return kernel.getArg(expr.paramIdx);
        case ArgRef::Kind::BinOp:
            break;
        case ArgRef::Kind::Cast:
        case ArgRef::Kind::Derived:
        case ArgRef::Kind::LoopIv:
        case ArgRef::Kind::FieldLoad:
            return nullptr;
    }
    if (expr.children.size() != 2) return nullptr;
    Value *left = evaluateDeviceInterval(builder, kernel, expr.children[0]);
    Value *right = evaluateDeviceInterval(builder, kernel, expr.children[1]);
    if (!left || !right) return nullptr;
    if (expr.opStr == "add") return builder.CreateAdd(left, right);
    if (expr.opStr == "sub") return builder.CreateSub(left, right);
    if (expr.opStr == "mul") return builder.CreateMul(left, right);
    if (expr.opStr == "shl") return builder.CreateShl(left, right);
    if (expr.opStr == "and") return builder.CreateAnd(left, right);
    if (expr.opStr == "or") return builder.CreateOr(left, right);
    if (expr.opStr == "xor") return builder.CreateXor(left, right);
    return nullptr;
}

bool materializableDeviceExpr(const DeviceExpr &expr,
                              const Function &kernel) {
    auto bits = metadataIntegerBits(expr.typeStr);
    if (!bits) return false;
    switch (expr.kind) {
        case DeviceExpr::Kind::Param:
            return expr.paramIdx < kernel.arg_size() &&
                kernel.getArg(expr.paramIdx)->getType()->isIntegerTy(*bits);
        case DeviceExpr::Kind::ConstI64:
            return true;
        case DeviceExpr::Kind::Builtin:
            return *bits == 32 &&
                (expr.opStr == "thread_id_x" ||
                 expr.opStr == "thread_id_y" ||
                 expr.opStr == "thread_id_z" ||
                 expr.opStr == "block_id_x" ||
                 expr.opStr == "block_id_y" ||
                 expr.opStr == "block_id_z" ||
                 expr.opStr == "block_size_x" ||
                 expr.opStr == "block_size_y" ||
                 expr.opStr == "block_size_z");
        case DeviceExpr::Kind::Cast:
            if (expr.children.size() != 1 ||
                !materializableDeviceExpr(expr.children.front(), kernel))
                return false;
            if (auto childBits =
                    metadataIntegerBits(expr.children.front().typeStr)) {
                if (expr.opStr == "trunc") return *bits < *childBits;
                if (expr.opStr == "sext" || expr.opStr == "zext")
                    return *bits > *childBits;
            }
            return false;
        case DeviceExpr::Kind::BinOp:
            if (expr.children.size() != 2 ||
                (expr.opStr != "add" && expr.opStr != "sub" &&
                 expr.opStr != "mul" && expr.opStr != "and" &&
                 expr.opStr != "or" && expr.opStr != "xor"))
                return false;
            break;
        case DeviceExpr::Kind::Compare:
            if (expr.children.size() != 2 ||
                (expr.opStr != "eq" && expr.opStr != "ne" &&
                 expr.opStr != "slt" && expr.opStr != "sle" &&
                 expr.opStr != "sgt" && expr.opStr != "sge" &&
                 expr.opStr != "ult" && expr.opStr != "ule" &&
                 expr.opStr != "ugt" && expr.opStr != "uge"))
                return false;
            break;
        case DeviceExpr::Kind::Select:
        case DeviceExpr::Kind::Unknown:
            return false;
    }
    return (expr.kind != DeviceExpr::Kind::Compare || *bits == 1) &&
        (expr.kind != DeviceExpr::Kind::BinOp ||
         expr.typeStr == expr.children[0].typeStr) &&
        materializableDeviceExpr(expr.children[0], kernel) &&
        materializableDeviceExpr(expr.children[1], kernel) &&
        expr.children[0].typeStr == expr.children[1].typeStr;
}

struct DeviceRegion {
    StoreInst *store = nullptr;
    BranchInst *branch = nullptr;
    unsigned activeSuccessor = 0;
    BasicBlock *merge = nullptr;
};

std::optional<DeviceRegion> auditProducerRegion(
        Function &kernel, const FissionPlan &plan, DominatorTree &DT,
        PostDominatorTree &PDT, std::string &reason) {
    if (plan.pointerParam >= kernel.arg_size() ||
        !kernel.getArg(plan.pointerParam)->getType()->isPointerTy()) {
        reason = "producer pointer formal is absent from final device IR";
        return std::nullopt;
    }
    Argument *root = kernel.getArg(plan.pointerParam);
    SmallVector<StoreInst *, 2> stores;
    for (BasicBlock &BB : kernel) {
        for (Instruction &I : BB) {
            auto *store = dyn_cast<StoreInst>(&I);
            if (!store || store->isAtomic() || store->isVolatile()) continue;
            const Value *base = getUnderlyingObject(
                store->getPointerOperand()->stripPointerCasts());
            if (base && base->stripPointerCasts() == root)
                stores.push_back(store);
        }
    }
    if (stores.size() != 1) {
        reason = "final device IR does not have one producer-rooted store";
        return std::nullopt;
    }
    StoreInst *store = stores.front();
    TypeSize storeSize = kernel.getParent()->getDataLayout().getTypeStoreSize(
        store->getValueOperand()->getType());
    if (storeSize.isScalable() ||
        storeSize.getFixedValue() != plan.producerStore.byteSize) {
        reason = "final producer store size disagrees with metadata";
        return std::nullopt;
    }

    std::optional<DeviceRegion> best;
    unsigned bestLevel = 0;
    bool tied = false;
    for (BasicBlock &BB : kernel) {
        auto *branch = dyn_cast<BranchInst>(BB.getTerminator());
        if (!branch || !branch->isConditional()) continue;
        const bool zero = DT.dominates(branch->getSuccessor(0),
                                       store->getParent());
        const bool one = DT.dominates(branch->getSuccessor(1),
                                      store->getParent());
        if (zero == one) continue;
        const unsigned active = zero ? 0 : 1;
        BasicBlock *target = branch->getSuccessor(active);
        const auto *domNode = DT.getNode(&BB);
        const auto *postNode = PDT.getNode(&BB);
        if (!target->hasNPredecessors(1) || !domNode || !postNode ||
            !postNode->getIDom())
            continue;
        BasicBlock *merge = postNode->getIDom()->getBlock();
        if (!merge || merge == target || !PDT.dominates(merge, target))
            continue;

        SmallPtrSet<BasicBlock *, 16> region;
        SmallVector<BasicBlock *, 16> work{target};
        bool valid = true;
        while (!work.empty() && valid) {
            BasicBlock *current = work.pop_back_val();
            if (current == merge || !region.insert(current).second) continue;
            if (!DT.dominates(target, current) || current == &BB) {
                valid = false;
                break;
            }
            for (BasicBlock *successor : successors(current))
                work.push_back(successor);
        }
        if (!valid || !region.contains(store->getParent())) continue;
        for (BasicBlock *current : region) {
            for (BasicBlock *predecessor : predecessors(current)) {
                if (current == target && predecessor == &BB) continue;
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
            best = DeviceRegion{store, branch, active, merge};
            bestLevel = level;
            tied = false;
        } else if (level == bestLevel) {
            tied = true;
        }
    }
    if (!best || tied) {
        reason = "final device IR has no unique producer-region edge";
        return std::nullopt;
    }
    const auto &domain = plan.producerStore;
    if (domain.partitionPredicateIndex >= domain.predicates.size() ||
        domain.predicates[domain.partitionPredicateIndex].requiredValue !=
            (best->activeSuccessor == 0)) {
        reason = "final producer-region edge disagrees with metadata";
        return std::nullopt;
    }
    reason = "final producer store and single-entry region re-proved";
    return best;
}

struct DeviceCommunication {
    SmallVector<CallInst *, 4> transfers;
    CallInst *lastTransfer = nullptr;
    CallInst *flush = nullptr;
    CallInst *quiet = nullptr;
};

bool callUsesContext(const CallInst &call, const Argument &context) {
    if (call.arg_empty()) return false;
    const Value *root = getUnderlyingObject(
        call.getArgOperand(0)->stripPointerCasts());
    return root && root->stripPointerCasts() == &context;
}

std::optional<DeviceCommunication> auditDeviceCommunication(
        const GICCKernelInfo &info, const FissionPlan &plan,
        DominatorTree &DT, PostDominatorTree &PDT, std::string &reason) {
    if (!info.kernel || info.kernel->arg_empty()) {
        reason = "device kernel has no context formal";
        return std::nullopt;
    }
    const Argument &context = *info.kernel->getArg(0);
    DeviceCommunication result;
    for (const GICCCallSite &site : info.sites) {
        const bool selected = std::find(plan.transferSiteIds.begin(),
                                        plan.transferSiteIds.end(),
                                        site.siteId) !=
                              plan.transferSiteIds.end();
        if (site.kind == GICCOpKind::PutNoDb && selected) {
            result.transfers.push_back(site.CI);
        } else if (site.kind == GICCOpKind::Flush &&
                   site.siteId == plan.completionSiteId && !result.flush) {
            result.flush = site.CI;
        } else if (site.kind == GICCOpKind::Quiet && !result.quiet) {
            result.quiet = site.CI;
        } else {
            reason = "kernel has communication outside the fission group";
            return std::nullopt;
        }
        if (!callUsesContext(*site.CI, context)) {
            reason = "a communication call is not rooted at context formal 0";
            return std::nullopt;
        }
    }
    if (result.transfers.size() != plan.transferSiteIds.size() ||
        !result.flush || !result.quiet) {
        reason = "fission group lacks one flush and one quiet";
        return std::nullopt;
    }

    for (CallInst *candidate : result.transfers) {
        const bool afterAll = std::all_of(
            result.transfers.begin(), result.transfers.end(),
            [&](CallInst *other) {
                return other == candidate || DT.dominates(other, candidate);
            });
        if (!afterAll) continue;
        if (result.lastTransfer) {
            reason = "fission group has no unique post-issue frontier";
            return std::nullopt;
        }
        result.lastTransfer = candidate;
    }
    if (!result.lastTransfer || !result.lastTransfer->getNextNode()) {
        reason = "fission group has no post-issue insertion instruction";
        return std::nullopt;
    }
    if (!DT.dominates(result.lastTransfer, result.flush)) {
        reason = "last transfer does not dominate its flush";
        return std::nullopt;
    }
    if (!PDT.dominates(result.flush->getParent(),
                       result.lastTransfer->getParent())) {
        reason = "flush does not post-dominate the transfer frontier";
        return std::nullopt;
    }
    if (result.lastTransfer->getParent() == result.flush->getParent() &&
        !result.lastTransfer->comesBefore(result.flush)) {
        reason = "flush precedes the transfer frontier";
        return std::nullopt;
    }
    if (!DT.dominates(result.flush, result.quiet)) {
        reason = "quiet is not ordered after flush";
        return std::nullopt;
    }
    reason = "one complete ordered device communication group re-proved";
    return result;
}

void gateCall(CallInst &call, Value *condition, StringRef stem) {
    BasicBlock *parent = call.getParent();
    BasicBlock *callBlock = parent->splitBasicBlock(
        &call, (stem + ".do").str());
    BasicBlock *continuation = callBlock->splitBasicBlock(
        call.getNextNode(), (stem + ".cont").str());
    Instruction *terminator = parent->getTerminator();
    BranchInst::Create(callBlock, continuation, condition, terminator);
    terminator->eraseFromParent();
}

bool materializeDeviceFission(Function &kernel, const FissionPlan &plan,
                              const DeviceRegion &region,
                              const DeviceCommunication &communication) {
    if (kernel.arg_empty() ||
        !kernel.getArg(0)->getType()->isPointerTy() ||
        plan.producerStore.byteSize == 0 ||
        !materializableDeviceExpr(plan.producerStore.byteOffset, kernel))
        return false;

    IRBuilder<> entryBuilder(&*kernel.getEntryBlock().getFirstInsertionPt());
    Value *phasePointer = entryBuilder.CreateGEP(
        entryBuilder.getInt8Ty(), kernel.getArg(0),
        entryBuilder.getInt64(kSchedulePhaseOffset), "gicc.phase.ptr");
    Value *phase = entryBuilder.CreateAlignedLoad(
        entryBuilder.getInt32Ty(), phasePointer, Align(4), false,
        "gicc.phase");
    Value *isProducer = entryBuilder.CreateICmpEQ(
        phase, entryBuilder.getInt32(kScheduleProducerFrontier),
        "gicc.phase.producer");
    Value *isRemainder = entryBuilder.CreateICmpEQ(
        phase, entryBuilder.getInt32(kScheduleRemainder),
        "gicc.phase.remainder");
    Value *isOriginal = entryBuilder.CreateNot(
        entryBuilder.CreateOr(isProducer, isRemainder),
        "gicc.phase.original");
    Value *communicationEnabled = entryBuilder.CreateOr(
        isOriginal, isRemainder, "gicc.phase.communication");

    IRBuilder<> regionBuilder(region.branch);
    Value *storeOffset = evaluateDeviceExpr(
        regionBuilder, kernel, plan.producerStore.byteOffset);
    if (!storeOffset || !storeOffset->getType()->isIntegerTy()) return false;
    if (!storeOffset->getType()->isIntegerTy(64))
        storeOffset = regionBuilder.CreateZExtOrTrunc(
            storeOffset, regionBuilder.getInt64Ty(), "gicc.store.offset");
    Value *storeEnd = regionBuilder.CreateAdd(
        storeOffset, regionBuilder.getInt64(plan.producerStore.byteSize),
        "gicc.store.end");
    Value *overlap = regionBuilder.getFalse();
    for (const TransferInterval &transfer : plan.intervals) {
        Value *offset = evaluateDeviceInterval(
            regionBuilder, kernel, transfer.offset);
        Value *size = evaluateDeviceInterval(
            regionBuilder, kernel, transfer.size);
        if (!offset || !size) return false;
        Value *end = regionBuilder.CreateAdd(offset, size,
                                             "gicc.transfer.end");
        Value *oneOverlap = regionBuilder.CreateAnd(
            regionBuilder.CreateICmpULT(storeOffset, end),
            regionBuilder.CreateICmpULT(offset, storeEnd),
            "gicc.interval.overlap");
        overlap = regionBuilder.CreateOr(overlap, oneOverlap,
                                         "gicc.any.overlap");
    }
    Value *phaseAllows = regionBuilder.CreateOr(
        isOriginal,
        regionBuilder.CreateOr(
            regionBuilder.CreateAnd(isProducer, overlap),
            regionBuilder.CreateAnd(isRemainder,
                                    regionBuilder.CreateNot(overlap))),
        "gicc.phase.compute");
    Value *originalEntry = region.activeSuccessor == 0
        ? region.branch->getCondition()
        : regionBuilder.CreateNot(region.branch->getCondition());
    Value *selectedEntry = regionBuilder.CreateAnd(
        originalEntry, phaseAllows, "gicc.compute.selected");
    region.branch->setCondition(
        region.activeSuccessor == 0
            ? selectedEntry
            : regionBuilder.CreateNot(selectedEntry));

    Instruction *insertBefore = communication.lastTransfer->getNextNode();
    auto *earlyFlush = cast<CallInst>(communication.flush->clone());
    earlyFlush->setArgOperand(
        0, communication.lastTransfer->getArgOperand(0));
    earlyFlush->setMetadata(
        kSyntheticFlushMD,
        MDNode::get(kernel.getContext(), MDString::get(kernel.getContext(),
                                                       "remainder")));
    earlyFlush->insertBefore(insertBefore);

    for (CallInst *transfer : communication.transfers)
        gateCall(*transfer, communicationEnabled, "gicc.fission.transfer");
    gateCall(*earlyFlush, isRemainder, "gicc.fission.early_flush");
    gateCall(*communication.flush, isOriginal, "gicc.fission.original_flush");
    gateCall(*communication.quiet, communicationEnabled,
             "gicc.fission.quiet");
    return true;
}

}  // namespace

PreservedAnalyses GICCProducerFissionHostPass::run(
        Module &module, ModuleAnalysisManager &) {
    const auto &config = getConfig();
    if (config.mode != Mode::Lower ||
        Triple(module.getTargetTriple()).getArch() != Triple::x86_64)
        return PreservedAnalyses::all();

    GICCLaunchInventory inventory =
        collectLaunchInventory(module, config.metaDir);
    SmallPtrSet<Function *, 4> visited;
    bool changed = false;
    for (const GICCLaunchSite &site : inventory.sites) {
        Function *wrapper = site.launchWrapper;
        if (!wrapper || !site.haveTemplate ||
            !visited.insert(wrapper).second)
            continue;
        std::string reason;
        if (!config.producerFissionOracle &&
            !site.kernelTemplate.producer_fission_device_materialized) {
            errs() << "[producer-fission-host] " << site.kernelMangled
                   << ": rejected: final device LTO did not attest the "
                      "producer/remainder partition\n";
            continue;
        }
        auto plan = buildFissionPlan(site.kernelTemplate, reason);
        if (!plan) {
            errs() << "[producer-fission-host] " << site.kernelMangled
                   << ": rejected: " << reason << "\n";
            continue;
        }
        if (!auditDelayedDwqRoute(*plan, reason)) {
            errs() << "[producer-fission-host] " << site.kernelMangled
                   << ": rejected: " << reason << "\n";
            continue;
        }
        AuditedLaunch launch = auditFinalLaunch(*wrapper,
                                                site.kernelTemplate);
        if (!launch.launch) {
            errs() << "[producer-fission-host] " << site.kernelMangled
                   << ": rejected: " << launch.reason << "\n";
            continue;
        }
        if (!materializeHostFission(*wrapper, *plan, launch)) {
            errs() << "[producer-fission-host] " << site.kernelMangled
                   << ": rejected during IR materialization\n";
            continue;
        }
        errs() << "[producer-fission-host] " << site.kernelMangled
               << ": materialized guarded two-phase launch\n";
        changed = true;
    }
    return changed ? PreservedAnalyses::none()
                   : PreservedAnalyses::all();
}

PreservedAnalyses GICCProducerFissionDevicePass::run(
        Module &module, ModuleAnalysisManager &MAM) {
    const auto &config = getConfig();
    if (config.mode != Mode::Lower ||
        !Triple(module.getTargetTriple()).isAMDGCN())
        return PreservedAnalyses::all();

    FunctionAnalysisManager &FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(module).getManager();
    bool changed = false;
    for (Function &kernel : module) {
        if (kernel.isDeclaration() || !isGPUKernel(kernel)) continue;

        KernelTemplate frozen;
        if (!readKernelTemplate(config.metaDir, kernel.getName().str(), frozen))
            continue;
        GICCKernelInfo info;
        collectGICCSites(kernel, info);
        if (info.sites.empty()) continue;

        LoopInfo &LI = FAM.getResult<LoopAnalysis>(kernel);
        DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(kernel);
        ScalarEvolution &SE =
            FAM.getResult<ScalarEvolutionAnalysis>(kernel);
        PostDominatorTree &PDT =
            FAM.getResult<PostDominatorTreeAnalysis>(kernel);
        KernelTemplate current =
            buildKernelTemplate(info, &LI, &DT, &SE, &PDT);

        std::string frozenReason;
        std::string currentReason;
        auto frozenPlan = buildFissionPlan(frozen, frozenReason);
        auto currentPlan = buildFissionPlan(current, currentReason);
        if (!frozenPlan || !currentPlan ||
            !sameFissionPlan(*frozenPlan, *currentPlan)) {
            errs() << "[producer-fission-device] " << kernel.getName()
                   << ": rejected: final compiler facts disagree with "
                      "persisted metadata"
                   << (currentPlan ? "" : (": " + currentReason)) << "\n";
            continue;
        }
        if (!auditDelayedDwqRoute(*currentPlan, currentReason)) {
            errs() << "[producer-fission-device] " << kernel.getName()
                   << ": rejected: " << currentReason << "\n";
            continue;
        }

        std::string regionReason;
        auto region = auditProducerRegion(
            kernel, *currentPlan, DT, PDT, regionReason);
        std::string communicationReason;
        auto communication = auditDeviceCommunication(
            info, *currentPlan, DT, PDT, communicationReason);
        if (!region || !communication) {
            errs() << "[producer-fission-device] " << kernel.getName()
                   << ": rejected: "
                   << (region ? communicationReason : regionReason) << "\n";
            continue;
        }
        if (!materializeDeviceFission(kernel, *currentPlan, *region,
                                      *communication)) {
            errs() << "[producer-fission-device] " << kernel.getName()
                   << ": rejected: device expressions are not materializable\n";
            continue;
        }
        current.producer_fission_device_materialized = true;
        if (!writeKernelTemplate(config.metaDir, current)) {
            errs() << "[producer-fission-device] " << kernel.getName()
                   << ": warning: could not persist device materialization "
                      "attestation; host fission will remain disabled\n";
        }
        errs() << "[producer-fission-device] " << kernel.getName()
               << ": materialized exact producer/remainder partition\n";
        changed = true;
    }
    return changed ? PreservedAnalyses::none()
                   : PreservedAnalyses::all();
}

}  // namespace gicc::pass
