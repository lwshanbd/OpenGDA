#include "GICCProducerFission.h"

#include "GICCHostDiscovery.h"
#include "GICCPassConfig.h"
#include "MetadataIO.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
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
    std::vector<TransferInterval> intervals;
};

std::optional<FissionPlan> buildFissionPlan(const KernelTemplate &kernel,
                                            std::string &reason) {
    std::vector<const OpTemplate *> transfers;
    std::string completion;
    for (const OpTemplate &op : kernel.ops) {
        if (op.kind != "put_no_db" && op.kind != "get_no_db") continue;
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
    for (const OpTemplate *op : transfers) {
        const ProducerFrontierFacts &other = op->producer_frontier;
        auto otherGuard = disablingGuard(kernel, other);
        if (!other.analyzed || !other.buffer_identity_guardable ||
            !other.producer_domains_known ||
            other.producer_pointer_param != result.pointerParam ||
            other.source_buffer_index_param != result.bufferParam ||
            other.atomic_write_sites != frontier.atomic_write_sites ||
            other.phase_sensitive_sites != frontier.phase_sensitive_sites ||
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
        result.intervals.push_back({offset->second, size->second});
    }
    reason = "all host-side producer-fission guards re-proved";
    return result;
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
        auto plan = buildFissionPlan(site.kernelTemplate, reason);
        if (!plan) {
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

}  // namespace gicc::pass
