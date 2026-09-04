#include "GICCFeatureExtraction.h"
#include "GICCHostDiscovery.h"
#include "GICCPassConfig.h"
#include "LaunchSiteInventory.h"
#include "MetadataIO.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

using namespace llvm;

namespace gicc::pass {

namespace {

struct Dim3Facts {
    std::optional<uint32_t> x;
    std::optional<uint32_t> y;
    std::optional<uint32_t> z;
};

struct LaunchGeometry {
    Dim3Facts grid;
    Dim3Facts block;
};

struct LaunchContextFacts {
    LaunchGeometry geometry;
    std::optional<int64_t> sizeBytes;
    std::optional<int64_t> tripCount;
    bool phaseLaunchSupported = false;
    std::string phaseLaunchStream = "unknown";
    std::string phaseLaunchMaterialization = "none";
    std::string phaseLaunchReason = "launch wrapper was not analyzed";
    bool kernelArgumentSlotsExact = false;
    unsigned kernelArgumentSlotCount = 0;
    std::string kernelArgumentSlotReason =
        "HIP kernel parameter array was not analyzed";
    unsigned staticCallsites = 1;
};

struct PhaseLaunchShape {
    bool supported = false;
    std::string stream = "unknown";
    std::string materialization = "none";
    std::string reason;
    bool kernelArgumentSlotsExact = false;
    unsigned kernelArgumentSlotCount = 0;
    std::string kernelArgumentSlotReason;
};

struct OverlapTransferFact {
    std::string siteId;
    ArgRef sourceBuffer;
    ArgRef byteOffset;
    ArgRef byteSize;
};

struct ProducerOverlapPartitionFacts {
    bool analyzed = false;
    bool exact = false;
    unsigned producerPointerParam = 0;
    unsigned sourceBufferParam = 0;
    ProducerStoreDomainFact producerStore;
    std::vector<OverlapTransferFact> transfers;
    bool sideEffectSafetyExact = false;
    std::string sideEffectSafetyMode = "unproved";
    bool hasSideEffectGuard = false;
    unsigned sideEffectGuardParam = 0;
    bool sideEffectGuardValue = false;
    std::string sideEffectSafetyReason;
    std::string reason;
};

std::optional<unsigned> metadataIntegerBits(StringRef metadataType) {
    if (metadataType.consume_front("i")) {
        unsigned bits = 0;
        if (!metadataType.empty() &&
            !metadataType.getAsInteger(10, bits) && bits > 0)
            return bits;
    }
    return std::nullopt;
}

bool parameterCellTypeMatches(const Type *cellType, StringRef metadataType) {
    if (metadataType == "ptr") return cellType->isPointerTy();
    if (auto bits = metadataIntegerBits(metadataType)) {
        if (cellType->isIntegerTy(*bits)) return true;
        // Clang materializes a by-value i1 kernel argument in an i8 host ABI
        // cell before handing its address to hipLaunchKernel.
        return *bits == 1 && cellType->isIntegerTy(8);
    }

    std::string printed;
    raw_string_ostream out(printed);
    cellType->print(out);
    out.flush();
    return printed == metadataType;
}

struct KernelArgumentSlotShape {
    bool exact = false;
    unsigned count = 0;
    std::string reason;
};

// Prove the ABI fact needed by compiler-owned runtime guards and repeated
// launches: slot i in hipLaunchKernel's void** parameter array is a distinct,
// launch-owner-local cell whose storage type matches device formal i.  The
// values need not be compile-time constants.  Device and host LTO therefore
// refer to the same runtime formal namespace without reconstructing source
// expressions such as a lambda capture.
KernelArgumentSlotShape kernelArgumentSlotShape(
        const CallInst &launch, const AllocaInst &params,
        const KernelTemplate &kernel) {
    KernelArgumentSlotShape result;
    result.count = kernel.params.size();
    if (kernel.params.empty()) {
        result.reason = "kernel parameter metadata is empty";
        return result;
    }

    const DataLayout &layout = launch.getModule()->getDataLayout();
    const uint64_t pointerBytes = layout.getPointerSize(
        params.getType()->getPointerAddressSpace());
    if (pointerBytes == 0) {
        result.reason = "target pointer size is unknown";
        return result;
    }
    const auto allocationSize = params.getAllocationSize(layout);
    const uint64_t requiredBytes =
        pointerBytes * static_cast<uint64_t>(kernel.params.size());
    if (!allocationSize || allocationSize->isScalable() ||
        allocationSize->getFixedValue() < requiredBytes) {
        result.reason = "kernel parameter array is smaller than metadata";
        return result;
    }

    std::vector<const AllocaInst *> cells(kernel.params.size(), nullptr);
    for (const BasicBlock &BB : *params.getFunction()) {
        for (const Instruction &I : BB) {
            const auto *store = dyn_cast<StoreInst>(&I);
            if (!store || store->getParent() != launch.getParent() ||
                !store->comesBefore(&launch))
                continue;
            int64_t offset = 0;
            const Value *base = GetPointerBaseWithConstantOffset(
                store->getPointerOperand(), offset, layout);
            if (base != &params || offset < 0 ||
                static_cast<uint64_t>(offset) % pointerBytes != 0)
                continue;
            const uint64_t slot =
                static_cast<uint64_t>(offset) / pointerBytes;
            if (slot >= cells.size()) {
                result.reason =
                    "kernel parameter array has an out-of-range slot store";
                return result;
            }
            const auto *cell = dyn_cast<AllocaInst>(
                store->getValueOperand()->stripPointerCasts());
            if (!cell || cell->getFunction() != params.getFunction()) {
                result.reason =
                    "kernel parameter slot does not name launch-local storage";
                return result;
            }
            if (cells[slot] && cells[slot] != cell) {
                result.reason = "kernel parameter slot is initialized twice";
                return result;
            }
            cells[slot] = cell;
        }
    }

    SmallPtrSet<const AllocaInst *, 16> distinct;
    for (unsigned slot = 0; slot < cells.size(); ++slot) {
        const AllocaInst *cell = cells[slot];
        if (!cell) {
            result.reason = "kernel parameter array has an uninitialized slot";
            return result;
        }
        if (!distinct.insert(cell).second) {
            result.reason = "kernel parameter slots alias the same value cell";
            return result;
        }
        if (!parameterCellTypeMatches(cell->getAllocatedType(),
                                      kernel.params[slot].typeStr)) {
            result.reason =
                "kernel parameter slot type disagrees with device metadata";
            return result;
        }
    }

    result.exact = true;
    result.reason =
        "every HIP parameter slot has distinct launch-local, metadata-typed storage";
    return result;
}

bool isBeforeInBlock(const Instruction &candidate,
                     const Instruction &limit) {
    return candidate.getParent() == limit.getParent() &&
           candidate.comesBefore(&limit);
}

// Before the regular optimizer has inlined dim3's constructor, clang lowers
// a source-level dim3 temporary as:
//
//   call dim3::dim3(tmp, x, y, z)
//   memcpy(abi.tmp, tmp, 12)
//   packed.xy = load i64, abi.tmp
//   launch(..., packed.xy, ...)
//
// Follow that IR def-use/memory chain conservatively.  This is needed because
// the GICC discovery passes intentionally run before the general inliner, so
// post-optimization ConstantInts are not always available yet.
std::optional<Dim3Facts> decodeConstructedDim3(const CallBase &launch,
                                               unsigned packedXYArg) {
    if (launch.arg_size() <= packedXYArg) return std::nullopt;
    const auto *packedLoad =
        dyn_cast<LoadInst>(launch.getArgOperand(packedXYArg));
    if (!packedLoad) return std::nullopt;

    const Value *abiStorage = getUnderlyingObject(
        packedLoad->getPointerOperand());
    // SROA commonly removes the memcpy and leaves the ABI load directly on
    // the dim3 alloca. Start with that form, then replace it if an explicit
    // ABI copy is still present.
    const Value *dimStorage = abiStorage;
    const Instruction *copyPoint = &launch;
    for (const Instruction &inst : *launch.getParent()) {
        if (!isBeforeInBlock(inst, launch)) break;
        const auto *copy = dyn_cast<MemCpyInst>(&inst);
        if (!copy || getUnderlyingObject(copy->getDest()) != abiStorage)
            continue;
        dimStorage = getUnderlyingObject(copy->getSource());
        copyPoint = copy;
    }
    const CallBase *constructor = nullptr;
    for (const Instruction &inst : *launch.getParent()) {
        if (!isBeforeInBlock(inst, *copyPoint)) break;
        const auto *call = dyn_cast<CallBase>(&inst);
        if (!call || call->arg_size() < 4) continue;
        const Function *callee = call->getCalledFunction();
        if (!callee || !callee->getName().contains("4dim3C")) continue;
        if (getUnderlyingObject(call->getArgOperand(0)) != dimStorage)
            continue;
        constructor = call;
    }
    if (!constructor) return std::nullopt;

    Dim3Facts result;
    auto read32 = [&](unsigned arg) -> std::optional<uint32_t> {
        const auto *constant =
            dyn_cast<ConstantInt>(constructor->getArgOperand(arg));
        if (!constant || constant->getBitWidth() > 32) return std::nullopt;
        return static_cast<uint32_t>(constant->getZExtValue());
    };
    result.x = read32(1);
    result.y = read32(2);
    result.z = read32(3);
    return result;
}

// On the x86-64 host ABI used by the LTO pipeline, HIP's 12-byte dim3 is
// lowered to two call operands: an i64 containing x (low 32 bits) and y
// (high 32 bits), followed by an i32 containing z.  Decode only literal
// ConstantInts.  A dynamic operand stays unknown rather than becoming an
// estimate; this is a compiler-fact interface, not a source heuristic.
Dim3Facts decodeDim3(const CallBase &call, unsigned packedXYArg,
                     unsigned zArg) {
    Dim3Facts result;
    if (call.arg_size() <= zArg) return result;

    if (const auto *xy = dyn_cast<ConstantInt>(call.getArgOperand(packedXYArg))) {
        if (xy->getBitWidth() <= 64) {
            const uint64_t packed = xy->getZExtValue();
            result.x = static_cast<uint32_t>(packed & 0xffffffffULL);
            result.y = static_cast<uint32_t>(packed >> 32);
        }
    }
    if (const auto *z = dyn_cast<ConstantInt>(call.getArgOperand(zArg))) {
        if (z->getBitWidth() <= 32)
            result.z = static_cast<uint32_t>(z->getZExtValue());
    }
    if ((!result.x || !result.y || !result.z)) {
        if (auto constructed = decodeConstructedDim3(call, packedXYArg)) {
            if (!result.x) result.x = constructed->x;
            if (!result.y) result.y = constructed->y;
            if (!result.z) result.z = constructed->z;
        }
    }
    return result;
}

LaunchGeometry launchGeometry(const GICCLaunchSite &site) {
    LaunchGeometry result;
    if (!site.callsite) return result;

    // The split i64/i32 representation below is the x86-64 SysV ABI shape.
    // Other host targets must stay unknown until their ABI is implemented.
    const Triple triple(site.callsite->getModule()->getTargetTriple());
    if (triple.getArch() != Triple::x86_64 || site.callsite->arg_size() < 5)
        return result;
    const unsigned widths[] = {64, 32, 64, 32};
    for (unsigned i = 0; i < 4; ++i) {
        const Type *type = site.callsite->getArgOperand(i + 1)->getType();
        if (!type->isIntegerTy(widths[i])) return result;
    }

    // arg 0 is Runtime&. Both launch overloads put grid and block next; the
    // optional shmem/stream operands follow them and do not affect this ABI.
    result.grid = decodeDim3(*site.callsite, 1, 2);
    result.block = decodeDim3(*site.callsite, 3, 4);
    return result;
}

SmallVector<const CallInst *, 2> hipLaunchesIn(const Function &function) {
    SmallVector<const CallInst *, 2> launches;
    for (const BasicBlock &BB : function) {
        for (const Instruction &I : BB) {
            const auto *call = dyn_cast<CallInst>(&I);
            const Function *callee = call ? call->getCalledFunction() : nullptr;
            if (callee && callee->getName() == "hipLaunchKernel")
                launches.push_back(call);
        }
    }
    return launches;
}

unsigned directCallsNamed(const Function &function, StringRef name) {
    unsigned count = 0;
    for (const BasicBlock &BB : function) {
        for (const Instruction &I : BB) {
            const auto *call = dyn_cast<CallBase>(&I);
            const Function *callee = call ? call->getCalledFunction() : nullptr;
            if (callee && callee->getName() == name) ++count;
        }
    }
    return count;
}

// At the early-simplification extension point, HIP still represents a
// source-level kernel launch as an indirect call through a constant global:
//
//   @kernel = constant ptr @__device_stub__kernel
//   %stub = load ptr, ptr @kernel
//   call void %stub(...)
//
// Later optimization normally folds and inlines that stub, leaving the direct
// hipLaunchKernel shape handled below. Resolve only this exact compiler-owned
// global form; arbitrary function-pointer calls remain unsupported.
const Function *indirectHipStub(const CallBase &call,
                                StringRef kernelMangled,
                                const GlobalVariable *&kernelGlobal) {
    kernelGlobal = nullptr;
    if (call.getCalledFunction()) return nullptr;
    const auto *load = dyn_cast<LoadInst>(
        call.getCalledOperand()->stripPointerCasts());
    if (!load) return nullptr;
    const auto *global = dyn_cast<GlobalVariable>(
        load->getPointerOperand()->stripPointerCasts());
    if (!global || global->getName() != kernelMangled ||
        !global->isConstant() || !global->hasInitializer())
        return nullptr;
    const auto *stub = dyn_cast<Function>(
        global->getInitializer()->stripPointerCasts());
    if (!stub || !stub->getName().contains("__device_stub__")) return nullptr;
    kernelGlobal = global;
    return stub;
}

bool hipLaunchTargetMatches(const CallInst &launch, StringRef kernelMangled) {
    const Value *kernel = launch.getArgOperand(0)->stripPointerCasts();
    return kernel->hasName() && kernel->getName() == kernelMangled;
}

unsigned indirectDispatchUses(const Module &module,
                              const GlobalVariable &kernelGlobal) {
    unsigned count = 0;
    for (const Function &function : module) {
        for (const BasicBlock &BB : function) {
            for (const Instruction &I : BB) {
                const auto *call = dyn_cast<CallBase>(&I);
                if (!call || call->getCalledFunction()) continue;
                const auto *load = dyn_cast<LoadInst>(
                    call->getCalledOperand()->stripPointerCasts());
                if (load && load->getPointerOperand()->stripPointerCasts() ==
                                &kernelGlobal)
                    ++count;
            }
        }
    }
    return count;
}

unsigned directDispatchUses(const Module &module, const Function &stub) {
    unsigned count = 0;
    for (const Function &function : module)
        for (const BasicBlock &BB : function)
            for (const Instruction &I : BB)
                if (const auto *call = dyn_cast<CallBase>(&I))
                    if (call->getCalledFunction() == &stub) ++count;
    return count;
}

PhaseLaunchShape phaseLaunchShape(const GICCLaunchSite &site) {
    PhaseLaunchShape result;
    if (!site.callsite || !site.launchWrapper) {
        result.reason = "missing launch call or annotated wrapper";
        return result;
    }
    if (site.kernelTemplate.params.empty()) {
        result.reason = "kernel parameter metadata is empty";
        return result;
    }

    const unsigned userParams = site.kernelTemplate.params.size() - 1;
    if (site.callsite->arg_size() == 5 + userParams)
        result.stream = "default";
    else if (site.callsite->arg_size() == 7 + userParams)
        result.stream = "explicit";
    else {
        result.reason =
            "launch operand count does not match a supported wrapper ABI";
        return result;
    }

    SmallVector<const CallInst *, 2> launches =
        hipLaunchesIn(*site.launchWrapper);
    const Function *launchOwner = site.launchWrapper;
    const CallBase *stubDispatch = nullptr;
    const Function *stub = nullptr;
    const GlobalVariable *kernelGlobal = nullptr;
    std::string materialization = "wrapper";

    // If the stub has not yet been folded/inlined, accept exactly one
    // compiler-generated wrapper-to-stub edge and prove that changing the
    // stub cannot affect another source-level kernel launch.
    SmallVector<const CallBase *, 2> stubDispatches;
    for (const BasicBlock &BB : *site.launchWrapper) {
        for (const Instruction &I : BB) {
            const auto *call = dyn_cast<CallBase>(&I);
            if (!call) continue;
            if (const Function *callee = call->getCalledFunction()) {
                if (callee->getName().contains("__device_stub__"))
                    stubDispatches.push_back(call);
                continue;
            }
            const GlobalVariable *candidateGlobal = nullptr;
            if (const Function *candidate = indirectHipStub(
                    *call, site.kernelMangled, candidateGlobal)) {
                if (!stub) {
                    stub = candidate;
                    kernelGlobal = candidateGlobal;
                } else if (stub != candidate || kernelGlobal != candidateGlobal) {
                    result.reason =
                        "wrapper dispatches through multiple HIP device stubs";
                    return result;
                }
                stubDispatches.push_back(call);
            }
        }
    }

    if (!launches.empty() && !stubDispatches.empty()) {
        result.reason = "wrapper contains both direct and stub kernel launches";
        return result;
    }
    if (launches.empty()) {
        if (stubDispatches.size() != 1 || !isa<CallInst>(stubDispatches.front())) {
            result.reason =
                "annotated wrapper must contain exactly one call-form HIP stub dispatch";
            return result;
        }
        stubDispatch = stubDispatches.front();
        if (!stub) stub = stubDispatch->getCalledFunction();
        if (!stub || stub->isDeclaration()) {
            result.reason = "HIP device stub body is unavailable";
            return result;
        }
        launches = hipLaunchesIn(*stub);
        launchOwner = stub;
        if (directCallsNamed(*site.launchWrapper,
                             "__hipPushCallConfiguration") != 1 ||
            directCallsNamed(*stub, "__hipPopCallConfiguration") != 1) {
            result.reason =
                "wrapper/stub launch configuration chain is not unique";
            return result;
        }
        const Module &module = *site.launchWrapper->getParent();
        const unsigned uses = directDispatchUses(module, *stub) +
            (kernelGlobal ? indirectDispatchUses(module, *kernelGlobal) : 0);
        if (uses != 1) {
            result.reason =
                "HIP device stub is shared by another kernel dispatch";
            return result;
        }
        materialization = "device_stub";
    }

    if (launches.size() != 1) {
        result.reason =
            "launch owner must contain exactly one hipLaunchKernel call";
        return result;
    }
    const CallInst *launch = launches.front();
    if (launch->arg_size() != 8 ||
        !launch->getArgOperand(0)->getType()->isPointerTy() ||
        !launch->getArgOperand(5)->getType()->isPointerTy() ||
        !launch->getArgOperand(6)->getType()->isIntegerTy() ||
        !launch->getArgOperand(7)->getType()->isPointerTy()) {
        result.reason =
            "hipLaunchKernel operands do not match the audited ABI";
        return result;
    }
    if (!launch->use_empty()) {
        result.reason = "hipLaunchKernel return value is observed";
        return result;
    }

    if (!hipLaunchTargetMatches(*launch, site.kernelMangled)) {
        result.reason =
            "hipLaunchKernel target does not match compiler metadata";
        return result;
    }
    const Value *params = getUnderlyingObject(
        launch->getArgOperand(5)->stripPointerCasts());
    const auto *alloca = dyn_cast<AllocaInst>(params);
    if (!alloca || alloca->getFunction() != launchOwner) {
        result.reason =
            "kernel parameter array lifetime is not launch-owner-local";
        return result;
    }

    const KernelArgumentSlotShape slots =
        kernelArgumentSlotShape(*launch, *alloca, site.kernelTemplate);
    result.kernelArgumentSlotsExact = slots.exact;
    result.kernelArgumentSlotCount = slots.count;
    result.kernelArgumentSlotReason = slots.reason;
    if (!slots.exact) {
        result.reason = slots.reason;
        return result;
    }

    result.supported = true;
    result.materialization = materialization;
    result.reason =
        "one original kernel launch with reusable parameters and unchanged stream";
    return result;
}

json::Value dim3Record(const Dim3Facts &dim) {
    json::Object record;
    if (dim.x) record["x"] = static_cast<int64_t>(*dim.x);
    else       record["x"] = nullptr;
    if (dim.y) record["y"] = static_cast<int64_t>(*dim.y);
    else       record["y"] = nullptr;
    if (dim.z) record["z"] = static_cast<int64_t>(*dim.z);
    else       record["z"] = nullptr;
    return json::Value(std::move(record));
}

std::optional<int64_t> dim3Product(const Dim3Facts &dim) {
    if (!dim.x || !dim.y || !dim.z) return std::nullopt;
    uint64_t product = *dim.x;
    for (uint32_t factor : {*dim.y, *dim.z}) {
        if (factor != 0 &&
            product > static_cast<uint64_t>(
                          std::numeric_limits<int64_t>::max()) / factor)
            return std::nullopt;
        product *= factor;
    }
    return static_cast<int64_t>(product);
}

std::optional<unsigned> wrapperArgForKernelParam(
        const GICCLaunchSite &site, unsigned paramIdx) {
    if (!site.callsite || paramIdx == 0 ||
        paramIdx >= site.kernelTemplate.params.size())
        return std::nullopt;

    // Kernel formal 0 is the DeviceCtx injected by gicc::launch. In the
    // ordinary overload, formal 1 starts at wrapper operand 5 (after Runtime&
    // and the four x86-64 dim3 ABI operands). The shmem/stream overload adds
    // two operands. Only accept an exact scalar-argument count: aggregate
    // kernel parameters need a target-specific ABI implementation.
    const unsigned userParams = site.kernelTemplate.params.size() - 1;
    if (site.callsite->arg_size() == 5 + userParams)
        return 4 + paramIdx;
    if (site.callsite->arg_size() == 7 + userParams)
        return 6 + paramIdx;
    return std::nullopt;
}

std::optional<int64_t> constantInteger(const Value *value) {
    const auto *constant = dyn_cast_or_null<ConstantInt>(value);
    if (!constant || constant->getBitWidth() > 64) return std::nullopt;
    if (constant->getBitWidth() < 64)
        return static_cast<int64_t>(constant->getZExtValue());
    return constant->getSExtValue();
}

std::optional<int64_t> argConstantAtLaunch(const ArgRef &arg,
                                           const GICCLaunchSite &site) {
    switch (arg.kind) {
        case ArgRef::Kind::ConstI64:
            return arg.constVal;
        case ArgRef::Kind::Param: {
            auto wrapperArg = wrapperArgForKernelParam(site, arg.paramIdx);
            if (!wrapperArg) return std::nullopt;
            return constantInteger(site.callsite->getArgOperand(*wrapperArg));
        }
        case ArgRef::Kind::Cast:
            if (arg.children.size() != 1) return std::nullopt;
            return argConstantAtLaunch(arg.children[0], site);
        case ArgRef::Kind::BinOp:
            break;
        case ArgRef::Kind::Derived:
        case ArgRef::Kind::LoopIv:
        case ArgRef::Kind::FieldLoad:
            return std::nullopt;
    }

    if (arg.children.size() != 2) return std::nullopt;
    auto left = argConstantAtLaunch(arg.children[0], site);
    auto right = argConstantAtLaunch(arg.children[1], site);
    if (!left || !right) return std::nullopt;

    __int128 value = 0;
    if (arg.opStr == "add") value = static_cast<__int128>(*left) + *right;
    else if (arg.opStr == "sub") value = static_cast<__int128>(*left) - *right;
    else if (arg.opStr == "mul") value = static_cast<__int128>(*left) * *right;
    else if (arg.opStr == "shl") {
        if (*right < 0 || *right > 62) return std::nullopt;
        value = static_cast<__int128>(*left) << *right;
    } else {
        return std::nullopt;
    }
    if (value < std::numeric_limits<int64_t>::min() ||
        value > std::numeric_limits<int64_t>::max())
        return std::nullopt;
    return static_cast<int64_t>(value);
}

std::optional<int64_t> tripCountForBound(const OpLoopInfo &loop,
                                         int64_t bound) {
    if (!loop.inLoop || loop.degraded || loop.ivStep <= 0 ||
        bound < loop.ivStart)
        return std::nullopt;
    const __int128 distance =
        static_cast<__int128>(bound) - loop.ivStart;
    const __int128 trips =
        (distance + loop.ivStep - 1) / loop.ivStep;
    if (trips > std::numeric_limits<int64_t>::max()) return std::nullopt;
    return static_cast<int64_t>(trips);
}

std::optional<int64_t> tripCountAtLaunch(const OpTemplate &op,
                                         const GICCLaunchSite &site) {
    if (op.trip_count >= 0) return op.trip_count;
    if (!op.loop.inLoop || !op.loop.ivBoundKnown || op.loop.degraded)
        return std::nullopt;
    if (op.loop.ivBoundIsConst)
        return tripCountForBound(op.loop, op.loop.ivBoundConst);

    ArgRef bound;
    bound.kind = ArgRef::Kind::Param;
    bound.paramIdx = op.loop.ivParamIdx;
    auto value = argConstantAtLaunch(bound, site);
    if (!value) return std::nullopt;
    return tripCountForBound(op.loop, *value);
}

LaunchContextFacts launchContext(const GICCLaunchSite &site,
                                 const OpTemplate &op) {
    LaunchContextFacts context;
    context.geometry = launchGeometry(site);
    const PhaseLaunchShape phase = phaseLaunchShape(site);
    context.phaseLaunchSupported = phase.supported;
    context.phaseLaunchStream = phase.stream;
    context.phaseLaunchMaterialization = phase.materialization;
    context.phaseLaunchReason = phase.reason;
    context.kernelArgumentSlotsExact = phase.kernelArgumentSlotsExact;
    context.kernelArgumentSlotCount = phase.kernelArgumentSlotCount;
    context.kernelArgumentSlotReason = phase.kernelArgumentSlotReason;
    if (auto size = op.args.find("size"); size != op.args.end())
        context.sizeBytes = argConstantAtLaunch(size->second, site);
    context.tripCount = tripCountAtLaunch(op, site);
    return context;
}

auto contextKey(const LaunchContextFacts &context) {
    return std::make_tuple(
        context.geometry.grid.x, context.geometry.grid.y,
        context.geometry.grid.z, context.geometry.block.x,
        context.geometry.block.y, context.geometry.block.z,
        context.sizeBytes, context.tripCount,
        context.phaseLaunchSupported, context.phaseLaunchStream,
        context.phaseLaunchMaterialization, context.phaseLaunchReason,
        context.kernelArgumentSlotsExact,
        context.kernelArgumentSlotCount,
        context.kernelArgumentSlotReason);
}

std::vector<LaunchContextFacts> coalesceContexts(
        std::vector<LaunchContextFacts> contexts) {
    std::sort(contexts.begin(), contexts.end(),
              [](const auto &left, const auto &right) {
                  return contextKey(left) < contextKey(right);
              });
    std::vector<LaunchContextFacts> result;
    for (const auto &context : contexts) {
        if (!result.empty() && contextKey(result.back()) == contextKey(context))
            result.back().staticCallsites += context.staticCallsites;
        else
            result.push_back(context);
    }
    return result;
}

template <typename T, typename Getter>
std::optional<T> commonKnownValue(
        const std::vector<LaunchContextFacts> &contexts, Getter get) {
    if (contexts.empty()) return std::nullopt;
    std::optional<T> common = get(contexts.front());
    if (!common) return std::nullopt;
    for (const auto &context : contexts)
        if (get(context) != common) return std::nullopt;
    return common;
}

LaunchGeometry commonGeometry(
        const std::vector<LaunchContextFacts> &contexts) {
    LaunchGeometry common;
    common.grid.x = commonKnownValue<uint32_t>(
        contexts, [](const auto &c) { return c.geometry.grid.x; });
    common.grid.y = commonKnownValue<uint32_t>(
        contexts, [](const auto &c) { return c.geometry.grid.y; });
    common.grid.z = commonKnownValue<uint32_t>(
        contexts, [](const auto &c) { return c.geometry.grid.z; });
    common.block.x = commonKnownValue<uint32_t>(
        contexts, [](const auto &c) { return c.geometry.block.x; });
    common.block.y = commonKnownValue<uint32_t>(
        contexts, [](const auto &c) { return c.geometry.block.y; });
    common.block.z = commonKnownValue<uint32_t>(
        contexts, [](const auto &c) { return c.geometry.block.z; });
    return common;
}

json::Value launchContextRecord(const LaunchContextFacts &context) {
    json::Object record;
    record["static_callsite_count"] =
        static_cast<int64_t>(context.staticCallsites);
    record["launch_grid"] = dim3Record(context.geometry.grid);
    record["launch_block"] = dim3Record(context.geometry.block);
    if (auto blocks = dim3Product(context.geometry.grid))
        record["grid_blocks"] = *blocks;
    else
        record["grid_blocks"] = nullptr;
    if (auto threads = dim3Product(context.geometry.block))
        record["threads_per_block"] = *threads;
    else
        record["threads_per_block"] = nullptr;
    if (context.sizeBytes) record["size_bytes"] = *context.sizeBytes;
    else                   record["size_bytes"] = nullptr;
    if (context.tripCount) record["trip_count"] = *context.tripCount;
    else                   record["trip_count"] = nullptr;
    record["phase_launch_supported"] = context.phaseLaunchSupported;
    record["phase_launch_stream"] = context.phaseLaunchStream;
    record["phase_launch_materialization"] =
        context.phaseLaunchMaterialization;
    record["phase_launch_reason"] = context.phaseLaunchReason;
    record["kernel_argument_slots_exact"] =
        context.kernelArgumentSlotsExact;
    record["kernel_argument_slot_count"] =
        static_cast<int64_t>(context.kernelArgumentSlotCount);
    record["kernel_argument_slot_reason"] =
        context.kernelArgumentSlotReason;
    return json::Value(std::move(record));
}

// Translate ArgRef::Kind to the JSON tag the decider consumes.
const char *argKindTag(ArgRef::Kind k) {
    switch (k) {
        case ArgRef::Kind::Param:     return "param";
        case ArgRef::Kind::ConstI64:  return "const";
        case ArgRef::Kind::BinOp:     return "binop";
        case ArgRef::Kind::Cast:      return "cast";
        case ArgRef::Kind::Derived:   return "derived";
        case ArgRef::Kind::LoopIv:    return "loop_iv";
        case ArgRef::Kind::FieldLoad: return "field_load";
    }
    return "derived";
}

// Preserve the compiler's symbolic descriptor expression for the model
// dossier.  This is deliberately a feature-only representation: the
// lowering still reads and re-proves the original ArgRef from kernel
// metadata, so a model can neither manufacture an expression nor turn this
// record into a legality assertion.
json::Value argRefFactRecord(const ArgRef &arg) {
    json::Object record;
    record["kind"] = argKindTag(arg.kind);
    switch (arg.kind) {
        case ArgRef::Kind::Param:
            record["param"] = static_cast<int64_t>(arg.paramIdx);
            break;
        case ArgRef::Kind::ConstI64:
            record["value"] = arg.constVal;
            break;
        case ArgRef::Kind::BinOp:
        case ArgRef::Kind::Cast: {
            record["op"] = arg.opStr;
            json::Array children;
            for (const auto &child : arg.children)
                children.push_back(argRefFactRecord(child));
            record["children"] = std::move(children);
            break;
        }
        case ArgRef::Kind::FieldLoad:
            record["base_formal"] = static_cast<int64_t>(arg.paramIdx);
            record["struct_size"] = arg.structElemSize;
            record["field_offset"] = arg.fieldByteOffset;
            record["field_type"] = arg.fieldTypeStr;
            if (!arg.children.empty())
                record["index"] = argRefFactRecord(arg.children.front());
            break;
        case ArgRef::Kind::Derived:
        case ArgRef::Kind::LoopIv:
            break;
    }
    return json::Value(std::move(record));
}

bool symbolicallyExact(const ArgRef &arg, const OpTemplate &op) {
    switch (arg.kind) {
        case ArgRef::Kind::Param:
        case ArgRef::Kind::ConstI64:
            return true;
        case ArgRef::Kind::LoopIv:
            return op.loop.inLoop && !op.loop.degraded &&
                   op.loop.ivBoundKnown;
        case ArgRef::Kind::Derived:
        case ArgRef::Kind::FieldLoad:
            return false;
        case ArgRef::Kind::BinOp: {
            static constexpr const char *Supported[] = {
                "add", "sub", "mul", "sdiv", "udiv", "srem", "urem",
                "shl", "ashr", "lshr", "and", "or", "xor",
            };
            if (arg.children.size() != 2 ||
                std::find(std::begin(Supported), std::end(Supported),
                          arg.opStr) == std::end(Supported))
                return false;
            break;
        }
        case ArgRef::Kind::Cast: {
            static constexpr const char *Supported[] = {
                "trunc", "zext", "sext", "fptoui", "fptosi", "uitofp",
                "sitofp", "fptrunc", "fpext", "ptrtoint", "inttoptr",
                "bitcast",
            };
            if (arg.children.size() != 1 ||
                std::find(std::begin(Supported), std::end(Supported),
                          arg.opStr) == std::end(Supported))
                return false;
            break;
        }
    }
    return std::all_of(arg.children.begin(), arg.children.end(),
                       [&](const auto &child) {
                           return symbolicallyExact(child, op);
                       });
}

// Conservative affine classification over kernel formals and a modeled loop
// IV.  It is intentionally narrower than "symbolically exact": exact bitwise
// or divide expressions are useful facts, but they do not satisfy the first
// producer-frontier candidate's affine-interval prerequisite.
bool affineArgRef(const ArgRef &arg, const OpTemplate &op) {
    switch (arg.kind) {
        case ArgRef::Kind::Param:
        case ArgRef::Kind::ConstI64:
            return true;
        case ArgRef::Kind::LoopIv:
            return op.loop.inLoop && !op.loop.degraded &&
                   op.loop.ivBoundKnown;
        case ArgRef::Kind::Derived:
        case ArgRef::Kind::FieldLoad:
            return false;
        case ArgRef::Kind::Cast:
            return arg.children.size() == 1 && arg.opStr != "gep" &&
                   symbolicallyExact(arg, op) &&
                   affineArgRef(arg.children.front(), op);
        case ArgRef::Kind::BinOp:
            if (arg.children.size() != 2 || !symbolicallyExact(arg, op))
                return false;
            if (arg.opStr == "add" || arg.opStr == "sub")
                return affineArgRef(arg.children[0], op) &&
                       affineArgRef(arg.children[1], op);
            if (arg.opStr == "mul") {
                int64_t ignored = 0;
                const bool lhsConst = constOf(arg.children[0], ignored);
                const bool rhsConst = constOf(arg.children[1], ignored);
                return (lhsConst && affineArgRef(arg.children[1], op)) ||
                       (rhsConst && affineArgRef(arg.children[0], op));
            }
            if (arg.opStr == "shl") {
                int64_t ignored = 0;
                return constOf(arg.children[1], ignored) &&
                       affineArgRef(arg.children[0], op);
            }
            return false;
    }
    return false;
}

bool kernelIntegerExpr(const ArgRef &arg, const KernelTemplate &kernel) {
    switch (arg.kind) {
        case ArgRef::Kind::Param:
            return arg.paramIdx < kernel.params.size() &&
                   metadataIntegerBits(
                       kernel.params[arg.paramIdx].typeStr).has_value();
        case ArgRef::Kind::ConstI64:
            return true;
        case ArgRef::Kind::BinOp:
        case ArgRef::Kind::Cast:
            return std::all_of(arg.children.begin(), arg.children.end(),
                               [&](const auto &child) {
                                   return kernelIntegerExpr(child, kernel);
                               });
        case ArgRef::Kind::Derived:
        case ArgRef::Kind::LoopIv:
        case ArgRef::Kind::FieldLoad:
            return false;
    }
    return false;
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

bool deviceExprFormalsValid(const DeviceExpr &expr,
                            const KernelTemplate &kernel) {
    if (expr.kind == DeviceExpr::Kind::Unknown || expr.typeStr.empty())
        return false;
    if (expr.kind == DeviceExpr::Kind::Param &&
        (expr.paramIdx >= kernel.params.size() ||
         kernel.params[expr.paramIdx].typeStr != expr.typeStr))
        return false;
    return std::all_of(expr.children.begin(), expr.children.end(),
                       [&](const auto &child) {
                           return deviceExprFormalsValid(child, kernel);
                       });
}

bool sameProducerDomain(const ProducerStoreDomainFact &left,
                        const ProducerStoreDomainFact &right) {
    if (left.pointerParam != right.pointerParam ||
        left.byteSize != right.byteSize ||
        left.addressExact != right.addressExact ||
        left.predicatesExact != right.predicatesExact ||
        left.domainExact != right.domainExact ||
        left.partitionRegionExact != right.partitionRegionExact ||
        left.partitionPredicateIndex != right.partitionPredicateIndex ||
        !sameDeviceExpr(left.byteOffset, right.byteOffset) ||
        left.predicates.size() != right.predicates.size())
        return false;
    for (size_t i = 0; i < left.predicates.size(); ++i) {
        if (left.predicates[i].requiredValue !=
                right.predicates[i].requiredValue ||
            !sameDeviceExpr(left.predicates[i].condition,
                            right.predicates[i].condition))
            return false;
    }
    return true;
}

bool sameAtomicDomain(const ProducerAtomicDomainFact &left,
                      const ProducerAtomicDomainFact &right) {
    if (left.pointerParam != right.pointerParam ||
        left.operation != right.operation ||
        left.resultUnused != right.resultUnused ||
        left.predicatesExact != right.predicatesExact ||
        left.domainExact != right.domainExact ||
        left.predicates.size() != right.predicates.size())
        return false;
    for (size_t i = 0; i < left.predicates.size(); ++i) {
        if (left.predicates[i].requiredValue !=
                right.predicates[i].requiredValue ||
            !sameDeviceExpr(left.predicates[i].condition,
                            right.predicates[i].condition))
            return false;
    }
    return true;
}

bool sameAtomicDomains(const ProducerFrontierFacts &left,
                       const ProducerFrontierFacts &right) {
    if (left.atomic_write_sites != right.atomic_write_sites ||
        left.atomic_domains_known != right.atomic_domains_known ||
        left.producer_atomic_domains.size() !=
            right.producer_atomic_domains.size())
        return false;
    for (size_t i = 0; i < left.producer_atomic_domains.size(); ++i)
        if (!sameAtomicDomain(left.producer_atomic_domains[i],
                              right.producer_atomic_domains[i]))
            return false;
    return true;
}

bool samePhaseSensitiveDomain(
        const ProducerPhaseSensitiveDomainFact &left,
        const ProducerPhaseSensitiveDomainFact &right) {
    if (left.operation != right.operation ||
        left.guardPredicatesExact != right.guardPredicatesExact ||
        left.domainExact != right.domainExact ||
        left.guardPredicates.size() != right.guardPredicates.size())
        return false;
    for (size_t i = 0; i < left.guardPredicates.size(); ++i) {
        if (left.guardPredicates[i].requiredValue !=
                right.guardPredicates[i].requiredValue ||
            !sameDeviceExpr(left.guardPredicates[i].condition,
                            right.guardPredicates[i].condition))
            return false;
    }
    return true;
}

bool samePhaseSensitiveDomains(const ProducerFrontierFacts &left,
                               const ProducerFrontierFacts &right) {
    if (left.phase_sensitive_sites != right.phase_sensitive_sites ||
        left.phase_sensitive_domains_known !=
            right.phase_sensitive_domains_known ||
        left.producer_phase_sensitive_domains.size() !=
            right.producer_phase_sensitive_domains.size())
        return false;
    for (size_t i = 0;
         i < left.producer_phase_sensitive_domains.size(); ++i)
        if (!samePhaseSensitiveDomain(
                left.producer_phase_sensitive_domains[i],
                right.producer_phase_sensitive_domains[i]))
            return false;
    return true;
}

void deriveSideEffectSafety(const KernelTemplate &kernel,
                            const ProducerFrontierFacts &frontier,
                            ProducerOverlapPartitionFacts &result) {
    if (frontier.atomic_write_sites == 0 &&
        frontier.phase_sensitive_sites == 0) {
        result.sideEffectSafetyExact = true;
        result.sideEffectSafetyMode = "no_nonduplicable_operations";
        result.sideEffectSafetyReason =
            "producer frontier contains no atomic writes or phase-sensitive calls";
        return;
    }
    if (frontier.atomic_write_sites != 0 &&
        (!frontier.atomic_domains_known ||
         frontier.producer_atomic_domains.empty())) {
        result.sideEffectSafetyReason =
            "atomic control domains are not exact";
        return;
    }
    if (frontier.phase_sensitive_sites != 0 &&
        (!frontier.phase_sensitive_domains_known ||
         frontier.producer_phase_sensitive_domains.empty())) {
        result.sideEffectSafetyReason =
            "phase-sensitive control domains are not exact";
        return;
    }

    // Search one site's direct i1-formal predicates for one common condition
    // that dominates every atomic and every phase-sensitive call.
    // Taking the opposite value at launch makes all such operations
    // unreachable; no FP reassociation or subgroup-semantic assumption is
    // needed.
    const std::vector<ProducerPredicateFact> &candidates =
        frontier.phase_sensitive_sites != 0
            ? frontier.producer_phase_sensitive_domains.front()
                  .guardPredicates
            : frontier.producer_atomic_domains.front().predicates;
    for (const auto &candidate : candidates) {
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
        const bool commonPhaseSensitive = std::all_of(
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
        if (!commonAtomic || !commonPhaseSensitive) continue;
        result.sideEffectSafetyExact = true;
        result.sideEffectSafetyMode =
            "all_nonduplicable_operations_disabled_by_formal_guard";
        result.hasSideEffectGuard = true;
        result.sideEffectGuardParam = condition.paramIdx;
        result.sideEffectGuardValue = !candidate.requiredValue;
        result.sideEffectSafetyReason =
            "one opposite i1 formal value makes every exact atomic and phase-sensitive domain unreachable";
        return;
    }
    result.sideEffectSafetyReason =
        "non-duplicable operations lack one shared direct i1-formal disabling predicate";
}

ProducerOverlapPartitionFacts producerOverlapPartition(
        const KernelTemplate &kernel, const OpTemplate &member) {
    ProducerOverlapPartitionFacts result;
    if (member.kind != "put_no_db" || member.completion_site_id.empty()) {
        result.reason = "operation is not a completed PUT group member";
        return result;
    }
    result.analyzed = true;

    const ProducerFrontierFacts &frontier = member.producer_frontier;
    if (!frontier.analyzed || !frontier.buffer_identity_guardable) {
        result.reason =
            "producer pointer and transfer buffer lack a compiler guard shape";
        return result;
    }
    if (!frontier.producer_domains_known ||
        frontier.ordinary_store_sites != 1 ||
        frontier.producer_store_domains.size() != 1 ||
        !frontier.producer_store_domains.front().domainExact) {
        result.reason =
            "first overlap partition requires one exact ordinary producer store";
        return result;
    }

    result.producerPointerParam = frontier.producer_pointer_param;
    result.sourceBufferParam = frontier.source_buffer_index_param;
    result.producerStore = frontier.producer_store_domains.front();
    if (result.producerStore.pointerParam != result.producerPointerParam ||
        result.producerPointerParam >= kernel.params.size() ||
        kernel.params[result.producerPointerParam].typeStr != "ptr" ||
        result.sourceBufferParam >= kernel.params.size() ||
        kernel.params[result.sourceBufferParam].typeStr != "i32" ||
        !deviceExprFormalsValid(result.producerStore.byteOffset, kernel) ||
        !std::all_of(result.producerStore.predicates.begin(),
                     result.producerStore.predicates.end(),
                     [&](const auto &predicate) {
                         return predicate.condition.typeStr == "i1" &&
                             deviceExprFormalsValid(predicate.condition,
                                                    kernel);
                     })) {
        result.reason =
            "producer store formals disagree with typed kernel metadata";
        return result;
    }

    for (const OpTemplate &op : kernel.ops) {
        if (op.completion_site_id != member.completion_site_id ||
            (op.kind != "put_no_db" && op.kind != "get_no_db"))
            continue;
        if (op.kind != "put_no_db") {
            result.reason =
                "mixed PUT/GET completion groups are not partitioned";
            return result;
        }
        const ProducerFrontierFacts &other = op.producer_frontier;
        if (!other.analyzed || !other.buffer_identity_guardable ||
            !other.producer_domains_known ||
            other.ordinary_store_sites != 1 ||
            other.producer_pointer_param != result.producerPointerParam ||
            other.source_buffer_index_param != result.sourceBufferParam ||
            other.producer_store_domains.size() != 1 ||
            !sameAtomicDomains(frontier, other) ||
            !samePhaseSensitiveDomains(frontier, other) ||
            !sameProducerDomain(other.producer_store_domains.front(),
                                result.producerStore)) {
            result.reason =
                "completion-group members disagree on the producer domain";
            return result;
        }

        const auto source = op.args.find("src_buf");
        const auto offset = op.args.find("src_off");
        const auto size = op.args.find("size");
        if (source == op.args.end() || offset == op.args.end() ||
            size == op.args.end() ||
            source->second.kind != ArgRef::Kind::Param ||
            source->second.paramIdx != result.sourceBufferParam ||
            !symbolicallyExact(offset->second, op) ||
            !symbolicallyExact(size->second, op) ||
            !affineArgRef(offset->second, op) ||
            !affineArgRef(size->second, op) ||
            !kernelIntegerExpr(offset->second, kernel) ||
            !kernelIntegerExpr(size->second, kernel)) {
            result.reason =
                "a group transfer lacks an exact kernel-formal byte interval";
            return result;
        }
        result.transfers.push_back({op.siteId, source->second,
                                    offset->second, size->second});
    }

    if (result.transfers.empty()) {
        result.reason = "completion group contains no transfer intervals";
        return result;
    }

    deriveSideEffectSafety(kernel, frontier, result);

    result.exact = true;
    result.reason =
        "exact store instances split by checked byte overlap and its logical complement";
    return result;
}

std::optional<json::Value> transferIntervalRecord(const OpTemplate &op) {
    if (op.kind != "put_no_db" && op.kind != "get_no_db")
        return std::nullopt;
    const auto srcBuf = op.args.find("src_buf");
    const auto srcOff = op.args.find("src_off");
    const auto size = op.args.find("size");
    if (srcBuf == op.args.end() || srcOff == op.args.end() ||
        size == op.args.end())
        return std::nullopt;

    json::Object record;
    record["semantics"] = "source_buffer_half_open_byte_interval";
    record["source_buffer"] = argRefFactRecord(srcBuf->second);
    record["byte_offset"] = argRefFactRecord(srcOff->second);
    record["byte_size"] = argRefFactRecord(size->second);
    const bool exact = symbolicallyExact(srcBuf->second, op) &&
                       symbolicallyExact(srcOff->second, op) &&
                       symbolicallyExact(size->second, op);
    record["symbolically_exact"] = exact;
    record["affine"] = exact && affineArgRef(srcBuf->second, op) &&
                        affineArgRef(srcOff->second, op) &&
                        affineArgRef(size->second, op);
    record["host_knowable"] = op.hk_capable && exact;
    return json::Value(std::move(record));
}

// log2 of a positive constant size; null for unknown or non-positive.
std::optional<int> sizeLog2(const std::optional<int64_t> &size) {
    if (!size || *size <= 0) return std::nullopt;
    return 63 - __builtin_clzll(static_cast<uint64_t>(*size));
}

double guardDensity(const GuardSpec &g) {
    return g.kind == GuardSpec::Kind::Always ? 1.0 : 0.5;
}

const char *guardKindName(const GuardSpec &g) {
    switch (g.kind) {
        case GuardSpec::Kind::Always:       return "always";
        case GuardSpec::Kind::ParamTruthy:  return "param_truthy";
        case GuardSpec::Kind::ParamEqConst: return "param_eq_const";
        case GuardSpec::Kind::BinOp:        return "binop";
        case GuardSpec::Kind::FieldNotNull: return "field_not_null";
        case GuardSpec::Kind::Unknown:      return "unknown";
    }
    return "unknown";
}

// Number of distinct param-indexed peers across the kernel's put_no_db /
// get_no_db ops. Coarse fan-out estimate; ops without a Param target_rank
// don't contribute (target == const_i64 doesn't fan out).
int kernelFanOut(const KernelTemplate &t) {
    SmallSet<unsigned, 8> peers;
    for (const auto &op : t.ops) {
        if (op.kind != "put_no_db" && op.kind != "get_no_db") continue;
        auto it = op.args.find("target_rank");
        if (it == op.args.end()) continue;
        if (it->second.kind == ArgRef::Kind::Param)
            peers.insert(it->second.paramIdx);
    }
    return static_cast<int>(peers.size());
}

// Structured loop descriptor for the ML decider. Only emitted when the
// site is actually inside a loop. The decider can combine this with
// runtime-side param values to reconstruct a trip-count estimate.
json::Value loopDescriptor(const OpLoopInfo &L, StringRef boundParamType) {
    json::Object o;
    o["iv_start"]    = L.ivStart;
    o["iv_step"]     = L.ivStep;
    o["bound_known"] = L.ivBoundKnown;
    if (L.ivBoundKnown) {
        if (L.ivBoundIsConst)
            o["bound_const"] = L.ivBoundConst;
        else {
            o["bound_param_idx"] = static_cast<int64_t>(L.ivParamIdx);
            if (!boundParamType.empty())
                o["bound_param_type"] = boundParamType;
        }
    }
    if (L.degraded) o["degraded"] = true;
    return json::Value(std::move(o));
}

const char *deviceExprKindTag(DeviceExpr::Kind kind) {
    switch (kind) {
        case DeviceExpr::Kind::Param:    return "param";
        case DeviceExpr::Kind::ConstI64: return "const";
        case DeviceExpr::Kind::Builtin:  return "builtin";
        case DeviceExpr::Kind::BinOp:    return "binop";
        case DeviceExpr::Kind::Cast:     return "cast";
        case DeviceExpr::Kind::Compare:  return "compare";
        case DeviceExpr::Kind::Select:   return "select";
        case DeviceExpr::Kind::Unknown:  return "unknown";
    }
    return "unknown";
}

json::Value deviceExprFactRecord(const DeviceExpr &expr) {
    json::Object record;
    record["kind"] = deviceExprKindTag(expr.kind);
    if (!expr.typeStr.empty()) record["type"] = expr.typeStr;
    if (expr.kind == DeviceExpr::Kind::Param)
        record["param"] = static_cast<int64_t>(expr.paramIdx);
    if (expr.kind == DeviceExpr::Kind::ConstI64)
        record["value"] = expr.constVal;
    if (expr.kind == DeviceExpr::Kind::Builtin)
        record["name"] = expr.opStr;
    if (expr.kind == DeviceExpr::Kind::BinOp ||
        expr.kind == DeviceExpr::Kind::Cast ||
        expr.kind == DeviceExpr::Kind::Compare)
        record["op"] = expr.opStr;
    if (!expr.children.empty()) {
        json::Array children;
        for (const auto &child : expr.children)
            children.push_back(deviceExprFactRecord(child));
        record["children"] = std::move(children);
    }
    return json::Value(std::move(record));
}

json::Value producerStoreDomainRecord(
        const ProducerStoreDomainFact &domain) {
    json::Object record;
    record["pointer_param"] = static_cast<int64_t>(domain.pointerParam);
    record["byte_offset"] = deviceExprFactRecord(domain.byteOffset);
    record["byte_size"] = static_cast<int64_t>(domain.byteSize);
    record["address_exact"] = domain.addressExact;
    json::Array predicates;
    for (const auto &predicate : domain.predicates) {
        json::Object item;
        item["condition"] = deviceExprFactRecord(predicate.condition);
        item["required_value"] = predicate.requiredValue;
        predicates.push_back(std::move(item));
    }
    record["predicates"] = std::move(predicates);
    record["predicates_exact"] = domain.predicatesExact;
    record["domain_exact"] = domain.domainExact;
    record["partition_region_exact"] = domain.partitionRegionExact;
    if (domain.partitionRegionExact) {
        record["partition_predicate_index"] =
            static_cast<int64_t>(domain.partitionPredicateIndex);
        record["partition_predicate"] = deviceExprFactRecord(
            domain.predicates[domain.partitionPredicateIndex].condition);
        record["partition_predicate_required_value"] =
            domain.predicates[domain.partitionPredicateIndex].requiredValue;
    } else {
        record["partition_predicate_index"] = nullptr;
        record["partition_predicate"] = nullptr;
        record["partition_predicate_required_value"] = nullptr;
    }
    record["partition_region_reason"] = domain.partitionRegionReason;
    record["reason"] = domain.reason;
    return json::Value(std::move(record));
}

json::Value producerAtomicDomainRecord(
        const ProducerAtomicDomainFact &domain) {
    json::Object record;
    record["pointer_param"] = static_cast<int64_t>(domain.pointerParam);
    record["operation"] = domain.operation;
    record["result_unused"] = domain.resultUnused;
    json::Array predicates;
    for (const auto &predicate : domain.predicates) {
        json::Object item;
        item["condition"] = deviceExprFactRecord(predicate.condition);
        item["required_value"] = predicate.requiredValue;
        predicates.push_back(std::move(item));
    }
    record["predicates"] = std::move(predicates);
    record["predicates_exact"] = domain.predicatesExact;
    record["domain_exact"] = domain.domainExact;
    record["reason"] = domain.reason;
    return json::Value(std::move(record));
}

json::Value producerPhaseSensitiveDomainRecord(
        const ProducerPhaseSensitiveDomainFact &domain) {
    json::Object record;
    record["operation"] = domain.operation;
    json::Array predicates;
    for (const auto &predicate : domain.guardPredicates) {
        json::Object item;
        item["condition"] = deviceExprFactRecord(predicate.condition);
        item["required_value"] = predicate.requiredValue;
        predicates.push_back(std::move(item));
    }
    record["guard_predicates"] = std::move(predicates);
    record["guard_predicates_exact"] = domain.guardPredicatesExact;
    record["domain_exact"] = domain.domainExact;
    record["reason"] = domain.reason;
    return json::Value(std::move(record));
}

json::Value producerOverlapPartitionRecord(
        const ProducerOverlapPartitionFacts &partition) {
    json::Object record;
    record["analyzed"] = partition.analyzed;
    record["exact"] = partition.exact;
    record["mode"] = partition.exact
        ? "checked_store_interval_overlap"
        : "none";
    record["formal_binding"] = partition.exact
        ? "same_kernel_formal_indices"
        : "unproved";
    if (partition.exact) {
        record["producer_pointer_param"] =
            static_cast<int64_t>(partition.producerPointerParam);
        record["source_buffer_index_param"] =
            static_cast<int64_t>(partition.sourceBufferParam);
        record["producer_store"] =
            producerStoreDomainRecord(partition.producerStore);
    } else {
        record["producer_pointer_param"] = nullptr;
        record["source_buffer_index_param"] = nullptr;
        record["producer_store"] = nullptr;
    }

    json::Array intervals;
    for (const auto &transfer : partition.transfers) {
        json::Object interval;
        interval["site_id"] = transfer.siteId;
        interval["source_buffer"] =
            argRefFactRecord(transfer.sourceBuffer);
        interval["byte_offset"] = argRefFactRecord(transfer.byteOffset);
        interval["byte_size"] = argRefFactRecord(transfer.byteSize);
        intervals.push_back(std::move(interval));
    }
    record["transfer_intervals"] = std::move(intervals);
    record["boundary_predicate"] = partition.exact
        ? "store_interval_overlaps_any_transfer_interval"
        : "unavailable";
    record["remainder_predicate"] = partition.exact
        ? "logical_complement_of_boundary"
        : "unavailable";
    record["checked_interval_ends_required"] = partition.exact;
    record["buffer_identity_guard_required"] = partition.exact;
    record["store_instance_partition_disjoint"] = partition.exact;
    record["store_instance_partition_complete"] = partition.exact;
    record["proof_scope"] = "ordinary_producer_store_instances";
    record["full_compute_region_partition_proved"] =
        partition.exact && partition.producerStore.partitionRegionExact;
    record["side_effect_safety_exact"] =
        partition.sideEffectSafetyExact;
    record["side_effect_safety_mode"] =
        partition.sideEffectSafetyMode;
    record["side_effect_safety_reason"] =
        partition.sideEffectSafetyReason;
    record["side_effects_excluded_on_optimized_path"] =
        partition.sideEffectSafetyExact;
    if (partition.hasSideEffectGuard) {
        json::Object guard;
        guard["kind"] = "param_eq";
        guard["param"] =
            static_cast<int64_t>(partition.sideEffectGuardParam);
        guard["value"] = partition.sideEffectGuardValue;
        record["side_effect_free_guard"] = std::move(guard);
    } else {
        record["side_effect_free_guard"] = nullptr;
    }
    record["side_effect_partition_proved"] = false;
    record["reason"] = partition.reason;
    return json::Value(std::move(record));
}

json::Value producerFrontierRecord(
        const ProducerFrontierFacts &facts,
        const ProducerOverlapPartitionFacts &partition) {
    json::Object record;
    record["analyzed"] = facts.analyzed;
    record["write_footprint_known"] = facts.write_footprint_known;
    record["completion_site_id"] = facts.completion_site_id;
    json::Array stores;
    for (unsigned param : facts.ordinary_store_params)
        stores.push_back(static_cast<int64_t>(param));
    record["ordinary_store_params"] = std::move(stores);
    json::Array atomics;
    for (unsigned param : facts.atomic_write_params)
        atomics.push_back(static_cast<int64_t>(param));
    record["atomic_write_params"] = std::move(atomics);
    record["buffer_identity_guardable"] =
        facts.buffer_identity_guardable;
    if (facts.buffer_identity_guardable) {
        record["producer_pointer_param"] =
            static_cast<int64_t>(facts.producer_pointer_param);
        record["source_buffer_index_param"] =
            static_cast<int64_t>(facts.source_buffer_index_param);
    } else {
        record["producer_pointer_param"] = nullptr;
        record["source_buffer_index_param"] = nullptr;
    }
    record["buffer_identity_guard_reason"] =
        facts.buffer_identity_guard_reason;
    record["source_identity_guardable"] =
        facts.source_identity_guardable;
    json::Array sourcePointers;
    for (unsigned param : facts.source_pointer_candidates)
        sourcePointers.push_back(static_cast<int64_t>(param));
    record["source_pointer_candidates"] = std::move(sourcePointers);
    if (facts.source_identity_guardable)
        record["source_identity_buffer_index_param"] =
            static_cast<int64_t>(
                facts.source_identity_buffer_index_param);
    else
        record["source_identity_buffer_index_param"] = nullptr;
    record["source_identity_guard_reason"] =
        facts.source_identity_guard_reason;
    record["producer_domains_known"] = facts.producer_domains_known;
    json::Array domains;
    for (const auto &domain : facts.producer_store_domains)
        domains.push_back(producerStoreDomainRecord(domain));
    record["producer_store_domains"] = std::move(domains);
    record["atomic_domains_known"] = facts.atomic_domains_known;
    json::Array atomicDomains;
    for (const auto &domain : facts.producer_atomic_domains)
        atomicDomains.push_back(producerAtomicDomainRecord(domain));
    record["producer_atomic_domains"] = std::move(atomicDomains);
    record["phase_sensitive_domains_known"] =
        facts.phase_sensitive_domains_known;
    json::Array phaseSensitiveDomains;
    for (const auto &domain : facts.producer_phase_sensitive_domains)
        phaseSensitiveDomains.push_back(
            producerPhaseSensitiveDomainRecord(domain));
    record["producer_phase_sensitive_domains"] =
        std::move(phaseSensitiveDomains);
    record["overlap_partition"] =
        producerOverlapPartitionRecord(partition);
    record["ordinary_store_sites"] =
        static_cast<int64_t>(facts.ordinary_store_sites);
    record["atomic_write_sites"] =
        static_cast<int64_t>(facts.atomic_write_sites);
    record["phase_sensitive_sites"] =
        static_cast<int64_t>(facts.phase_sensitive_sites);
    record["unknown_write_sites"] =
        static_cast<int64_t>(facts.unknown_write_sites);
    record["reason"] = facts.reason;
    json::Array remaining;
    const char *identityProof = facts.buffer_identity_guardable
        ? "buffer_identity_guarded_fallback_materialization"
        : "registered_buffer_identity";
    remaining.push_back(identityProof);
    if (!partition.exact) {
        remaining.push_back("exact_transfer_intervals");
        remaining.push_back("exact_producer_domains");
    } else {
        remaining.push_back("checked_interval_guard_materialization");
    }
    if (!partition.exact || !partition.producerStore.partitionRegionExact)
        remaining.push_back("complete_disjoint_partition");
    else
        remaining.push_back("device_phase_partition_materialization");
    if (!partition.sideEffectSafetyExact)
        remaining.push_back("side_effect_partition");
    else if (partition.hasSideEffectGuard)
        remaining.push_back("side_effect_guarded_fallback_materialization");
    remaining.push_back("launch_phase_materialization");
    record["remaining_proofs"] = std::move(remaining);
    return json::Value(std::move(record));
}

json::Value toRecord(const std::string &siteId,
                     const std::string &simpleKernel,
                     const OpTemplate  &op,
                     const std::string &boundParamType,
                     int                fanOut,
                     const ProducerOverlapPartitionFacts &partition,
                     std::vector<LaunchContextFacts> contexts) {
    contexts = coalesceContexts(std::move(contexts));
    const auto geometry = commonGeometry(contexts);
    const auto sizeBytes = commonKnownValue<int64_t>(
        contexts, [](const auto &c) { return c.sizeBytes; });
    const auto launchTripCount = commonKnownValue<int64_t>(
        contexts, [](const auto &c) { return c.tripCount; });

    json::Object r;
    // v6 aggregates all host launch contexts for one device operation and
    // binds constant wrapper arguments back to kernel formals. The hint key
    // names device code, so emitting duplicate records for two calls of the
    // same kernel would falsely imply the pass can choose two lowerings.
    //
    // Jumps 2 → 4 on purpose: the emitter had been left at 2 while the
    // schema doc already described a v3 (flops_to_first_use, trip_count,
    // distance_exact), so anything claiming 2 may or may not carry those.
    // Skipping the number keeps "3" from meaning two different things.
    r["schema_version"] = 6;
    r["site_id"]        = siteId;
    r["kernel"]         = simpleKernel;
    r["op_kind"]        = op.kind;
    // HK Analysis capability bit. Sites with hk_capable=false MUST be
    // routed to CPU_PROXY_ENQUEUE by the decider — routing them to
    // IPC_PUSH / DWQ_TRIGGER / IPC_OR_DWQ / DWQ_BATCHED is a build error
    // (enforced by GICCDispatchLowering in Task 2).
    r["hk_capable"]     = op.hk_capable;

    if (auto it = op.args.find("size"); it != op.args.end()) {
        r["size_kind"] = argKindTag(it->second.kind);
        if (auto l = sizeLog2(sizeBytes))
            r["size_log2"] = *l;
        else
            r["size_log2"] = nullptr;
    } else {
        r["size_kind"] = nullptr;
        r["size_log2"] = nullptr;
    }
    if (sizeBytes) r["size_bytes"] = *sizeBytes;
    else           r["size_bytes"] = nullptr;

    if (auto interval = transferIntervalRecord(op))
        r["transfer_interval"] = std::move(*interval);
    else
        r["transfer_interval"] = nullptr;

    if (auto it = op.args.find("target_rank"); it != op.args.end()) {
        r["peer_kind"] = argKindTag(it->second.kind);
    } else {
        r["peer_kind"] = nullptr;
    }
    // peer_locality still requires a runtime topology side-band (which
    // peer ranks live on the same node as self). Filled in v2.x by the
    // decider after Runtime::exchange() dumps the mapping; until then
    // we report null and the runtime branch in IPC_OR_DWQ handles it.
    r["peer_locality"] = nullptr;

    r["in_loop"]       = op.loop.inLoop;
    if (op.loop.inLoop)
        r["loop"] = loopDescriptor(op.loop, boundParamType);

    r["guard_density"] = guardDensity(op.guard);
    // A scalar density was enough for the original per-site cost model, but
    // structural transforms need to distinguish a loop-invariant degraded
    // guard from a per-iteration FieldNotNull guard.  The plan materializer
    // still rechecks the full GuardSpec from kernel metadata.
    r["guard_kind"] = guardKindName(op.guard);
    r["fan_out"]       = fanOut;

    unsigned staticLaunchSites = 0;
    json::Array launchContexts;
    for (const auto &context : contexts) {
        staticLaunchSites += context.staticCallsites;
        launchContexts.push_back(launchContextRecord(context));
    }
    r["static_launch_sites"] = static_cast<int64_t>(staticLaunchSites);
    r["launch_contexts"] = std::move(launchContexts);

    const bool phaseLaunchSupported = !contexts.empty() &&
        std::all_of(contexts.begin(), contexts.end(), [](const auto &context) {
            return context.phaseLaunchSupported;
        });
    r["phase_launch_supported"] = phaseLaunchSupported;
    const bool kernelArgumentSlotsExact = !contexts.empty() &&
        std::all_of(contexts.begin(), contexts.end(), [](const auto &context) {
            return context.kernelArgumentSlotsExact;
        });
    r["kernel_argument_slots_exact"] = kernelArgumentSlotsExact;
    if (auto count = commonKnownValue<unsigned>(
            contexts, [](const auto &context) {
                return std::optional<unsigned>(
                    context.kernelArgumentSlotCount);
            }))
        r["kernel_argument_slot_count"] = static_cast<int64_t>(*count);
    else
        r["kernel_argument_slot_count"] = nullptr;
    if (!contexts.empty()) {
        const std::string &stream = contexts.front().phaseLaunchStream;
        const std::string &materialization =
            contexts.front().phaseLaunchMaterialization;
        const std::string &reason = contexts.front().phaseLaunchReason;
        const std::string &slotReason =
            contexts.front().kernelArgumentSlotReason;
        bool sameStream = std::all_of(
            contexts.begin(), contexts.end(), [&](const auto &context) {
                return context.phaseLaunchStream == stream;
            });
        bool sameReason = std::all_of(
            contexts.begin(), contexts.end(), [&](const auto &context) {
                return context.phaseLaunchReason == reason;
            });
        bool sameMaterialization = std::all_of(
            contexts.begin(), contexts.end(), [&](const auto &context) {
                return context.phaseLaunchMaterialization == materialization;
            });
        bool sameSlotReason = std::all_of(
            contexts.begin(), contexts.end(), [&](const auto &context) {
                return context.kernelArgumentSlotReason == slotReason;
            });
        r["phase_launch_stream"] = sameStream ? stream : "mixed";
        r["phase_launch_materialization"] =
            sameMaterialization ? materialization : "mixed";
        r["phase_launch_reason"] =
            sameReason ? reason : "launch contexts disagree";
        r["kernel_argument_slot_reason"] = sameSlotReason
            ? slotReason : "launch contexts disagree";
    } else {
        r["phase_launch_stream"] = "unknown";
        r["phase_launch_materialization"] = "none";
        r["phase_launch_reason"] = "no host launch context";
        r["kernel_argument_slot_reason"] = "no host launch context";
    }

    r["launch_grid"]  = dim3Record(geometry.grid);
    r["launch_block"] = dim3Record(geometry.block);
    if (auto blocks = dim3Product(geometry.grid))
        r["grid_blocks"] = *blocks;
    else
        r["grid_blocks"] = nullptr;
    if (auto threads = dim3Product(geometry.block))
        r["threads_per_block"] = *threads;
    else
        r["threads_per_block"] = nullptr;

    // compute_before_flops: static count of arithmetic / FP ops in BBs
    // dominating the call site. -1 sentinel from the JSON means the
    // device pass didn't have DT available — emit null in that case so
    // the decider can tell "0 compute" apart from "not measured".
    if (op.compute_before >= 0)
        r["compute_before_flops"] = static_cast<int64_t>(op.compute_before);
    else
        r["compute_before_flops"] = nullptr;

    // flops_to_first_use: the same currency as compute_before_flops, but
    // counted BETWEEN this op and the kernel's completion point. This is
    // the static form of issue-to-first-use distance, which decides
    // whether the transfer can hide behind compute — measurements show
    // the best dispatch flips on it at fixed message size and trip count.
    if (op.compute_after >= 0)
        r["flops_to_first_use"] = static_cast<int64_t>(op.compute_after);
    else
        r["flops_to_first_use"] = nullptr;

    // trip_count: ops issued per communication phase, when ScalarEvolution
    // can prove it. Both dispatch paths pay a per-op issue cost with
    // different constants, so this scales the decision.
    if (launchTripCount) r["trip_count"] = *launchTripCount;
    else                 r["trip_count"] = nullptr;
    r["distance_exact"] = op.distance_exact;

    if (op.producer_frontier.analyzed)
        r["producer_frontier"] =
            producerFrontierRecord(op.producer_frontier, partition);

    if (op.trip_count < 0 && launchTripCount)
        r["iter_estimate"] = *launchTripCount;
    else
        r["iter_estimate"] = nullptr;

    // Reuse and batching. These are the properties a runtime cannot
    // establish at the moment of the call: it sees one transfer, not the
    // loop it sits in nor the group it belongs to.
    r["descriptor_reusable"] = descriptorReusable(op);
    r["buffer_reusable"]     = bufferReusable(op);
    // Consecutive iterations land exactly `size` apart at both ends, so any
    // run of this loop's transfers may be issued as one larger transfer.
    // A runtime cannot establish this: when it sees transfer i it does not
    // know where i+1 will go, and by then i has already been issued.
    r["coalescable"]         = transfersAreAdjacent(op);
    // Widest power-of-two element a copy of this transfer may legally use.
    // Exceeding the contiguous run does not run slowly on a strided face,
    // it faults, so this is a legality bound rather than a preference.
    r["max_vector_bytes"]    = static_cast<int64_t>(maxVectorBytes(op));
    // For a completion point: the weakest sound fence scope. 3 (system) is
    // what every site emitted before this existed, so it is also what the
    // analysis reports whenever it cannot prove something weaker.
    if (op.kind == "quiet" || op.kind == "flush")
        r["fence_scope"] = static_cast<int64_t>(op.fence_scope);
    if (op.batch_size >= 0)
        r["batch_size"] = static_cast<int64_t>(op.batch_size);
    else
        r["batch_size"] = nullptr;

    // Legality, not preference. The proxy path is always available: the
    // device pushes a command and the worker reads the descriptor at
    // submit time, so nothing has to be knowable in advance. The trigger
    // and IPC paths need the host to reconstruct the descriptor before
    // the kernel launches, which is exactly what HK analysis proves —
    // and a loop the pass failed to model means trace synthesis would
    // drop the transfer, so that disqualifies them too.
    const bool hostCanStage =
        op.hk_capable && !(op.loop.inLoop && op.loop.degraded);
    // A modeled loop is synthesized as one batched placeholder carrying
    // arrays of descriptors. The current host lowering can materialize that
    // shape as DWQ or erase it for CPU proxy, but it cannot issue per-element
    // IPC (nor the IPC_OR_DWQ hybrid). Do not advertise an action that the
    // real lowering will reject later.
    const bool batchedLoop =
        op.loop.inLoop && op.loop.ivBoundKnown && !op.loop.degraded;
    json::Array legal;
    legal.push_back("proxy");
    if (hostCanStage) {
        legal.push_back("trigger");
        if (!batchedLoop) legal.push_back("ipc");
    }
    r["legal_paths"] = json::Value(std::move(legal));
    return json::Value(std::move(r));
}

std::string pickFeaturesPath(const Config &cfg) {
    if (!cfg.featuresOut.empty()) return cfg.featuresOut;
    SmallString<256> p(cfg.metaDir);
    sys::path::append(p, "features.json");
    return p.str().str();
}

}  // namespace

PreservedAnalyses GICCFeatureExtractionPass::run(Module &M,
                                                 ModuleAnalysisManager &) {
    const auto &cfg = getConfig();
    if (cfg.mode != Mode::FeatureExtract && cfg.mode != Mode::Lower)
        return PreservedAnalyses::all();

    auto inv = collectLaunchInventory(M, cfg.metaDir);
    if (inv.sites.empty()) return PreservedAnalyses::all();

    struct PendingRecord {
        std::string kernel;
        OpTemplate op;
        std::string boundParamType;
        int fanOut = 0;
        bool initialized = false;
        ProducerOverlapPartitionFacts overlapPartition;
        std::vector<LaunchContextFacts> contexts;
    };
    std::map<std::string, PendingRecord> pending;
    bool collision = false;
    for (const auto &s : inv.sites) {
        if (!s.haveTemplate) continue;
        int fanOut = kernelFanOut(s.kernelTemplate);
        for (const auto &op : s.kernelTemplate.ops) {
            auto &record = pending[op.siteId];
            if (!record.initialized) {
                record.kernel = s.kernelTemplate.simpleName;
                record.op = op;
                if (op.loop.inLoop && op.loop.ivBoundKnown &&
                    !op.loop.ivBoundIsConst &&
                    op.loop.ivParamIdx < s.kernelTemplate.params.size())
                    record.boundParamType =
                        s.kernelTemplate.params[op.loop.ivParamIdx].typeStr;
                record.fanOut = fanOut;
                record.overlapPartition =
                    producerOverlapPartition(s.kernelTemplate, op);
                record.initialized = true;
            } else if (record.kernel != s.kernelTemplate.simpleName ||
                       record.op.kind != op.kind) {
                errs() << "[feature-extract] ERROR: site_id collision for "
                       << op.siteId << "\n";
                collision = true;
                continue;
            }
            record.contexts.push_back(launchContext(s, op));
        }
    }
    if (collision) return PreservedAnalyses::all();

    json::Array records;
    for (auto &[siteId, record] : pending)
        records.push_back(toRecord(siteId, record.kernel, record.op,
                                   record.boundParamType,
                                   record.fanOut,
                                   record.overlapPartition,
                                   std::move(record.contexts)));
    if (records.empty()) return PreservedAnalyses::all();

    std::string outPath = pickFeaturesPath(cfg);
    if (auto ec = sys::fs::create_directories(
            sys::path::parent_path(outPath))) {
        errs() << "[feature-extract] WARN: cannot create dir for "
               << outPath << "\n";
        return PreservedAnalyses::all();
    }

    std::error_code ec;
    raw_fd_ostream  out(outPath, ec, sys::fs::OF_Text);
    if (ec) {
        errs() << "[feature-extract] WARN: cannot open " << outPath
               << " for writing\n";
        return PreservedAnalyses::all();
    }
    out << formatv("{0:2}", json::Value(std::move(records))) << "\n";
    errs() << "[feature-extract] wrote " << outPath << "\n";
    return PreservedAnalyses::all();
}

}  // namespace gicc::pass
