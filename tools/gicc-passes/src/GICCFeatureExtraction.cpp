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
    unsigned staticCallsites = 1;
};

struct PhaseLaunchShape {
    bool supported = false;
    std::string stream = "unknown";
    std::string materialization = "none";
    std::string reason;
};

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
    if (!isa<CallInst>(site.callsite)) {
        result.reason = "invoke launch sites are not supported";
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
        context.phaseLaunchMaterialization, context.phaseLaunchReason);
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
json::Value loopDescriptor(const OpLoopInfo &L) {
    json::Object o;
    o["iv_start"]    = L.ivStart;
    o["iv_step"]     = L.ivStep;
    o["bound_known"] = L.ivBoundKnown;
    if (L.ivBoundKnown) {
        if (L.ivBoundIsConst)
            o["bound_const"] = L.ivBoundConst;
        else
            o["bound_param_idx"] = static_cast<int64_t>(L.ivParamIdx);
    }
    if (L.degraded) o["degraded"] = true;
    return json::Value(std::move(o));
}

json::Value producerFrontierRecord(const ProducerFrontierFacts &facts) {
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
    record["ordinary_store_sites"] =
        static_cast<int64_t>(facts.ordinary_store_sites);
    record["atomic_write_sites"] =
        static_cast<int64_t>(facts.atomic_write_sites);
    record["unknown_write_sites"] =
        static_cast<int64_t>(facts.unknown_write_sites);
    record["reason"] = facts.reason;
    json::Array remaining;
    for (const char *proof : {
             "registered_buffer_identity", "exact_transfer_intervals",
             "exact_producer_domains", "complete_disjoint_partition",
             "side_effect_partition", "launch_phase_materialization"})
        remaining.push_back(proof);
    record["remaining_proofs"] = std::move(remaining);
    return json::Value(std::move(record));
}

json::Value toRecord(const std::string &siteId,
                     const std::string &simpleKernel,
                     const OpTemplate  &op,
                     int                fanOut,
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
    if (op.loop.inLoop) r["loop"] = loopDescriptor(op.loop);

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
    if (!contexts.empty()) {
        const std::string &stream = contexts.front().phaseLaunchStream;
        const std::string &materialization =
            contexts.front().phaseLaunchMaterialization;
        const std::string &reason = contexts.front().phaseLaunchReason;
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
        r["phase_launch_stream"] = sameStream ? stream : "mixed";
        r["phase_launch_materialization"] =
            sameMaterialization ? materialization : "mixed";
        r["phase_launch_reason"] =
            sameReason ? reason : "launch contexts disagree";
    } else {
        r["phase_launch_stream"] = "unknown";
        r["phase_launch_materialization"] = "none";
        r["phase_launch_reason"] = "no host launch context";
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
            producerFrontierRecord(op.producer_frontier);

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
        int fanOut = 0;
        bool initialized = false;
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
                record.fanOut = fanOut;
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
                                   record.fanOut,
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
