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

#include <cstdint>
#include <limits>
#include <optional>

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

// log2 of a positive constant size; null for non-constant or non-positive.
std::optional<int> sizeLog2(const ArgRef &size) {
    if (size.kind != ArgRef::Kind::ConstI64) return std::nullopt;
    if (size.constVal <= 0) return std::nullopt;
    return 63 - __builtin_clzll(static_cast<uint64_t>(size.constVal));
}

double guardDensity(const GuardSpec &g) {
    return g.kind == GuardSpec::Kind::Always ? 1.0 : 0.5;
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

// Compile-time estimate of the loop's trip count, when both bound and
// start/step are integer constants. Today only the constant-step part
// is recorded — bounds are typically kernel formals, so this returns
// null and the decider has to fall back to runtime values. The hook
// stays here so the schema is forward-compatible once analyzeLoop
// learns to recognize const-bound loops.
std::optional<int64_t> iterEstimate(const OpLoopInfo &L) {
    if (!L.inLoop || !L.ivBoundKnown || L.degraded) return std::nullopt;
    // bound is a kernel formal in v1 — no compile-time numeric estimate.
    return std::nullopt;
}

// Structured loop descriptor for the ML decider. Only emitted when the
// site is actually inside a loop. The decider can combine this with
// runtime-side param values to reconstruct a trip-count estimate.
json::Value loopDescriptor(const OpLoopInfo &L) {
    json::Object o;
    o["iv_start"]    = L.ivStart;
    o["iv_step"]     = L.ivStep;
    o["bound_known"] = L.ivBoundKnown;
    if (L.ivBoundKnown)
        o["bound_param_idx"] = static_cast<int64_t>(L.ivParamIdx);
    if (L.degraded) o["degraded"] = true;
    return json::Value(std::move(o));
}

json::Value toRecord(const std::string &siteId,
                     const std::string &simpleKernel,
                     const OpTemplate  &op,
                     int                fanOut,
                     const LaunchGeometry &geometry) {
    json::Object r;
    // v5 adds the launch geometry visible at the host LTO callsite. This
    // matters for dispatch throughput (for example, how many proxy producers
    // can issue concurrently) and cannot be recovered from per-kernel device
    // metadata alone.
    //
    // Jumps 2 → 4 on purpose: the emitter had been left at 2 while the
    // schema doc already described a v3 (flops_to_first_use, trip_count,
    // distance_exact), so anything claiming 2 may or may not carry those.
    // Skipping the number keeps "3" from meaning two different things.
    r["schema_version"] = 5;
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
        if (auto l = sizeLog2(it->second))
            r["size_log2"] = *l;
        else
            r["size_log2"] = nullptr;
    } else {
        r["size_kind"] = nullptr;
        r["size_log2"] = nullptr;
    }

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
    r["fan_out"]       = fanOut;

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
    if (op.trip_count >= 0)
        r["trip_count"] = static_cast<int64_t>(op.trip_count);
    else
        r["trip_count"] = nullptr;
    r["distance_exact"] = op.distance_exact;

    if (auto est = iterEstimate(op.loop))
        r["iter_estimate"] = *est;
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
    json::Array legal;
    legal.push_back("proxy");
    if (hostCanStage) {
        legal.push_back("trigger");
        legal.push_back("ipc");
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

    json::Array records;
    for (const auto &s : inv.sites) {
        if (!s.haveTemplate) continue;
        int fanOut = kernelFanOut(s.kernelTemplate);
        const auto geometry = launchGeometry(s);
        for (const auto &op : s.kernelTemplate.ops) {
            records.push_back(
                toRecord(op.siteId, s.kernelTemplate.simpleName, op, fanOut,
                         geometry));
        }
    }
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
