#include "GICCFeatureExtraction.h"
#include "GICCHostDiscovery.h"
#include "GICCPassConfig.h"
#include "LaunchSiteInventory.h"
#include "MetadataIO.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/SmallSet.h"

#include <cstdint>

using namespace llvm;

namespace gicc::pass {

namespace {

// Translate ArgRef::Kind to the JSON tag the decider consumes.
const char *argKindTag(ArgRef::Kind k) {
    switch (k) {
        case ArgRef::Kind::Param:    return "param";
        case ArgRef::Kind::ConstI64: return "const";
        case ArgRef::Kind::BinOp:    return "binop";
        case ArgRef::Kind::Cast:     return "cast";
        case ArgRef::Kind::Derived:  return "derived";
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

json::Value toRecord(const std::string &siteId,
                     const std::string &simpleKernel,
                     const OpTemplate  &op,
                     int                fanOut) {
    json::Object r;
    r["schema_version"] = 1;
    r["site_id"]        = siteId;
    r["kernel"]         = simpleKernel;
    r["op_kind"]        = op.kind;

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
    // peer_locality is filled by the decider / runtime introspection in
    // v2; v1 always reports null.
    r["peer_locality"] = nullptr;

    r["in_loop"]              = false;
    r["guard_density"]        = guardDensity(op.guard);
    r["fan_out"]              = fanOut;
    r["compute_before_flops"] = 0;
    r["iter_estimate"]        = nullptr;
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
        for (const auto &op : s.kernelTemplate.ops) {
            records.push_back(
                toRecord(op.siteId, s.kernelTemplate.simpleName, op, fanOut));
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
