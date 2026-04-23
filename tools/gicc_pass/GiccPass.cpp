/**
 * GiccPass.cpp — LLVM analysis pass for the GICC-Pilot selector.
 *
 * Module pass that:
 *   1. Loads a policy_<platform>.json at pass initialisation (path via
 *      $GICC_POLICY_FILE env var).
 *   2. For each call to gicc::put, gicc::barrier, or gicc::Runtime::prepare,
 *      extracts the feature tuple (peer_class, size_class, freq_class).
 *   3. Feeds the features to libgicc_policy::eval() → decision.
 *   4. Checks CXI cap feasibility (when $GICC_RANK_HINT is set). If
 *      GICC_PASS_CAP_PRUNE=1 at pass build time, infeasible decisions
 *      error out; if 0, they warn and fall back to conservative.
 *   5. Attaches a !gicc.decision metadata node to the call instruction.
 *   6. When $GICC_EXPLAIN=1, emits a one-line human-readable decision
 *      summary to stderr for each site.
 *
 * Registered as a NewPM plugin under the name `gicc-analysis`. Runs at
 * PipelineStartEP so that gicc::put / gicc::barrier call sites are still
 * visible (pre-inline). Emits only metadata — PreservedAnalyses::all().
 */

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "gicc/policy/policy.hpp"

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;
namespace gp = gicc::policy;

#ifndef GICC_PASS_CAP_PRUNE
#define GICC_PASS_CAP_PRUNE 1
#endif

namespace {

// ---------- Identifying a GICC call site -----------------------------------

// Return "gicc::put" / "gicc::barrier" / "gicc::Runtime::prepare" (or ""
// if this isn't a call we care about). We check demangled names because the
// pass runs pre-inline and the symbol names are still the Itanium-mangled
// forms corresponding to exactly these three APIs.
static std::string classifyCallSite(const CallBase& call) {
    const Function* f = call.getCalledFunction();
    if (!f) return "";
    StringRef mangled = f->getName();
    std::string demangled = demangle(mangled.str());
    StringRef dr(demangled);
    // Match the three APIs we reason about. Accept common signatures.
    if (dr.contains("gicc::put"))     return "gicc::put";
    if (dr.contains("gicc::barrier")) return "gicc::barrier";
    if (dr.contains("gicc::quiet"))   return "gicc::quiet";
    if (dr.contains("gicc::flush"))   return "gicc::flush";
    if (dr.contains("gicc::Runtime::prepare")) return "gicc::Runtime::prepare";
    // Also catch the wrapper/thunk names when inlining left a stub.
    if (mangled.contains("gicc3put")  && !mangled.contains("put_no_db"))    return "gicc::put";
    if (mangled.contains("gicc7barrier"))                                    return "gicc::barrier";
    if (mangled.contains("gicc5quiet"))                                      return "gicc::quiet";
    if (mangled.contains("gicc5flush"))                                      return "gicc::flush";
    if (mangled.contains("7Runtime7prepare"))                                return "gicc::Runtime::prepare";
    return "";
}

// ---------- Feature extraction (deliberately coarse v1) --------------------

// peer_class: Look at the peer slot specifically — in gicc::put / gicc::
// Runtime::prepare the peer argument is a 32-bit integer (the last one of
// those, after the 64-bit size argument for put; the first for prepare).
// If that specific slot is a ConstantInt → k_const; else if there is a
// surrounding llvm.var.annotation with "gicc_topology" on any arg →
// k_from_topology_hint; else dynamic.
static gp::PeerClass extractPeerClass(const CallBase& call) {
    // Scan args: only 32-bit integer operands. If at least one exists and
    // the LAST such is a ConstantInt, treat it as k_const. This matches
    // gicc::put(..., uint64_t size, int peer) and gicc::Runtime::prepare(int peer, ...).
    const Value* peer_operand = nullptr;
    for (const Use& op : call.args()) {
        Type* t = op.get()->getType();
        if (t && t->isIntegerTy(32)) peer_operand = op.get();
    }
    if (peer_operand) {
        if (isa<ConstantInt>(peer_operand)) return gp::PeerClass::KConst;
    }
    // Look for llvm.var.annotation on any of the call's Value arguments.
    // The Clang lowering of __attribute__((annotate("gicc_topology:..."))) emits
    // a call to llvm.var.annotation(ptr, annotation_cstr, ...).
    for (const Use& op : call.operands()) {
        const Value* v = op.get();
        if (!v) continue;
        for (const User* u : v->users()) {
            if (const auto* intrin = dyn_cast<IntrinsicInst>(u)) {
                if (intrin->getIntrinsicID() == Intrinsic::var_annotation ||
                    intrin->getIntrinsicID() == Intrinsic::ptr_annotation) {
                    if (intrin->arg_size() >= 2) {
                        const Value* ann = intrin->getArgOperand(1)->stripPointerCasts();
                        if (const auto* gv = dyn_cast<GlobalVariable>(ann)) {
                            if (gv->hasInitializer()) {
                                if (const auto* da = dyn_cast<ConstantDataArray>(gv->getInitializer())) {
                                    if (da->isString() && da->getAsCString().contains("gicc_topology"))
                                        return gp::PeerClass::KFromTopologyHint;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return gp::PeerClass::Dynamic;
}

// size_class: bucket a ConstantInt-sized argument; else dynamic. gicc::put
// takes the byte size as a 64-bit (size_t / unsigned long) operand — look
// only at 64-bit integer args and pick the largest ConstantInt. That avoids
// conflating size with the 32-bit peer rank.
static gp::SizeClass extractSizeClass(const CallBase& call) {
    uint64_t best = 0;
    bool found = false;
    for (const Use& op : call.args()) {
        Type* t = op.get()->getType();
        if (!t || !t->isIntegerTy(64)) continue;
        if (const auto* ci = dyn_cast<ConstantInt>(op.get())) {
            uint64_t v = ci->getZExtValue();
            if (v > best) { best = v; found = true; }
        }
    }
    if (!found) return gp::SizeClass::Dynamic;
    if (best <= (1ULL << 10))  return gp::SizeClass::SmallConst;   // ≤ 1 KB
    if (best <= (1ULL << 16))  return gp::SizeClass::MediumConst;  // ≤ 64 KB
    if (best <= (1ULL << 23))  return gp::SizeClass::LargeConst;   // ≤ 8 MB
    return gp::SizeClass::LargeConst;
}

// freq_class: walk enclosing loops via LoopInfo and ScalarEvolution. If the
// trip count is computable and ≥ 100 → hot_loop; ≥ 10 → outer_loop; else
// infrequent. If SE fails, Dynamic.
static gp::FreqClass extractFreqClass(const Instruction& I,
                                      LoopInfo& LI,
                                      ScalarEvolution& SE) {
    const BasicBlock* bb = I.getParent();
    Loop* L = LI.getLoopFor(bb);
    if (!L) return gp::FreqClass::Infrequent;
    const SCEV* tc = SE.getBackedgeTakenCount(L);
    if (!tc || isa<SCEVCouldNotCompute>(tc)) return gp::FreqClass::Dynamic;
    if (const auto* c = dyn_cast<SCEVConstant>(tc)) {
        uint64_t v = c->getAPInt().getZExtValue();
        if (v >= 100) return gp::FreqClass::HotLoop;
        if (v >= 10)  return gp::FreqClass::OuterLoop;
        return gp::FreqClass::Infrequent;
    }
    return gp::FreqClass::Dynamic;
}

// ---------- Metadata emission ----------------------------------------------

static MDNode* buildDecisionMD(LLVMContext& ctx,
                               StringRef api,
                               const gp::Features& f,
                               gp::Fabric fab,
                               const gp::Decision& d) {
    auto str = [&](StringRef s) { return MDString::get(ctx, s); };
    auto i32 = [&](int v) {
        return ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), v));
    };
    return MDNode::get(ctx, {
        str("api"),          str(api),
        str("peer_class"),   str(gp::to_cstr(f.peer)),
        str("size_class"),   str(gp::to_cstr(f.size)),
        str("freq_class"),   str(gp::to_cstr(f.freq)),
        str("platform"),     str(gp::to_cstr(fab)),
        str("rule_id"),      i32(d.rule_id),
        str("path"),         str(gp::to_cstr(d.path)),
        str("slot_depth"),   i32(d.slot_depth),
        str("pool_size"),    i32(d.pool_size),
        str("channel_map"),  str(gp::to_cstr(d.channel_map)),
        str("conservative"), i32(d.conservative ? 1 : 0),
    });
}

// ---------- --gicc-explain diagnostic (M2-E) --------------------------------

static void explain(raw_ostream& os,
                    const Instruction& I,
                    StringRef api,
                    const gp::Features& f,
                    gp::Fabric fab,
                    const gp::Decision& d) {
    os << "gicc-explain: ";
    if (DILocation* loc = I.getDebugLoc()) {
        os << loc->getFilename() << ":" << loc->getLine() << ":" << loc->getColumn() << "  ";
    } else {
        os << "<no-debug-loc>  ";
    }
    os << api << "  "
       << "features={peer=" << gp::to_cstr(f.peer)
       << ",size="          << gp::to_cstr(f.size)
       << ",freq="          << gp::to_cstr(f.freq) << "}  "
       << "platform="       << gp::to_cstr(fab)    << "  "
       << "rule_id="        << d.rule_id           << "  "
       << "decision={path=" << gp::to_cstr(d.path);
    if (d.path == gp::Path::OfiTriggered) os << ",slot_depth=" << d.slot_depth;
    if (d.path == gp::Path::OfiTriggered || d.path == gp::Path::OfiProxy)
        os << ",pool_size=" << d.pool_size;
    if (d.path == gp::Path::OfiProxy) os << ",channel_map=" << gp::to_cstr(d.channel_map);
    if (d.conservative) os << ",conservative=1";
    os << "}\n";
}

// ---------- The pass -------------------------------------------------------

class GiccAnalysisPass : public PassInfoMixin<GiccAnalysisPass> {
public:
    PreservedAnalyses run(Module& M, ModuleAnalysisManager& MAM) {
        // Load the policy on first invocation. A single opt/clang process
        // usually sees one module, but guard anyway so re-invocations don't
        // re-parse the JSON or duplicate the diagnostic banner.
        if (!policy_loaded_ && !policy_load_attempted_) {
            loadPolicy();
            policy_load_attempted_ = true;
        }
        if (!policy_loaded_) {
            return PreservedAnalyses::all();
        }

        bool explain_mode = explainModeFromEnv();
        int  rank_hint    = rankHintFromEnv();

        FunctionAnalysisManager& FAM =
            MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

        int sites = 0;
        for (Function& F : M) {
            if (F.isDeclaration()) continue;
            LoopInfo& LI = FAM.getResult<LoopAnalysis>(F);
            ScalarEvolution& SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
            for (Instruction& I : instructions(F)) {
                auto* call = dyn_cast<CallBase>(&I);
                if (!call) continue;
                std::string api = classifyCallSite(*call);
                if (api.empty()) continue;

                gp::Features f;
                f.peer = extractPeerClass(*call);
                f.size = extractSizeClass(*call);
                f.freq = extractFreqClass(*call, LI, SE);

                gp::Decision d = gp::eval(policy_, f);

                if (rank_hint > 0 && !gp::check_feasible(d, policy_.platform_desc, rank_hint)) {
#if GICC_PASS_CAP_PRUNE
                    errs() << "gicc-pass: error: infeasible decision for call to " << api
                           << " at P=" << rank_hint
                           << " (path=" << gp::to_cstr(d.path)
                           << " slot_depth=" << d.slot_depth
                           << " pool_size=" << d.pool_size
                           << ") — would violate CXI NIC caps\n";
                    d.conservative = true;
                    d.pool_size = 16;
                    d.slot_depth = 2;
                    d.path = gp::Path::OfiProxy;
                    d.rule_id = -3;
#else
                    errs() << "gicc-pass: warning: cap-unaware build lets through "
                              "infeasible decision for "
                           << api << " at P=" << rank_hint << "\n";
#endif
                }

                call->setMetadata("gicc.decision",
                    buildDecisionMD(M.getContext(), api,
                                    f, policy_.platform_desc.fabric, d));

                if (explain_mode)
                    explain(errs(), I, api, f, policy_.platform_desc.fabric, d);

                ++sites;
            }
        }

        if (explain_mode) {
            errs() << "gicc-pass: processed " << sites
                   << " GICC call site(s) in module "
                   << M.getName() << "\n";
        }
        return PreservedAnalyses::all();
    }

private:
    void loadPolicy() {
        const char* path = std::getenv("GICC_POLICY_FILE");
        if (!path || !*path) {
            errs() << "gicc-pass: $GICC_POLICY_FILE not set; pass is a no-op.\n";
            return;
        }
        try {
            policy_ = gp::load_policy(path);
            policy_loaded_ = true;
            if (std::getenv("GICC_EXPLAIN")) {
                errs() << "gicc-pass: loaded policy " << path
                       << " (" << policy_.rules.size() << " rules, platform="
                       << gp::to_cstr(policy_.platform_desc.fabric) << ")\n";
            }
        } catch (const std::exception& e) {
            errs() << "gicc-pass: failed to load policy '" << path << "': "
                   << e.what() << "\n";
        }
    }

    static bool explainModeFromEnv() {
        const char* e = std::getenv("GICC_EXPLAIN");
        return e && *e && std::string(e) != "0";
    }
    static int rankHintFromEnv() {
        const char* e = std::getenv("GICC_RANK_HINT");
        if (!e || !*e) return 0;
        try { return std::stoi(e); } catch (...) { return 0; }
    }

    gp::Policy policy_;
    bool policy_loaded_ = false;
    bool policy_load_attempted_ = false;
};

}  // namespace

// ---------- NewPM plugin registration --------------------------------------

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "gicc-analysis", "0.1",
        [](PassBuilder& PB) {
            // Explicit opt-style invocation: `opt -passes=gicc-analysis ...`.
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager& MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "gicc-analysis") {
                        MPM.addPass(GiccAnalysisPass());
                        return true;
                    }
                    return false;
                });
            // Auto-run via -fpass-plugin at -O2+ compile lines. Runs at
            // PipelineStartEP so user-facing gicc::* call sites are still
            // direct (pre-inline).
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager& MPM, OptimizationLevel) {
                    MPM.addPass(GiccAnalysisPass());
                });
        }};
}
