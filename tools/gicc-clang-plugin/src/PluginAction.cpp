/**
 * PluginAction.cpp — gicc-clang-plugin entry point.
 *
 * Registered as a Clang FrontendAction under the name "gicc". When loaded
 * via -fplugin=gicc-clang-plugin.so it instantiates a GiccConsumer for
 * each TU. The consumer drives Phase 1 discovery (KernelDiscoveryVisitor
 * and LaunchSiteVisitor); later phases will add validation and trace
 * generation.
 */
#include "Diagnostics.h"
#include "HKAnalysis.h"
#include "KernelDiscovery.h"
#include "TraceEmitter.h"
#include "Validator.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {

class GiccConsumer : public ASTConsumer {
public:
    GiccConsumer(CompilerInstance& CI, bool debug_hk, std::string sidecar_dir)
        : CI_(CI), debug_hk_(debug_hk),
          sidecar_dir_(std::move(sidecar_dir)) {}
    void HandleTranslationUnit(ASTContext& Ctx) override {
        // HIP/CUDA invokes the frontend twice per TU (host + device pass).
        // We only want to analyze the host AST: that's where launch sites
        // live, and host is also where Phase 3 will inject trace
        // specializations. Skipping the device pass also avoids duplicate
        // Phase 2 diagnostics and Phase 3 ODR violations.
        if (Ctx.getLangOpts().CUDAIsDevice) return;

        llvm::errs() << "[gicc-plugin] consumer ran\n";

        // Build the per-TU diagnostic ID table once; passed by value
        // (cheap) into validators in later phases.
        gicc_plugin::Diags diags = gicc_plugin::Diags::create(
            CI_.getDiagnostics());

        gicc_plugin::KernelDiscoveryVisitor kd;
        kd.TraverseDecl(Ctx.getTranslationUnitDecl());

        gicc_plugin::LaunchSiteVisitor ls;
        ls.TraverseDecl(Ctx.getTranslationUnitDecl());

        // Phase 2.a + 2.b: HK propagation + validation per kernel.
        // Phase 3 (F1+): for each validated GICC kernel, emit a sidecar
        // file with a kernel_trace specialization. We skip kernels that
        // don't call into gicc:: at all (pure HIP/CUDA — not our concern)
        // and kernels that fail validation (errors already diagnosed).
        gicc_plugin::Validator val(CI_, diags);
        gicc_plugin::TraceEmitter te(CI_, sidecar_dir_);
        bool emitted_any = false;
        for (auto& ki : kd.kernels()) {
            gicc_plugin::HKAnalysis hk(ki.decl);
            hk.debug = debug_hk_;
            hk.run();
            const bool ok = val.validate(ki, hk);
            if (!ok) continue;
            if (ki.calls.empty()) continue;
            te.emit(ki, hk);
            emitted_any = true;
        }
        // Phase G: every TU built with -fplugin must produce a predictable
        // sidecar so CMake can wire it into the build graph. If nothing
        // was lifted above, the buffer for this TU is empty — call
        // ensure_sidecar() to seed an empty header-only buffer.
        if (!emitted_any) {
            auto& sm = CI_.getSourceManager();
            auto fid = sm.getMainFileID();
            auto fref = sm.getFileEntryRefForID(fid);
            std::string main_file = fref
                ? std::string(fref->getName())
                : std::string("unknown");
            te.ensure_sidecar(main_file);
        }
        // Commit buffered sidecar content to disk. Writes are skipped
        // when the on-disk content is byte-for-byte identical, so the
        // file's mtime is preserved across no-op rebuilds — this is what
        // lets the CMake helper use mtime-based dependency tracking
        // without thrashing.
        te.flush();
    }
private:
    CompilerInstance& CI_;
    bool debug_hk_;
    std::string sidecar_dir_;
};

class GiccPluginAction : public PluginASTAction {
protected:
    std::unique_ptr<ASTConsumer>
    CreateASTConsumer(CompilerInstance& CI, llvm::StringRef) override {
        return std::make_unique<GiccConsumer>(CI, debug_hk_, sidecar_dir_);
    }
    bool ParseArgs(const CompilerInstance&,
                   const std::vector<std::string>& args) override {
        // Accepted flags:
        //   debug-hk                          — verbose HK propagation
        //   sidecar-dir=<path>                — write sidecars to <path>
        // Empty sidecar-dir keeps the legacy /tmp default (lit tests).
        const llvm::StringRef sidecar_prefix = "sidecar-dir=";
        for (const auto& a : args) {
            if (a == "debug-hk") {
                debug_hk_ = true;
            } else if (llvm::StringRef(a).starts_with(sidecar_prefix)) {
                sidecar_dir_ = a.substr(sidecar_prefix.size());
            }
        }
        return true;
    }
    PluginASTAction::ActionType getActionType() override {
        return AddBeforeMainAction;
    }
private:
    bool debug_hk_ = false;
    std::string sidecar_dir_;
};

} // namespace

static FrontendPluginRegistry::Add<GiccPluginAction>
    X("gicc", "GICC unified-API code-generation plugin");
