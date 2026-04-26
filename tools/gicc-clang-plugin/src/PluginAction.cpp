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

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {

class GiccConsumer : public ASTConsumer {
public:
    GiccConsumer(CompilerInstance& CI, bool debug_hk)
        : CI_(CI), debug_hk_(debug_hk) {}
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
        (void)diags;

        gicc_plugin::KernelDiscoveryVisitor kd;
        kd.TraverseDecl(Ctx.getTranslationUnitDecl());

        gicc_plugin::LaunchSiteVisitor ls;
        ls.TraverseDecl(Ctx.getTranslationUnitDecl());

        // Phase 2.a: HK propagation per kernel.
        for (auto& ki : kd.kernels()) {
            gicc_plugin::HKAnalysis hk(ki.decl);
            hk.debug = debug_hk_;
            hk.run();
        }
    }
private:
    CompilerInstance& CI_;
    bool debug_hk_;
};

class GiccPluginAction : public PluginASTAction {
protected:
    std::unique_ptr<ASTConsumer>
    CreateASTConsumer(CompilerInstance& CI, llvm::StringRef) override {
        return std::make_unique<GiccConsumer>(CI, debug_hk_);
    }
    bool ParseArgs(const CompilerInstance&,
                   const std::vector<std::string>& args) override {
        for (const auto& a : args) {
            if (a == "debug-hk") debug_hk_ = true;
        }
        return true;
    }
    PluginASTAction::ActionType getActionType() override {
        return AddBeforeMainAction;
    }
private:
    bool debug_hk_ = false;
};

} // namespace

static FrontendPluginRegistry::Add<GiccPluginAction>
    X("gicc", "GICC unified-API code-generation plugin");
