/**
 * PluginAction.cpp — gicc-clang-plugin entry point.
 *
 * Registered as a Clang FrontendAction under the name "gicc". When loaded
 * via -fplugin=gicc-clang-plugin.so it instantiates a GiccConsumer for
 * each TU. The consumer drives Phase 1 discovery (KernelDiscoveryVisitor
 * and LaunchSiteVisitor); later phases will add validation and trace
 * generation.
 */
#include "KernelDiscovery.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {

class GiccConsumer : public ASTConsumer {
public:
    explicit GiccConsumer(CompilerInstance& CI) : CI_(CI) {}
    void HandleTranslationUnit(ASTContext& Ctx) override {
        // HIP/CUDA invokes the frontend twice per TU (host + device pass).
        // We only want to analyze the host AST: that's where launch sites
        // live, and host is also where Phase 3 will inject trace
        // specializations. Skipping the device pass also avoids duplicate
        // Phase 2 diagnostics and Phase 3 ODR violations.
        if (Ctx.getLangOpts().CUDAIsDevice) return;

        llvm::errs() << "[gicc-plugin] consumer ran\n";
        gicc_plugin::KernelDiscoveryVisitor kd;
        kd.TraverseDecl(Ctx.getTranslationUnitDecl());

        gicc_plugin::LaunchSiteVisitor ls;
        ls.TraverseDecl(Ctx.getTranslationUnitDecl());
    }
private:
    CompilerInstance& CI_;
};

class GiccPluginAction : public PluginASTAction {
protected:
    std::unique_ptr<ASTConsumer>
    CreateASTConsumer(CompilerInstance& CI, llvm::StringRef) override {
        return std::make_unique<GiccConsumer>(CI);
    }
    bool ParseArgs(const CompilerInstance&,
                   const std::vector<std::string>&) override {
        return true;
    }
    PluginASTAction::ActionType getActionType() override {
        return AddBeforeMainAction;
    }
};

} // namespace

static FrontendPluginRegistry::Add<GiccPluginAction>
    X("gicc", "GICC unified-API code-generation plugin");
