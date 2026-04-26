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
        llvm::errs() << "[gicc-plugin] consumer ran\n";
        gicc_plugin::KernelDiscoveryVisitor kd;
        kd.TraverseDecl(Ctx.getTranslationUnitDecl());
        // D2/D3 visitors added in subsequent commits.
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
