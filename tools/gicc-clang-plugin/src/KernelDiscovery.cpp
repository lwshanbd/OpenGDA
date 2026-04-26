/**
 * KernelDiscovery.cpp — Phase 1 discovery for the gicc-clang-plugin.
 *
 * D1: KernelDiscoveryVisitor finds every FunctionDecl with the
 *     CUDAGlobalAttr (__global__) and records a KernelInfo entry.
 * D2: CallCollector walks each kernel's body and records every CallExpr
 *     whose direct callee is in the gicc:: namespace.
 * D3: LaunchSiteVisitor finds CallExprs that resolve to gicc::launch and
 *     pulls the kernel FunctionDecl out of template argument 0. The
 *     launch wrapper is `template<auto Kernel> void launch(...)`, so the
 *     kernel is a non-type template parameter, not a runtime arg.
 */
#include "KernelDiscovery.h"
#include "llvm/Support/raw_ostream.h"

namespace gicc_plugin {

namespace {

// Walks a kernel body and records every direct call into the gicc::
// namespace. Owned by the KernelInfo it appends to.
class CallCollector : public clang::RecursiveASTVisitor<CallCollector> {
public:
    explicit CallCollector(KernelInfo& ki) : ki_(ki) {}
    bool VisitCallExpr(clang::CallExpr* ce) {
        auto* fd = ce->getDirectCallee();
        if (!fd) return true;
        std::string qn = fd->getQualifiedNameAsString();
        if (qn.rfind("gicc::", 0) == 0) {
            ki_.calls.push_back({qn, ce});
            llvm::errs() << "[gicc-plugin]   call: " << qn << "\n";
        }
        return true;
    }
private:
    KernelInfo& ki_;
};

// Pull a FunctionDecl* out of a TemplateArgument that should hold a
// pointer to a kernel function. For `template<auto Kernel>` instantiated
// with `&my_kernel`, Clang typically materializes this as a Declaration
// argument; older / different forms can land as Expression (a
// DeclRefExpr or address-of of one). Handle both.
clang::FunctionDecl* extractKernel(const clang::TemplateArgument& targ) {
    if (targ.getKind() == clang::TemplateArgument::Declaration) {
        if (auto* fd = llvm::dyn_cast_or_null<clang::FunctionDecl>(
                targ.getAsDecl())) {
            return fd;
        }
    }
    if (targ.getKind() == clang::TemplateArgument::Expression) {
        const clang::Expr* e = targ.getAsExpr()->IgnoreParenImpCasts();
        if (auto* uo = llvm::dyn_cast<clang::UnaryOperator>(e)) {
            e = uo->getSubExpr()->IgnoreParenImpCasts();
        }
        if (auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(e)) {
            if (auto* fd = llvm::dyn_cast<clang::FunctionDecl>(dre->getDecl())) {
                return const_cast<clang::FunctionDecl*>(fd);
            }
        }
    }
    return nullptr;
}

} // namespace

bool KernelDiscoveryVisitor::VisitFunctionDecl(clang::FunctionDecl* fd) {
    if (!fd->hasBody()) return true;
    if (!fd->hasAttr<clang::CUDAGlobalAttr>()) return true;
    KernelInfo ki;
    ki.decl = fd;
    kernels_.push_back(ki);
    llvm::errs() << "[gicc-plugin] kernel: " << fd->getNameAsString() << "\n";
    CallCollector cc(kernels_.back());
    cc.TraverseStmt(fd->getBody());
    return true;
}

bool LaunchSiteVisitor::VisitCallExpr(clang::CallExpr* ce) {
    auto* callee = ce->getDirectCallee();
    if (!callee) return true;
    if (callee->getQualifiedNameAsString() != "gicc::launch") return true;

    const auto* tsi = callee->getTemplateSpecializationArgs();
    if (!tsi || tsi->size() < 1) return true;

    clang::FunctionDecl* kernel = extractKernel(tsi->get(0));
    if (!kernel) return true;

    sites_.push_back({ce, kernel});
    llvm::errs() << "[gicc-plugin] launch_site -> kernel: "
                 << kernel->getNameAsString() << "\n";
    return true;
}

} // namespace gicc_plugin
