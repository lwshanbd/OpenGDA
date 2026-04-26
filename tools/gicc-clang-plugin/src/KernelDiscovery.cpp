/**
 * KernelDiscovery.cpp — Phase 1 discovery for the gicc-clang-plugin.
 *
 * D1: KernelDiscoveryVisitor finds every FunctionDecl with the
 *     CUDAGlobalAttr (__global__) and records a KernelInfo entry.
 * D2: CallCollector walks each kernel's body and records every CallExpr
 *     whose direct callee is in the gicc:: namespace.
 * D3: LaunchSiteVisitor finds gicc::launch<K> CallExprs and pulls the
 *     kernel FunctionDecl out of the template args (added in a later
 *     commit).
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

// LaunchSiteVisitor::VisitCallExpr — added in D3 commit.
bool LaunchSiteVisitor::VisitCallExpr(clang::CallExpr*) { return true; }

} // namespace gicc_plugin
