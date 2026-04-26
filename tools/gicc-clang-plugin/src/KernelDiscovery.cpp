/**
 * KernelDiscovery.cpp — Phase 1 discovery for the gicc-clang-plugin.
 *
 * D1: KernelDiscoveryVisitor finds every FunctionDecl with the
 *     CUDAGlobalAttr (__global__) and records a KernelInfo entry.
 * D2: per-kernel call-site collection (added in a later commit).
 * D3: LaunchSiteVisitor finds gicc::launch<K> CallExprs and pulls the
 *     kernel FunctionDecl out of the template args (added in a later
 *     commit).
 */
#include "KernelDiscovery.h"
#include "llvm/Support/raw_ostream.h"

namespace gicc_plugin {

bool KernelDiscoveryVisitor::VisitFunctionDecl(clang::FunctionDecl* fd) {
    if (!fd->hasBody()) return true;
    if (!fd->hasAttr<clang::CUDAGlobalAttr>()) return true;
    KernelInfo ki;
    ki.decl = fd;
    kernels_.push_back(ki);
    llvm::errs() << "[gicc-plugin] kernel: " << fd->getNameAsString() << "\n";
    return true;
}

// LaunchSiteVisitor::VisitCallExpr — added in D3 commit.
bool LaunchSiteVisitor::VisitCallExpr(clang::CallExpr*) { return true; }

} // namespace gicc_plugin
