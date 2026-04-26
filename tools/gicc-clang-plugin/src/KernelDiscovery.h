/**
 * KernelDiscovery.h — Phase 1 discovery for the gicc-clang-plugin.
 *
 * Walks a translation unit's AST and collects:
 *   - All FunctionDecl with __global__ attribute (kernels)
 *   - For each kernel, its body's calls to gicc::* symbols (added in D2)
 *   - All gicc::launch<K> call sites + the K they target  (added in D3)
 */
#pragma once

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include <string>
#include <vector>

namespace gicc_plugin {

struct CallInfo {
    std::string qualified_name;   // e.g. "gicc::put_no_db"
    clang::CallExpr* expr;
};

struct KernelInfo {
    clang::FunctionDecl* decl;
    std::vector<CallInfo> calls;
};

struct LaunchSite {
    clang::CallExpr* expr;
    clang::FunctionDecl* kernel;  // the K in gicc::launch<K>
};

class KernelDiscoveryVisitor
    : public clang::RecursiveASTVisitor<KernelDiscoveryVisitor> {
public:
    bool VisitFunctionDecl(clang::FunctionDecl* fd);
    const std::vector<KernelInfo>& kernels() const { return kernels_; }
private:
    std::vector<KernelInfo> kernels_;
};

class LaunchSiteVisitor
    : public clang::RecursiveASTVisitor<LaunchSiteVisitor> {
public:
    bool VisitCallExpr(clang::CallExpr* ce);
    const std::vector<LaunchSite>& sites() const { return sites_; }
private:
    std::vector<LaunchSite> sites_;
};

} // namespace gicc_plugin
