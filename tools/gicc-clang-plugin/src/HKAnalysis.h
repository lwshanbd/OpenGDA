/**
 * HKAnalysis.h — Phase 2.a: Host-Known value propagation.
 *
 * Per spec §5.1 the plugin annotates each AST expression with a
 * "host-known" (HK) bit. An expression is HK iff its enumeration over
 * the launch grid can be computed at the launch site without executing
 * any device code. The seed set is:
 *   - kernel function parameters
 *   - integer / float / character / boolean literals
 *   - CUDA/HIP builtins: threadIdx.{x,y,z}, blockIdx.{x,y,z},
 *     blockDim.*, gridDim.* (HK with grid-dim tag — tag is informative
 *     for Phase F trace generation, not enforced here)
 *
 * Propagation:
 *   - Pure arithmetic (+, -, *, /, %, shifts, bitwise, comparisons) of
 *     HK operands is HK.
 *   - Unary -, +, !, ~ of HK operands is HK.
 *   - VarDecl initializers that are HK make the VarDecl HK; subsequent
 *     DeclRefExprs to that VarDecl are HK.
 *
 * NOT HK (conservative for v1):
 *   - ArraySubscriptExpr (memory load)
 *   - MemberExpr on a non-builtin base (struct field load)
 *   - UnaryOperator UO_Deref (*ptr)
 *   - CallExpr return values
 *   - Atomic op return values
 */
#pragma once

#include "clang/AST/AST.h"
#include "llvm/ADT/DenseSet.h"

namespace gicc_plugin {

class HKAnalysis {
public:
    explicit HKAnalysis(clang::FunctionDecl* kernel);

    // Run the analysis: walks the kernel body once to populate the
    // VarDecl HK set. Idempotent.
    void run();

    // Query whether an expression is host-known. Safe to call after run().
    bool isHK(const clang::Expr* e) const;

    // Debug toggle — when true, run() prints per-parameter and
    // per-DeclStmt-init HK status for visual inspection in lit tests.
    bool debug = false;

private:
    bool exprIsHK(const clang::Expr* e) const;
    bool declIsHK(const clang::ValueDecl* d) const;

    clang::FunctionDecl* kernel_;
    llvm::DenseSet<const clang::ValueDecl*> hk_decls_;
};

} // namespace gicc_plugin
