/**
 * Validator.h — Phase 2.b–2.e: enforce the v1 lift-able subset.
 *
 * The validator runs once per discovered KernelInfo and emits Clang
 * diagnostics for any code that violates spec §5 / §9.1. After this
 * phase, kernels that pass validation are eligible for Phase 3 trace
 * generation; kernels that fail produce a non-zero compiler exit.
 *
 * The validator owns no state across kernels — every check reads from
 * the KernelInfo (call sites discovered in phase 1) and the HKAnalysis
 * for that kernel.
 */
#pragma once

#include "Diagnostics.h"
#include "HKAnalysis.h"
#include "KernelDiscovery.h"

#include "clang/Frontend/CompilerInstance.h"

namespace gicc_plugin {

class Validator {
public:
    Validator(clang::CompilerInstance& CI, Diags d) : CI_(CI), diags_(d) {}

    // Run all enabled checks on `ki`. Returns false if any check
    // emitted an error. The caller does not need to short-circuit on
    // false — Clang's DiagnosticsEngine will fail the compile once the
    // error count is non-zero, and other kernels can still be checked.
    bool validate(const KernelInfo& ki, const HKAnalysis& hk);

private:
    clang::CompilerInstance& CI_;
    Diags diags_;
};

} // namespace gicc_plugin
