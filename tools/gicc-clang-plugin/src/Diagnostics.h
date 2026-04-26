/**
 * Diagnostics.h — Custom Clang diagnostic IDs for the GICC plugin.
 *
 * Each member of Diags is a DiagnosticsEngine custom ID created at TU
 * setup time. Validators emit diagnostics by calling
 *   eng.Report(loc, ids.<member>) << arg0 << arg1 ...;
 *
 * Format strings use Clang's %0/%1 placeholder convention, matching
 * spec §9.2. All errors are Level::Error except launch_via_triple_chevron
 * which is Warning per spec §9.3.
 */
#pragma once

#include "clang/Basic/Diagnostic.h"

namespace gicc_plugin {

struct Diags {
    unsigned non_hk_arg = 0;
    unsigned put_in_non_hk_branch = 0;
    unsigned non_whitelist_call = 0;
    unsigned cross_tu_helper = 0;
    unsigned recursion = 0;
    unsigned bad_first_arg = 0;
    unsigned launch_via_triple_chevron = 0;

    static Diags create(clang::DiagnosticsEngine& eng);
};

} // namespace gicc_plugin
