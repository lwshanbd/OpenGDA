/**
 * Diagnostics.cpp — register the GICC plugin's custom diagnostic IDs.
 *
 * IDs returned by getCustomDiagID are stable for the lifetime of the
 * DiagnosticsEngine, so we build a Diags struct once per TU and pass it
 * to validators that need to report errors. Format strings match spec
 * §9.2.
 */
#include "Diagnostics.h"

namespace gicc_plugin {

Diags Diags::create(clang::DiagnosticsEngine& eng) {
    using L = clang::DiagnosticsEngine::Level;
    Diags d;
    d.non_hk_arg = eng.getCustomDiagID(
        L::Error,
        "gicc::%0 argument '%1' is not host-known");
    d.put_in_non_hk_branch = eng.getCustomDiagID(
        L::Error,
        "gicc::%0 inside a control-flow construct gated by a "
        "non-host-known condition");
    d.non_whitelist_call = eng.getCustomDiagID(
        L::Error,
        "gicc::%0 cannot be lifted to OFI; not in v1 whitelist "
        "(use put_no_db/get_no_db/flush/quiet)");
    d.cross_tu_helper = eng.getCustomDiagID(
        L::Error,
        "helper '%0' is called along a put_no_db path but is opaque "
        "(cross-TU or __noinline__)");
    d.recursion = eng.getCustomDiagID(
        L::Error,
        "kernel '%0' is recursive; not supported by v1 lift");
    d.bad_first_arg = eng.getCustomDiagID(
        L::Error,
        "kernel '%0' must take gicc::DeviceCtx* as its first parameter");
    d.launch_via_triple_chevron = eng.getCustomDiagID(
        L::Warning,
        "kernel '%0' is launched via <<<>>>; the OFI build will not "
        "pre-stage RDMA — use gicc::launch instead");
    return d;
}

} // namespace gicc_plugin
