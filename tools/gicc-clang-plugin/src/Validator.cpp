/**
 * Validator.cpp — see Validator.h for design intent.
 *
 * E3 (this commit): every put_no_db / get_no_db call has all non-ctx,
 * non-signaled arguments host-known. Argument names by index:
 *   0 ctx          (skip)
 *   1 local_addr
 *   2 local_lkey
 *   3 remote_addr
 *   4 remote_rkey
 *   5 size
 *   6 signaled     (optional, skip)
 *
 * Later commits add E4 (whitelist), E5 (non-HK control flow), and E6
 * (first-arg type) checks.
 */
#include "Validator.h"

namespace gicc_plugin {

namespace {

// Map argument index → human-readable name for put_no_db / get_no_db.
// Used in the diagnostic message to point the user at the wrong arg.
const char* arg_name_for_put(unsigned i) {
    static const char* names[] = {
        "ctx", "local_addr", "local_lkey", "remote_addr",
        "remote_rkey", "size", "signaled"
    };
    if (i < sizeof(names) / sizeof(names[0])) return names[i];
    return "<?>";
}

// Strip the leading "gicc::" namespace from a qualified name so the
// diagnostic prints "gicc::put_no_db" cleanly via the format string
// (which already includes the "gicc::" prefix).
std::string strip_gicc_prefix(const std::string& qn) {
    static const std::string prefix = "gicc::";
    if (qn.compare(0, prefix.size(), prefix) == 0) {
        return qn.substr(prefix.size());
    }
    return qn;
}

} // namespace

bool Validator::validate(const KernelInfo& ki, const HKAnalysis& hk) {
    bool ok = true;
    auto& diag = CI_.getDiagnostics();

    for (const auto& call : ki.calls) {
        const bool is_put = (call.qualified_name == "gicc::put_no_db" ||
                             call.qualified_name == "gicc::get_no_db");
        if (!is_put) continue;

        auto* ce = call.expr;
        const unsigned n = ce->getNumArgs();
        const std::string short_fn = strip_gicc_prefix(call.qualified_name);

        // Skip ctx (arg 0). We check arg 6 (signaled) only if the
        // user passes a 7th arg, but per the spec it is allowed to be
        // any bool literal — still, exclude index 6 explicitly so a
        // common pattern like `, /*signaled*/ false` is never flagged.
        for (unsigned i = 1; i < n && i < 6; i++) {
            auto* a = ce->getArg(i);
            if (!hk.isHK(a)) {
                diag.Report(a->getBeginLoc(), diags_.non_hk_arg)
                    << short_fn
                    << arg_name_for_put(i);
                ok = false;
            }
        }
    }

    return ok;
}

} // namespace gicc_plugin
