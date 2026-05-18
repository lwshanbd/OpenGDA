/*
 * DispatchDecision.h - Shared dispatch-routing utilities.
 *
 * The dispatch routing (which dispatch each put_no_db site uses) is a
 * pure function of (site_id, hint.json). Both the device-side and
 * host-side lowering passes need this answer, but they previously
 * exchanged it through the kernel JSON's `proxy_aware` field. That
 * cross-pass channel created an ordering hazard: in a single compile
 * invocation the device pass runs first and reads the JSON before the
 * host pass has written the `proxy_aware` bit. Direction 3 of the
 * mixed-dispatch fix is to have both passes compute the decision
 * directly from hint.json instead of routing it through JSON.
 *
 * This header exposes the minimum that both passes need:
 *   - DispatchKind                 enum of dispatch kinds
 *   - HintFile                     parsed hint.json
 *   - readHintFile / hintFor       loader + per-site lookup
 *   - kernelHasProxySite           "should this kernel's device-side
 *                                   put_no_db body be preserved?"
 */
#pragma once

#include "KernelInventory.h"

#include "llvm/ADT/StringRef.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace gicc {
namespace pass {

// Order preserved from the pre-refactor definition in
// GICCDispatchLowering.cpp so any sentinel-comparison or default-value
// behavior stays identical. The enum is enum class so no implicit int
// conversion can leak, but keeping the layout stable is the
// defensive choice.
enum class DispatchKind {
    IpcPush,         // force IPC path (assumes peer is mapped)
    DwqTrigger,      // force DWQ path
    DwqBatched,      // currently lowered same as DwqTrigger
    IpcOrDwq,        // hybrid: same-node IPC, off-node DWQ — default
                     //   when no hint.json supplied
    CpuProxyEnqueue, // device-side enqueue to CPU proxy ring; host
                     //   trace emits nothing for the site (the actual
                     //   work is performed device-side and serviced
                     //   by the CPU proxy thread). Required for
                     //   HK-incapable sites.
    Unknown,
};

DispatchKind parseDispatch(llvm::StringRef s);
const char  *dispatchName(DispatchKind d);

struct SiteHint {
    DispatchKind dispatch    = DispatchKind::Unknown;
    int          streamIndex = 0;
};

struct HintFile {
    // Default when hint.json is missing or the site has no entry. The
    // hybrid IPC_OR_DWQ matches the v1 production default — same-node
    // peers go through IPC, off-node fall back to DWQ.
    DispatchKind                            defaultDispatch = DispatchKind::IpcOrDwq;
    std::unordered_map<std::string, SiteHint> sites;
};

// Load a hint.json from disk into `out`. Returns false on I/O or
// parse error; `out` is left in default state on failure (caller can
// then treat sites as defaultDispatch).
bool readHintFile(const std::string &path, HintFile &out);

// Look up the SiteHint for `siteId`. Falls back to defaultDispatch
// when the site isn't explicitly listed.
SiteHint hintFor(const HintFile &h, llvm::StringRef siteId);

// Returns true if any put_no_db / get_no_db / quiet site in `sites`
// is routed to CPU_PROXY_ENQUEUE under the given hint. Used by
// device-side lowering to decide whether to preserve the in-kernel
// put_no_db body (so the device-pushed proxy ring enqueue stays in
// the kernel). Computed locally from hint + site list — no JSON
// read-back required.
bool kernelHasProxySite(const std::vector<GICCCallSite> &sites,
                        const HintFile                  &hint);

}  // namespace pass
}  // namespace gicc
