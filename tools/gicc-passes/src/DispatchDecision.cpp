/*
 * DispatchDecision.cpp - shared dispatch-routing logic.
 *
 * Extracted from GICCDispatchLowering.cpp so device-side passes can
 * compute the same answer without going through the kernel JSON's
 * proxy_aware field. See header for rationale.
 */
#include "DispatchDecision.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

namespace gicc {
namespace pass {

DispatchKind parseDispatch(llvm::StringRef s) {
    if (s == "IPC_PUSH")          return DispatchKind::IpcPush;
    if (s == "DWQ_TRIGGER")       return DispatchKind::DwqTrigger;
    if (s == "DWQ_BATCHED")       return DispatchKind::DwqBatched;
    if (s == "IPC_OR_DWQ")        return DispatchKind::IpcOrDwq;
    if (s == "CPU_PROXY_ENQUEUE") return DispatchKind::CpuProxyEnqueue;
    return DispatchKind::Unknown;
}

const char *dispatchName(DispatchKind d) {
    switch (d) {
        case DispatchKind::IpcPush:         return "IPC_PUSH";
        case DispatchKind::DwqTrigger:      return "DWQ_TRIGGER";
        case DispatchKind::DwqBatched:      return "DWQ_BATCHED";
        case DispatchKind::IpcOrDwq:        return "IPC_OR_DWQ";
        case DispatchKind::CpuProxyEnqueue: return "CPU_PROXY_ENQUEUE";
        case DispatchKind::Unknown:         return "UNKNOWN";
    }
    return "UNKNOWN";
}

bool readHintFile(const std::string &path, HintFile &out) {
    auto bufOr = llvm::MemoryBuffer::getFile(path);
    if (!bufOr) return false;
    auto parsed = llvm::json::parse((*bufOr)->getBuffer());
    if (!parsed) {
        llvm::consumeError(parsed.takeError());
        return false;
    }
    const auto *root = parsed->getAsObject();
    if (!root) return false;
    if (auto def = root->getString("default_dispatch"))
        out.defaultDispatch = parseDispatch(*def);
    if (const auto *sites = root->getObject("sites")) {
        for (const auto &kv : *sites) {
            const auto *entry = kv.second.getAsObject();
            if (!entry) continue;
            auto disp = entry->getString("dispatch");
            if (!disp) continue;
            SiteHint sh;
            sh.dispatch = parseDispatch(*disp);
            if (auto v = entry->getInteger("stream_index"))
                sh.streamIndex = static_cast<int>(*v);
            out.sites[kv.first.str()] = sh;
        }
    }
    return true;
}

SiteHint hintFor(const HintFile &h, llvm::StringRef siteId) {
    auto it = h.sites.find(siteId.str());
    if (it != h.sites.end()) return it->second;
    SiteHint sh;
    sh.dispatch    = h.defaultDispatch;
    sh.streamIndex = 0;
    return sh;
}

bool kernelHasProxySite(const std::vector<GICCCallSite> &sites,
                        const HintFile                  &hint) {
    // Match GICCDispatchLowering's placeholder-lowering scope:
    // dispatch decisions exist only for put_no_db / get_no_db sites.
    // Flush is unconditional infrastructure (MMIO trigger), and quiet
    // has no independent dispatch — it is preserved on device iff
    // SOME put/get in the same kernel is proxy-routed (which is what
    // this function answers).
    for (const auto &s : sites) {
        if (s.kind != GICCOpKind::PutNoDb
            && s.kind != GICCOpKind::GetNoDb)
            continue;
        SiteHint sh = hintFor(hint, s.siteId);
        if (sh.dispatch == DispatchKind::CpuProxyEnqueue)
            return true;
    }
    return false;
}

}  // namespace pass
}  // namespace gicc
