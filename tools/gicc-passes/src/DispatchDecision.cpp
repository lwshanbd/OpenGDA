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

CommunicationTransform parseCommunicationTransform(llvm::StringRef s) {
    if (s == "NONE")                return CommunicationTransform::None;
    if (s == "COALESCE_LOOP")       return CommunicationTransform::CoalesceLoop;
    if (s == "COALESCE_LOOP_EARLY") return CommunicationTransform::CoalesceLoopEarly;
    if (s == "TRIGGER_GROUP_EARLY") return CommunicationTransform::TriggerGroupEarly;
    return CommunicationTransform::Unknown;
}

const char *communicationTransformName(CommunicationTransform t) {
    switch (t) {
        case CommunicationTransform::None:         return "NONE";
        case CommunicationTransform::CoalesceLoop: return "COALESCE_LOOP";
        case CommunicationTransform::CoalesceLoopEarly:
            return "COALESCE_LOOP_EARLY";
        case CommunicationTransform::TriggerGroupEarly:
            return "TRIGGER_GROUP_EARLY";
        case CommunicationTransform::Unknown:      return "UNKNOWN";
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

    // Treat the hint as an untrusted decision boundary.  In particular, do
    // not let an unknown model-produced string become DispatchKind::Unknown
    // and then silently lower as DWQ_TRIGGER.  Parse into a temporary object
    // so a late error cannot leave `out` half populated.
    auto version = root->getInteger("version");
    auto schema  = root->getString("schema_version");
    if (!version || *version != 1 || !schema || *schema != "gicc-hint-v1")
        return false;

    HintFile candidate;
    if (auto def = root->getString("default_dispatch")) {
        candidate.defaultDispatch = parseDispatch(*def);
        if (candidate.defaultDispatch == DispatchKind::Unknown) return false;
    }
    if (const auto *sites = root->getObject("sites")) {
        for (const auto &kv : *sites) {
            const auto *entry = kv.second.getAsObject();
            if (!entry) return false;
            auto disp = entry->getString("dispatch");
            if (!disp) return false;
            SiteHint sh;
            sh.dispatch = parseDispatch(*disp);
            if (sh.dispatch == DispatchKind::Unknown) return false;
            if (auto transform = entry->getString("transform")) {
                sh.transform = parseCommunicationTransform(*transform);
                if (sh.transform == CommunicationTransform::Unknown)
                    return false;
            }
            if (auto v = entry->getInteger("stream_index")) {
                if (*v < 0 || *v > 1024) return false;
                sh.streamIndex = static_cast<int>(*v);
            }
            candidate.sites[kv.first.str()] = sh;
        }
    } else if (root->get("sites")) {
        return false;
    }

    out = std::move(candidate);
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
