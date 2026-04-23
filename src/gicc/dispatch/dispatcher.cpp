/**
 * dispatcher.cpp — Implementation of the host-side multiversion dispatcher.
 */
#include "gicc/dispatch/dispatcher.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace gicc {
namespace dispatch {

namespace {
// ---------- Feature classification (deliberately coarse v1) -----------------

policy::PeerClass classifyPeer(const PrepareObs& obs) {
    if (obs.has_topology_hint) return policy::PeerClass::KFromTopologyHint;
    if (obs.peer_rank >= 0)    return policy::PeerClass::KConst;
    return policy::PeerClass::Dynamic;
}

policy::SizeClass classifySize(const PrepareObs& obs) {
    std::size_t s = obs.transfer_size ? obs.transfer_size : obs.buffer_size;
    if (s == 0)             return policy::SizeClass::Dynamic;
    if (s <= (1ULL << 10))  return policy::SizeClass::SmallConst;
    if (s <= (1ULL << 16))  return policy::SizeClass::MediumConst;
    return policy::SizeClass::LargeConst;
}

// freq_class is not observable at prepare() time; the dispatcher treats it
// as Dynamic so that rule-list wildcard matching handles it. The LLVM pass
// is the place where freq_class gets statically refined via LoopInfo/SCEV.
policy::FreqClass classifyFreq(const PrepareObs&) {
    return policy::FreqClass::Dynamic;
}

// Map a Decision to a small variant index. For M2-G v1 we use 4 buckets:
//   0 — conservative
//   1 — ofi_proxy primary
//   2 — ofi_triggered primary
//   3 — ib_native primary
// Downstream device-side specialisation can refine this.
uint8_t variantIndex(const policy::Decision& d) {
    if (d.conservative) return 0;
    switch (d.path) {
        case policy::Path::OfiProxy:     return 1;
        case policy::Path::OfiTriggered: return 2;
        case policy::Path::IbNative:     return 3;
    }
    return 0;
}
}  // namespace

policy::Features Dispatcher::classify(const PrepareObs& obs) const {
    policy::Features f;
    f.peer = classifyPeer(obs);
    f.size = classifySize(obs);
    f.freq = classifyFreq(obs);
    return f;
}

Dispatcher::Dispatcher(const std::string& policy_path) {
    std::string p = policy_path;
    if (p.empty()) {
        if (const char* env = std::getenv("GICC_POLICY_FILE")) p = env;
    }
    if (p.empty()) {
        std::fprintf(stderr,
            "gicc-dispatch: no policy file supplied; bind() will fall back "
            "to conservative.\n");
        return;
    }
    try {
        policy_ = policy::load_policy(p);
        policy_loaded_ = true;
    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "gicc-dispatch: failed to load policy '%s': %s\n",
            p.c_str(), e.what());
    }
    if (const char* env = std::getenv("GICC_EXPLAIN")) {
        if (env[0] && std::strcmp(env, "0") != 0) explain_ = true;
    }
}

Binding Dispatcher::bind(const PrepareObs& obs) {
    Binding out;
    if (!policy_loaded_) {
        out.decision.path = policy::Path::OfiProxy;
        out.decision.pool_size = 16;
        out.decision.conservative = true;
        out.decision.rule_id = -1;
        out.decision.source = "fallback";
        out.variant_idx = 0;
        return out;
    }
    policy::Features f = classify(obs);
    out.decision = policy::eval(policy_, f);
    out.variant_idx = variantIndex(out.decision);

    if (explain_) {
        std::fprintf(stderr,
            "gicc-dispatch: prepare peer=%d size=%zu topo=%d -> rule_id=%d path=%s"
            " slot_depth=%d pool_size=%d variant_idx=%u%s\n",
            obs.peer_rank, (obs.transfer_size ? obs.transfer_size : obs.buffer_size),
            obs.has_topology_hint ? 1 : 0,
            out.decision.rule_id, policy::to_cstr(out.decision.path),
            out.decision.slot_depth, out.decision.pool_size,
            (unsigned)out.variant_idx, out.decision.conservative ? " CONSERVATIVE" : "");
    }
    return out;
}

}  // namespace dispatch
}  // namespace gicc
