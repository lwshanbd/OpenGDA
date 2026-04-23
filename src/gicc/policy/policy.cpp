/**
 * policy.cpp — Implementation of the selector rule-list evaluator.
 */
#include "gicc/policy/policy.hpp"
#include "gicc/policy/mini_json.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace gicc {
namespace policy {

// ---------- Enum <-> string conversions -------------------------------------

const char* to_cstr(Fabric f) {
    switch (f) {
        case Fabric::IbMlx5: return "ib_mlx5";
        case Fabric::OfiCxi: return "ofi_cxi";
    }
    return "?";
}
const char* to_cstr(Path p) {
    switch (p) {
        case Path::IbNative:     return "ib_native";
        case Path::OfiTriggered: return "ofi_triggered";
        case Path::OfiProxy:     return "ofi_proxy";
    }
    return "?";
}
const char* to_cstr(PeerClass pc) {
    switch (pc) {
        case PeerClass::KConst:            return "k_const";
        case PeerClass::KFromTopologyHint: return "k_from_topology_hint";
        case PeerClass::Dynamic:           return "dynamic";
    }
    return "?";
}
const char* to_cstr(SizeClass sc) {
    switch (sc) {
        case SizeClass::SmallConst:  return "small_const";
        case SizeClass::MediumConst: return "medium_const";
        case SizeClass::LargeConst:  return "large_const";
        case SizeClass::Dynamic:     return "dynamic";
    }
    return "?";
}
const char* to_cstr(FreqClass fc) {
    switch (fc) {
        case FreqClass::HotLoop:    return "hot_loop";
        case FreqClass::OuterLoop:  return "outer_loop";
        case FreqClass::Infrequent: return "infrequent";
        case FreqClass::Dynamic:    return "dynamic";
    }
    return "?";
}
const char* to_cstr(ChannelMap cm) {
    switch (cm) {
        case ChannelMap::StaticGraphAware: return "static_graph_aware";
        case ChannelMap::ModularHash:      return "modular_hash";
    }
    return "?";
}

Fabric fabric_from_str(const std::string& s) {
    if (s == "ib_mlx5") return Fabric::IbMlx5;
    if (s == "ofi_cxi") return Fabric::OfiCxi;
    throw std::runtime_error("unknown fabric '" + s + "'");
}
Path path_from_str(const std::string& s) {
    if (s == "ib_native")     return Path::IbNative;
    if (s == "ofi_triggered") return Path::OfiTriggered;
    if (s == "ofi_proxy")     return Path::OfiProxy;
    throw std::runtime_error("unknown path '" + s + "'");
}
PeerClass peer_from_str(const std::string& s) {
    if (s == "k_const")              return PeerClass::KConst;
    if (s == "k_from_topology_hint") return PeerClass::KFromTopologyHint;
    if (s == "dynamic")              return PeerClass::Dynamic;
    throw std::runtime_error("unknown peer_class '" + s + "'");
}
SizeClass size_from_str(const std::string& s) {
    if (s == "small_const")  return SizeClass::SmallConst;
    if (s == "medium_const") return SizeClass::MediumConst;
    if (s == "large_const")  return SizeClass::LargeConst;
    if (s == "dynamic")      return SizeClass::Dynamic;
    throw std::runtime_error("unknown size_class '" + s + "'");
}
FreqClass freq_from_str(const std::string& s) {
    if (s == "hot_loop")   return FreqClass::HotLoop;
    if (s == "outer_loop") return FreqClass::OuterLoop;
    if (s == "infrequent") return FreqClass::Infrequent;
    if (s == "dynamic")    return FreqClass::Dynamic;
    throw std::runtime_error("unknown freq_class '" + s + "'");
}
ChannelMap channel_map_from_str(const std::string& s) {
    if (s == "static_graph_aware") return ChannelMap::StaticGraphAware;
    if (s == "modular_hash")       return ChannelMap::ModularHash;
    throw std::runtime_error("unknown channel_map '" + s + "'");
}

// ---------- JSON loader ------------------------------------------------------

namespace {
[[noreturn]] void bail(const std::string& where, const std::string& reason) {
    throw std::runtime_error("policy.json: " + where + ": " + reason);
}

const mini_json::Value& required(const mini_json::Value& obj,
                                 const std::string& key,
                                 const std::string& where) {
    const mini_json::Value* v = obj.find(key);
    if (!v) bail(where, "missing required key '" + key + "'");
    return *v;
}

PlatformDesc parse_platform(const mini_json::Value& v) {
    if (!v.is_object()) bail("platform_desc", "expected object");
    PlatformDesc pd;
    pd.fabric     = fabric_from_str(required(v, "fabric", "platform_desc").as_string());
    pd.gpu_family = required(v, "gpu_family", "platform_desc").as_string();
    const mini_json::Value& caps = required(v, "nic_caps", "platform_desc");
    if (!caps.is_object()) bail("platform_desc.nic_caps", "expected object");
    if (const auto* x = caps.find("counter_max"))    pd.nic_caps.counter_max    = x->as_int();
    if (const auto* x = caps.find("dwq_max"))        pd.nic_caps.dwq_max        = x->as_int();
    if (const auto* x = caps.find("ctrs_per_op"))    pd.nic_caps.ctrs_per_op    = x->as_int();
    if (const auto* x = caps.find("qp_max_per_peer")) pd.nic_caps.qp_max_per_peer = x->as_int();
    return pd;
}

Decision parse_decision(const mini_json::Value& v) {
    if (!v.is_object()) bail("rule.then", "expected object");
    Decision d;
    d.path = path_from_str(required(v, "path", "rule.then").as_string());
    if (const auto* x = v.find("slot_depth"))    d.slot_depth = x->as_int();
    if (const auto* x = v.find("pool_size"))     d.pool_size  = x->as_int();
    if (const auto* x = v.find("channel_map"))   d.channel_map = channel_map_from_str(x->as_string());
    if (const auto* x = v.find("conservative"))  d.conservative = x->as_bool();
    return d;
}

Rule parse_rule(const mini_json::Value& v) {
    if (!v.is_object()) bail("rule", "expected object");
    Rule r;
    if (const auto* x = v.find("id")) r.id = x->as_int();
    if (const auto* cond = v.find("if")) {
        if (!cond->is_object()) bail("rule.if", "expected object");
        if (const auto* x = cond->find("peer_class")) r.if_peer = peer_from_str(x->as_string());
        if (const auto* x = cond->find("size_class")) r.if_size = size_from_str(x->as_string());
        if (const auto* x = cond->find("freq_class")) r.if_freq = freq_from_str(x->as_string());
    }
    r.then_decision = parse_decision(required(v, "then", "rule"));
    r.then_decision.rule_id = r.id;
    return r;
}
}  // namespace

Policy load_policy(const std::string& path) {
    mini_json::Value root = mini_json::parse_file(path);
    if (!root.is_object()) bail(path, "top level must be an object");
    Policy p;
    p.source_path = path;
    if (const auto* v = root.find("version"))         p.version = v->as_int();
    p.platform_desc = parse_platform(required(root, "platform_desc", path));
    const auto& rules_v = required(root, "rules", path);
    if (!rules_v.is_array()) bail(path + ".rules", "expected array");
    for (const auto& r : rules_v.as_array()) {
        p.rules.push_back(parse_rule(r));
        p.rules.back().then_decision.source = path;
    }
    if (p.rules.empty()) bail(path, "rules[] must contain at least a catch-all");
    return p;
}

// ---------- Evaluator --------------------------------------------------------

namespace {
// A rule's optional predicate matches iff it is unset (wildcard) OR
// the feature is Dynamic (caller couldn't resolve it — match anything)
// OR the predicate equals the feature literally.
template <class T>
bool matches(const std::optional<T>& pred, T feature, T dyn_sentinel) {
    if (!pred.has_value()) return true;
    if (feature == dyn_sentinel) return true;
    return *pred == feature;
}
}  // namespace

Decision eval(const Policy& p, const Features& f) {
    for (const auto& r : p.rules) {
        if (matches(r.if_peer, f.peer, PeerClass::Dynamic) &&
            matches(r.if_size, f.size, SizeClass::Dynamic) &&
            matches(r.if_freq, f.freq, FreqClass::Dynamic)) {
            return r.then_decision;
        }
    }
    // Should never happen: validation in load_policy ensures ≥1 rule.
    // Be defensive and emit a minimum-feasible decision rather than crash.
    Decision fallback;
    fallback.path = (p.platform_desc.fabric == Fabric::IbMlx5) ? Path::IbNative : Path::OfiProxy;
    fallback.pool_size = 16;
    fallback.conservative = true;
    fallback.rule_id = -2;  // sentinel for "exhausted rule list"
    fallback.source = p.source_path;
    return fallback;
}

// ---------- Cap feasibility --------------------------------------------------

bool check_feasible(const Decision& d, const PlatformDesc& pd, int P) {
    if (P <= 1) return true;
    // ib_native only legal on mlx5.
    if (d.path == Path::IbNative && pd.fabric != Fabric::IbMlx5) return false;
    if (pd.fabric != Fabric::OfiCxi) return true;  // no caps on mlx5 QPs here.

    // R(P) = ceil(log2(P))
    int r = 0;
    for (int x = 1; x < P; x <<= 1) ++r;
    if (r == 0) r = 1;

    const NicCaps& c = pd.nic_caps;
    if (d.path == Path::OfiTriggered) {
        if (d.slot_depth <= 0) return false;
        if ((long long)d.slot_depth * r > (long long)c.counter_max / c.ctrs_per_op) return false;
        if (d.pool_size <= 0) return false;
        if ((long long)d.pool_size * r > (long long)c.dwq_max) return false;
    } else if (d.path == Path::OfiProxy) {
        if (d.pool_size <= 0) return false;
        if ((long long)d.pool_size * r > (long long)c.dwq_max) return false;
    }
    return true;
}

}  // namespace policy
}  // namespace gicc
