/**
 * policy.hpp — Schema and evaluator for the GICC-Pilot selector rule list.
 *
 * This file owns the public surface of libgicc_policy. Two consumers link it:
 *   1. The LLVM analysis pass (tools/gicc_pass) — loads the policy at compile
 *      time and evaluates it per call site to attach !gicc.decision metadata.
 *   2. The host-side runtime (src/gicc/runtime/...) — evaluates the same policy
 *      at rt.prepare() time to bind one pre-compiled variant per DeviceCtx.
 *
 * The schema is intentionally narrow (FINAL_PROPOSAL §"Selector Interface"):
 *   path ∈ {ib_native, ofi_triggered, ofi_proxy}
 *   ib_native:     no tunables
 *   ofi_triggered: slot_depth ∈ {2,4,8}, pool_size ∈ feasible
 *   ofi_proxy:     pool_size ∈ feasible, channel_map ∈ {static_graph_aware, modular_hash}
 *
 * Cap feasibility (check_feasible) enforces HPDC'26 §"NIC Resource Limits":
 *   slot_depth * R(P) <= counter_max / ctrs_per_op
 *   pool_size  * R(P) <= dwq_max
 * where R(P) = ceil(log2(P)) is the dissemination barrier round count.
 */
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace gicc {
namespace policy {

// ---------- Enumerations -----------------------------------------------------

enum class Fabric { IbMlx5, OfiCxi };

enum class Path { IbNative, OfiTriggered, OfiProxy };

enum class PeerClass { KConst, KFromTopologyHint, Dynamic };

enum class SizeClass { SmallConst, MediumConst, LargeConst, Dynamic };

enum class FreqClass { HotLoop, OuterLoop, Infrequent, Dynamic };

enum class ChannelMap { StaticGraphAware = 0, ModularHash = 1 };

// ---------- Core types -------------------------------------------------------

struct NicCaps {
    int counter_max    = 2047;
    int dwq_max        = 256;
    int ctrs_per_op    = 2;
    int qp_max_per_peer = 0;  ///< 0 = N/A (only meaningful on ib_mlx5)
};

struct PlatformDesc {
    Fabric       fabric;
    std::string  gpu_family;
    NicCaps      nic_caps;
};

/// Per-call-site feature tuple. A class marked Dynamic was not statically
/// resolvable; the evaluator must treat it as a wildcard match.
struct Features {
    PeerClass peer = PeerClass::Dynamic;
    SizeClass size = SizeClass::Dynamic;
    FreqClass freq = FreqClass::Dynamic;
};

/// Tagged-union decision. Only the fields meaningful for the selected path
/// are consulted; others must be left at their zero default to avoid schema
/// drift. See FINAL_PROPOSAL §"Selector Interface" for the authoritative
/// tagged-union contract.
struct Decision {
    Path path          = Path::OfiProxy;
    int  slot_depth    = 0;  ///< ofi_triggered only
    int  pool_size     = 0;  ///< ofi_triggered, ofi_proxy
    ChannelMap channel_map = ChannelMap::StaticGraphAware;  ///< ofi_proxy only
    bool conservative  = false;
    int  rule_id       = -1;     ///< policy rule that matched, -1 = default
    std::string source;          ///< e.g. "policy_tioga.json"
};

struct Rule {
    int id = 0;
    std::optional<PeerClass> if_peer;
    std::optional<SizeClass> if_size;
    std::optional<FreqClass> if_freq;
    Decision then_decision;
};

struct Policy {
    int version = 1;
    PlatformDesc platform_desc;
    std::vector<Rule> rules;
    std::string source_path;  ///< original file path, used by --gicc-explain
};

// ---------- API --------------------------------------------------------------

/// Parse a policy file. Throws std::runtime_error with a human-readable
/// message on any schema violation.
Policy load_policy(const std::string& path);

/// Walk the rule list; return the first rule whose `if` conjunction matches
/// `f`. Dynamic features match any `if_*` predicate. The last rule is
/// expected to be a catch-all (empty `if`) and marks its decision with
/// `conservative = true`.
Decision eval(const Policy& p, const Features& f);

/// True iff `d` respects the platform's NIC caps at the given rank count
/// `P`. Used both at compile time (in the LLVM pass) and offline (in the
/// rule-list fitter).
bool check_feasible(const Decision& d, const PlatformDesc& pd, int P);

// ---------- String conversions for --gicc-explain ---------------------------

const char* to_cstr(Fabric);
const char* to_cstr(Path);
const char* to_cstr(PeerClass);
const char* to_cstr(SizeClass);
const char* to_cstr(FreqClass);
const char* to_cstr(ChannelMap);

/// Parse an enum from a JSON string (throws on unknown literal).
Fabric      fabric_from_str(const std::string&);
Path        path_from_str(const std::string&);
PeerClass   peer_from_str(const std::string&);
SizeClass   size_from_str(const std::string&);
FreqClass   freq_from_str(const std::string&);
ChannelMap  channel_map_from_str(const std::string&);

}  // namespace policy
}  // namespace gicc
