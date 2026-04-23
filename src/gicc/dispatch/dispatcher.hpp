/**
 * dispatcher.hpp — Host-side multiversion dispatcher (M2-G).
 *
 * This is the runtime companion to the LLVM pass and the policy evaluator.
 * At `rt.prepare()` time the runtime calls Dispatcher::bind() with the
 * prepare-time-observable features (peer and buffer size at minimum, and
 * an optional topology hint supplied by the caller). The dispatcher runs
 * the same libgicc_policy rule list the LLVM pass uses, returns the
 * Decision that binds this DeviceCtx to one pre-compiled variant index,
 * and — when configured — dumps the decision trace for diagnostics.
 *
 * Scope note. The *device-side* variant-specific code path (switch on
 * ctx->variant_idx inside gicc::put / gicc::barrier) is future work tied
 * to the template-specialisation refactor. M2-G ships only the host-side
 * selection logic + microbenchmark for the C2 overhead claim (target:
 * median ≤ 0.5 ms, p99 ≤ 2 ms). The variant index produced here can be
 * consumed by a later device-side specialisation without API changes.
 */
#pragma once

#include "gicc/policy/policy.hpp"

#include <cstdint>
#include <string>

namespace gicc {
namespace dispatch {

/// Observable features at the prepare() call site.
struct PrepareObs {
    int         peer_rank      = -1;   ///< destination peer, -1 if unknown
    std::size_t buffer_size    = 0;    ///< registered buffer size, 0 if unknown
    std::size_t transfer_size  = 0;    ///< if different from buffer size, 0 otherwise
    bool        has_topology_hint = false;
};

/// The binding returned by bind(). variant_idx is what the device struct
/// should store; decision is the full tagged-union for --gicc-explain and
/// for downstream introspection.
struct Binding {
    uint8_t          variant_idx = 0;
    policy::Decision decision{};
};

/// A Dispatcher is constructed once per gicc::Runtime, owns a loaded
/// policy, and is called O(nprepares) times during program lifetime.
class Dispatcher {
public:
    /// Load the platform policy. If `path` is empty, consult the
    /// GICC_POLICY_FILE env var. If neither is set, bind() returns a
    /// catch-all conservative decision (variant_idx == 0).
    explicit Dispatcher(const std::string& policy_path = "");

    /// Perform the per-site feature classification + policy eval. Cheap:
    /// target overhead is ≤ 0.5 ms median per call (C2 gate).
    Binding bind(const PrepareObs& obs);

    /// Diagnostic knob. When true, every bind() call prints a one-line
    /// summary to stderr. Off by default; can be flipped from env at
    /// construction via GICC_EXPLAIN.
    void set_explain(bool on) { explain_ = on; }

    bool policy_loaded() const { return policy_loaded_; }
    const policy::Policy& policy() const { return policy_; }

private:
    policy::Features classify(const PrepareObs& obs) const;

    policy::Policy policy_;
    bool policy_loaded_ = false;
    bool explain_       = false;
};

}  // namespace dispatch
}  // namespace gicc
