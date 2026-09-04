#pragma once

#include <string>

namespace gicc::pass {

enum class Mode {
    Discover,
    FeatureExtract,
    Lower,
    Passthrough,
};

enum class Target {
    OfiTriggered,   // AMDGCN + Slingshot/CXI
    IbNative,       // NVPTX + ConnectX/IB
    Auto,           // infer from module triple
};

// Plugin-wide configuration. Read once from environment variables on
// first access; subsequent calls return the cached value. Env vars
// (instead of -mllvm cl::opt) sidestep a Clang/LLVM 19 ordering bug
// where -fpass-plugin loads the plugin after -mllvm parsing has run,
// so plugin-defined cl::opt's would be reported as unknown options.
//
// Recognized variables (all optional; defaults shown):
//   GICC_MODE          ("passthrough")
//   GICC_TARGET        ("auto")
//   GICC_META_DIR      ("/tmp/gicc-meta")
//   GICC_FEATURES_OUT  ("")
//   GICC_HINT_IN       ("")
//   GICC_COLLECTIVE_OUT     ("")
//   GICC_COLLECTIVE_HINT_IN ("")
//   GICC_COLLECTIVE_ONLY    (false)
//   GICC_PRODUCER_FISSION_ORACLE (false)
struct Config {
    Mode        mode           = Mode::Passthrough;
    Target      target         = Target::Auto;
    std::string metaDir        = "/tmp/gicc-meta";
    std::string featuresOut;
    std::string hintIn;
    std::string collectiveOut;
    std::string collectiveHintIn;
    // Restrict the automatically attached pipeline to collective planning.
    // Collective-policy experiments intentionally have no ordinary
    // communication hint or synthesized host trace; running device lowering
    // in that situation would erase proxy put/quiet calls without replacing
    // them. Named passes remain available for focused tests.
    bool        collectiveOnly = false;
    // Backward-compatible research-oracle switch. The production selector
    // instead activates the passes with an explicit
    // PRODUCER_FRONTIER_TWO_PHASE site transform. Both paths still re-prove
    // the DWQ route and every compiler legality fact.
    bool        producerFissionOracle = false;
};

// Returns a singleton Config, lazily populated from getenv() on first call.
const Config &getConfig();

const char *modeName(Mode m);
const char *targetName(Target t);

}  // namespace gicc::pass
