#include "GICCPassConfig.h"

#include <cstdlib>
#include <string>
#include <string_view>

namespace gicc::pass {

namespace {

Mode parseMode(const char *e) {
    if (!e) return Mode::Passthrough;
    std::string_view s(e);
    if (s == "discover")        return Mode::Discover;
    if (s == "feature-extract") return Mode::FeatureExtract;
    if (s == "lower")           return Mode::Lower;
    return Mode::Passthrough;
}

Target parseTarget(const char *e) {
    if (!e) return Target::Auto;
    std::string_view s(e);
    if (s == "ofi-triggered") return Target::OfiTriggered;
    if (s == "ib-native")     return Target::IbNative;
    return Target::Auto;
}

std::string envOr(const char *name, const char *fallback) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string(fallback);
}

Config buildConfig() {
    Config c;
    c.mode        = parseMode(std::getenv("GICC_MODE"));
    c.target      = parseTarget(std::getenv("GICC_TARGET"));
    c.metaDir     = envOr("GICC_META_DIR", "/tmp/gicc-meta");
    c.featuresOut = envOr("GICC_FEATURES_OUT", "");
    c.hintIn      = envOr("GICC_HINT_IN", "");
    return c;
}

}  // namespace

const Config &getConfig() {
    static const Config cached = buildConfig();
    return cached;
}

const char *modeName(Mode m) {
    switch (m) {
        case Mode::Discover:        return "discover";
        case Mode::FeatureExtract:  return "feature-extract";
        case Mode::Lower:           return "lower";
        case Mode::Passthrough:     return "passthrough";
    }
    return "?";
}

const char *targetName(Target t) {
    switch (t) {
        case Target::OfiTriggered: return "ofi-triggered";
        case Target::IbNative:     return "ib-native";
        case Target::Auto:         return "auto";
    }
    return "?";
}

}  // namespace gicc::pass
