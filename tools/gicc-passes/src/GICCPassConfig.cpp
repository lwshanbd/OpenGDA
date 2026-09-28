#include "GICCPassConfig.h"

#include "llvm/Support/ErrorHandling.h"

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
    if (s == "omp-dwq")         return Mode::OmpDwq;
    if (s == "chunk-analyze")   return Mode::ChunkAnalyze;
    if (s == "chunk-lower")     return Mode::ChunkLower;
    if (s == "write-summary")   return Mode::WriteSummary;
    if (s == "put-discover")    return Mode::PutDiscover;
    return Mode::Passthrough;
}

Target parseTarget(const char *e) {
    if (!e) return Target::Auto;
    std::string_view s(e);
    if (s == "ofi-triggered") return Target::OfiTriggered;
    if (s == "ib-native")     return Target::IbNative;
    return Target::Auto;
}

ChunkGrain parseGrain(const char *e) {
    if (!e || std::string_view(e) == "element") return ChunkGrain::Element;
    if (std::string_view(e) == "block") return ChunkGrain::Block;
    llvm::report_fatal_error("GICC_CHUNK_GRAIN must be 'element' or 'block'");
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
    c.metaDirSet  = std::getenv("GICC_META_DIR") != nullptr;
    c.featuresOut = envOr("GICC_FEATURES_OUT", "");
    c.hintIn      = envOr("GICC_HINT_IN", "");
    c.chunkGrain  = parseGrain(std::getenv("GICC_CHUNK_GRAIN"));
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
        case Mode::OmpDwq:          return "omp-dwq";
        case Mode::ChunkAnalyze:    return "chunk-analyze";
        case Mode::ChunkLower:      return "chunk-lower";
        case Mode::WriteSummary:    return "write-summary";
        case Mode::PutDiscover:     return "put-discover";
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
