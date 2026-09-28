#pragma once

#include <string>

namespace gicc::pass {

enum class Mode {
    Discover,
    FeatureExtract,
    Lower,
    OmpDwq,        // Phase 2: DWQ-from-OpenMP (post-inline, gated)
    ChunkAnalyze,  // report whether pipelined puts can be split per block
    ChunkLower,    // ... and split the ones that can
    WriteSummary,  // report every kernel's write sets
    PutDiscover,   // host: record the puts that follow a kernel launch
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
//   GICC_CHUNK_GRAIN   ("element")
// How chunk-lower sends a pipelined put: every store mirrored to the peer
// as it happens, or one send per distribute block as the block completes.
enum class ChunkGrain { Element, Block };

struct Config {
    Mode        mode           = Mode::Passthrough;
    Target      target         = Target::Auto;
    std::string metaDir        = "/tmp/gicc-meta";
    // GICC_META_DIR was given. The puts a kernel is followed by (put-discover)
    // are read back only from a directory the build named, never from the
    // default one, where another build's files may lie.
    bool        metaDirSet     = false;
    std::string featuresOut;
    std::string hintIn;
    ChunkGrain  chunkGrain     = ChunkGrain::Element;
};

// Returns a singleton Config, lazily populated from getenv() on first call.
const Config &getConfig();

const char *modeName(Mode m);
const char *targetName(Target t);

}  // namespace gicc::pass
