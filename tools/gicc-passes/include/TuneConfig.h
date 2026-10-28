#pragma once

// Performance knobs an external tuner sets per site, and the report of the
// sites it may set them on.
//
// GICC_SITES_OUT names a file the passes append one JSON object per line to:
// every put site the chunk analysis read (its verdict and the options that
// are legal there) and every kernel whose distribute chunk can be widened.
// GICC_TUNE_IN names a JSON file of choices keyed by those ids:
//
//   {"sites":   {"<kernel>#<n>": {"pipeline": false, "grain": "block"}},
//    "kernels": {"<kernel>":     {"dist_chunk": 16384, "split": false}}}
//
// A site id of "*" applies to every site; a site's own entry overrides it
// field by field.
//
// A choice only ever selects among lowerings the analysis proved correct: a
// site that is not LEGAL is never pipelined whatever it asks, and one that
// asks for what its site cannot do is reported and ignored.

#include "GICCPassConfig.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>
#include <string>

namespace gicc::pass::tune {

struct SiteChoice {
    std::optional<bool>       pipeline;
    std::optional<ChunkGrain> grain;
};

// The choice for a put site, or an empty one when the file names none.
SiteChoice site(llvm::StringRef id);

// The distribute chunk chosen for a kernel, when the file names one.
std::optional<uint64_t> distChunk(llvm::StringRef kernel);

// Whether any kernel has a chunk chosen, so the dist-chunk pass runs even
// without GICC_DIST_CHUNK.
bool anyDistChunk();

// Whether a kernel may be split around a deferred fence (GICCSplitLaunch.h),
// when the file says.
std::optional<bool> split(llvm::StringRef kernel);

// Appends one record to GICC_SITES_OUT; does nothing when it is unset.
void record(llvm::json::Object rec);

}  // namespace gicc::pass::tune
