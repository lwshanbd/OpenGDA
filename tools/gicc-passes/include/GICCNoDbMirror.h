#pragma once
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
namespace gicc::pass {
// Makes ompx_put_no_db cheap without changing what it means. A posted put
// sends its source at the next quiet; this pass instruments every store the
// device code makes so that a store into a posted source is repeated into
// the same-node peer's destination and its words are marked as already
// sent. The quiet then sends only the words nobody marked. Writes the pass
// cannot repeat (atomics, memory intrinsics, calls it cannot see into)
// poison the epoch instead, and the quiet sends everything. The state the
// instrumentation reads is the device global `ompx__nodb` (gicc/omp.h).
// Runs in GICC_MODE=chunk-lower, on device modules only.
class GICCNoDbMirrorPass : public llvm::PassInfoMixin<GICCNoDbMirrorPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCNoDbMirrorPass"; }
};
}  // namespace gicc::pass
