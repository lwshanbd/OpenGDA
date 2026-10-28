#pragma once
// The team and thread counts a tuning file chooses for one kernel's
// launches (GICC_TUNE_IN, "kernels": {"<kernel>": {"teams": N, "threads": M}}).
//
// OMP_NUM_TEAMS and OMP_TEAMS_THREAD_LIMIT set them for every kernel of a
// program, and a kernel that sends what it writes wants other counts than
// the rest: with one wave of teams every piece of its puts is written at the
// end, while fewer teams per wave let the first waves' pieces go out under
// the later ones, which the kernels that send nothing only pay for.
//
// Without a num_teams or thread_limit clause the counts are the
// implementation's to choose, so a conforming program is correct with any:
// the pass sets them on a launch only where clang passes 0, and records for
// the tuner, per launched kernel, which of the two the source fixes.

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

struct GICCLaunchShapePass : llvm::PassInfoMixin<GICCLaunchShapePass> {
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCLaunchShapePass"; }
};

}  // namespace gicc::pass
