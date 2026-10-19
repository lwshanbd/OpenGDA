#pragma once
// Widens the distribute chunk of every offload kernel with a combined
// `teams distribute parallel for`, GICC calls or not, when GICC_DIST_CHUNK
// names a chunk (in iterations).
//
// Without a dist_schedule clause clang deals the loop to teams round robin
// one element per thread at a time: a chunk of 128 iterations for 128
// threads, each its own pass through the distribute loop and its own
// for_static_init. On the Jacobi stencil (16384^2, H100) that schedule
// takes 2.98 s; dist_schedule(static, 8192) in the source takes 1.84. This
// pass makes the same change without touching the source: the chunk
// becomes a multiple of clang's of up to GICC_DIST_CHUNK iterations, but
// small enough that every team still gets OmpKernel::kChunksPerTeam of
// them (an iteration may be a whole row).
//
// The chunk a team runs is not observable to a program that never asks
// which team it is in, so a kernel that does (omp_get_team_num, a block id)
// is left alone; so is one whose schedule is not chunked
// (dist_schedule(static) without a chunk), and one chunk-lower already
// widened. A dist_schedule(static, c) the source wrote itself looks the same
// as clang's default in IR, and is widened as well.

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

struct GICCDistChunkPass : llvm::PassInfoMixin<GICCDistChunkPass> {
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCDistChunkPass"; }
};

}  // namespace gicc::pass
