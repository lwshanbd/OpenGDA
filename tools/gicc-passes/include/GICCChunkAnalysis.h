#pragma once
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
namespace gicc::pass {
// Decides, per ompx_pipelined_put in an OpenMP offload kernel, whether the
// put can be split along the kernel's `distribute` blocks and issued as
// each block finishes -- and reports why not when it cannot. In
// GICC_MODE=chunk-analyze the IR is left unchanged; in chunk-lower each
// splittable put becomes one ompx__block_put per block (gicc/omp_pipeline.h)
// and an unsplittable one is a compile error.
class GICCChunkAnalysisPass
    : public llvm::PassInfoMixin<GICCChunkAnalysisPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCChunkAnalysisPass"; }
};
}  // namespace gicc::pass
