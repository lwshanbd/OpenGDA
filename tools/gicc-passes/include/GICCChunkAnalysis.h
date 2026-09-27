#pragma once
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
namespace gicc::pass {
// Decides, per ompx_pipelined_put in an OpenMP offload kernel, whether the
// put can be split into sends issued as the kernel's data becomes final --
// and reports why not when it cannot. GICC_MODE=chunk-analyze leaves the IR
// unchanged; chunk-lower rewrites each splittable put at the grain
// GICC_CHUNK_GRAIN selects (element by default, or block; see
// gicc/omp_pipeline.h) and makes an unsplittable one a compile error.
class GICCChunkAnalysisPass
    : public llvm::PassInfoMixin<GICCChunkAnalysisPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCChunkAnalysisPass"; }
};

// Runs before any inlining in chunk-lower mode: keeps the outlined parallel
// regions of kernels with an ompx_pipelined_put out of their wrappers, so
// the element-grain lowering rewrites the code the workers run. The
// lowering restores the regions' inlining attributes.
class GICCChunkPrepPass : public llvm::PassInfoMixin<GICCChunkPrepPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCChunkPrepPass"; }
};
}  // namespace gicc::pass
