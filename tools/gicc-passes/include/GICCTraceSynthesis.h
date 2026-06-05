#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace gicc::pass {

struct KernelTemplate;  // from MetadataIO.h

class GICCTraceSynthesisPass
    : public llvm::PassInfoMixin<GICCTraceSynthesisPass> {
public:
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &);
    static llvm::StringRef name() { return "GICCTraceSynthesisPass"; }
};

// Reusable synthesis primitives, shared with the OpenMP host-discovery
// pass (Phase 2). getOrCreateTraceFn builds (or returns) the
// `@gicc_trace_<simpleName>` function whose signature is `ptr %rt` followed
// by one parameter per kernel formal EXCEPT the leading DeviceCtx* (formal
// 0). emitTraceBody fills in the body from the template's op list. Both
// live in MetadataIO.h's KernelTemplate.
llvm::Function *getOrCreateTraceFn(llvm::Module &M, const KernelTemplate &t);
void emitTraceBody(llvm::Module &M, llvm::Function *traceFn,
                   const KernelTemplate &t);

}  // namespace gicc::pass
