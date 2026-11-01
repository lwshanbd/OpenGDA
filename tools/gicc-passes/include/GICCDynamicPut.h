#pragma once
// A put after a kernel sent from the kernel by run-time counting.
//
// GICCChunkAnalysis sends a put after the kernel (GICCAfterPut.h) from the
// kernel only when it can prove which iterations write which bytes of the
// source: a distribute loop whose stores are affine in its iteration. A
// kernel whose stores land where data says -- an MoE expert writing the rows
// of the tokens routed to it, at offsets read from the routing -- defeats
// that proof. For such a kernel, when the tuning file sets "dynamic": true
// on it (or GICC_DYNAMIC=1 for every kernel), this pass, in chunk-lower mode
// on the device side:
//
//   - calls ompx__dyn_begin at the start of the kernel's user code, which
//     reads the host's post of the put into team memory;
//   - calls ompx__dyn_count after every store to memory that is neither a
//     stack slot nor team memory, in the kernel and in every function only
//     it reaches (its outlined parallel regions), and inlines it.
//
// At run time the put's range is cut into pieces and each piece goes out
// once the stores into it add up to its size; for an ompx_put_signal the
// signal follows the last piece on the same QP (gicc/omp_pipeline_impl.h).
// That needs every byte of the range written exactly once, which the tuning
// file asserts and the pass cannot check. A kernel that writes memory any
// other way -- memset or memcpy, an atomic, a call the pass cannot see into
// -- or shares a function with another kernel is left alone, and the put
// stays on the host.

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

struct GICCDynamicPutPass : llvm::PassInfoMixin<GICCDynamicPutPass> {
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
    static llvm::StringRef name() { return "GICCDynamicPutPass"; }
    static bool isRequired() { return true; }
};

}  // namespace gicc::pass
