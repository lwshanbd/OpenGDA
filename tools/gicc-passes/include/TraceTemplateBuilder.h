#pragma once

#include "KernelInventory.h"
#include "MetadataIO.h"

namespace llvm {
class LoopInfo;
class DominatorTree;
class PostDominatorTree;
class ScalarEvolution;
}

namespace gicc::pass {

// Convert a kernel's GICCKernelInfo (from collectGICCSites) into a
// KernelTemplate suitable for JSON serialization. Walks each call's
// operands via DefUseChain to produce ArgRef expressions; resolves
// guards from the call's controlling branch.
//
// `LI` (optional): if provided, the builder uses LLVM's LoopInfo to
// detect call sites inside canonical loops `for(i=lo; i<hi; i+=step)`,
// fills `OpTemplate::loop`, and rewrites iv-dependent ArgRefs into a
// LoopIv-rooted expression so the host trace can substitute the
// runtime loop counter. Without LI all calls appear as non-loop ops
// (the legacy behavior, used by lit tests on tiny IR).
//
// `DT` (optional): if provided, populates `OpTemplate::compute_before`
// with the count of arithmetic / FP instructions in BBs dominating the
// call site, and `OpTemplate::compute_after` with the count between the
// call and the kernel's completion point (its static issue-to-first-use
// distance). Without DT both stay at their -1 sentinel.
//
// `SE` (optional): if provided, populates `OpTemplate::trip_count` with
// the enclosing loop's compile-time trip count when one can be proven.
KernelTemplate buildKernelTemplate(const GICCKernelInfo  &info,
                                   llvm::LoopInfo        *LI = nullptr,
                                   llvm::DominatorTree   *DT = nullptr,
                                   llvm::ScalarEvolution *SE = nullptr,
                                   llvm::PostDominatorTree *PDT = nullptr);

}  // namespace gicc::pass
