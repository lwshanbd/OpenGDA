#pragma once

#include "KernelInventory.h"
#include "MetadataIO.h"

namespace llvm { class LoopInfo; class DominatorTree; }

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
// call site. Without DT the field stays at its -1 sentinel.
KernelTemplate buildKernelTemplate(const GICCKernelInfo &info,
                                   llvm::LoopInfo       *LI = nullptr,
                                   llvm::DominatorTree  *DT = nullptr);

}  // namespace gicc::pass
