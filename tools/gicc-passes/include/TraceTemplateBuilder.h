#pragma once

#include "KernelInventory.h"
#include "MetadataIO.h"

namespace llvm { class LoopInfo; }

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
KernelTemplate buildKernelTemplate(const GICCKernelInfo &info,
                                   llvm::LoopInfo      *LI = nullptr);

}  // namespace gicc::pass
