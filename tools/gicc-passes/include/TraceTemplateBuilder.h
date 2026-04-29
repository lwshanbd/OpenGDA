#pragma once

#include "KernelInventory.h"
#include "MetadataIO.h"

namespace gicc::pass {

// Convert a kernel's GICCKernelInfo (from collectGICCSites) into a
// KernelTemplate suitable for JSON serialization. Walks each call's
// operands via DefUseChain to produce ArgRef expressions; resolves
// guards from the call's controlling branch.
KernelTemplate buildKernelTemplate(const GICCKernelInfo &info);

}  // namespace gicc::pass
