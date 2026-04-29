#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"

#include <string>

namespace gicc::pass {

// Build site_id of form "<TU_basename>:<line>:<kernel_simple>::<call_index>".
// TU_basename + line come from I->getDebugLoc(); when debug info is missing,
// the corresponding fields are "?".
std::string buildSiteId(const llvm::Instruction *I,
                        const llvm::Function    *kernel,
                        unsigned                 callIndex);

}  // namespace gicc::pass
