#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"

#include <string>

namespace gicc::pass {

struct HKResult {
    bool                ok           = true;
    std::string         failReason;          // "depends on threadIdx.x" etc.
    llvm::Instruction  *failOrigin   = nullptr;
};

// Per spec §5.2: a value is host-knowable when it's a constant, a kernel
// formal parameter, or a pure composition of host-knowable values via
// arithmetic, casts, GEPs, PHIs, or HK-pure intrinsic calls. Reads from
// device memory and references to thread/block IDs make a value
// non-host-knowable.
HKResult isHK(llvm::Value *V, llvm::Function *kernelF);

}  // namespace gicc::pass
