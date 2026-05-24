#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"

#include <cstdint>
#include <string>
#include <vector>

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
//
// `hostMirrored` (optional): per-formal bits indicating which formals
// carry host-side mirrors. When non-null, loads matching the
// `host_mirror[iv].field` pattern are accepted as HK (the trace
// synthesizer reads the field from the host mirror at trace time). See
// matchHostMirroredFieldLoad below.
HKResult isHK(llvm::Value *V, llvm::Function *kernelF,
              const std::vector<bool> *hostMirrored = nullptr);

// Result of recognizing a host-mirrored gep+load pattern on a kernel
// formal. The matcher rejects anything that doesn't reduce to
// `host_mirror_of(formal[iv]).field` with constant field indices.
struct FieldLoadMatch {
    bool        matched         = false;
    unsigned    formalIdx       = 0;
    int64_t     structElemSize  = 0;
    int64_t     fieldByteOffset = 0;
    std::string fieldTypeStr;     // "i32" / "i64" / "ptr" / ...
    llvm::Value *iv              = nullptr;  // the loop-iv expression
};

// Try to recognize `LI` as a host-mirrored field load. `LI` must load
// from a GEP (Pattern P1) or chained GEPs (Pattern P2) rooted at a
// kernel formal flagged as host_mirrored in `hostMirrored`. On success
// returns FieldLoadMatch with matched=true and the resolved
// (formalIdx, structElemSize, fieldByteOffset, fieldTypeStr, iv).
FieldLoadMatch
matchHostMirroredFieldLoad(const llvm::LoadInst *LI,
                           const llvm::Function *kernelF,
                           const std::vector<bool> &hostMirrored);

}  // namespace gicc::pass
